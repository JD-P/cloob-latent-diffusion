#!/usr/bin/env python3

"""Classifier-free guidance sampling from a diffusion model."""

import argparse
from functools import partial
from pathlib import Path
import sys

from omegaconf import OmegaConf
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import trange

from CLIP import clip
import train_latent_diffusion as train

sys.path.append('./cloob-training')
from cloob_training import model_pt, pretrained

sys.path.append('./latent-diffusion')
import ldm.models.autoencoder

sys.path.append('./v-diffusion-pytorch')
from diffusion import sampling, utils

MODULE_DIR = Path(__file__).resolve().parent


def parse_prompt(prompt, default_weight=3.):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', default_weight][len(vals):]
    return vals[0], float(vals[1])


def resize_and_center_crop(image, size):
    fac = max(size[0] / image.size[0], size[1] / image.size[1])
    image = image.resize((int(fac * image.size[0]), int(fac * image.size[1])), Image.LANCZOS)
    return TF.center_crop(image, size[::-1])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('prompts', type=str, nargs="+",
                   help='the text prompt to use')
    p.add_argument('--images', type=str, default=[], nargs="+",
                   help='the image prompts to use')
    # p.add_argument('--batch-size', '-bs', type=int, default=1,
    #                help='the number of images per batch')
    p.add_argument('--autoencoder', type=str,
                   help='the autoencoder to use, e.g. "model" for model[.ckpt] and model[.yaml]')
    p.add_argument('--checkpoint', type=str,
                   help='the checkpoint to use')
    p.add_argument('--cond-scale', type=float, default=3.,
                   help='the conditioning scale')
    p.add_argument('--device', type=str,
                   help='the device to use')
    p.add_argument('--eta', type=float, default=0.,
                   help='the amount of noise to add during sampling (0-1)')
    p.add_argument('--method', type=str, default='plms',
                   choices=['ddpm', 'ddim', 'prk', 'plms', 'pie', 'plms2'],
                   help='the sampling method to use')
    p.add_argument('-n', type=int, default=4,
                   help='the number of images to sample')
    p.add_argument('--seed', type=int, default=0,
                   help='the random seed')
    # p.add_argument('--size', type=int, nargs=2,
    #                help='the output image size')
    p.add_argument('--steps', type=int, default=50,
                   help='the number of timesteps')
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    ae_config = OmegaConf.load(args.autoencoder + '.yaml')
    ae_model = ldm.models.autoencoder.AutoencoderKL(**ae_config.model.params)
    ae_model.eval().requires_grad_(False).to("cuda:0")
    ae_model.init_from_ckpt(args.autoencoder + '.ckpt')
    n_ch, side_y, side_x = 4, 32, 32
    checkpoint = args.checkpoint
    model = train.DiffusionModel(192, [4,4,8,8], autoencoder_scale=torch.tensor(2.55))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model = model.to(device).eval().requires_grad_(False)

    cloob_config = pretrained.get_config('cloob_laion_400m_vit_b_16_16_epochs')
    cloob = model_pt.get_pt_model(cloob_config)
    checkpoint = pretrained.download_checkpoint(cloob_config)
    cloob.load_state_dict(model_pt.get_pt_params(cloob_config, checkpoint))
    cloob.eval().requires_grad_(False).to('cuda')

    zero_embed = torch.zeros([1, cloob.config['d_embed']], device=device)
    target_embeds, weights = [zero_embed], []

    for prompt in args.prompts:
        txt, weight = parse_prompt(prompt)
        target_embeds.append(cloob.text_encoder(cloob.tokenize(txt).to(device)).float())
        weights.append(weight)

    for prompt in args.images:
        path, weight = parse_prompt(prompt)
        img = Image.open(utils.fetch(path)).convert('RGB')
        clip_size = cloob.config['image_encoder']['image_size']
        img = resize_and_center_crop(img, (clip_size, clip_size))
        batch = TF.to_tensor(img)[None].to(device)
        embed = F.normalize(cloob.image_encoder(cloob.normalize(batch)).float(), dim=-1)
        target_embeds.append(embed)
        weights.append(weight)

    weights = torch.tensor([1 - sum(weights), *weights], device=device)

    torch.manual_seed(args.seed)

    def cfg_model_fn(x, t):
        n = x.shape[0]
        n_conds = len(target_embeds)
        x_in = x.repeat([n_conds, 1, 1, 1])
        t_in = t.repeat([n_conds])
        clip_embed_in = torch.cat([*target_embeds]).repeat_interleave(n, 0)
        vs = model(x_in, t_in, clip_embed_in).view([n_conds, n, *x.shape[1:]])
        v = vs.mul(weights[:, None, None, None, None]).sum(0)
        return v

    def run(x, steps):
        if args.method == 'ddpm':
            return sampling.sample(cfg_model_fn, x, steps, 1., {})
        if args.method == 'ddim':
            return sampling.sample(cfg_model_fn, x, steps, args.eta, {})
        if args.method == 'prk':
            return sampling.prk_sample(cfg_model_fn, x, steps, {})
        if args.method == 'plms':
            return sampling.plms_sample(cfg_model_fn, x, steps, {})
        if args.method == 'pie':
            return sampling.pie_sample(cfg_model_fn, x, steps, {})
        if args.method == 'plms2':
            return sampling.plms2_sample(cfg_model_fn, x, steps, {})
        assert False

    def run_all(n):
        batch_size = n
        x = torch.randn([n, n_ch, side_y, side_x], device=device)
        t = torch.linspace(1, 0, args.steps + 1, device=device)[:-1]
        steps = utils.get_spliced_ddpm_cosine_schedule(t)
        for i in trange(0, n, batch_size):
            cur_batch_size = min(n - i, batch_size)
            out_latents = run(x[i:i+cur_batch_size], steps)
            outs = ae_model.decode(out_latents * torch.tensor(2.55).to("cuda:0"))
            for j, out in enumerate(outs):
                utils.to_pil_image(out).save(f'out_{i + j:05}.png')

    try:
        run_all(args.n)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
