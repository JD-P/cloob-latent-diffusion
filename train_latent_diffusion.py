#!/usr/bin/env python3

import argparse
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import math
import random
from pathlib import Path
import json
import pickle
import sys

from omegaconf import OmegaConf
from PIL import Image
sys.path.append('./taming-transformers')
from taming.models import cond_transformer, vqgan
sys.path.append('./latent-diffusion')
import ldm.models.autoencoder
sys.path.append('./v-diffusion-pytorch')
from diffusion import sampling
from diffusion import utils as diffusion_utils
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from torchvision.io import read_image
from torchvision import transforms, utils, datasets
from torchvision.transforms import functional as TF
import torchvision.transforms as T
from tqdm import trange
import wandb

from CLIP import clip

sys.path.append('./cloob-training')
from cloob_training import model_pt, pretrained

# Define utility functions

def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model

@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


# Define the diffusion noise schedule

def get_alphas_sigmas(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


# Define the model (a residual U-Net)

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResLinearBlock(ResidualBlock):
    def __init__(self, f_in, f_mid, f_out, is_last=False):
        skip = None if f_in == f_out else nn.Linear(f_in, f_out, bias=False)
        super().__init__([
            nn.Linear(f_in, f_mid),
            nn.ReLU(inplace=True),
            nn.Linear(f_mid, f_out),
            nn.ReLU(inplace=True) if not is_last else nn.Identity(),
        ], skip)


class Modulation2d(nn.Module):
    def __init__(self, state, feats_in, c_out):
        super().__init__()
        self.state = state
        self.layer = nn.Linear(feats_in, c_out * 2, bias=False)

    def forward(self, input):
        scales, shifts = self.layer(self.state['cond']).chunk(2, dim=-1)
        return torch.addcmul(shifts[..., None, None], input, scales[..., None, None] + 1)


class ResModConvBlock(ResidualBlock):
    def __init__(self, state, feats_in, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.GroupNorm(1, c_mid, affine=False),
            Modulation2d(state, feats_in, c_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.GroupNorm(1, c_out, affine=False) if not is_last else nn.Identity(),
            Modulation2d(state, feats_in, c_out) if not is_last else nn.Identity(),
            nn.ReLU(inplace=True) if not is_last else nn.Identity(),
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)
        self.weight.requires_grad_(False)
        # self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SelfAttention2d(nn.Module):
    def __init__(self, c_in, n_head=1, dropout_rate=0.1):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = nn.GroupNorm(1, c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)
        self.dropout = nn.Identity()  # nn.Dropout2d(dropout_rate, inplace=True)

    def forward(self, input):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(self.norm(input))
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3]**-0.25
        att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.dropout(self.out_proj(y))


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class DiffusionModel(nn.Module):
    def __init__(self, base_channels, cm, autoencoder_scale=1):
        super().__init__()
        c = base_channels  # The base channel count
        cs = [c * cm[0], c * cm[1], c * cm[2], c * cm[3]]

        self.mapping_timestep_embed = FourierFeatures(1, 128)
        self.mapping = nn.Sequential(
            ResLinearBlock(512 + 128, 1024, 1024),
            ResLinearBlock(1024, 1024, 1024, is_last=True),
        )

        with torch.no_grad():
            for param in self.mapping.parameters():
                param *= 0.5**0.5

        self.state = {}
        conv_block = partial(ResModConvBlock, self.state, 1024)

        self.register_buffer('autoencoder_scale', autoencoder_scale)
        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.net = nn.Sequential( # 32x32
            conv_block(4 + 16, cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            SkipBlock([
                self.down,  # 16x16
                conv_block(cs[0], cs[1], cs[1]),
                SelfAttention2d(cs[1], cs[1] // 64),
                conv_block(cs[1], cs[1], cs[1]),
                SelfAttention2d(cs[1], cs[1] // 64),
                conv_block(cs[1], cs[1], cs[1]),
                SelfAttention2d(cs[1], cs[1] // 64),
                conv_block(cs[1], cs[1], cs[1]),
                SelfAttention2d(cs[1], cs[1] // 64),
                SkipBlock([
                    self.down,  # 8x8
                    conv_block(cs[1], cs[2], cs[2]),
                    SelfAttention2d(cs[2], cs[2] // 64),
                    conv_block(cs[2], cs[2], cs[2]),
                    SelfAttention2d(cs[2], cs[2] // 64),
                    conv_block(cs[2], cs[2], cs[2]),
                    SelfAttention2d(cs[2], cs[2] // 64),
                    conv_block(cs[2], cs[2], cs[2]),
                    SelfAttention2d(cs[2], cs[2] // 64),
                    SkipBlock([
                        self.down,  # 4x4
                        conv_block(cs[2], cs[3], cs[3]),
                        SelfAttention2d(cs[3], cs[3] // 64),
                        conv_block(cs[3], cs[3], cs[3]),
                        SelfAttention2d(cs[3], cs[3] // 64),
                        conv_block(cs[3], cs[3], cs[3]),
                        SelfAttention2d(cs[3], cs[3] // 64),
                        conv_block(cs[3], cs[3], cs[3]),
                        SelfAttention2d(cs[3], cs[3] // 64),
                        conv_block(cs[3], cs[3], cs[3]),
                        SelfAttention2d(cs[3], cs[3] // 64),
                        conv_block(cs[3], cs[3], cs[3]),
                        SelfAttention2d(cs[3], cs[3] // 64),
                        conv_block(cs[3], cs[3], cs[3]),
                        SelfAttention2d(cs[3], cs[3] // 64),
                        conv_block(cs[3], cs[3], cs[2]),
                        SelfAttention2d(cs[2], cs[2] // 64),
                        self.up,
                    ]),
                    conv_block(cs[2] * 2, cs[2], cs[2]),
                    SelfAttention2d(cs[2], cs[2] // 64),
                    conv_block(cs[2], cs[2], cs[2]),
                    SelfAttention2d(cs[2], cs[2] // 64),
                    conv_block(cs[2], cs[2], cs[2]),
                    SelfAttention2d(cs[2], cs[2] // 64),
                    conv_block(cs[2], cs[2], cs[1]),
                    SelfAttention2d(cs[1], cs[1] // 64),
                    self.up,
                ]),
                conv_block(cs[1] * 2, cs[1], cs[1]),
                SelfAttention2d(cs[1], cs[1] // 64),
                conv_block(cs[1], cs[1], cs[1]),
                SelfAttention2d(cs[1], cs[1] // 64),
                conv_block(cs[1], cs[1], cs[1]),
                SelfAttention2d(cs[1], cs[1] // 64),
                conv_block(cs[1], cs[1], cs[0]),
                SelfAttention2d(cs[0], cs[0] // 64),
                self.up,
            ]),
            conv_block(cs[0] * 2, cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], 4, is_last=True),)
        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5**0.5

    def forward(self, input, t, clip_embed):
        clip_embed = F.normalize(clip_embed, dim=-1) * clip_embed.shape[-1]**0.5
        mapping_timestep_embed = self.mapping_timestep_embed(t[:, None])
        self.state['cond'] = self.mapping(torch.cat([clip_embed, mapping_timestep_embed], dim=1))
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        out = self.net(torch.cat([input, timestep_embed], dim=1))
        self.state.clear()
        return out


class TokenizerWrapper:
    def __init__(self, max_len=None):
        self.tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        self.sot_token = self.tokenizer.encoder['<|startoftext|>']
        self.eot_token = self.tokenizer.encoder['<|endoftext|>']
        self.context_length = 77
        self.max_len = self.context_length - 2 if max_len is None else max_len

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        result = torch.zeros([len(texts), self.context_length], dtype=torch.long)
        for i, text in enumerate(texts):
            tokens_trunc = self.tokenizer.encode(text)[:self.max_len]
            tokens = [self.sot_token, *tokens_trunc, self.eot_token]
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result


class ToMode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class LightningDiffusion(pl.LightningModule):
    def __init__(self, cloob_checkpoint, vqgan_model, train_dl, ae_model, autoencoder_scale,
                 base_channels=128, channel_multipliers="4,4,8,8", ema_decay_at=200000):
        super().__init__()
        """
        # autoencoder
        ae_config = OmegaConf.load(vqgan_model + '.yaml')
        self.ae_model = ldm.models.autoencoder.AutoencoderKL(**ae_config.model.params)
        # ae_model = ldm.models.autoencoder.VQModel(**ae_config.model.params)
        self.ae_model.eval().requires_grad_(False)
        self.ae_model.init_from_ckpt(vqgan_model + '.ckpt')

        # autoencoder scale
        print("Getting autoencoder scale for {}.ckpt".format(vqgan_model))
        self.scale_factor = 1
        var_accum = 0.
        dl_iter = iter(train_dl)
        for i in trange(32):
            batch = next(dl_iter)
            reals, _ = batch
            reals = self.encode(reals * 2 - 1)
            var_accum += reals.var().item()
        """
        # autoencoder
        self.ae_model = ae_model
        self.register_buffer('scale_factor', autoencoder_scale)

        # CLOOB
        cloob_config = pretrained.get_config(cloob_checkpoint)
        self.cloob = model_pt.get_pt_model(cloob_config)
        checkpoint = pretrained.download_checkpoint(cloob_config)
        self.cloob.load_state_dict(model_pt.get_pt_params(cloob_config, checkpoint))
        self.cloob.eval().requires_grad_(False).to('cuda')

        # Diffusion model
        self.model = DiffusionModel(base_channels,
                                    [int(i) for i in channel_multipliers.strip().split(",")],
                                    autoencoder_scale)
        self.model_ema = deepcopy(self.model)
        self.ema_decay_at = ema_decay_at

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    def encode(self, image):
        return self.ae_model.encode(image).sample() / self.scale_factor

    def decode(self, latent):
        return self.ae_model.decode(latent * self.scale_factor)
        
    def forward(self, *args, **kwargs):
        if self.training:
            return self.model(*args, **kwargs)
        return self.model_ema(*args, **kwargs)

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=3e-5, weight_decay=0.01)
        # return optim.AdamW(self.model.parameters(), lr=5e-6, weight_decay=0.01)

    def eval_batch(self, batch):
        reals, _ = batch
        cloob_reals = F.interpolate(reals, (224, 224), mode='bicubic', align_corners=False)
        cond = self.cloob.image_encoder(self.cloob.normalize(cloob_reals))
        del cloob_reals
        reals = self.encode(reals * 2 - 1)
        p = torch.rand([reals.shape[0], 1], device=reals.device)
        cond = torch.where(p > 0.2, cond, torch.zeros_like(cond))

        # Sample timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(reals)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas

        # Compute the model output and the loss.
        v = self(noised_reals, t, cond)
        return F.mse_loss(v, targets)

    def training_step(self, batch, batch_idx):
        loss = self.eval_batch(batch)
        log_dict = {'train/loss': loss.detach()}
        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.trainer.global_step < 20000:
            decay = 0.99
        elif self.trainer.global_step < self.ema_decay_at:
            decay = 0.999
        else:
            decay = 0.9999
        ema_update(self.model, self.model_ema, decay)


class DemoCallback(pl.Callback):
    def __init__(self, prompts, prompts_toks, demo_every=2000):
        super().__init__()
        self.prompts = prompts
        self.prompts_toks = prompts_toks
        self.demo_every = demo_every

    @rank_zero_only
    @torch.no_grad()
    def on_batch_end(self, trainer, module):
        if trainer.global_step % self.demo_every != 0:
            return

        lines = [f'({i // 4}, {i % 4}) {line}' for i, line in enumerate(self.prompts)]
        lines_text = '\n'.join(lines)
        Path('demo_prompts_out.txt').write_text(lines_text)

        noise = torch.randn([16, 4, 32, 32], device=module.device)
        clip_embed = module.cloob.text_encoder(self.prompts_toks.to(module.device))
        t = torch.linspace(1, 0, 50 + 1)[:-1]
        steps = diffusion_utils.get_spliced_ddpm_cosine_schedule(t)
        def model_fn(x, t, clip_embed):
            x_in = torch.cat([x, x])
            t_in = torch.cat([t, t])
            clip_embed_in = torch.cat([torch.zeros_like(clip_embed), clip_embed])
            v_uncond, v_cond = module(x_in, t_in, clip_embed_in).chunk(2, dim=0)
            return v_uncond + (v_cond - v_uncond) * 3
        with eval_mode(module):
            fakes = sampling.plms_sample(model_fn, noise, steps, {'clip_embed': clip_embed})
            # fakes = sample(module, noise, 1000, 1, {'clip_embed': clip_embed}, guidance_scale=3.)
            fakes = module.decode(fakes)
            
        grid = utils.make_grid(fakes, 4, padding=0).cpu()
        image = TF.to_pil_image(grid.add(1).div(2).clamp(0, 1))
        filename = f'demo_{trainer.global_step:08}.png'
        image.save(filename)
        log_dict = {'demo_grid': wandb.Image(image),
                    'prompts': wandb.Html(f'<pre>{lines_text}</pre>')}
        trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        del(clip_embed)


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err!s}', file=sys.stderr)


def worker_init_fn(worker_id):
    random.seed(torch.initial_seed())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cloob-checkpoint", type=str,
                   default='cloob_laion_400m_vit_b_16_16_epochs',
                   help="the CLOOB to condition with")
    p.add_argument("--vqgan-model", type=str, required=True,
                   help="the VQGAN checkpoint")
    p.add_argument('--train-set', type=Path, required=True,
                   help='path to the text file containing your training paths')
    p.add_argument('--checkpoint-every', type=int, default=50000,
                   help='output a model checkpoint every N steps')
    p.add_argument('--resume-from', type=str, default=None,
                   help='resume from (or finetune) the checkpoint at path') 
    p.add_argument('--demo-prompts', type=Path, required=True,
                   help='the demo prompts')
    p.add_argument('--demo-every', type=int, default=2000,
                   help='output a demo grid every N steps')
    p.add_argument('--wandb-project', type=str, required=True,
                   help='the wandb project to log to for this run')
    p.add_argument('--fprecision', type=int, default=32,
                   help='The precision to train in (32, 16, etc)')
    p.add_argument('--num-gpus', type=int, default=1,
                   help='the number of gpus to train with')
    p.add_argument('--num-workers', type=int, default=12,
                   help='the number of workers to load batches with')
    p.add_argument('--batch-size', type=int, default=64,
                   help='the batch size to use per step')
    p.add_argument('--base-channels', type=int, default=128,
                   help='the base channel count (width) for the model')
    p.add_argument('--channel-multipliers', type=str, default="4,4,8,8",
                   help='comma separated multiplier constants for the four model resolutions')
    p.add_argument('--ema-decay-at', type=int, default=200000,
                   help='the step to tighten ema decay at')
    args = p.parse_args()
    
    batch_size = args.batch_size
    size = 256

    TRAIN_PATHS = args.train_set

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    
    def tf(image):       
        return transforms.Compose([
            ToMode('RGB'),
            transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])(image)
    tok_wrap = TokenizerWrapper()


    class CustomDataset(data.Dataset):
        def __init__(self, train_paths, transform=None, target_transform=None):
            with open(train_paths) as infile:
                self.paths = [line.strip() for line in infile.readlines() if line.strip()]
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img_path = self.paths[idx]
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            return image, 0 # Pretend this is a None

    train_set = CustomDataset(TRAIN_PATHS, transform=tf)
    train_dl = data.DataLoader(train_set, batch_size, shuffle=True, drop_last=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    demo_prompts = Path(args.demo_prompts).read_text().strip().split('\n')
    demo_prompts = tok_wrap(demo_prompts)


    # We need to get the autoencoder scale outside the init of LightningDiffusion
    # because it runs on CPU
    ae_config = OmegaConf.load(args.vqgan_model + '.yaml')
    ae_model = ldm.models.autoencoder.AutoencoderKL(**ae_config.model.params)
    ae_model.eval().requires_grad_(False).to("cuda:0")
    ae_model.init_from_ckpt(args.vqgan_model + '.ckpt')

    # autoencoder scale
    print("Getting autoencoder scale for {}.ckpt".format(args.vqgan_model))
    var_accum = 0.
    dl_iter = iter(train_dl)
    for i in trange(32):
        batch = next(dl_iter)
        reals, _ = batch
        reals = reals.to("cuda:0")
        reals = ae_model.encode(reals).sample() * 2 - 1
        var_accum += reals.var().item()
    autoencoder_scale = torch.tensor(var_accum ** 0.5)
        
    model = LightningDiffusion(args.cloob_checkpoint, args.vqgan_model, train_dl,
                               ae_model, autoencoder_scale,
                               args.base_channels, args.channel_multipliers, args.ema_decay_at)

    wandb_logger = pl.loggers.WandbLogger(project=args.wandb_project)
    wandb_logger.watch(model.model)
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, save_top_k=-1)
    demo_callback = DemoCallback(demo_prompts, demo_prompts, args.demo_every)
    exc_callback = ExceptionCallback()
    trainer = pl.Trainer(
        gpus=args.num_gpus,
        num_nodes=1,
        strategy='ddp',
        precision=args.fprecision,
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        resume_from_checkpoint=args.resume_from,
    )

    trainer.fit(model, train_dl)


if __name__ == '__main__':
    main()
