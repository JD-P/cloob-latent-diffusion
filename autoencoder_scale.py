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

parser = argparse.ArgumentParser()
parser.add_argument("vqgan_model")
parser.add_argument("dataset_paths")
args = parser.parse_args()

class ToMode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)

def tf(image):       
    return transforms.Compose([
        ToMode('RGB'),
        transforms.Resize(256, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])(image)

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

train_set = CustomDataset(args.dataset_paths, transform=tf)
train_dl = data.DataLoader(train_set, 64, shuffle=True, drop_last=True,
                           num_workers=12, persistent_workers=True, pin_memory=True)

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
    reals = ae_model.encode(reals * 2 - 1).sample()
    var_accum += reals.var().item()
autoencoder_scale = torch.tensor((var_accum / 32) ** 0.5)
print(autoencoder_scale)
