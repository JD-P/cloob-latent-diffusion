# CLOOB Conditioned Latent Diffusion: Convenient High Quality Diffusion Models

## Introduction

This repository contains the training code for CLOOB conditioned latent diffusion.
CCLD is similar in approach to the CLIP conditioned diffusion [trained by
Katherine Crowson](https://github.com/crowsonkb/v-diffusion-pytorch) with a few
key differences:

- The use of **latent diffusion** cuts training costs by something like a factor of
ten, allowing a high quality 1.2 billion parameter model to converge in as few as 5 days
on a single 8x A100 pod.

- **CLOOB conditioning** can take advantage of CLOOB's unified latent space. CLOOB
text and image embeds on the same inputs share a high similarity of somewhere around 0.9. **This
makes it possible to train the model without captions** by using image embeds in the
training loop and text embeds during inference.

This combination of traits makes the CCLD training approach extremely attractive
to hobbyists, academics, and newcomers due to its high quality results, low
finetune/training costs, and easy setup. It is the StyleGAN of diffusion models. 

## Pretrained Models

We plan to release a variety of pretrained models in the near future, but right
now we have a 1.2 billion parameter classifier-free-guidance model [trained on yfcc 100m](https://paperswithcode.com/dataset/yfcc100m):

[yfcc_cfg](https://the-eye.eu/public/AI/models/yfcc-latent-diffusion-f8-e2-s250k.ckpt) (ViT-B/16 CLOOB 16 epochs, 192 base channels, 4-4-8-8 resolution multipliers) - [CLOOB checkpoint](https://the-eye.eu/public/AI/models/cloob/cloob_laion_400m_vit_b_16_16_epochs-405a3c31572e0a38f8632fa0db704d0e4521ad663555479f86babd3d178b1892.pkl) | [Autoencoder](https://ommer-lab.com/files/latent-diffusion/kl-f8.zip) | [Autoencoder Config](https://raw.githubusercontent.com/CompVis/latent-diffusion/main/configs/autoencoder/autoencoder_kl_32x32x4.yaml) | [Model Mirror](https://mystic.the-eye.eu/public/AI/models/yfcc-latent-diffusion-f8-e2-s250k.ckpt)

## Training

### Training setup

First recursively git clone this repo to get it and its submodules:

`git clone --recursive https://github.com/JD-P/cloob-latent-diffusion`

If you don't already have pytorch you'll need to install it, for most datacenter
GPUs the command looks like:

`pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`

Then pip install our other dependencies:

`pip3 install omegaconf pillow pytorch-lightning einops wandb ftfy regex pycocotools ./CLIP`

You are now ready to prepare your training run.

### Preparing The Dataset

First prepare your training set by creating a `.txt` of filepaths that the
images to train on will be loaded from. For example this is how you make such
a list for the [MS COCO dataset](https://paperswithcode.com/dataset/coco):

```
find /datasets/coco/train2017/ -type f >> train_paths.txt
find /datasets/coco/val2017/ -type f >> train_paths.txt
shuf train_paths.txt > train_paths_2.txt
mv train_paths_2.txt train_paths.txt 
```

The `find` command is run over the top level directory where images are stored
in the dataset. The `-type f` flag filters the search so that only files are returned,
if the images are stored only with other images this is equivalent to getting the
filepaths for every image in the dataset by themselves. If the data is not conveniently
organized this way it is possible to do further filtering by piping the results
of find into utilities like `grep`.


**Training Tip**: It's important to shuffle your dataset so that the net generalizes
during training. This is why the `shuf` utility is used on the training paths.

### Demo Prompts

You will also need demo prompts for the grids displayed in wandb during your
training run. These grids are cheap to generate with PLMS sampling and massively
improve your ability to diagnose problems with your run. Here's some written by us:

```
A portrait of Friedrich Nietzsche wearing an open double breasted suit with a bowtie
A portrait of a man in a flight jacket leaning against a biplane
a vision of paradise. unreal engine
the gateway between dreams, trending on ArtStation
A fantasy painting of a city in a deep valley by Ivan Aivazovsky
a rainy city street in the style of cyberpunk noir, trending on ArtStation
An oil painting of A Vase Of Flowers
oil painting of a candy dish of glass candies, mints, and other assorted sweets
The Human Utility Function
the Tower of Babel by J.M.W. Turner
sketch of a 3D printer by Leonardo da Vinci
The US Capitol Building in the style of Kandinsky
Metaphysics in the style of WPAP
a watercolor painting of a Christmas tree
control room monitors televisions screens computers hacker lab, concept art, matte painting, trending on artstation
illustration of airship zepplins in the skies, trending on artstation
```

**Training Tip**: You may want to modify these prompts if you're training on
a photorealistic dataset, as these are optimized more for getting results from
models that do illustration and paintings.

### Autoencoder

In order to train latent diffusion you need a latent space to train in. The
architecture of the training code is set up for an f=8 KL autoencoder. You can
[get a photorealistic autoencoder here](https://ommer-lab.com/files/latent-diffusion/kl-f8.zip)
among with others in the [CompVis latent diffusion repo](https://github.com/CompVis/latent-diffusion).
You will also need [the configuration file for it](https://raw.githubusercontent.com/CompVis/latent-diffusion/main/configs/autoencoder/autoencoder_kl_32x32x4.yaml).

If you're not training on a photorealistic dataset, you will either need to find an
appropriate pretrained KL autoencoder or train your own. [The training repo for
these models](https://github.com/CompVis/latent-diffusion) is unfortunately pretty
nasty for a beginner and requires modification before you can easily train an
arbitrary dataset with it. We plan to release some pretrained models of our own
along with a more friendly fork of that repo in the future.

**Training Tip**: From a compute perspective if you only have an A6000 or 3090
your best bet is probably to finetune an existing KL f=8 autoencoder on the dataset
you want to train on. This still requires working training code however.

**Training Tip**: You must(?) use a low dimensional autoencoder for latent diffusion
to work, our experiments with higher dimensional autoencoders did not work well.

### Training The Model

Once you have the setup, training set, autoencoder, demo prompts, and wandb project ready
starting the training run is as simple as:

`python3 yfcc_latent_diffusion.py --train-set train_paths.txt --vqgan-model kl_f8 --demo-prompts coco_demo_prompts.txt --wandb-project jdp-latent-diffusion`

For the YFCC CLOOB conditioned latent diffusion training took about five and a
half days to reach the 250k checkpoint with a base channel count of 192 and
channel multipliers of 4,4,8,8. You can analyze the logs from these runs at the
following links:

[0-150k step training run](https://wandb.ai/jdp/jdp-latent-diffusion/runs/1dv7xxrg?workspace=user-jdp)
[150k-250k step training run](https://wandb.ai/jdp/jdp-latent-diffusion/runs/258cmlpw?workspace=user-jdp)

**Training Tip**: The loss curve has a small scale past the initial warmup, if it
seems to be stuck in the same loss regime this doesn't necessarily mean it isn't
improving. Make sure to use your demo grids to monitor progress.

**Training Tip**: It's possible to train in fp16 and then resume in fp32
once the run begins to explode or diverge. This is especially useful if you're VRAM
constrained and would like to use a higher batch size in the early training. It
also makes early training go faster if you're compute constrained or impatient.

**Training Tip**: Once the loss converges it is often possible to get it down lower
by restarting the run with a lower learning rate. You need to overwrite the learning
rate in the checkpoint so it doesn't get overwritten when you resume. You can do that
from a python prompt like so:

```
Python 3.8.10 (default, Nov 26 2021, 20:14:08) 
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> ckpt = torch.load("jdp-latent-diffusion/1dv7xxrg/checkpoints/epoch=1-step=149999.ckpt")
>>> ckpt['optimizer_states'][0]['param_groups'][0]['lr']
3e-05
>>> ckpt['optimizer_states'][0]['param_groups'][0]['lr'] = 3e-06
>>> torch.save(ckpt, "yfcc_resume.ckpt")
>>>
```

### Finetuning

It's possible to save time (and money) by retraining an existing model on a new
dataset rather than starting from scratch. This is called finetuning a model. If
you would like to finetune an existing model this is easily accomplished using
the `--resume-from` flag:

`python3 train_latent_diffusion.py --train-set train_paths.txt --vqgan-model kl_f8 --demo-prompts coco_demo_prompts.txt --resume-from to_finetune.ckpt --wandb-project jdp-latent-diffusion`

**Training Tip**: As a rule of thumb, finetunes tend to take 10-20% of the resources
that the original training run did in compute time.