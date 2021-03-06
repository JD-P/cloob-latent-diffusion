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

We currently have two models and plan to release more in the near future. Right
now there is a 1.2 billion parameter classifier-free-guidance model [trained on yfcc 100m](https://paperswithcode.com/dataset/yfcc100m):

[yfcc_cfg](https://the-eye.eu/public/AI/models/yfcc-latent-diffusion-f8-e2-s250k.ckpt) (ViT-B/16 CLOOB 16 epochs, 192 base channels, 4-4-8-8 resolution multipliers) - [CLOOB checkpoint](https://the-eye.eu/public/AI/models/cloob/cloob_laion_400m_vit_b_16_16_epochs-405a3c31572e0a38f8632fa0db704d0e4521ad663555479f86babd3d178b1892.pkl) | [Autoencoder](https://ommer-lab.com/files/latent-diffusion/kl-f8.zip) | [Autoencoder Config](https://raw.githubusercontent.com/CompVis/latent-diffusion/main/configs/autoencoder/autoencoder_kl_32x32x4.yaml) | [Model Mirror](https://mystic.the-eye.eu/public/AI/models/yfcc-latent-diffusion-f8-e2-s250k.ckpt)

[danbooru_cfg](https://the-eye.eu/public/AI/models/danbooru-latent-diffusion-e88.ckpt) (ViT-B/16 CLOOB 32 epochs, 128 base channels, 4-4-8-8 resolution multipliers)

And a stage one LAION 5b autoencoder which makes a good general base to
train your latent diffusion model on top of if you can't train your own.
LAION 5b contains a wide variety of images and should therefore have textures
for your dataset in its distribution:

[LAION 5b Autoencoder](https://the-eye.eu/public/AI/cah/laion-5B-kl-f8.ckpt) (autoencoder scale 8.0779) - [Config](https://the-eye.eu/public/AI/cah/laion-5B-kl-f8.yaml)

**Note**: The LAION 5b autoencoder was not trained on all of LAION 5b, but the
laion2b-en and laion1b-nolang subsets.

[Danbooru Autoencoder](https://the-eye.eu/public/AI/models/danbooru-kl-f8.ckpt) (autoencoder scale 9.3154) - [Config](https://the-eye.eu/public/AI/models/danbooru-kl-f8.yaml)

## Setup

First recursively git clone this repo to get it and its submodules:

`git clone --recursive https://github.com/JD-P/cloob-latent-diffusion`

If you don't already have pytorch you'll need to install it, for most datacenter
GPUs the command looks like:

`pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`

Then pip install our other dependencies:

`pip3 install omegaconf pillow pytorch-lightning einops wandb ftfy regex pycocotools ./CLIP`

You are now ready to sample or prepare your training run.

## Sampling

It is possible to sample from a model like so:

```rm -f out*.png; ./cfg_sample.py "A photorealist detailed snarling goblin" --autoencoder kl_f8 --checkpoint yfcc-latent-diffusion-f8-e2-s250k.ckpt -n 128 --seed 4485 && v-diffusion-pytorch/make_grid.py out_*.png```

Or in the case of something like the danbooru latent diffusion model:

```rm -f out*.png; ./cfg_sample.py "anime portrait of a man in a flight jacket leaning against a biplane" --autoencoder danbooru-kl-f8 --checkpoint danbooru-latent-diffusion-e88.ckpt --cloob-checkpoint cloob_laion_400m_vit_b_16_32_epochs --base-channels 128 --channel-multipliers 4,4,8,8 -n 16 --seed 4485 && v-diffusion-pytorch/make_grid.py out_*.png```


## Training

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
You will also need [the configuration file for it](https://raw.githubusercontent.com/CompVis/latent-diffusion/main/configs/autoencoder/autoencoder_kl_32x32x4.yaml)
which can be found in the latent-diffusion repo recursively cloned along with cloob-latent-diffusion.
It should have the same name as your autoencoder with the file extension changed. For example:

`cp latent-diffusion/configs/autoencoder/autoencoder_kl_32x32x4.yaml ./2022_04_04_wikiart_kl_f8.yaml`

Before training you must get the scale for your autoencoder like so:

`python3 autoencoder_scale.py 2022_04_04_wikiart_kl_f8 train.txt`

Write down the number you obtain from this and use it in your training run, this
same number must be used in inference for the model to work. The model checkpoint
retains a copy of the autoencoder scale but it's best to keep your own record of it
in your lab notes.

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

`python3 train_latent_diffusion.py --train-set train.txt --vqgan-model kl_f8 --autoencoder-scale 8.0779 --demo-prompts demo_prompts.txt --wandb-project jdp-latent-diffusion --batch-size 128 --num-gpus 8`

For the YFCC CLOOB conditioned latent diffusion training took about five and a
half days to reach the 250k checkpoint with a base channel count of 192 and
channel multipliers of 4,4,8,8. You can analyze the logs from these runs at the
following links:

[0-150k step training run](https://wandb.ai/jdp/jdp-latent-diffusion/runs/1dv7xxrg?workspace=user-jdp)

[150k-250k step training run](https://wandb.ai/jdp/jdp-latent-diffusion/runs/258cmlpw?workspace=user-jdp)

**Training Tip**: Your model is likely to overfit/memorize the training set if
it's too big in relation to your dataset size. The rule of thumb for overfitting
is the parameter count shouldn't be more than 2/3 the datapoints in the set. You
can calculate datapoints (floats) from the size of your latents times the size of
your dataset. For the f=8 kl autoencoder used by this training repo it's
32x32x4xDataSetSize. So for example WikiArt which has 80k training items should
be trained on a model no more than 0.66 * 32 * 32 * 4 * 80000 parameters large,
or 216.2688 million. You should pick your base channel count and channel
multipliers to respect this rule. Base channel count must be a multiple of 64
for this architecture.

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

`python3 train_latent_diffusion.py --train-set train_paths.txt --vqgan-model kl_f8 --autoencoder-scale 8.0779 --demo-prompts coco_demo_prompts.txt --resume-from to_finetune.ckpt --wandb-project jdp-latent-diffusion`

**Training Tip**: As a rule of thumb, finetunes tend to take 10-20% of the resources
that the original training run did in compute time.