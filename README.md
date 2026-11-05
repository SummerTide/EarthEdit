# EarthEdit

Code for the manuscript **Editing the Earth: Source-Preserving Remote Sensing Change Synthesis via Context-Aware Inpainting**.

**JianChong Guo, YuTing Wan, AiLong Ma, YanFei Zhong**

[Code repository](https://github.com/SummerTide/EarthEdit)

## Overview

EarthEdit synthesizes a post-change remote sensing image from a real pre-change image, a target layout and a change mask. A context-aware ControlNet with a frozen SD3-medium backbone generates the change while preserving source pixels outside the editing and blending region.

## Contents

| Path | Purpose |
| --- | --- |
| `earthedit/` | model, data adapters, masks, captions, loss and sampler |
| `sample.py` | sampling |
| `train_layout.py`, `train.py` | single-GPU layout and generator training |
| `configs/` | final configuration per dataset |

## Install

```bash
pip install -r requirements.txt
```

`diffusers` is pinned at 0.30.1 because the SD3 ControlNet code depends on its APIs.

SD3 Medium has its own licence. Accept it on the [Hugging Face model page](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) and log in before the first run. The model id is set by `pretrained_model` in each config, and `--pretrained_model` also accepts a local path.

## Weights

Trained EarthEdit model weights are not publicly released. To use `sample.py`, train a generator following the [training instructions](#training) and pass the resulting checkpoint folder with `--checkpoint`.

A generator checkpoint folder holds `config.json`, `inpaint_controlnet.safetensors` and `inpaint_meta.json`.

## Input formats

All images are 512x512 RGB PNGs. Layouts and masks have the same size as their image.

| Dataset | `layout` | `mask` |
| --- | --- | --- |
| SECOND | RGB label with the SECOND palette, white = no change (a `label2` file) | single-channel PNG, white = changed |
| Hi-UCD | 3-channel Hi-UCD mask, post class read from G, or a single-channel PNG of class ids 0 to 9 | 3-channel Hi-UCD mask, changed where B = 2, or a single-channel PNG, white = changed |
| xBD | single-channel PNG of class ids: 0 background, 1 intact, 2 minor, 3 major, 4 destroyed | single-channel PNG, white = changed |

On SECOND the change mask is the union of the nonwhite pixels of both dates. On xBD it holds the buildings with class 2, 3 or 4, and unclassified buildings take class 1. xBD tiles are cut into four 512x512 crops.

A caption can be given in a `caption` column. If it is empty, it is derived from the layout. On xBD the derived caption takes the disaster type from the `stem`, which must then contain the event name, as in `santa-rosa-wildfire_00000131_0`.

## Sampling

`sample.py` needs a locally trained generator. The dataset comes from `--config`, and `--checkpoint` is a path to the generator checkpoint folder.

One image:

```bash
python sample.py --config configs/second.yaml --checkpoint runs/second/checkpoint-16000 \
    --pre pre.png --layout layout.png --mask mask.png --output_dir outputs/one
```

Many images, from a CSV with the columns `stem,pre,layout,mask` and an optional `caption`:

```csv
stem,pre,layout,mask
00003,im1/00003.png,label2/00003.png,masks/00003.png
```

```bash
python sample.py --config configs/second.yaml --checkpoint runs/second/checkpoint-16000 \
    --csv pairs.csv --output_dir outputs/second
```

Results are written as `<output_dir>/<stem>.png`. Pairs without any change return the source image.

## Training

Both training scripts run on one GPU, take `--batch_size`, `--grad_accum` and `--mixed_precision` (bf16 by default), and write `checkpoint-<step>` folders.

### 1. Layout ControlNet

The generator is initialized from a layout ControlNet trained on single-date images and VAE-encoded layouts. `train_layout.py` reads a CSV with the columns `image,layout,caption` and an optional `weight`, the relative sampling weight of the row. Each row is one date, so a pair gives two rows. `layout` follows the table above, with a single-channel PNG of class ids for the Hi-UCD pre date (R of the mask).

```csv
image,layout,caption,weight
im1/00003.png,label1/00003.png,an aerial remote sensing image with buildings,4.2
im2/00003.png,label2/00003.png,an aerial remote sensing image with tree,4.2
```

The original recipe used these stages. Their settings are in the `layout_base` and `layout_refine` sections of the configs.

| Stage | Start | Timesteps | Dropout | Learning rate | Total batch | Steps run | Checkpoint used |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SECOND base | SD3 transformer | uniform | none | 5e-6 | 16 | 5,000 | 5,000 |
| Hi-UCD base | SD3 transformer | uniform | none | 5e-6 | 16 | 10,000 | 10,000 |
| SECOND refine | SECOND base | logit-normal | 0.1 | 1e-5 | 16 | 6,000 | 4,000 |
| Hi-UCD refine | Hi-UCD base | logit-normal | 0.1 | 1e-5 | 16 | 6,000 | 6,000 |
| xBD refine | SECOND refine, checkpoint 4,000 | logit-normal | 0.1 | 1e-5 | 16 | 8,000 | 2,000 |

The learning rate and total batch of the SECOND base stage are taken from the Hi-UCD base stage, since only its step count is recorded.

Captions and sampling differ by stage:

- Base stage: `earthedit.captions.class_caption` on the layout of that date, with uniform sampling over all training pairs.
- Refine stage: `earthedit.captions.layout_caption` (`xbd_caption` on xBD) on the layout of that date. The `weight` is `earthedit.dataset.rarity` of the pair, multiplied by 4 for pairs with change. Pairs held back for selection and evaluation are left out.

```bash
python train_layout.py --stage base --config configs/second.yaml --csv second_dates.csv \
    --output_dir runs/second_base
python train_layout.py --stage refine --config configs/second.yaml --csv second_dates_refine.csv \
    --init_controlnet runs/second_base/checkpoint-5000 --output_dir runs/second_refine
```

### 2. Context-aware generator

`train.py` reads a CSV with the columns `stem,pre,post,layout,mask` and the optional `layout_pre` and `caption`. `layout` is the post layout. Give `layout_pre` to count the classes of both dates in the rare-class weights, as in the paper. Leave out pairs held back for selection and evaluation.

```csv
stem,pre,post,layout,mask,layout_pre
00003,im1/00003.png,im2/00003.png,label2/00003.png,masks/00003.png,label1/00003.png
```

```bash
python train.py --config configs/second.yaml --index second_pairs.csv \
    --init_controlnet runs/second_refine/checkpoint-4000 --output_dir runs/second
```

The paper trained with a total batch of 32. The defaults reach it on one GPU with `batch_size` 4 and `grad_accum` 8. Checkpoints are in the format `sample.py` loads.

## Configuration

Training and sampling parameters are defined in `configs/{second,hiucd,xbd}.yaml`. Sampling uses 28 steps with the presets below; pass the selected generator checkpoint explicitly through `--checkpoint`.

| Dataset | Checkpoint step | Guidance | `cfg_mode` | Dilate | Feather |
| --- | --- | --- | --- | --- | --- |
| SECOND | 16000 | 1.5 | both | 2 | 1 |
| Hi-UCD | 18000 | 1.5 | both | 6 | 3 |
| xBD | 20000 | 2.0 | text | 6 | 3 |

## Citation

If you use EarthEdit in your research, please cite the manuscript:

```bibtex
@misc{guo2026earthedit,
  title  = {Editing the {Earth}: Source-Preserving Remote Sensing Change Synthesis via Context-Aware Inpainting},
  author = {Guo, JianChong and Wan, YuTing and Ma, AiLong and Zhong, YanFei},
  year   = {2026},
  note   = {Manuscript},
  url    = {https://github.com/SummerTide/EarthEdit}
}
```

