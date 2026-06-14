# CoTAM: A Coding Paradigm Tailored to MLLMs

<p align="center">
  <strong>Official PyTorch implementation of the ICLR 2026 paper</strong><br>
  <em>When MLLMs Meet Compression Distortion: A Coding Paradigm Tailored to MLLMs</em>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2509.24258">Paper</a> |
</p>

## Overview

Multimodal Large Language Models (MLLMs) are increasingly deployed as cloud-side intelligence, while images and videos are often captured on edge devices. This creates a practical need for codecs that reduce bandwidth while preserving the visual information that MLLMs actually use.

CoTAM, short for **Codec TAilored to MLLMs**, is an image compression framework designed for MLLM downstream tasks. Instead of optimizing only for human visual fidelity, CoTAM adapts bit allocation and reconstruction objectives to protect multi-level visual features used by MLLM vision encoders.

## Highlights

- **MLLM-oriented compression**: studies how compression distortion affects MLLM performance through different visual feature levels.
- **Semantic bit allocation**: uses shallow CLIP attention to guide spatial bit allocation toward important regions.
- **Multi-level feature preservation**: combines decoded image priors, a lightweight latent adapter, and multi-level visual feature losses.
- **Codec- and adapter-based design**: supports training a base codec and an MLLM-oriented adapter for compressed visual representations.

## Installation

```bash
conda create -n cotam python=3.9 -y
conda activate cotam

pip install -r requirements.txt
```

Run commands from the repository root:

```bash
cd CoTAM
```

## Data Preparation

The adapter training code expects image-text data for training and COCO captions for validation.
We recommend using [rom1504/img2dataset](https://github.com/rom1504/img2dataset) to download and organize CC3M-style image-text training data.

Example layout:

```text
data/
  cc3m/
    train/
      sample_000001.jpg
      sample_000001.txt
  coco/
    annotations/
      captions_val2017.json
    val2017/
      000000000001.jpg
```

Each CC3M image should have a sidecar `.txt` file with the corresponding caption.

## Configuration

Most paths and training options are configured in:

```text
configs/cotam_elic_clip_adapter.yaml
```

Users typically need to set:

```yaml
data:
  train_path: /path/to/cc3m
  val_path: /path/to/coco

codec:
  checkpoint: /path/to/base_codec.pth.tar

experiment:
  output_dir: ./runs/cotam
```

The same options can also be overridden from the command line with `--override key=value`.

## Training

### Base Codec

Train the base codec:

```bash
bash scripts/train_base_codec.sh \
  --override data.dataset_path=/path/to/images \
  --override experiment.output_dir=./runs/base_codec
```

### CoTAM Adapter

Train the CoTAM adapter on top of a pretrained codec:

```bash
bash scripts/train_cotam_adapter.sh \
  --override data.train_path=/path/to/cc3m \
  --override data.val_path=/path/to/coco \
  --override codec.checkpoint=/path/to/base_codec.pth.tar \
  --override experiment.output_dir=./runs/cotam
```

The default configuration is in:

```text
configs/cotam_elic_clip_adapter.yaml
```

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{liu2026cotam,
  title     = {When MLLMs Meet Compression Distortion: A Coding Paradigm Tailored to MLLMs},
  author    = {Jinming Liu and Zhaoyang Jia and Jiahao Li and Bin Li and Xin Jin and Wenjun Zeng and Yan Lu},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```
