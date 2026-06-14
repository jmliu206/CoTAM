from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cotam.adapters import LinearEncoder
from cotam.clip import build_clip_wrapper
from cotam.codecs import build_vbr_codec
from cotam.datasets import (
    CC3MDataset,
    CocoCaptionDataset,
    FakeCaptionDataset,
    build_image_transform,
    collate_image_text,
    load_text_tokenizer,
)
from cotam.losses import build_loss, select_loss_names
from cotam.masks import build_mask_provider
from cotam.utils.checkpoint import load_adapter_state, load_checkpoint, load_model_state, save_training_checkpoint
from cotam.utils.config import deep_get, load_config
from cotam.utils.distributed import cleanup_distributed, init_distributed, maybe_wrap_ddp, unwrap_model
from cotam.utils.logging import setup_logger
from cotam.utils.seed import set_seed


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean CoTAM adapter training.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--use-fake-data", action="store_true")
    parser.add_argument("--use-fake-clip", action="store_true")
    parser.add_argument("--use-tiny-codec", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args(argv)


def build_loaders(config: dict, use_fake_data: bool, distributed: bool, rank: int, world_size: int):
    data_cfg = config.get("data", {})
    image_size = int(data_cfg.get("image_size", 384))
    batch_size = int(data_cfg.get("batch_size", 8))
    val_batch_size = int(data_cfg.get("val_batch_size", data_cfg.get("test_batch_size", batch_size)))
    num_workers = int(data_cfg.get("num_workers", 0))

    if use_fake_data:
        train_dataset = FakeCaptionDataset(length=max(batch_size * 2, 4), image_size=image_size)
        val_dataset = FakeCaptionDataset(length=max(val_batch_size, 4), image_size=image_size)
    else:
        transform = build_image_transform(image_size)
        tokenizer = load_text_tokenizer(context_length=int(data_cfg.get("context_length", 77)))
        train_dataset = CC3MDataset(
            data_cfg.get("train_path", "./data/cc3m"),
            tokenizer=tokenizer,
            split=data_cfg.get("train_split", "train"),
            transform=transform,
        )
        val_dataset = CocoCaptionDataset(
            data_cfg.get("val_path", "./data/coco"),
            tokenizer=tokenizer,
            split=data_cfg.get("val_split", "val2017"),
            transform=transform,
        )

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_image_text,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_image_text,
        drop_last=False,
    )
    return train_loader, val_loader


def build_models(config: dict, device: torch.device, use_fake_clip: bool, use_tiny_codec: bool):
    codec_cfg = dict(config.get("codec", {}))
    if use_tiny_codec:
        codec_cfg["use_tiny"] = True
        codec_cfg["name"] = "tiny"
    codec = build_vbr_codec(codec_cfg).to(device)
    for param in codec.parameters():
        param.requires_grad = False
    codec.eval()

    adapter_cfg = config.get("adapter", {})
    adapter = LinearEncoder(
        in_features=int(adapter_cfg.get("in_features", codec_cfg.get("m", 320))),
        out_features=int(adapter_cfg.get("out_features", 1024)),
        input_resolution=int(adapter_cfg.get("input_resolution", deep_get(config, "data.image_size", 384))),
        patch_size=int(adapter_cfg.get("patch_size", 16)),
        transformer_layers=int(adapter_cfg.get("transformer_layers", 1)),
    ).to(device)

    clip_cfg = dict(config.get("clip", {}))
    if use_fake_clip:
        clip_cfg["use_fake"] = True
    clip_wrapper = build_clip_wrapper(clip_cfg, device=device)
    criterion = build_loss(config.get("loss", {}), clip_wrapper)
    return codec, adapter, criterion, clip_wrapper


def run_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    codec: torch.nn.Module,
    adapter: torch.nn.Module,
    criterion: torch.nn.Module,
    clip_wrapper,
    mask_provider,
    config: dict,
    epoch: int,
) -> dict[str, torch.Tensor]:
    device = images.device
    mask = mask_provider(images)
    with torch.no_grad():
        out = codec(images, mask=mask)
        decode_prior = clip_wrapper.token_features_from_image(out["x_hat"])
    student_tokens = adapter(out["y_hat"])
    if bool(deep_get(config, "loss.use_decode_prior", True)):
        student_tokens = student_tokens + decode_prior.to(device)
    loss_names = select_loss_names(
        epoch=epoch,
        mode=str(deep_get(config, "loss.mode", "low_then_low_high")),
        warmup_epochs=int(deep_get(config, "loss.warmup_epochs", 5)),
    )
    return criterion(
        student_tokens=student_tokens,
        labels=labels,
        original_images=images,
        loss_names=loss_names,
        mask=mask,
    )


def train_or_eval(config: dict, args: argparse.Namespace) -> dict:
    set_seed(deep_get(config, "experiment.seed", 42))
    context = init_distributed(config.get("distributed", {}))
    output_dir = Path(deep_get(config, "experiment.output_dir", "./runs"))
    exp_name = str(deep_get(config, "experiment.name", "cotam_clean"))
    run_dir = output_dir / exp_name
    if context.is_main:
        setup_logger(run_dir / "train.log")

    train_loader, val_loader = build_loaders(
        config=config,
        use_fake_data=args.use_fake_data,
        distributed=context.enabled,
        rank=context.rank,
        world_size=context.world_size,
    )
    codec, adapter, criterion, clip_wrapper = build_models(
        config,
        context.device,
        use_fake_clip=args.use_fake_clip,
        use_tiny_codec=args.use_tiny_codec,
    )
    checkpoint_path = args.checkpoint or deep_get(config, "codec.checkpoint")
    if checkpoint_path and not args.use_tiny_codec:
        ckpt = load_checkpoint(checkpoint_path, map_location=context.device)
        load_model_state(codec, ckpt, strict=bool(deep_get(config, "checkpoint.strict_codec", True)))
        load_adapter_state(adapter, ckpt, strict=bool(deep_get(config, "checkpoint.strict_adapter", True)))

    optimizer = torch.optim.Adam(
        adapter.parameters(),
        lr=float(deep_get(config, "adapter.learning_rate", 5e-4)),
    )
    adapter = maybe_wrap_ddp(adapter, context)
    mask_provider = build_mask_provider(config.get("mask", {}))

    max_steps = args.max_steps
    metrics: dict[str, float] = {}
    try:
        if args.eval_only:
            metrics = evaluate(val_loader, codec, adapter, criterion, clip_wrapper, mask_provider, config, context.device)
        else:
            metrics = train_loop(
                train_loader,
                codec,
                adapter,
                criterion,
                clip_wrapper,
                mask_provider,
                optimizer,
                config,
                context.device,
                max_steps=max_steps,
            )
            if context.is_main:
                save_training_checkpoint(
                    run_dir / "checkpoint.pth.tar",
                    epoch=0,
                    codec=codec,
                    adapter=unwrap_model(adapter),
                    optimizer=optimizer,
                )
    finally:
        cleanup_distributed()
    return {"metrics": metrics, "run_dir": str(run_dir)}


def train_loop(
    loader,
    codec,
    adapter,
    criterion,
    clip_wrapper,
    mask_provider,
    optimizer,
    config,
    device,
    max_steps: Optional[int] = None,
) -> dict[str, float]:
    adapter.train()
    last_loss = 0.0
    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = run_batch(images, labels, codec, adapter, criterion, clip_wrapper, mask_provider, config, epoch=0)
        losses["total_loss"].backward()
        optimizer.step()
        last_loss = float(losses["total_loss"].detach().cpu())
        logging.info("step=%s total_loss=%.6f", step, last_loss)
        if max_steps is not None and step >= max_steps:
            break
    return {"total_loss": last_loss}


@torch.no_grad()
def evaluate(loader, codec, adapter, criterion, clip_wrapper, mask_provider, config, device) -> dict[str, float]:
    adapter.eval()
    values = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        losses = run_batch(images, labels, codec, adapter, criterion, clip_wrapper, mask_provider, config, epoch=0)
        values.append(float(losses["total_loss"].detach().cpu()))
    return {"total_loss": sum(values) / max(len(values), 1)}


def main(argv: Optional[list[str]] = None) -> dict:
    args = parse_args(argv)
    config = load_config(args.config, args.override)
    return train_or_eval(config, args)


if __name__ == "__main__":
    main()
