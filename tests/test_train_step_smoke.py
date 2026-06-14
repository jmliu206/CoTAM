from __future__ import annotations

from pathlib import Path

import torch

from cotam.clip import FakeClipWrapper
from cotam.codecs.elic_vbr import TinyELICVBRCodec
from cotam.datasets import FakeCaptionDataset
from cotam.losses import build_loss
from cotam.masks import RandomMaskProvider
from cotam.train_adapter import main as train_adapter_main, run_batch
from cotam.adapters import LinearEncoder


def test_one_train_step_updates_only_adapter() -> None:
    codec = TinyELICVBRCodec()
    adapter = LinearEncoder()
    for param in codec.parameters():
        param.requires_grad = False
    criterion = build_loss({"total_weight": 1.0}, FakeClipWrapper())
    mask_provider = RandomMaskProvider(seed=1)
    dataset = FakeCaptionDataset(length=2)
    images, labels = dataset[0]
    images = images.unsqueeze(0)
    labels = labels.unsqueeze(0)

    losses = run_batch(images, labels, codec, adapter, criterion, criterion.clip, mask_provider, {}, epoch=0)
    losses["total_loss"].backward()
    assert all(param.grad is None for param in codec.parameters())
    assert any(param.grad is not None for param in adapter.parameters())


def test_train_adapter_smoke_saves_checkpoint(tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "cotam_elic_clip_adapter.yaml"
    result = train_adapter_main(
        [
            "-c",
            str(config),
            "--use-fake-data",
            "--use-fake-clip",
            "--use-tiny-codec",
            "--max-steps",
            "1",
            "--override",
            f"experiment.output_dir={tmp_path}",
            "--override",
            "experiment.name=smoke",
            "--override",
            "data.batch_size=1",
            "--override",
            "data.val_batch_size=1",
        ]
    )
    ckpt = tmp_path / "smoke" / "checkpoint.pth.tar"
    assert ckpt.exists()
    assert "metrics" in result
