from __future__ import annotations

import torch

from cotam.codecs.elic import TinyBaseCodec
from cotam.codecs.elic_vbr import TinyELICVBRCodec


def test_base_codec_forward_contract() -> None:
    model = TinyBaseCodec()
    out = model(torch.randn(1, 3, 256, 256))
    assert out["x_hat"].shape == (1, 3, 256, 256)
    assert "y" in out["likelihoods"]
    assert "z" in out["likelihoods"]


def test_vbr_codec_forward_contract() -> None:
    model = TinyELICVBRCodec()
    images = torch.randn(1, 3, 384, 384)
    mask = torch.zeros(24, 24, dtype=torch.long)
    out = model(images, mask=mask)
    assert out["x_hat"].shape == (1, 3, 384, 384)
    assert out["y_hat"].shape == (1, 320, 24, 24)
    assert out["likelihoods"]["y"].shape == out["y_hat"].shape
