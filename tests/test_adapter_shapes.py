from __future__ import annotations

import torch

from cotam.adapters import LinearEncoder


def test_adapter_output_shape() -> None:
    model = LinearEncoder(in_features=320, out_features=1024, input_resolution=384, transformer_layers=1)
    x = torch.randn(2, 320, 24, 24)
    y = model(x)
    assert y.shape == (2, 577, 1024)
    assert model.positional_embedding.shape == (577, 1024)
