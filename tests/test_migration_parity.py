from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from cotam.adapters import LinearEncoder


def test_linear_encoder_matches_legacy_when_importable() -> None:
    train_root = Path(__file__).resolve().parents[2]
    examples = train_root / "examples"
    for path in [str(train_root), str(examples)]:
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        from adapter_model import Linear_Encoder as LegacyLinearEncoder  # type: ignore
    except Exception as exc:
        pytest.skip(f"legacy adapter import unavailable: {exc}")

    torch.manual_seed(123)
    clean = LinearEncoder(in_features=320, out_features=1024, input_resolution=384, transformer_layers=1)
    legacy = LegacyLinearEncoder(in_features=320, out_features=1024, input_resolution=384, layer=1)
    legacy.load_state_dict(clean.state_dict(), strict=True)
    x = torch.randn(1, 320, 24, 24)
    clean_out = clean(x)
    legacy_out = legacy(x)
    assert torch.allclose(clean_out, legacy_out, atol=1e-5, rtol=1e-5)
