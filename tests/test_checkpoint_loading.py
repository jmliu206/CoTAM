from __future__ import annotations

import torch

from cotam.adapters import LinearEncoder
from cotam.utils.checkpoint import load_adapter_state, load_model_state, normalize_state_dict


def test_normalize_state_dict_strips_legacy_prefixes() -> None:
    state = {
        "module.base_coder.encoder.weight": torch.ones(1),
        "base_coder.decoder.bias": torch.zeros(1),
    }
    normalized = normalize_state_dict(state)
    assert "encoder.weight" in normalized
    assert "decoder.bias" in normalized


def test_checkpoint_loading_codec_and_adapter() -> None:
    adapter = LinearEncoder()
    ckpt = {
        "state_dict": {},
        "adapter": {"module." + key: value.clone() for key, value in adapter.state_dict().items()},
    }
    result = load_adapter_state(adapter, ckpt, strict=True)
    assert result is not None
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_missing_adapter_does_not_crash() -> None:
    adapter = LinearEncoder()
    assert load_adapter_state(adapter, {"state_dict": {}}, strict=False) is None


def test_load_model_state_non_strict() -> None:
    model = torch.nn.Linear(2, 2)
    result = load_model_state(model, {"state_dict": {}}, strict=False)
    assert result.missing_keys
