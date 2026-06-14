from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Union

import torch


LEGACY_PREFIXES = ("module.", "base_coder.")


def strip_legacy_prefixes(key: str, prefixes: Iterable[str] = LEGACY_PREFIXES) -> str:
    name = key
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {strip_legacy_prefixes(key): value for key, value in state_dict.items()}


def extract_state_dict(checkpoint: Union[dict, torch.Tensor]) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def load_checkpoint(path: Union[str, Path], map_location: Union[str, torch.device] = "cpu") -> dict:
    return torch.load(Path(path), map_location=map_location)


def load_model_state(
    model: torch.nn.Module,
    checkpoint: dict,
    strict: bool = True,
    state_key: str = "state_dict",
) -> torch.nn.modules.module._IncompatibleKeys:
    state = checkpoint[state_key] if state_key in checkpoint else extract_state_dict(checkpoint)
    normalized = normalize_state_dict(state)
    result = model.load_state_dict(normalized, strict=strict)
    if result.missing_keys or result.unexpected_keys:
        logging.info("State load missing=%s unexpected=%s", result.missing_keys, result.unexpected_keys)
    return result


def load_adapter_state(
    adapter: torch.nn.Module,
    checkpoint: dict,
    strict: bool = True,
) -> Optional[torch.nn.modules.module._IncompatibleKeys]:
    if "adapter" not in checkpoint:
        logging.info("Checkpoint has no adapter state; adapter will be randomly initialized.")
        return None
    adapter_state = normalize_state_dict(checkpoint["adapter"])
    return adapter.load_state_dict(adapter_state, strict=strict)


def save_training_checkpoint(
    path: Union[str, Path],
    epoch: int,
    codec: torch.nn.Module,
    adapter: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
) -> None:
    payload = {
        "epoch": epoch,
        "state_dict": codec.state_dict(),
        "adapter": adapter.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["lr_scheduler"] = scheduler.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
