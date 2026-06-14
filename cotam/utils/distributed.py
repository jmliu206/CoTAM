from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch


@dataclass
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(config: Optional[Mapping[str, Any]] = None) -> DistributedContext:
    cfg = config or {}
    mode = cfg.get("enabled", "auto")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    enabled = world_size > 1 if mode == "auto" else bool(mode)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if enabled and not torch.distributed.is_initialized():
        backend = str(cfg.get("backend", "nccl" if torch.cuda.is_available() else "gloo"))
        torch.distributed.init_process_group(backend=backend)

    return DistributedContext(
        enabled=enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def cleanup_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def maybe_wrap_ddp(module: torch.nn.Module, context: DistributedContext) -> torch.nn.Module:
    if not context.enabled:
        return module
    if context.device.type == "cuda":
        return torch.nn.parallel.DistributedDataParallel(
            module,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
        )
    return torch.nn.parallel.DistributedDataParallel(module)


def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module
