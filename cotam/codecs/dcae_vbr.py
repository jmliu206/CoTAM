from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .elic_vbr import TinyELICVBRCodec
from .legacy import import_legacy


def build_dcae_vbr(config: Optional[dict] = None) -> nn.Module:
    cfg = config or {}
    if cfg.get("use_tiny", False):
        return TinyELICVBRCodec(latent_channels=int(cfg.get("m", 320)))
    module = import_legacy("dcae_vbr", kind="examples")
    return module.DCAE()
