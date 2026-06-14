from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .legacy import import_legacy


class TinyELICVBRCodec(nn.Module):
    """Small VBR codec with the same forward contract as the ELIC VBR path."""

    def __init__(self, latent_channels: int = 320, levels: int = 10) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.levels = levels
        self.encoder = nn.Conv2d(3, latent_channels, kernel_size=16, stride=16)
        self.decoder = nn.ConvTranspose2d(latent_channels, 3, kernel_size=16, stride=16)
        self.mask_params_encode = nn.Parameter(torch.ones(4, levels))
        self.mask_params_decode = nn.Parameter(torch.ones(4, levels))

    def _mask_gain(self, mask: Optional[torch.Tensor], shape: tuple[int, int], device: torch.device) -> torch.Tensor:
        if mask is None:
            return torch.ones(1, 1, shape[0], shape[1], device=device)
        resized = F.interpolate(
            mask.to(device=device).unsqueeze(0).unsqueeze(0).float(),
            size=shape,
            mode="nearest",
        )
        return 1.0 + resized / max(float(self.levels - 1), 1.0) * 0.01

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **_: object) -> dict:
        y_hat = self.encoder(x)
        gain = self._mask_gain(mask, y_hat.shape[-2:], x.device)
        y_hat = y_hat * gain
        x_hat = torch.sigmoid(self.decoder(y_hat))
        y_likelihood = torch.full_like(y_hat, 0.5).clamp_min(1e-9)
        z_likelihood = torch.full(
            (x.shape[0], 1, max(1, y_hat.shape[2] // 4), max(1, y_hat.shape[3] // 4)),
            0.5,
            device=x.device,
            dtype=x.dtype,
        )
        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihood, "z": z_likelihood},
        }

    def aux_loss(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)


def build_vbr_codec(config: Optional[dict] = None) -> nn.Module:
    cfg = config or {}
    name = cfg.get("name", "elic_vbr")
    if cfg.get("use_tiny", False) or name == "tiny":
        return TinyELICVBRCodec(
            latent_channels=int(cfg.get("m", 320)),
            levels=int(cfg.get("levels", 10)),
        )
    if name in {"elic", "elic_vbr"}:
        module = import_legacy("Network_mul_gain", kind="examples")
        return module.TestModel(N=int(cfg.get("n", 192)), M=int(cfg.get("m", 320)), vbr=True)
    if name == "dcae":
        module = import_legacy("dcae_vbr", kind="examples")
        return module.DCAE()
    raise ValueError(f"Unknown VBR codec: {name}")
