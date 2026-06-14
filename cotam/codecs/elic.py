from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .legacy import import_legacy


class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda: float = 1e-2) -> None:
        super().__init__()
        self.lmbda = float(lmbda)
        self.mse = nn.MSELoss()

    def forward(self, output: dict, target: torch.Tensor) -> dict[str, torch.Tensor]:
        num_pixels = target.size(0) * target.size(2) * target.size(3)
        bpp = sum(
            torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            for likelihoods in output["likelihoods"].values()
        )
        mse = self.mse(output["x_hat"], target) * 255 ** 2
        return {
            "bpp_loss": bpp,
            "mse_loss": mse,
            "loss": self.lmbda * mse + bpp,
        }


class TinyBaseCodec(nn.Module):
    """Tiny codec for tests; it keeps the same output contract as the real codec."""

    def __init__(self, latent_channels: int = 320) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, latent_channels, kernel_size=16, stride=16)
        self.decoder = nn.ConvTranspose2d(latent_channels, 3, kernel_size=16, stride=16)

    def forward(self, x: torch.Tensor, noisequant: bool = False) -> dict:
        y_hat = self.encoder(x)
        x_hat = torch.sigmoid(self.decoder(y_hat))
        likelihood = torch.full_like(y_hat, 0.5).clamp_min(1e-9)
        z_likelihood = torch.full(
            (x.shape[0], 1, max(1, y_hat.shape[2] // 4), max(1, y_hat.shape[3] // 4)),
            0.5,
            device=x.device,
            dtype=x.dtype,
        )
        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "likelihoods": {"y": likelihood, "z": z_likelihood},
        }

    def aux_loss(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)


def build_base_codec(config: Optional[dict] = None) -> nn.Module:
    cfg = config or {}
    if cfg.get("use_tiny", False) or cfg.get("name", "elic") == "tiny":
        return TinyBaseCodec(latent_channels=int(cfg.get("m", 320)))
    module = import_legacy("Network", kind="base")
    return module.TestModel(N=int(cfg.get("n", 192)), M=int(cfg.get("m", 320)))
