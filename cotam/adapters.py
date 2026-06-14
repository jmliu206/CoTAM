from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return super().forward(x.float()).to(dtype)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = LayerNorm(width)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(width, width * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(width * 4, width)),
                ]
            )
        )
        self.ln_2 = LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.ln_1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class LinearEncoder(nn.Module):
    """Map codec latents to CLIP ViT-L/14 token features."""

    def __init__(
        self,
        in_features: int = 320,
        out_features: int = 1024,
        input_resolution: int = 384,
        patch_size: int = 16,
        transformer_layers: int = 1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        token_count = (input_resolution // patch_size) ** 2 + 1
        scale = out_features ** -0.5
        self.trans_latent_dim = nn.Linear(in_features, out_features)
        self.class_embedding = nn.Parameter(scale * torch.randn(out_features))
        self.positional_embedding = nn.Parameter(scale * torch.randn(token_count, out_features))
        self.ln_pre = LayerNorm(out_features)
        self.transformer = Transformer(out_features, transformer_layers, out_features // 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected latent [B, C, H, W], got {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        if channels != self.in_features:
            raise ValueError(f"Expected {self.in_features} channels, got {channels}")
        x = x.view(batch, channels, height * width).permute(0, 2, 1)
        x = self.trans_latent_dim(x)
        cls = self.class_embedding.to(x.dtype).expand(batch, 1, -1)
        x = torch.cat([cls, x], dim=1)
        if x.shape[1] != self.positional_embedding.shape[0]:
            raise ValueError(
                "Latent spatial size does not match adapter positional embedding: "
                f"{x.shape[1]} vs {self.positional_embedding.shape[0]}"
            )
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        return x.permute(1, 0, 2)


# Backward-compatible alias for the name used by the original scripts.
Linear_Encoder = LinearEncoder
