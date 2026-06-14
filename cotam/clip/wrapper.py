from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize

from cotam.codecs.legacy import add_legacy_paths


@dataclass
class ClipOutputs:
    image_features: torch.Tensor
    tokens: torch.Tensor
    all_layer_tokens: list[torch.Tensor]


class FakeClipWrapper:
    """Deterministic CLIP-shaped wrapper for tests and smoke training."""

    def __init__(
        self,
        token_dim: int = 1024,
        feature_dim: int = 768,
        token_count: int = 577,
        layers: int = 4,
    ) -> None:
        self.token_dim = token_dim
        self.feature_dim = feature_dim
        self.token_count = token_count
        self.layers = layers

    def token_features_from_image(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        pooled = F.adaptive_avg_pool2d(images, (24, 24)).mean(dim=1)
        patches = pooled.view(batch, -1, 1).repeat(1, 1, self.token_dim)
        cls = images.mean(dim=(1, 2, 3), keepdim=False).view(batch, 1, 1).repeat(1, 1, self.token_dim)
        return torch.cat([cls, patches], dim=1)

    def run_from_tokens(self, tokens: torch.Tensor) -> ClipOutputs:
        all_layers = []
        current = tokens
        for layer in range(self.layers):
            current = current + (layer + 1) * 0.001
            all_layers.append(current.permute(1, 0, 2))
        image_features = current[:, 0, : self.feature_dim]
        projected_tokens = current[:, :, : self.feature_dim]
        return ClipOutputs(image_features=image_features, tokens=projected_tokens, all_layer_tokens=all_layers)

    def run_image(self, images: torch.Tensor) -> ClipOutputs:
        return self.run_from_tokens(self.token_features_from_image(images))

    def encode_text(self, labels: torch.Tensor) -> torch.Tensor:
        batch = labels.shape[0]
        text = labels.float()
        base = text.mean(dim=1, keepdim=True) / 1000.0
        offsets = torch.linspace(0, 1, self.feature_dim, device=labels.device).view(1, -1)
        return base.repeat(1, self.feature_dim) + offsets.repeat(batch, 1) * 0.001


class ModifiedClipWrapper:
    """Adapter around the repo's modified CLIP implementation."""

    def __init__(self, model_name: str = "ViT-L/14@336px", device: Optional[torch.device] = None) -> None:
        add_legacy_paths("examples")
        import CLIP_modify.clip as clip_modify  # type: ignore

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = clip_modify.load(model_name, device=self.device)
        self.model.eval()
        self.processing = Compose(
            [
                Resize((336, 336), interpolation=InterpolationMode.BICUBIC),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

    def token_features_from_image(self, images: torch.Tensor) -> torch.Tensor:
        images = self.processing(images.to(self.device))
        return self.model.visual(images.type(self.model.dtype), start_layer=0, return_layer=0, feature_input=False)

    def run_from_tokens(self, tokens: torch.Tensor) -> ClipOutputs:
        image_features, tokens_out, all_tokens = self.model.encode_image(
            tokens.to(self.device),
            start_layer=0,
            output_all_layer_tokens=True,
            feature_input=True,
        )
        return ClipOutputs(image_features=image_features, tokens=tokens_out, all_layer_tokens=all_tokens)

    def run_image(self, images: torch.Tensor) -> ClipOutputs:
        images = self.processing(images.to(self.device))
        image_features, tokens_out, all_tokens = self.model.encode_image(
            images,
            start_layer=0,
            output_all_layer_tokens=True,
            feature_input=False,
        )
        return ClipOutputs(image_features=image_features, tokens=tokens_out, all_layer_tokens=all_tokens)

    def encode_text(self, labels: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(labels.to(self.device))


def build_clip_wrapper(config: Optional[dict] = None, device: Optional[torch.device] = None):
    cfg = config or {}
    if cfg.get("use_fake", False):
        return FakeClipWrapper(
            token_dim=int(cfg.get("token_dim", 1024)),
            feature_dim=int(cfg.get("feature_dim", 768)),
            token_count=int(cfg.get("token_count", 577)),
        )
    return ModifiedClipWrapper(model_name=cfg.get("model_name", "ViT-L/14@336px"), device=device)
