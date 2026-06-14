from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


MaskShape = Tuple[int, int]


def _image_to_mask_shape(images: torch.Tensor, patch_size: int) -> MaskShape:
    if images.ndim != 4:
        raise ValueError(f"Expected image tensor [B, C, H, W], got {tuple(images.shape)}")
    height, width = int(images.shape[-2]), int(images.shape[-1])
    return height // patch_size, width // patch_size


@dataclass
class RandomMaskProvider:
    """Random VBR mask provider compatible with the original 0..9 mask contract."""

    levels: int = 10
    patch_size: int = 16
    modes: Sequence[str] = ("uniform", "patch", "gradient")
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.levels <= 1:
            raise ValueError("levels must be greater than 1")
        self._generator = torch.Generator()
        if self.seed is not None:
            self._generator.manual_seed(int(self.seed))

    def __call__(self, images: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        h, w = _image_to_mask_shape(images, self.patch_size)
        selected = mode or self._choose_mode()
        mask = self.generate(h, w, selected)
        return mask.to(device=images.device)

    def _choose_mode(self) -> str:
        index = int(torch.randint(len(self.modes), (1,), generator=self._generator).item())
        return str(self.modes[index])

    def generate(self, height: int, width: int, mode: str) -> torch.Tensor:
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid mask shape: {(height, width)}")
        if mode == "uniform":
            return self._uniform(height, width)
        if mode == "patch":
            return self._patch(height, width)
        if mode == "gradient":
            return self._gradient(height, width)
        raise ValueError(f"Unknown mask mode: {mode}")

    def _uniform(self, height: int, width: int) -> torch.Tensor:
        value = int(torch.randint(self.levels, (1,), generator=self._generator).item())
        return torch.full((height, width), value, dtype=torch.long)

    def _patch(self, height: int, width: int) -> torch.Tensor:
        patch_options = torch.tensor([1, 2, 4], dtype=torch.long)
        index = int(torch.randint(len(patch_options), (1,), generator=self._generator).item())
        patch = int(patch_options[index].item())
        grid_h = (height + patch - 1) // patch
        grid_w = (width + patch - 1) // patch
        values = torch.randint(self.levels, (grid_h, grid_w), generator=self._generator)
        return values.repeat_interleave(patch, dim=0).repeat_interleave(patch, dim=1)[:height, :width]

    def _gradient(self, height: int, width: int) -> torch.Tensor:
        ramp_y = torch.linspace(0, 1, steps=height).unsqueeze(1)
        ramp_x = torch.linspace(0, 1, steps=width).unsqueeze(0)
        variants = (
            ramp_x.expand(height, width),
            ramp_y.expand(height, width),
            (ramp_x + ramp_y) / 2,
            (1 - ramp_x + ramp_y) / 2,
        )
        index = int(torch.randint(len(variants), (1,), generator=self._generator).item())
        values = (variants[index] * (self.levels - 1)).round().long()
        return values.clamp(0, self.levels - 1)


class ClipGuidedMaskProvider:
    """Placeholder for the paper's CLIP-guided semantic mask path."""

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "CLIP-guided masks are intentionally left as a separate migration task."
        )


def resize_mask_for_feature(mask: torch.Tensor, feature_hw: tuple[int, int], mode: str = "nearest") -> torch.Tensor:
    """Resize an HxW integer mask to a feature-map shape."""

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {tuple(mask.shape)}")
    resized = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(),
        size=feature_hw,
        mode=mode,
    )
    return resized.squeeze(0).squeeze(0).long()


def token_mask(mask: Optional[torch.Tensor], num_tokens: int, device: torch.device) -> torch.Tensor:
    """Convert a spatial 0..9 mask to CLIP token weights with a prepended CLS weight."""

    if mask is None:
        return torch.ones(1, num_tokens, 1, device=device)
    token_size = int((num_tokens - 1) ** 0.5)
    if token_size * token_size != num_tokens - 1:
        raise ValueError(f"Cannot infer square token grid from num_tokens={num_tokens}")
    weights = F.interpolate(
        mask.to(device=device).unsqueeze(0).unsqueeze(0).float(),
        size=(token_size, token_size),
        mode="bilinear",
        align_corners=False,
    ).view(1, -1, 1)
    max_value = max(float(mask.max().item()), 1.0)
    weights = weights / max_value
    cls = torch.ones(1, 1, 1, device=device, dtype=weights.dtype)
    return torch.cat([cls, weights], dim=1)


def build_mask_provider(config: dict) -> Union[RandomMaskProvider, ClipGuidedMaskProvider]:
    provider = config.get("provider", "random")
    if provider == "random":
        modes: Iterable[str] = config.get("modes", ("uniform", "patch", "gradient"))
        return RandomMaskProvider(
            levels=int(config.get("levels", 10)),
            patch_size=int(config.get("patch_size", 16)),
            modes=tuple(modes),
            seed=config.get("seed"),
        )
    if provider == "clip_guided":
        return ClipGuidedMaskProvider()
    raise ValueError(f"Unknown mask provider: {provider}")
