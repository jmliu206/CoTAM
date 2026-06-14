from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cotam.clip.wrapper import ClipOutputs
from cotam.masks import token_mask


def select_loss_names(epoch: int, mode: str = "low_then_low_high", warmup_epochs: int = 5) -> list[str]:
    if mode == "low_only":
        return ["clip_layer1_distill"]
    if mode == "high_only":
        return ["clip_feature_distill"]
    if mode == "other_layer":
        return ["other_layer_loss"] if epoch < warmup_epochs else ["clip_feature_distill", "other_layer_loss"]
    if mode == "low_then_low_high":
        return ["clip_layer1_distill"] if epoch < warmup_epochs else [
            "clip_feature_distill",
            "clip_layer1_distill",
        ]
    if mode == "all":
        return ["clip", "clip_distill", "clip_feature_distill", "clip_layer1_distill", "clip_layer2_cls_distill"]
    raise ValueError(f"Unknown loss mode: {mode}")


@dataclass
class LossWeights:
    total: float = 11.52
    layer1: float = 0.1
    other_layer: float = 0.1


class MultiLevelClipLoss(nn.Module):
    def __init__(
        self,
        clip_wrapper,
        weights: Optional[LossWeights] = None,
        other_layer_index: int = 2,
    ) -> None:
        super().__init__()
        self.clip = clip_wrapper
        self.weights = weights or LossWeights()
        self.other_layer_index = other_layer_index

    def dist_loss(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean()

    def forward(
        self,
        student_tokens: torch.Tensor,
        labels: torch.Tensor,
        original_images: torch.Tensor,
        loss_names: Iterable[str],
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        selected = set(loss_names)
        device = student_tokens.device

        student = self.clip.run_from_tokens(student_tokens)
        with torch.no_grad():
            teacher = self.clip.run_image(original_images)
            text_features = self.clip.encode_text(labels)

        result: dict[str, torch.Tensor] = {}
        zero = student_tokens.sum() * 0.0

        student_image = F.normalize(student.image_features, dim=-1)
        teacher_image = F.normalize(teacher.image_features, dim=-1)
        text = F.normalize(text_features, dim=-1)
        logits_student_i = student_image @ text.t()
        logits_teacher_i = teacher_image @ text.t()
        logits_student_t = text @ student_image.t()
        logits_teacher_t = text @ teacher_image.t()

        if "clip_layer1_distill" in selected:
            weights = token_mask(mask, student.all_layer_tokens[0].shape[0], device)
            lhs = student.all_layer_tokens[0].permute(1, 0, 2)
            rhs = teacher.all_layer_tokens[0].permute(1, 0, 2)
            result["clip_layer1_distill"] = (
                F.mse_loss(lhs, rhs, reduction="none") * weights
            ).mean() * self.weights.layer1
        else:
            result["clip_layer1_distill"] = zero

        if "clip_layer2_cls_distill" in selected:
            lhs = student.all_layer_tokens[1].permute(1, 0, 2)[:, :1, :]
            rhs = teacher.all_layer_tokens[1].permute(1, 0, 2)[:, :1, :]
            result["clip_layer2_cls_distill"] = F.mse_loss(lhs, rhs)
        else:
            result["clip_layer2_cls_distill"] = zero

        if "other_layer_loss" in selected:
            idx = min(self.other_layer_index, len(student.all_layer_tokens) - 1)
            weights = token_mask(mask, student.all_layer_tokens[idx].shape[0], device)
            lhs = student.all_layer_tokens[idx].permute(1, 0, 2)
            rhs = teacher.all_layer_tokens[idx].permute(1, 0, 2)
            result["other_layer_loss"] = (
                F.mse_loss(lhs, rhs, reduction="none") * weights
            ).mean() * self.weights.other_layer
        else:
            result["other_layer_loss"] = zero

        if "clip_feature_distill" in selected:
            result["clip_feature_distill"] = F.mse_loss(student.image_features, teacher.image_features)
        else:
            result["clip_feature_distill"] = zero

        if "clip_distill" in selected:
            result["clip_distill"] = (
                self.dist_loss(logits_teacher_i, logits_student_i)
                + self.dist_loss(logits_teacher_t, logits_student_t)
            ) / 2
        else:
            result["clip_distill"] = zero

        if "clip" in selected:
            targets = torch.arange(student_tokens.shape[0], device=device)
            result["clip"] = (
                F.cross_entropy(logits_student_i, targets)
                + F.cross_entropy(logits_student_t, targets)
            ) / 2
        else:
            result["clip"] = zero

        result["total_loss"] = sum(result[name] for name in selected) * self.weights.total
        return result


def build_loss(config: dict, clip_wrapper) -> MultiLevelClipLoss:
    weights = LossWeights(
        total=float(config.get("total_weight", 11.52)),
        layer1=float(config.get("layer1_weight", 0.1)),
        other_layer=float(config.get("other_layer_weight", 0.1)),
    )
    return MultiLevelClipLoss(
        clip_wrapper=clip_wrapper,
        weights=weights,
        other_layer_index=int(config.get("other_layer_index", 2)),
    )
