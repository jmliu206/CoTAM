from __future__ import annotations

import torch

from cotam.clip import FakeClipWrapper
from cotam.losses import LossWeights, MultiLevelClipLoss, select_loss_names


def test_loss_modes_select_expected_names() -> None:
    assert select_loss_names(0, "low_then_low_high", 5) == ["clip_layer1_distill"]
    assert select_loss_names(5, "low_then_low_high", 5) == [
        "clip_feature_distill",
        "clip_layer1_distill",
    ]
    assert select_loss_names(0, "high_only", 5) == ["clip_feature_distill"]


def test_multilevel_clip_loss_all_modes_forward() -> None:
    wrapper = FakeClipWrapper()
    criterion = MultiLevelClipLoss(wrapper, weights=LossWeights(total=2.0, layer1=0.1, other_layer=0.1))
    student = torch.randn(2, 577, 1024, requires_grad=True)
    labels = torch.ones(2, 77, dtype=torch.long)
    images = torch.rand(2, 3, 384, 384)
    mask = torch.ones(24, 24, dtype=torch.long)

    for names in [
        ["clip_layer1_distill"],
        ["clip_feature_distill"],
        ["clip_feature_distill", "clip_layer1_distill"],
        ["other_layer_loss"],
        ["clip", "clip_distill", "clip_layer2_cls_distill"],
    ]:
        losses = criterion(student, labels, images, names, mask=mask)
        assert "total_loss" in losses
        assert torch.isfinite(losses["total_loss"])

    losses = criterion(student, labels, images, ["other_layer_loss"], mask=None)
    losses["total_loss"].backward()
    assert student.grad is not None
