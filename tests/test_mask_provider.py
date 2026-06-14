from __future__ import annotations

import torch

from cotam.masks import RandomMaskProvider, resize_mask_for_feature, token_mask


def test_random_mask_contract_and_modes() -> None:
    images = torch.zeros(2, 3, 384, 384)
    provider = RandomMaskProvider(seed=7)
    for mode in ["uniform", "patch", "gradient"]:
        mask = provider(images, mode=mode)
        assert mask.shape == (24, 24)
        assert mask.dtype == torch.long
        assert int(mask.min()) >= 0
        assert int(mask.max()) <= 9


def test_random_mask_seed_reproducible() -> None:
    images = torch.zeros(1, 3, 384, 384)
    a = RandomMaskProvider(seed=3)(images)
    b = RandomMaskProvider(seed=3)(images)
    assert torch.equal(a, b)


def test_mask_resize_and_token_weights() -> None:
    mask = torch.arange(24 * 24).view(24, 24) % 10
    resized = resize_mask_for_feature(mask, (12, 12))
    assert resized.shape == (12, 12)
    weights = token_mask(mask, 577, torch.device("cpu"))
    assert weights.shape == (1, 577, 1)
    assert weights[:, :1].eq(1).all()
