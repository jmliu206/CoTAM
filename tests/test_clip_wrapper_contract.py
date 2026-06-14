from __future__ import annotations

import torch

from cotam.clip import FakeClipWrapper


def test_fake_clip_wrapper_contract() -> None:
    wrapper = FakeClipWrapper()
    images = torch.rand(2, 3, 384, 384)
    tokens = wrapper.token_features_from_image(images)
    assert tokens.shape == (2, 577, 1024)

    out_from_image = wrapper.run_image(images)
    out_from_tokens = wrapper.run_from_tokens(tokens)
    assert out_from_image.image_features.shape == (2, 768)
    assert out_from_tokens.image_features.shape == (2, 768)
    assert len(out_from_tokens.all_layer_tokens) >= 3
    assert out_from_tokens.all_layer_tokens[0].shape == (577, 2, 1024)

    labels = torch.ones(2, 77, dtype=torch.long)
    text = wrapper.encode_text(labels)
    assert text.shape == (2, 768)
