from __future__ import annotations

import torch

from thinkflow_rdt.data import RDTBatchCollator


def test_collator_masks_padding():
    collator = RDTBatchCollator(
        max_lang_tokens=4,
        image_tokens=6,
        pred_horizon=5,
        feature_dim=8,
        state_dim=7,
        action_dim=7,
    )
    sample = {
        "qwen_kv": torch.randn(1, 8),
        "lang_tokens": torch.randn(3, 8),
        "img_tokens": torch.randn(4, 8),
        "state": torch.randn(7),
        "actions": torch.randn(2, 7),
        "ctrl_freq": 20.0,
    }
    batch = collator([sample])
    assert batch["lang_mask"].tolist() == [[True, True, True, False]]
    assert batch["img_mask"].tolist() == [[True, True, True, True, False, False]]
    assert batch["action_time_mask"].tolist() == [[True, True, False, False, False]]
    assert batch["actions"].shape == (1, 5, 7)
    assert batch["qwen_kv"].shape == (1, 1, 8)


def test_collator_supports_separate_language_and_image_widths():
    collator = RDTBatchCollator(
        max_lang_tokens=2,
        image_tokens=3,
        pred_horizon=1,
        feature_dim=8,
        state_dim=7,
        action_dim=7,
        lang_token_dim=4,
        img_token_dim=6,
    )
    sample = {
        "qwen_kv": torch.randn(1, 8),
        "lang_tokens": torch.randn(2, 4),
        "img_tokens": torch.randn(3, 6),
        "state": torch.randn(7),
        "actions": torch.randn(1, 7),
        "ctrl_freq": 10.0,
    }

    batch = collator([sample])

    assert batch["lang_tokens"].shape == (1, 2, 4)
    assert batch["img_tokens"].shape == (1, 3, 6)
