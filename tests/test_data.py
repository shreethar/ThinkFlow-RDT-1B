from __future__ import annotations

import json

import torch

from thinkflow_rdt.data import (
    ONLINE_SIGLIP_REQUIRED_KEYS,
    CachedFeatureDataset,
    RDTBatchCollator,
)


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


def test_cached_feature_dataset_reads_episode_pack(tmp_path):
    pack_path = tmp_path / "episode_000000000.pt"
    manifest_path = tmp_path / "manifest.jsonl"
    torch.save(
        {
            "cache_layout": "episode_pack",
            "dataset_id": "dummy",
            "episode_id": "episode_a",
            "num_samples": 2,
            "sample_step_idx": ["3", "7"],
            "sample_anchor_index": torch.tensor([0, 1]),
            "qwen_anchor_kv": torch.stack(
                [torch.ones(1, 8), torch.full((1, 8), 2.0)],
                dim=0,
            ),
            "qwen_anchor_step_idx": ["3", "7"],
            "qwen_anchor_kind": ["first_step", "first_gripper_change"],
            "lang_tokens": torch.randn(3, 4),
            "lang_mask": torch.tensor([True, True, True]),
            "state": torch.randn(2, 7),
            "actions": torch.randn(2, 5, 7),
            "action_time_mask": torch.ones(2, 5, dtype=torch.bool),
            "action_dim_mask": torch.ones(2, 7),
            "ctrl_freq": torch.tensor([10.0, 10.0]),
            "image_jpegs": [b"image-a", b"image-b"],
            "sample_image_indices": torch.tensor([[0, 1], [1, 0]]),
            "sample_image_mask": torch.tensor([[True, False], [True, True]]),
        },
        pack_path,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "path": pack_path.name,
                "cache_layout": "episode_pack",
                "num_samples": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = CachedFeatureDataset(
        manifest_path,
        required_keys=ONLINE_SIGLIP_REQUIRED_KEYS,
    )

    assert len(dataset) == 2
    sample = dataset[1]
    assert sample["episode_id"] == "episode_a"
    assert sample["step_idx"] == "7"
    assert sample["qwen_anchor_kind"] == "first_gripper_change"
    assert sample["qwen_kv"].shape == (1, 8)
    assert sample["qwen_kv"].unique().item() == 2.0
    assert sample["image_slot_jpegs"] == [b"image-b", b"image-a"]
    assert sample["image_slot_mask"].tolist() == [True, True]
