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
    assert batch["dataset_id"] == ["unknown"]


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


def test_collator_flips_normalized_closed_gripper_by_sign() -> None:
    collator = RDTBatchCollator(
        max_lang_tokens=1,
        image_tokens=1,
        pred_horizon=2,
        feature_dim=8,
        state_dim=7,
        action_dim=7,
    )
    actions = torch.zeros(2, 7)
    actions[:, 6] = torch.tensor([-1.0, 1.0])
    sample = {
        "qwen_kv": torch.zeros(1, 8),
        "lang_tokens": torch.zeros(1, 8),
        "img_tokens": torch.zeros(1, 8),
        "state": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "actions": actions,
        "actions_normalized": True,
        "ctrl_freq": 10.0,
    }

    batch = collator([sample])

    assert batch["state"][0, 6].item() == 1.0
    assert batch["actions"][0, :, 6].tolist() == [1.0, -1.0]


def test_collator_preserves_unequal_raw_libero_state_and_action_dimensions():
    collator = RDTBatchCollator(
        max_lang_tokens=2,
        image_tokens=3,
        pred_horizon=2,
        feature_dim=8,
        state_dim=11,
        action_dim=10,
        convert_cached_gripper_closed_to_open=False,
    )
    state = torch.tensor([0.0] * 9 + [0.04, -0.04])
    actions = torch.zeros(1, 10)
    actions[0, 9] = -1.0
    sample = {
        "qwen_kv": torch.randn(1, 8),
        "lang_tokens": torch.randn(2, 8),
        "img_tokens": torch.randn(3, 8),
        "state": state,
        "state_dim_mask": torch.ones(11),
        "actions": actions,
        "ctrl_freq": 20.0,
    }

    batch = collator([sample])

    torch.testing.assert_close(batch["state"][0], state)
    assert batch["state_dim_mask"].shape == (1, 11)
    assert batch["actions"][0, 0, 9].item() == -1.0
    assert batch["action_dim_mask"].shape == (1, 10)


def test_collator_builds_native_128d_targets_with_exact_ten_dim_mask(
    tmp_path,
) -> None:
    stats_path = tmp_path / "audit.json"
    stats_path.write_text(
        json.dumps(
            {
                "action_normalization": {
                    "q01": [-1.0] * 6 + [0.0],
                    "q99": [1.0] * 7,
                }
            }
        )
    )
    collator = RDTBatchCollator(
        max_lang_tokens=1,
        image_tokens=1,
        pred_horizon=2,
        feature_dim=8,
        state_dim=128,
        action_dim=128,
        cache_state_dim=7,
        cache_action_dim=7,
        native_rdt_128=True,
        convert_cached_gripper_closed_to_open=False,
        action_stats_paths={"dummy": str(stats_path)},
    )
    sample = {
        "qwen_kv": torch.zeros(1, 8),
        "lang_tokens": torch.zeros(1, 8),
        "img_tokens": torch.zeros(1, 8),
        "state": torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]),
        "actions": torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0],
                [0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        "actions_normalized": True,
        "dataset_id": "dummy",
        "ctrl_freq": 10.0,
    }

    batch = collator([sample])

    expected_indices = [10, *range(30, 39)]
    assert batch["state"].shape == (1, 128)
    assert batch["actions"].shape == (1, 2, 128)
    assert torch.nonzero(batch["state_dim_mask"][0]).flatten().tolist() == expected_indices
    assert torch.nonzero(batch["action_dim_mask"][0]).flatten().tolist() == expected_indices
    torch.testing.assert_close(
        batch["actions"][0, 0, 33:39],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    )
    assert batch["actions"][0, 0, 10].item() == 1.0
    assert batch["actions"][0, 1, 10].item() == -1.0


def test_collator_maps_raw_libero_11d_10d_cache_to_native_128() -> None:
    collator = RDTBatchCollator(
        max_lang_tokens=1,
        image_tokens=1,
        pred_horizon=3,
        feature_dim=8,
        state_dim=128,
        action_dim=128,
        cache_state_dim=11,
        cache_action_dim=10,
        native_rdt_128=True,
        convert_cached_gripper_closed_to_open=False,
    )
    state = torch.tensor(
        [1.0, 2.0, 3.0, *range(4, 10), 0.04, -0.04],
        dtype=torch.float32,
    )
    action = torch.tensor(
        [[0.1, 0.2, 0.3, *range(4, 10), -1.0]],
        dtype=torch.float32,
    )
    sample = {
        "qwen_kv": torch.zeros(1, 8),
        "lang_tokens": torch.zeros(1, 8),
        "img_tokens": torch.zeros(1, 8),
        "state": state,
        "actions": action,
        "actions_normalized": False,
        "dataset_id": "libero_10",
        "ctrl_freq": 20.0,
    }

    batch = collator([sample])

    assert torch.nonzero(batch["state_dim_mask"][0]).flatten().tolist() == [
        10,
        11,
        *range(30, 39),
    ]
    assert torch.nonzero(batch["action_dim_mask"][0]).flatten().tolist() == [
        10,
        *range(30, 39),
    ]
    torch.testing.assert_close(batch["state"][0, 30:39], state[:9])
    torch.testing.assert_close(batch["state"][0, 10:12], state[9:11])
    torch.testing.assert_close(batch["actions"][0, 0, 30:39], action[0, :9])
    assert batch["actions"][0, 0, 10].item() == -1.0
    assert batch["action_time_mask"].tolist() == [[True, False, False]]
    # Padded commands remain zero but are not supervised; the model separately
    # supplies a full temporal conditioning mask to the action adaptor.
    assert batch["actions"][0, 1:].count_nonzero().item() == 0


def test_collator_maps_libero_joint_state_and_eef_delta_to_author_slots() -> None:
    collator = RDTBatchCollator(
        max_lang_tokens=1,
        image_tokens=1,
        pred_horizon=2,
        feature_dim=8,
        state_dim=128,
        action_dim=128,
        cache_state_dim=8,
        cache_action_dim=7,
        native_rdt_128=True,
        native_rdt_128_mapping="libero_joint_eef_delta",
        convert_cached_gripper_closed_to_open=False,
    )
    joints = torch.arange(7, dtype=torch.float32) / 10
    state = torch.tensor(
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -0.04245, 0.05185]
    )
    actions = torch.tensor(
        [[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 1.0]],
        dtype=torch.float32,
    )
    sample = {
        "qwen_kv": torch.zeros(1, 8),
        "lang_tokens": torch.zeros(1, 8),
        "img_tokens": torch.zeros(1, 8),
        "state": state,
        "joint_state": joints,
        "actions": actions,
        "actions_normalized": False,
        "dataset_id": "libero_spatial",
        "ctrl_freq": 20.0,
    }

    batch = collator([sample])

    assert torch.nonzero(batch["state_dim_mask"][0]).flatten().tolist() == [
        *range(7),
        10,
        11,
    ]
    assert torch.nonzero(batch["action_dim_mask"][0]).flatten().tolist() == [
        10,
        *range(39, 45),
    ]
    torch.testing.assert_close(batch["state"][0, :7], joints)
    torch.testing.assert_close(
        batch["state"][0, 10:12], torch.tensor([0.0, 1.0])
    )
    torch.testing.assert_close(batch["actions"][0, 0, 39:45], actions[0, :6])
    assert batch["actions"][0, 0, 10].item() == 1.0


def test_collator_maps_cached_joint7_normalized_gripper2_without_side_channel() -> None:
    collator = RDTBatchCollator(
        max_lang_tokens=1,
        image_tokens=1,
        pred_horizon=2,
        feature_dim=8,
        state_dim=128,
        action_dim=128,
        cache_state_dim=9,
        cache_action_dim=7,
        native_rdt_128=True,
        native_rdt_128_mapping="libero_joint_eef_delta",
        convert_cached_gripper_closed_to_open=False,
    )
    state = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0, 1.0])
    actions = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, -1.0]])
    batch = collator(
        [
            {
                "qwen_kv": torch.zeros(1, 8),
                "lang_tokens": torch.zeros(1, 8),
                "img_tokens": torch.zeros(1, 8),
                "state": state,
                "actions": actions,
                "actions_normalized": False,
                "dataset_id": "libero_spatial",
                "ctrl_freq": 20.0,
            }
        ]
    )

    torch.testing.assert_close(batch["state"][0, :7], state[:7])
    torch.testing.assert_close(batch["state"][0, 10:12], state[7:9])
    torch.testing.assert_close(batch["actions"][0, 0, 39:45], actions[0, :6])
    assert batch["actions"][0, 0, 10].item() == -1.0


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


def test_cached_feature_dataset_reads_sample_shard(tmp_path):
    shard_path = tmp_path / "shard_000000000.pt"
    manifest_path = tmp_path / "manifest.jsonl"
    torch.save(
        {
            "cache_layout": "sample_shard",
            "feature_type": "latent_student_spatial_kv",
            "num_samples": 2,
            "qwen_kv": torch.stack(
                [torch.ones(5, 8), torch.full((5, 8), 3.0)],
                dim=0,
            ),
            "lang_tokens": [torch.randn(2, 4), torch.randn(3, 4)],
            "lang_mask": [torch.ones(2, dtype=torch.bool), torch.ones(3, dtype=torch.bool)],
            "sample_lang_index": torch.tensor([0, 1]),
            "instructions": ["first instruction", "second instruction"],
            "state": torch.randn(2, 7),
            "actions": torch.randn(2, 5, 7),
            "action_time_mask": torch.ones(2, 5, dtype=torch.bool),
            "action_dim_mask": torch.ones(2, 7),
            "ctrl_freq": torch.tensor([10.0, 3.0]),
            "metadata": [
                {"dataset_id": "dummy", "episode_id": "ep_a", "step_idx": "1"},
                {"dataset_id": "dummy", "episode_id": "ep_a", "step_idx": "2"},
            ],
            "image_jpegs": [b"image-a", b"image-b"],
            "sample_image_indices": torch.tensor([[0, 1], [1, 0]]),
            "sample_image_mask": torch.tensor([[True, False], [True, True]]),
            "latent_waypoints": torch.randn(2, 5, 2),
            "qwen_hidden_states": torch.randn(2, 5, 8),
        },
        shard_path,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "path": shard_path.name,
                "cache_layout": "sample_shard",
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
    assert sample["episode_id"] == "ep_a"
    assert sample["step_idx"] == "2"
    assert sample["qwen_kv"].shape == (5, 8)
    assert sample["qwen_kv"].unique().item() == 3.0
    assert sample["lang_tokens"].shape == (3, 4)
    assert sample["instruction"] == "second instruction"
    assert sample["image_slot_jpegs"] == [b"image-b", b"image-a"]
    assert sample["image_slot_mask"].tolist() == [True, True]
    assert sample["latent_waypoints"].shape == (5, 2)
    assert sample["qwen_hidden_states"].shape == (5, 8)


def test_cached_feature_dataset_reads_lossless_image_arrays(tmp_path):
    shard_path = tmp_path / "shard_000000000.pt"
    manifest_path = tmp_path / "manifest.jsonl"
    image_a = torch.zeros((8, 9, 3), dtype=torch.uint8)
    image_b = torch.full((8, 9, 3), 127, dtype=torch.uint8)
    torch.save(
        {
            "cache_layout": "sample_shard",
            "feature_type": "latent_student_spatial_kv",
            "num_samples": 1,
            "qwen_kv": torch.ones(1, 5, 8),
            "lang_tokens": [torch.randn(2, 4)],
            "lang_mask": [torch.ones(2, dtype=torch.bool)],
            "sample_lang_index": torch.tensor([0]),
            "state": torch.randn(1, 9),
            "actions": torch.randn(1, 64, 7),
            "action_time_mask": torch.ones(1, 64, dtype=torch.bool),
            "action_dim_mask": torch.ones(1, 7),
            "ctrl_freq": torch.tensor([20.0]),
            "metadata": [
                {"dataset_id": "libero_spatial", "episode_id": "demo_0", "step_idx": "0"}
            ],
            "image_arrays": [image_a, image_b],
            "sample_image_indices": torch.tensor([[0, 1]]),
            "sample_image_mask": torch.tensor([[True, True]]),
        },
        shard_path,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "path": shard_path.name,
                "cache_layout": "sample_shard",
                "num_samples": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = CachedFeatureDataset(
        manifest_path,
        required_keys=ONLINE_SIGLIP_REQUIRED_KEYS,
    )
    sample = dataset[0]

    assert len(sample["image_slot_jpegs"]) == 2
    assert torch.equal(sample["image_slot_jpegs"][0], image_a)
    assert torch.equal(sample["image_slot_jpegs"][1], image_b)


def test_collator_batches_required_hidden_waypoint_plan_features():
    collator = RDTBatchCollator(
        max_lang_tokens=2,
        image_tokens=3,
        pred_horizon=2,
        feature_dim=8,
        state_dim=7,
        action_dim=7,
        plan_hidden_dim=8,
        spatial_token_count=5,
        waypoint_dim=2,
        require_plan_features=True,
    )
    sample = {
        "qwen_kv": torch.randn(5, 8),
        "qwen_hidden_states": torch.randn(5, 8),
        "latent_waypoints": torch.rand(5, 2),
        "lang_tokens": torch.randn(2, 8),
        "img_tokens": torch.randn(3, 8),
        "state": torch.randn(7),
        "actions": torch.randn(2, 7),
        "ctrl_freq": 20.0,
    }

    batch = collator([sample])

    assert batch["qwen_hidden_states"].shape == (1, 5, 8)
    assert batch["latent_waypoints"].shape == (1, 5, 2)
    assert batch["plan_mask"].tolist() == [[True] * 5]


def test_collator_batches_b0_hidden_without_waypoints():
    collator = RDTBatchCollator(
        max_lang_tokens=2,
        image_tokens=3,
        pred_horizon=2,
        feature_dim=8,
        state_dim=7,
        action_dim=7,
        plan_hidden_dim=8,
        spatial_token_count=1,
        require_hidden_features=True,
    )
    sample = {
        "qwen_kv": torch.randn(1, 8),
        "qwen_hidden_states": torch.randn(1, 8),
        "lang_tokens": torch.randn(2, 8),
        "img_tokens": torch.randn(3, 8),
        "state": torch.randn(7),
        "actions": torch.randn(2, 7),
        "ctrl_freq": 20.0,
    }

    batch = collator([sample])

    assert batch["qwen_hidden_states"].shape == (1, 1, 8)
    assert batch["plan_mask"].tolist() == [[True]]
    assert "latent_waypoints" not in batch
