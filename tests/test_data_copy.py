from __future__ import annotations

import json
from itertools import groupby

import pytest
import torch

from thinkflow_rdt.data import (
    ONLINE_SIGLIP_REQUIRED_KEYS,
    CachedFeatureDataset,
    EpisodePackSampler,
    FixedStratifiedSampler,
    RDTBatchCollator,
    RDTOnlineSiglipBatchCollator,
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


def test_collators_preserve_cached_bfloat16_features():
    sample = {
        "qwen_kv": torch.randn(1, 8, dtype=torch.bfloat16),
        "lang_tokens": torch.randn(2, 4, dtype=torch.bfloat16),
        "img_tokens": torch.randn(3, 6, dtype=torch.bfloat16),
        "state": torch.randn(7, dtype=torch.float64),
        "actions": torch.randn(2, 7, dtype=torch.float64),
        "ctrl_freq": 10.0,
    }
    collator = RDTBatchCollator(
        max_lang_tokens=3,
        image_tokens=3,
        pred_horizon=2,
        feature_dim=8,
        state_dim=7,
        action_dim=7,
        lang_token_dim=4,
        img_token_dim=6,
        qwen_kv_dim=8,
    )

    batch = collator([sample])

    assert batch["qwen_kv"].dtype == torch.bfloat16
    assert batch["lang_tokens"].dtype == torch.bfloat16
    assert batch["state"].dtype == torch.float32
    assert batch["actions"].dtype == torch.float32

    online_collator = RDTOnlineSiglipBatchCollator(
        max_lang_tokens=3,
        pred_horizon=2,
        feature_dim=8,
        state_dim=7,
        action_dim=7,
        lang_token_dim=4,
        qwen_kv_dim=8,
    )
    online_sample = {
        key: value for key, value in sample.items() if key != "img_tokens"
    }
    online_sample.update(
        {
            "image_slot_jpegs": [b"image"],
            "image_slot_mask": torch.tensor([True]),
        }
    )

    online_batch = online_collator([online_sample])

    assert online_batch["qwen_kv"].dtype == torch.bfloat16
    assert online_batch["lang_tokens"].dtype == torch.bfloat16
    assert online_batch["state"].dtype == torch.float32
    assert online_batch["actions"].dtype == torch.float32


def test_collator_validates_qwen_width():
    collator = RDTBatchCollator(
        max_lang_tokens=2,
        image_tokens=2,
        pred_horizon=1,
        feature_dim=8,
        state_dim=7,
        action_dim=7,
        qwen_kv_dim=7,
    )
    sample = {
        "qwen_kv": torch.randn(1, 8),
        "lang_tokens": torch.randn(2, 8),
        "img_tokens": torch.randn(2, 8),
        "state": torch.randn(7),
        "actions": torch.randn(1, 7),
        "ctrl_freq": 10.0,
    }

    with pytest.raises(ValueError, match="Expected qwen_kv width 7, got 8"):
        collator([sample])


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


def test_cached_feature_dataset_remaps_legacy_uniform_anchor(tmp_path):
    pack_path = tmp_path / "episode.pt"
    manifest_path = tmp_path / "manifest.jsonl"
    torch.save(
        {
            "cache_layout": "episode_pack",
            "dataset_id": "bc_z",
            "episode_id": "episode_a",
            "num_samples": 2,
            "sample_step_idx": ["0", "5"],
            "sample_anchor_index": torch.tensor([0, 1]),
            "qwen_anchor_kv": torch.stack(
                [torch.ones(1, 8), torch.full((1, 8), 9.0)]
            ),
            "qwen_anchor_step_idx": ["0", "5"],
            "qwen_anchor_kind": ["first_step", "uniform"],
            "lang_tokens": torch.ones(2, 4),
            "lang_mask": torch.ones(2, dtype=torch.bool),
            "state": torch.zeros(2, 7),
            "actions": torch.zeros(2, 4, 7),
            "action_time_mask": torch.ones(2, 4, dtype=torch.bool),
            "action_dim_mask": torch.ones(2, 7),
            "ctrl_freq": torch.full((2,), 10.0),
            "image_jpegs": [b"jpeg"],
            "sample_image_indices": torch.zeros(2, 1, dtype=torch.long),
            "sample_image_mask": torch.ones(2, 1, dtype=torch.bool),
        },
        pack_path,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "path": pack_path.name,
                "cache_layout": "episode_pack",
                "dataset_id": "bc_z",
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
    sample = dataset[1]
    assert sample["qwen_kv"].unique().item() == 1.0
    assert sample["qwen_anchor_kind"] == "first_step"
    assert sample["qwen_anchor_original_kind"] == "uniform"
    assert sample["qwen_anchor_count"] == 1


def _write_episode_pack(
    path,
    *,
    dataset_id: str,
    num_samples: int,
    episode_id: str | None = None,
) -> None:
    torch.save(
        {
            "cache_layout": "episode_pack",
            "dataset_id": dataset_id,
            "episode_id": episode_id or f"{dataset_id}_episode",
            "num_samples": num_samples,
            "sample_step_idx": [str(index) for index in range(num_samples)],
            "sample_anchor_index": torch.zeros(num_samples, dtype=torch.long),
            "qwen_anchor_kv": torch.ones(1, 1, 8, dtype=torch.bfloat16),
            "qwen_anchor_step_idx": ["0"],
            "qwen_anchor_kind": ["first_step"],
            "lang_tokens": torch.ones(2, 4, dtype=torch.bfloat16),
            "lang_mask": torch.ones(2, dtype=torch.bool),
            "state": torch.ones(num_samples, 7),
            "actions": torch.ones(num_samples, 5, 7),
            "action_time_mask": torch.ones(num_samples, 5, dtype=torch.bool),
            "action_dim_mask": torch.ones(num_samples, 7),
            "ctrl_freq": torch.full((num_samples,), 10.0),
            "image_jpegs": [b"image"],
            "sample_image_indices": torch.zeros(num_samples, 1, dtype=torch.long),
            "sample_image_mask": torch.ones(num_samples, 1, dtype=torch.bool),
        },
        path,
    )


def test_dataset_filtering_ranges_and_episode_pack_sampler(tmp_path):
    first_path = tmp_path / "first.pt"
    second_path = tmp_path / "second.pt"
    _write_episode_pack(first_path, dataset_id="bc_z", num_samples=2)
    _write_episode_pack(second_path, dataset_id="fractal", num_samples=3)
    manifest_path = tmp_path / "manifest.jsonl"
    rows = [
        {
            "path": "excluded-does-not-need-to-exist.pt",
            "cache_layout": "episode_pack",
            "dataset_id": "bridge",
            "num_samples": 4,
        },
        {
            "path": first_path.name,
            "cache_layout": "episode_pack",
            "dataset_id": "bc_z",
            "num_samples": 2,
        },
        {
            "path": second_path.name,
            "cache_layout": "episode_pack",
            "dataset_id": "fractal",
            "num_samples": 3,
        },
    ]
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    dataset = CachedFeatureDataset(
        manifest_path,
        required_keys=ONLINE_SIGLIP_REQUIRED_KEYS,
        excluded_dataset_ids={"bridge"},
    )

    assert len(dataset) == 5
    assert dataset.episode_pack_ranges == (range(0, 2), range(2, 5))
    assert dataset.contiguous_ranges == dataset.episode_pack_ranges
    assert {dataset[index]["dataset_id"] for index in range(len(dataset))} == {
        "bc_z",
        "fractal",
    }

    sequential = list(EpisodePackSampler(dataset, shuffle=False))
    assert sequential == list(range(5))

    sampler = EpisodePackSampler(dataset, shuffle=True, seed=7)
    first_order = list(sampler)
    assert sorted(first_order) == list(range(5))
    pack_labels = [0 if index < 2 else 1 for index in first_order]
    assert len(list(groupby(pack_labels))) == 2

    sampler.set_epoch(1)
    assert list(sampler) != first_order


def test_fixed_stratified_sampler_balances_dataset_prefix(tmp_path):
    rows = []
    for dataset_id in ("bc_z", "fractal", "kuka"):
        path = tmp_path / f"{dataset_id}.pt"
        _write_episode_pack(path, dataset_id=dataset_id, num_samples=3)
        rows.append(
            {
                "path": path.name,
                "cache_layout": "episode_pack",
                "dataset_id": dataset_id,
                "num_samples": 3,
            }
        )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    dataset = CachedFeatureDataset(
        manifest_path,
        required_keys=ONLINE_SIGLIP_REQUIRED_KEYS,
    )

    first_order = list(FixedStratifiedSampler(dataset, seed=11))
    second_order = list(FixedStratifiedSampler(dataset, seed=11))
    assert first_order == second_order
    assert sorted(first_order) == list(range(len(dataset)))
    first_six_ids = [dataset[index]["dataset_id"] for index in first_order[:6]]
    assert first_six_ids == ["bc_z", "fractal", "kuka"] * 2


def test_fixed_stratified_sampler_reads_merged_manifest_and_diversifies_tasks(
    tmp_path,
):
    rows = []
    for suite in ("libero_goal", "libero_object"):
        for task_number in range(2):
            for demo_number in range(2):
                episode_id = (
                    f"{suite}_task_{task_number}_demo:demo_{demo_number}"
                )
                path = tmp_path / f"{suite}_{task_number}_{demo_number}.pt"
                _write_episode_pack(
                    path,
                    dataset_id=suite,
                    num_samples=2,
                    episode_id=episode_id,
                )
                rows.append(
                    {
                        "path": path.name,
                        "cache_layout": "episode_pack",
                        "first_dataset_id": suite,
                        "first_episode_id": episode_id,
                        "num_samples": 2,
                    }
                )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    dataset = CachedFeatureDataset(
        manifest_path,
        required_keys=ONLINE_SIGLIP_REQUIRED_KEYS,
    )

    first_order = list(FixedStratifiedSampler(dataset, seed=123))
    assert first_order == list(FixedStratifiedSampler(dataset, seed=123))
    assert dataset.contiguous_range_dataset_ids == (
        "libero_goal",
        "libero_goal",
        "libero_goal",
        "libero_goal",
        "libero_object",
        "libero_object",
        "libero_object",
        "libero_object",
    )
    first_four = [dataset[index] for index in first_order[:4]]
    assert [sample["dataset_id"] for sample in first_four] == [
        "libero_goal",
        "libero_object",
        "libero_goal",
        "libero_object",
    ]
    assert len({sample["episode_id"] for sample in first_four}) == 4
    assert len({sample["episode_id"].split("_demo:")[0] for sample in first_four}) == 4


def test_cached_feature_dataset_uses_restricted_torch_load(tmp_path, monkeypatch):
    pack_path = tmp_path / "pack.pt"
    _write_episode_pack(pack_path, dataset_id="dummy", num_samples=1)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "path": pack_path.name,
                "cache_layout": "episode_pack",
                "dataset_id": "dummy",
                "num_samples": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    original_load = torch.load
    calls = []

    def tracked_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    dataset = CachedFeatureDataset(
        manifest_path,
        required_keys=ONLINE_SIGLIP_REQUIRED_KEYS,
    )

    dataset[0]

    assert calls
    assert all(call.get("weights_only") is True for call in calls)


def test_episode_pack_sampler_supports_legacy_sample_manifest(tmp_path):
    sample_path = tmp_path / "sample.pt"
    torch.save(
        {
            "qwen_kv": torch.ones(1, 8),
            "lang_tokens": torch.ones(2, 8),
            "img_tokens": torch.ones(3, 8),
            "state": torch.ones(7),
            "actions": torch.ones(5, 7),
            "ctrl_freq": 10.0,
        },
        sample_path,
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(sample_path.name) + "\n", encoding="utf-8")

    dataset = CachedFeatureDataset(manifest_path)

    assert dataset.contiguous_ranges == (range(0, 1),)
    assert dataset.episode_pack_ranges == ()
    assert list(EpisodePackSampler(dataset, seed=3)) == [0]
    assert dataset[0]["qwen_kv"].shape == (1, 8)
