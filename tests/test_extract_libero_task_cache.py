from __future__ import annotations

import torch

from scripts.extract_libero_task_cache import subset_sample_shard
from scripts.train_b0_cached_features import parse_horizon_loss_schedule
from thinkflow_rdt.train import (
    attach_training_objective,
    infer_gripper_release_mask,
)


def test_subset_sample_shard_slices_only_sample_level_values() -> None:
    pack = {
        "cache_layout": "sample_shard",
        "num_samples": 4,
        "sample_start_index": 10,
        "sample_stop_index": 14,
        "qwen_kv": torch.arange(8).reshape(4, 2),
        "state": torch.arange(12).reshape(4, 3),
        "actions": torch.arange(20).reshape(4, 5),
        "ctrl_freq": torch.tensor([10.0, 11.0, 12.0, 13.0]),
        "metadata": [{"step_idx": str(index)} for index in range(4)],
        "sample_lang_index": torch.tensor([0, 0, 0, 0]),
        "lang_tokens": torch.ones(1, 2, 3),
        "image_jpegs": [b"shared-image-pool"],
    }

    result = subset_sample_shard(pack, [1, 3], sample_start_index=20)

    assert result["num_samples"] == 2
    assert result["sample_start_index"] == 20
    assert result["sample_stop_index"] == 22
    torch.testing.assert_close(result["state"], pack["state"][[1, 3]])
    torch.testing.assert_close(result["ctrl_freq"], torch.tensor([11.0, 13.0]))
    assert result["metadata"] == [{"step_idx": "1"}, {"step_idx": "3"}]
    torch.testing.assert_close(result["lang_tokens"], pack["lang_tokens"])
    assert result["image_jpegs"] == pack["image_jpegs"]


def test_parse_horizon_loss_schedule() -> None:
    weights = parse_horizon_loss_schedule(
        "1-4:5,5-8:3,9-16:2,17-64:1",
        64,
    )
    assert weights == [5.0] * 4 + [3.0] * 4 + [2.0] * 8 + [1.0] * 48


def test_attach_training_objective_uses_action_dtype_and_device() -> None:
    batch: dict[str, object] = {
        "actions": torch.zeros(2, 4, 10, dtype=torch.bfloat16),
    }

    attach_training_objective(
        batch,
        horizon_loss_weights=[5.0, 3.0, 2.0, 1.0],
        gripper_bce_weight=1.0,
        gripper_bce_logit_scale=5.0,
        rotation_geodesic_weight=1.0,
    )

    for key in (
        "horizon_loss_weights",
        "gripper_bce_weight",
        "gripper_bce_logit_scale",
        "rotation_geodesic_weight",
    ):
        value = batch[key]
        assert isinstance(value, torch.Tensor)
        assert value.dtype == torch.bfloat16
    torch.testing.assert_close(
        batch["horizon_loss_weights"],
        torch.tensor([5.0, 3.0, 2.0, 1.0], dtype=torch.bfloat16),
    )


def test_infer_gripper_release_mask_distinguishes_approach_and_release() -> None:
    target = torch.tensor(
        [
            [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        ]
    )
    time_mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, False, False, False, False],
            [True, True, True, True, True, True],
        ]
    )

    release = infer_gripper_release_mask(target, time_mask)

    torch.testing.assert_close(
        release,
        torch.tensor(
            [
                [False, False, False, False, True, True],
                [True, True, False, False, False, False],
                [False, False, False, False, False, False],
            ]
        ),
    )
