from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.overfit_libero_cached import (
    RAW_ACTION_NAMES,
    attach_gripper_training_weights,
    build_gripper_release_masks,
    build_release_group_sampling_probabilities,
    decoded_command_metrics,
    horizon_weight_vector,
    wandb_sampling_metrics,
)
from thinkflow_rdt.adapters.libero import libero_action_to_rdt


def test_overfit_horizon_weights_use_non_overlapping_ranges() -> None:
    weights = horizon_weight_vector(64).numpy()
    np.testing.assert_array_equal(weights[:4], 5.0)
    np.testing.assert_array_equal(weights[4:8], 3.0)
    np.testing.assert_array_equal(weights[8:16], 2.0)
    np.testing.assert_array_equal(weights[16:], 1.0)


def test_real_sampling_metrics_are_exact_for_perfect_raw_commands() -> None:
    raw = np.zeros((2, 64, 7), dtype=np.float32)
    raw[..., :6] = np.linspace(-0.8, 0.8, 64, dtype=np.float32)[None, :, None]
    raw[..., 6] = np.where(np.arange(64) < 32, -1.0, 1.0)
    encoded = libero_action_to_rdt(raw)
    mask = np.ones((2, 64), dtype=bool)

    metrics = decoded_command_metrics(encoded, encoded, mask)

    for horizon in (1, 4, 8, 64):
        assert metrics["horizon_rmse"][str(horizon)]["command_rmse_7d"] == 0.0
    assert metrics["overall_sign_agreement"] == 1.0
    assert metrics["overall_saturation_fraction"] == np.mean(np.abs(raw) >= 1 - 1e-6)
    assert metrics["gripper"]["accuracy"] == 1.0
    assert metrics["gripper"]["f1"] == 1.0
    assert set(metrics["per_dimension_correlation"]) == set(RAW_ACTION_NAMES)


def test_gripper_metrics_treat_nonnegative_raw_command_as_positive() -> None:
    target_raw = np.zeros((1, 4, 7), dtype=np.float32)
    target_raw[..., 6] = [-1.0, 1.0, 1.0, -1.0]
    predicted_raw = target_raw.copy()
    predicted_raw[..., 6] = [-1.0, -1.0, 1.0, 1.0]
    mask = np.ones((1, 4), dtype=bool)

    metrics = decoded_command_metrics(
        libero_action_to_rdt(predicted_raw),
        libero_action_to_rdt(target_raw),
        mask,
    )

    assert metrics["gripper"]["accuracy"] == 0.5
    assert metrics["gripper"]["precision"] == 0.5
    assert metrics["gripper"]["recall"] == 0.5
    assert metrics["gripper"]["f1"] == 0.5


def test_phase_metrics_separate_approach_hold_and_release() -> None:
    target_raw = np.zeros((1, 6, 7), dtype=np.float32)
    target_raw[..., 6] = [-1, -1, 1, 1, -1, -1]
    predicted_raw = target_raw.copy()
    predicted_raw[..., 6] = [-1, -1, 1, 1, 1, -1]
    release_mask = np.array([[False, False, False, False, True, True]])

    metrics = decoded_command_metrics(
        libero_action_to_rdt(predicted_raw),
        libero_action_to_rdt(target_raw),
        np.ones((1, 6), dtype=bool),
        release_mask,
    )

    phases = metrics["gripper_phase"]
    assert phases["approach_open"]["accuracy"] == 1.0
    assert phases["close_hold"]["accuracy"] == 1.0
    assert phases["release_open"]["accuracy"] == 0.5
    assert phases["release_transition_timing"]["mae_steps"] == 1.0


def test_release_masks_find_final_positive_to_negative_transition() -> None:
    samples = []
    signs = [-1, -1, 1, 1, -1, -1]
    for step, sign in enumerate(signs):
        actions = torch.zeros(4, 10)
        actions[:, 9] = torch.tensor(signs[step : step + 4] + [-1] * 4)[:4]
        valid = torch.arange(4) < min(4, len(signs) - step)
        samples.append(
            {
                "episode_id": "episode",
                "step_idx": str(step),
                "actions": actions,
                "action_time_mask": valid,
            }
        )

    masks, audit = build_gripper_release_masks(samples, horizon=4)
    assert audit["episodes_with_release"] == 1
    assert masks[2].tolist() == [False, False, True, True]
    batches = [{"state": torch.zeros(6, 11)}]
    attach_gripper_training_weights(batches, masks, release_weight=5.0)
    assert batches[0]["gripper_loss_weights"][2].tolist() == [1.0, 1.0, 5.0, 5.0]


def test_release_oversampling_prefers_groups_with_nearby_release() -> None:
    masks = torch.zeros(8, 4, dtype=torch.bool)
    masks[4:, 0] = True

    probabilities, audit = build_release_group_sampling_probabilities(
        masks,
        batch_size=4,
        oversample_factor=4.0,
        oversample_horizon=1,
    )

    np.testing.assert_allclose(probabilities, [0.2, 0.8])
    assert audit["release_relevant_samples"] == 4
    assert audit["natural_release_window_fraction"] == pytest.approx(0.5)
    assert audit["expected_oversampled_release_fraction"] == pytest.approx(0.8)


def test_wandb_sampling_metrics_include_requested_diagnostics() -> None:
    raw = np.zeros((2, 64, 7), dtype=np.float32)
    raw[..., 0] = np.linspace(-0.5, 0.5, 64)
    raw[..., 6] = np.where(np.arange(64) < 32, -1.0, 1.0)
    encoded = libero_action_to_rdt(raw)
    metrics = decoded_command_metrics(
        encoded,
        encoded,
        np.ones((2, 64), dtype=bool),
    )
    metrics["diffusion_sampling_repeats"] = 1
    metrics["sampled_trajectories"] = 2

    flattened = wandb_sampling_metrics(metrics)

    assert flattened["sampling/horizon_1/command_rmse_7d"] == 0.0
    assert flattened["sampling/horizon_64/command_rmse_7d"] == 0.0
    assert flattened["sampling/correlation/dx"] == pytest.approx(1.0)
    assert flattened["sampling/sign_agreement/overall"] == 1.0
    assert "sampling/saturation_fraction/gripper" in flattened
    assert flattened["sampling/gripper/accuracy"] == 1.0
    assert flattened["sampling/gripper/f1"] == 1.0
