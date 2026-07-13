from __future__ import annotations

import json

import numpy as np

from thinkflow_rdt.adapters.action_stats import (
    ActionNormalizationStats,
    compute_action_quantile_stats,
    denormalize_action_array,
    load_action_stats,
    normalize_action_array,
    normalize_action_horizon,
    write_action_stats_to_audit,
)
from thinkflow_rdt.adapters.fractal import FractalEpisode, FractalStandardizedDataset


def test_action_normalization_round_trips_inside_quantile_range():
    stats = ActionNormalizationStats(
        q01=np.array([-2.0, 0.0, 0.0], dtype=np.float32),
        q99=np.array([2.0, 10.0, 1.0], dtype=np.float32),
    )
    actions = np.array([[-2.0, 5.0, 0.0], [2.0, 10.0, 1.0]], dtype=np.float32)

    normalized = normalize_action_array(actions, stats)
    restored = denormalize_action_array(normalized, stats)

    np.testing.assert_allclose(
        normalized,
        np.array([[-1.0, 0.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(restored, actions, atol=1e-6)


def test_action_normalization_clips_outliers():
    stats = ActionNormalizationStats(
        q01=np.array([0.0], dtype=np.float32),
        q99=np.array([10.0], dtype=np.float32),
    )

    normalized = normalize_action_array(np.array([[-5.0], [15.0]], dtype=np.float32), stats)

    np.testing.assert_allclose(normalized, np.array([[-1.0], [1.0]], dtype=np.float32))


def test_normalize_action_horizon_keeps_padding_zero():
    stats = ActionNormalizationStats(
        q01=np.zeros((2,), dtype=np.float32),
        q99=np.ones((2,), dtype=np.float32),
    )
    actions = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    mask = np.array([1.0, 0.0], dtype=np.float32)

    normalized = normalize_action_horizon(actions, mask, stats)

    np.testing.assert_allclose(
        normalized,
        np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
    )


def test_compute_action_quantile_stats_and_audit_io(tmp_path):
    actions = np.arange(100, dtype=np.float32).reshape(50, 2)

    stats = compute_action_quantile_stats(actions)

    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({"dataset_name": "toy"}), encoding="utf-8")
    write_action_stats_to_audit(
        audit_path,
        stats,
        source={"dataset_id": "toy", "num_steps": int(actions.shape[0])},
    )
    loaded = load_action_stats(audit_path)

    np.testing.assert_allclose(loaded.q01, stats.q01)
    np.testing.assert_allclose(loaded.q99, stats.q99)


def test_dataset_normalizes_valid_action_rows_but_keeps_padding_zero():
    stats = ActionNormalizationStats(
        q01=np.zeros((7,), dtype=np.float32),
        q99=np.ones((7,), dtype=np.float32),
    )
    episode = FractalEpisode(
        episode_id="train_000000",
        instructions=["pick", "pick"],
        images=[
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.uint8),
        ],
        states=np.ones((2, 7), dtype=np.float32),
        actions=np.array(
            [[1.0] * 7, [0.0] * 7],
            dtype=np.float32,
        ),
    )
    dataset = FractalStandardizedDataset.from_episodes(
        [episode],
        horizon=3,
        normalize_actions=True,
        action_stats=stats,
    )

    sample = dataset[0]

    np.testing.assert_allclose(sample["actions"][0], np.ones((7,), dtype=np.float32))
    np.testing.assert_allclose(sample["actions"][1], -np.ones((7,), dtype=np.float32))
    np.testing.assert_allclose(sample["actions"][2], np.zeros((7,), dtype=np.float32))
