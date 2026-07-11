from __future__ import annotations

import numpy as np

from thinkflow_rdt.adapters.fractal import (
    FractalEpisode,
    FractalStandardizedDataset,
    pad_action_horizon,
    quat_xyzw_to_rpy_rad,
    standardize_binary_gripper_action,
)


def test_quat_xyzw_to_rpy_identity():
    rpy = quat_xyzw_to_rpy_rad(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(rpy, np.zeros((1, 3), dtype=np.float32), atol=1e-6)


def test_standardize_binary_gripper_action_holds_last_target():
    raw = np.array([0.0, 1.0, 0.82, 0.12, 0.0, 0.0, -1.0, -0.8, 0.0])
    closed = np.array([0.0, 0.0, 0.18, 0.88, 1.0, 1.0, 1.0, 0.5, 0.0])

    target = standardize_binary_gripper_action(raw, closed)

    np.testing.assert_array_equal(
        target,
        np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def test_pad_action_horizon_zero_pads_and_masks():
    actions = np.ones((3, 7), dtype=np.float32)

    padded, mask = pad_action_horizon(actions, 1, horizon=4)

    assert padded.shape == (4, 7)
    assert mask.tolist() == [1.0, 1.0, 0.0, 0.0]
    np.testing.assert_array_equal(padded[:2], np.ones((2, 7), dtype=np.float32))
    np.testing.assert_array_equal(padded[2:], np.zeros((2, 7), dtype=np.float32))


def test_fractal_dataset_sample_schema_from_episode_arrays():
    episode = FractalEpisode(
        episode_id="train_000000",
        instructions=["pick", "pick"],
        images=[
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.full((2, 2, 3), 255, dtype=np.uint8),
        ],
        states=np.ones((2, 7), dtype=np.float32),
        actions=np.ones((2, 7), dtype=np.float32),
    )
    dataset = FractalStandardizedDataset.from_episodes([episode], horizon=4)

    sample = dataset[0]

    assert sample["dataset_id"] == "fractal"
    assert sample["episode_id"] == "train_000000"
    assert sample["step_idx"] == "0"
    assert sample["instruction"] == "pick"
    assert sample["image_mask"] == {"primary": 1, "wrist": 0, "secondary": 0}
    assert sample["images"]["primary"].size == (2, 2)
    assert sample["images"]["wrist"] is None
    assert sample["images"]["secondary"] is None
    assert sample["state"].shape == (7,)
    assert sample["state_mask"].tolist() == [1.0] * 7
    assert sample["actions"].shape == (4, 7)
    assert sample["actions_mask"].tolist() == [1.0, 1.0, 0.0, 0.0]
