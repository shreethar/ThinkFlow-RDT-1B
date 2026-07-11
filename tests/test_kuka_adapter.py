from __future__ import annotations

import numpy as np

from thinkflow_rdt.adapters.kuka import KukaEpisode, KukaStandardizedDataset
from thinkflow_rdt.adapters.fractal import standardize_binary_gripper_action


def test_kuka_gripper_binary_target_matches_observed_command_sign():
    raw = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    closed = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)

    target = standardize_binary_gripper_action(raw, closed)

    np.testing.assert_array_equal(
        target,
        np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )


def test_kuka_dataset_sample_schema_from_episode_arrays():
    episode = KukaEpisode(
        episode_id="train_000000",
        instructions=["pick", "pick", "pick"],
        images=[
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.full((2, 2, 3), 128, dtype=np.uint8),
            np.full((2, 2, 3), 255, dtype=np.uint8),
        ],
        states=np.ones((3, 7), dtype=np.float32),
        actions=np.ones((3, 7), dtype=np.float32),
    )
    dataset = KukaStandardizedDataset.from_episodes([episode], horizon=4)

    sample = dataset[1]

    assert sample["dataset_id"] == "kuka"
    assert sample["episode_id"] == "train_000000"
    assert sample["step_idx"] == "1"
    assert sample["instruction"] == "pick"
    assert sample["image_mask"] == {"primary": 1, "wrist": 0, "secondary": 0}
    assert sample["images"]["primary"].size == (2, 2)
    assert sample["images"]["wrist"] is None
    assert sample["images"]["secondary"] is None
    assert sample["state"].shape == (7,)
    assert sample["state_mask"].tolist() == [1.0] * 7
    assert sample["actions"].shape == (4, 7)
    assert sample["actions_mask"].tolist() == [1.0, 1.0, 0.0, 0.0]
