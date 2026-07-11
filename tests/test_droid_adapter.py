from __future__ import annotations

import numpy as np

from thinkflow_rdt.adapters.droid import (
    DroidEpisode,
    DroidStandardizedDataset,
    standardize_droid_action,
    standardize_droid_state,
)


def test_droid_state_uses_cartesian_position_and_binary_closed_gripper():
    cartesian = np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]], dtype=np.float32)
    gripper = np.array([[0.9]], dtype=np.float32)

    state = standardize_droid_state(cartesian, gripper)

    np.testing.assert_allclose(
        state,
        np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.0]], dtype=np.float32),
    )


def test_droid_absolute_action_converts_to_relative_delta():
    cartesian = np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]], dtype=np.float32)
    absolute_action = np.array(
        [[1.5, 1.0, 4.0, 0.2, 0.1, 0.8, 0.0]],
        dtype=np.float32,
    )

    action = standardize_droid_action(absolute_action, cartesian)

    np.testing.assert_allclose(
        action,
        np.array([[0.5, -1.0, 1.0, 0.1, -0.1, 0.5, 0.0]], dtype=np.float32),
        atol=1e-6,
    )


def test_droid_dataset_sample_schema_from_episode_arrays():
    episode = DroidEpisode(
        episode_id="train_000000",
        instructions=["pick", "pick", "pick"],
        primary_images=[
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.full((2, 2, 3), 64, dtype=np.uint8),
            np.full((2, 2, 3), 255, dtype=np.uint8),
        ],
        wrist_images=[
            np.full((2, 2, 3), 11, dtype=np.uint8),
            np.full((2, 2, 3), 12, dtype=np.uint8),
            np.full((2, 2, 3), 13, dtype=np.uint8),
        ],
        secondary_images=[
            np.full((2, 2, 3), 21, dtype=np.uint8),
            np.full((2, 2, 3), 22, dtype=np.uint8),
            np.full((2, 2, 3), 23, dtype=np.uint8),
        ],
        states=np.ones((3, 7), dtype=np.float32),
        actions=np.ones((3, 7), dtype=np.float32),
    )
    dataset = DroidStandardizedDataset.from_episodes([episode], horizon=4)

    sample = dataset[1]

    assert sample["dataset_id"] == "droid"
    assert sample["episode_id"] == "train_000000"
    assert sample["step_idx"] == "1"
    assert sample["instruction"] == "pick"
    assert sample["image_mask"] == {"primary": 1, "wrist": 1, "secondary": 1}
    assert sample["images"]["primary"].size == (2, 2)
    assert sample["images"]["wrist"].size == (2, 2)
    assert sample["images"]["secondary"].size == (2, 2)
    assert sample["state"].shape == (7,)
    assert sample["state_mask"].tolist() == [1.0] * 7
    assert sample["actions"].shape == (4, 7)
    assert sample["actions_mask"].tolist() == [1.0, 1.0, 0.0, 0.0]
