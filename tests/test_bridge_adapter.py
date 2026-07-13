from __future__ import annotations

import numpy as np

from thinkflow_rdt.adapters.bridge import (
    BridgeEpisode,
    BridgeStandardizedDataset,
    standardize_bridge_gripper_open_to_closed,
    zero_image_to_none,
)


def test_bridge_gripper_inverts_open_to_closed_convention():
    raw = np.array([1.0, 0.0, 0.9, 0.1, 0.5], dtype=np.float32)

    target = standardize_bridge_gripper_open_to_closed(raw)

    np.testing.assert_array_equal(
        target,
        np.array([0.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )


def test_zero_image_to_none_handles_bridge_placeholders():
    assert zero_image_to_none(np.zeros((2, 2, 3), dtype=np.uint8)) is None

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0, 0] = 1

    assert zero_image_to_none(image) is not None


def test_bridge_dataset_sample_schema_from_episode_arrays():
    states = np.ones((3, 7), dtype=np.float32)
    actions = np.ones((3, 7), dtype=np.float32)
    episode = BridgeEpisode(
        episode_id="123",
        instructions=["pick", "pick", ""],
        primary_images=[
            np.full((2, 2, 3), 255, dtype=np.uint8),
            None,
            np.full((2, 2, 3), 64, dtype=np.uint8),
        ],
        secondary_images=[
            None,
            np.full((2, 2, 3), 128, dtype=np.uint8),
            None,
        ],
        states=states,
        actions=actions,
    )
    dataset = BridgeStandardizedDataset.from_episodes([episode], horizon=4)

    first = dataset[0]
    second = dataset[1]

    assert first["dataset_id"] == "bridge"
    assert first["episode_id"] == "123"
    assert first["step_idx"] == "0"
    assert first["instruction"] == "pick"
    assert first["image_mask"] == {"primary": 1, "wrist": 0, "secondary": 0}
    assert first["images"]["primary"].size == (2, 2)
    assert first["images"]["wrist"] is None
    assert first["images"]["secondary"] is None
    assert first["state"].shape == (7,)
    assert first["state_mask"].tolist() == [1.0] * 7
    assert first["actions"].shape == (4, 7)
    assert first["actions_mask"].tolist() == [1.0, 1.0, 1.0, 0.0]

    assert second["image_mask"] == {"primary": 0, "wrist": 0, "secondary": 1}
    assert second["images"]["primary"] is None
    assert second["images"]["secondary"].size == (2, 2)


def test_bridge_dataset_filters_empty_language_steps():
    episode = BridgeEpisode(
        episode_id="123",
        instructions=["pick", "", "place"],
        primary_images=[np.ones((2, 2, 3), dtype=np.uint8)] * 3,
        secondary_images=[None, None, None],
        states=np.ones((3, 7), dtype=np.float32),
        actions=np.ones((3, 7), dtype=np.float32),
    )
    dataset = BridgeStandardizedDataset.from_episodes([episode], horizon=2)

    assert len(dataset) == 2
    assert [dataset[index]["step_idx"] for index in range(len(dataset))] == ["0", "2"]


def test_bridge_dataset_caps_episode_samples_and_preserves_gripper_window():
    actions = np.zeros((100, 7), dtype=np.float32)
    actions[50:, 6] = 1.0
    episode = BridgeEpisode(
        episode_id="123",
        instructions=["pick"] * 100,
        primary_images=[np.ones((2, 2, 3), dtype=np.uint8)] * 100,
        secondary_images=[None] * 100,
        states=np.ones((100, 7), dtype=np.float32),
        actions=actions,
    )
    dataset = BridgeStandardizedDataset.from_episodes(
        [episode],
        horizon=2,
        max_samples_per_episode=10,
    )

    step_indices = [int(dataset[index]["step_idx"]) for index in range(len(dataset))]

    assert len(dataset) == 10
    for special_step in [47, 48, 49, 50, 51, 52]:
        assert special_step in step_indices
