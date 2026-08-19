from __future__ import annotations

import numpy as np

from thinkflow_rdt.adapters.sample_filtering import (
    build_episode_sample_indices,
    directional_gripper_change_window_indices,
    first_directional_gripper_change_window_indices,
    gripper_change_window_indices,
)


def test_gripper_change_window_keeps_three_before_and_three_after_transition():
    actions = np.zeros((10, 7), dtype=np.float32)
    actions[5:, 6] = 1.0

    selected = gripper_change_window_indices(actions)

    assert selected == [2, 3, 4, 5, 6, 7]


def test_episode_sampling_filters_empty_language_and_keeps_all_when_short():
    actions = np.zeros((4, 7), dtype=np.float32)
    instructions = ["pick", "", "place", "  "]

    selected = build_episode_sample_indices(
        instructions,
        actions,
        max_samples_per_episode=64,
    )

    assert selected == [0, 2]


def test_episode_sampling_prioritizes_gripper_change_windows_under_cap():
    actions = np.zeros((100, 7), dtype=np.float32)
    actions[50:, 6] = 1.0
    instructions = ["pick"] * 100

    selected = build_episode_sample_indices(
        instructions,
        actions,
        max_samples_per_episode=10,
    )

    for special_step in [47, 48, 49, 50, 51, 52]:
        assert special_step in selected
    assert len(selected) == 10
    assert selected == sorted(selected)


def test_episode_sampling_can_prioritize_only_first_wide_gripper_window():
    actions = np.zeros((120, 7), dtype=np.float32)
    actions[50:80, 6] = 1.0
    actions[80:, 6] = 0.0
    instructions = ["pick"] * 120

    special = gripper_change_window_indices(
        actions,
        before=10,
        after=11,
        change_scope="first",
    )

    selected = build_episode_sample_indices(
        instructions,
        actions,
        max_samples_per_episode=64,
        gripper_window_before=10,
        gripper_window_after=11,
        gripper_change_scope="first",
    )

    assert special == list(range(40, 61))
    for special_step in special:
        assert special_step in selected
    assert len(selected) == 64
    assert selected == sorted(selected)


def test_episode_sampling_can_use_directional_gripper_windows():
    actions = np.zeros((80, 7), dtype=np.float32)
    actions[30:55, 6] = 1.0
    instructions = ["pick"] * 80

    special = directional_gripper_change_window_indices(
        actions,
        open_to_close_before=5,
        open_to_close_after=4,
        close_to_open_before=3,
        close_to_open_after=0,
    )
    selected = build_episode_sample_indices(
        instructions,
        actions,
        max_samples_per_episode=32,
        gripper_change_scope="directional",
        open_to_close_before=5,
        open_to_close_after=4,
        close_to_open_before=3,
        close_to_open_after=0,
    )

    assert special == [25, 26, 27, 28, 29, 30, 31, 32, 33, 52, 53, 54]
    for special_step in special:
        assert special_step in selected
    assert len(selected) == 32
    assert selected == sorted(selected)


def test_first_directional_sampling_keeps_four_frames_on_each_side_and_fills_uniformly():
    actions = np.zeros((100, 7), dtype=np.float32)
    actions[30:60, 6] = 1.0
    # These later transitions must not create additional priority windows.
    actions[75:85, 6] = 1.0
    instructions = ["pick and place"] * len(actions)

    special = first_directional_gripper_change_window_indices(
        actions,
        open_to_close_before=4,
        open_to_close_after=4,
        close_to_open_before=4,
        close_to_open_after=4,
    )
    selected = build_episode_sample_indices(
        instructions,
        actions,
        max_samples_per_episode=32,
        gripper_change_scope="first_directional",
        open_to_close_before=4,
        open_to_close_after=4,
        close_to_open_before=4,
        close_to_open_after=4,
    )

    assert special == list(range(26, 34)) + list(range(56, 64))
    assert len(special) == 16
    assert len(selected) == 32
    assert set(special).issubset(selected)
    assert selected == sorted(selected)
