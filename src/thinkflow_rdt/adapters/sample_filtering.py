from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_MAX_SAMPLES_PER_EPISODE = 64
DEFAULT_GRIPPER_WINDOW_BEFORE = 3
DEFAULT_GRIPPER_WINDOW_AFTER = 3


def has_language_instruction(instruction: str) -> bool:
    return bool(str(instruction).strip())


def uniform_sample_indices(indices: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    sorted_indices = sorted(set(int(index) for index in indices))
    if count >= len(sorted_indices):
        return sorted_indices

    positions = np.linspace(0, len(sorted_indices) - 1, count, dtype=int)
    selected = [sorted_indices[int(position)] for position in positions]
    if len(set(selected)) < count:
        selected_set = set(selected)
        for index in sorted_indices:
            if len(selected_set) >= count:
                break
            selected_set.add(index)
        selected = sorted(selected_set)
    return sorted(set(selected))


def gripper_change_window_indices(
    actions: np.ndarray,
    *,
    before: int = DEFAULT_GRIPPER_WINDOW_BEFORE,
    after: int = DEFAULT_GRIPPER_WINDOW_AFTER,
    gripper_threshold: float = 0.5,
) -> list[int]:
    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.ndim != 2 or action_array.shape[0] == 0:
        return []
    if action_array.shape[1] < 7:
        raise ValueError(f"Expected action dim at least 7, got {action_array.shape}")

    gripper = (action_array[:, 6] >= gripper_threshold).astype(np.int8)
    change_steps = np.flatnonzero(gripper[1:] != gripper[:-1]) + 1
    total_steps = action_array.shape[0]

    selected: set[int] = set()
    for change_step in change_steps:
        start = max(0, int(change_step) - before)
        stop = min(total_steps, int(change_step) + after)
        selected.update(range(start, stop))
    return sorted(selected)


def build_episode_sample_indices(
    instructions: list[str],
    actions: np.ndarray,
    *,
    max_samples_per_episode: int | None = DEFAULT_MAX_SAMPLES_PER_EPISODE,
    filter_empty_language: bool = True,
    gripper_window_before: int = DEFAULT_GRIPPER_WINDOW_BEFORE,
    gripper_window_after: int = DEFAULT_GRIPPER_WINDOW_AFTER,
    gripper_threshold: float = 0.5,
) -> list[int]:
    total_steps = len(instructions)
    if total_steps == 0:
        return []

    if filter_empty_language:
        valid_steps = [
            step_index
            for step_index, instruction in enumerate(instructions)
            if has_language_instruction(instruction)
        ]
    else:
        valid_steps = list(range(total_steps))

    if not valid_steps:
        return []

    if max_samples_per_episode is None or len(valid_steps) <= max_samples_per_episode:
        return valid_steps
    if max_samples_per_episode <= 0:
        return []

    valid_set = set(valid_steps)
    special_steps = [
        step_index
        for step_index in gripper_change_window_indices(
            actions,
            before=gripper_window_before,
            after=gripper_window_after,
            gripper_threshold=gripper_threshold,
        )
        if step_index in valid_set
    ]
    if len(special_steps) >= max_samples_per_episode:
        return uniform_sample_indices(special_steps, max_samples_per_episode)

    special_set = set(special_steps)
    remaining_budget = max_samples_per_episode - len(special_set)
    uniform_candidates = [
        step_index for step_index in valid_steps if step_index not in special_set
    ]
    uniform_steps = uniform_sample_indices(uniform_candidates, remaining_budget)
    return sorted(special_set.union(uniform_steps))


def build_dataset_sample_index(
    episodes: list[Any],
    *,
    max_samples_per_episode: int | None = DEFAULT_MAX_SAMPLES_PER_EPISODE,
    filter_empty_language: bool = True,
    gripper_window_before: int = DEFAULT_GRIPPER_WINDOW_BEFORE,
    gripper_window_after: int = DEFAULT_GRIPPER_WINDOW_AFTER,
) -> list[tuple[int, int]]:
    index: list[tuple[int, int]] = []
    for episode_index, episode in enumerate(episodes):
        step_indices = build_episode_sample_indices(
            episode.instructions,
            episode.actions,
            max_samples_per_episode=max_samples_per_episode,
            filter_empty_language=filter_empty_language,
            gripper_window_before=gripper_window_before,
            gripper_window_after=gripper_window_after,
        )
        index.extend((episode_index, step_index) for step_index in step_indices)
    return index
