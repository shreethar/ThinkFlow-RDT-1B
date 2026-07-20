from __future__ import annotations

import numpy as np

from thinkflow_rdt.adapters.combined import (
    CombinedStandardizedDataset,
    split_indices_for_dataset,
    split_standardized_datasets,
)


class ToyStandardizedDataset:
    def __init__(self, dataset_id: str, episode_lengths: list[int]) -> None:
        self.dataset_id = dataset_id
        self.episodes = [object() for _ in episode_lengths]
        self.index: list[tuple[int, int]] = []
        for episode_index, episode_length in enumerate(episode_lengths):
            self.index.extend((episode_index, step_index) for step_index in range(episode_length))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict:
        episode_index, step_index = self.index[index]
        return {
            "dataset_id": self.dataset_id,
            "episode_id": f"episode_{episode_index}",
            "step_idx": str(step_index),
            "instruction": "pick",
            "images": {"primary": None, "wrist": None, "secondary": None},
            "image_mask": {"primary": 0, "wrist": 0, "secondary": 0},
            "state": np.zeros((7,), dtype=np.float32),
            "state_mask": np.ones((7,), dtype=np.float32),
            "actions": np.zeros((32, 7), dtype=np.float32),
            "actions_mask": np.ones((32,), dtype=np.float32),
        }


def test_split_indices_are_episode_level_and_roughly_80_10_10():
    dataset = ToyStandardizedDataset("toy", [2] * 10)

    train = split_indices_for_dataset(dataset, "train", dataset_id="toy", seed=123)
    validation = split_indices_for_dataset(dataset, "validation", dataset_id="toy", seed=123)
    test = split_indices_for_dataset(dataset, "test", dataset_id="toy", seed=123)

    assert len(train) == 16
    assert len(validation) == 2
    assert len(test) == 2

    train_episodes = {dataset.index[index][0] for index in train}
    validation_episodes = {dataset.index[index][0] for index in validation}
    test_episodes = {dataset.index[index][0] for index in test}

    assert train_episodes.isdisjoint(validation_episodes)
    assert train_episodes.isdisjoint(test_episodes)
    assert validation_episodes.isdisjoint(test_episodes)


def test_combined_dataset_concatenates_standardized_samples():
    left = ToyStandardizedDataset("left", [1] * 10)
    right = ToyStandardizedDataset("right", [1] * 10)

    combined = CombinedStandardizedDataset({"left": left, "right": right})

    assert len(combined) == 20
    assert combined.dataset_lengths == {"left": 10, "right": 10}
    assert combined[0]["dataset_id"] == "left"
    assert combined[10]["dataset_id"] == "right"


def test_split_standardized_datasets_returns_combined_splits_per_dataset():
    left = ToyStandardizedDataset("left", [1] * 10)
    right = ToyStandardizedDataset("right", [1] * 10)

    splits = split_standardized_datasets({"left": left, "right": right}, seed=123)

    assert set(splits) == {"train", "validation", "test"}
    assert len(splits["train"]) == 16
    assert len(splits["validation"]) == 2
    assert len(splits["test"]) == 2
    assert splits["train"].dataset_lengths == {"left": 8, "right": 8}
