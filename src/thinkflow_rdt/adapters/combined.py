from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bc_z import BcZStandardizedDataset
from .bridge import BridgeStandardizedDataset
from .droid import DroidStandardizedDataset
from .fractal import DEFAULT_HORIZON, FractalStandardizedDataset
from .kuka import KukaStandardizedDataset
from .sample_filtering import (
    DEFAULT_GRIPPER_WINDOW_AFTER,
    DEFAULT_GRIPPER_WINDOW_BEFORE,
    DEFAULT_MAX_SAMPLES_PER_EPISODE,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)
SPLIT_NAMES = ("train", "validation", "test")


STANDARDIZED_DATASET_CLASSES: dict[str, type] = {
    "bc_z": BcZStandardizedDataset,
    "bridge": BridgeStandardizedDataset,
    "droid": DroidStandardizedDataset,
    "fractal": FractalStandardizedDataset,
    "kuka": KukaStandardizedDataset,
}


@dataclass
class StandardizedDatasetConfig:
    dataset_id: str
    data_dir: str | Path
    split: str = "train"
    max_episodes: int | None = None
    shard_pattern: str | None = None
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CombinedMember:
    dataset_id: str
    dataset: Any
    length: int


class StandardizedDatasetView:
    """Index view over a standardized map-style dataset."""

    def __init__(
        self,
        dataset: Any,
        indices: Sequence[int],
        *,
        dataset_id: str,
        split_name: str,
    ) -> None:
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.dataset_id = dataset_id
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.dataset[self.indices[index]]


class CombinedStandardizedDataset:
    """Concatenate standardized dataset views while preserving sample schema."""

    def __init__(
        self,
        datasets: Mapping[str, Any] | Sequence[tuple[str, Any]],
        *,
        split_name: str | None = None,
    ) -> None:
        items = datasets.items() if isinstance(datasets, Mapping) else datasets
        self.members = [
            CombinedMember(dataset_id=dataset_id, dataset=dataset, length=len(dataset))
            for dataset_id, dataset in items
            if len(dataset) > 0
        ]
        self.split_name = split_name
        lengths = [member.length for member in self.members]
        self._cumulative_lengths = np.cumsum(lengths).astype(np.int64).tolist()

    @property
    def dataset_lengths(self) -> dict[str, int]:
        return {member.dataset_id: member.length for member in self.members}

    def __len__(self) -> int:
        if not self._cumulative_lengths:
            return 0
        return int(self._cumulative_lengths[-1])

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        member_index = bisect.bisect_right(self._cumulative_lengths, index)
        previous_total = 0 if member_index == 0 else self._cumulative_lengths[member_index - 1]
        local_index = index - previous_total
        member = self.members[member_index]
        return member.dataset[local_index]


def default_standardized_dataset_configs(
    dataset_ids: Sequence[str] | None = None,
    *,
    root: str | Path | None = None,
) -> list[StandardizedDatasetConfig]:
    root_path = Path(root).expanduser().resolve() if root is not None else REPO_ROOT
    mock_root = root_path / "dataset" / "mock_dataset"
    configs = {
        "bc_z": StandardizedDatasetConfig(
            dataset_id="bc_z",
            data_dir=mock_root / "bc_z_dataset" / "data",
        ),
        "bridge": StandardizedDatasetConfig(
            dataset_id="bridge",
            data_dir=_first_existing(
                mock_root / "bridge_dataset" / "data",
                mock_root / "bridge_dataset" / "bridge_subset",
            ),
        ),
        "droid": StandardizedDatasetConfig(
            dataset_id="droid",
            data_dir=_first_existing(
                mock_root / "droid_dataset" / "data",
                mock_root / "droid_dataset" / "droid_100" / "1.0.0",
            ),
        ),
        "fractal": StandardizedDatasetConfig(
            dataset_id="fractal",
            data_dir=mock_root / "fractal_dataset" / "data",
        ),
        "kuka": StandardizedDatasetConfig(
            dataset_id="kuka",
            data_dir=mock_root / "kuka_dataset" / "data",
        ),
    }
    selected = list(configs) if dataset_ids is None else list(dataset_ids)
    return [configs[dataset_id] for dataset_id in selected]


def build_standardized_dataset_from_config(
    config: StandardizedDatasetConfig,
    *,
    horizon: int = DEFAULT_HORIZON,
    normalize_actions: bool = True,
    filter_empty_language: bool = True,
    max_samples_per_episode: int | None = DEFAULT_MAX_SAMPLES_PER_EPISODE,
    gripper_window_before: int = DEFAULT_GRIPPER_WINDOW_BEFORE,
    gripper_window_after: int = DEFAULT_GRIPPER_WINDOW_AFTER,
    common_adapter_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    if config.dataset_id not in STANDARDIZED_DATASET_CLASSES:
        raise KeyError(f"Unknown dataset_id: {config.dataset_id}")

    dataset_cls = STANDARDIZED_DATASET_CLASSES[config.dataset_id]
    kwargs: dict[str, Any] = {
        "split": config.split,
        "horizon": horizon,
        "dataset_id": config.dataset_id,
        "max_episodes": config.max_episodes,
        "shard_pattern": config.shard_pattern,
        "normalize_actions": normalize_actions,
        "filter_empty_language": filter_empty_language,
        "max_samples_per_episode": max_samples_per_episode,
        "gripper_window_before": gripper_window_before,
        "gripper_window_after": gripper_window_after,
    }
    if common_adapter_kwargs is not None:
        kwargs.update(common_adapter_kwargs)
    kwargs.update(config.adapter_kwargs)
    return dataset_cls(config.data_dir, **kwargs)


def build_combined_standardized_splits(
    configs: Sequence[StandardizedDatasetConfig] | None = None,
    *,
    dataset_ids: Sequence[str] | None = None,
    root: str | Path | None = None,
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 0,
    horizon: int = DEFAULT_HORIZON,
    normalize_actions: bool = True,
    filter_empty_language: bool = True,
    max_samples_per_episode: int | None = DEFAULT_MAX_SAMPLES_PER_EPISODE,
    gripper_window_before: int = DEFAULT_GRIPPER_WINDOW_BEFORE,
    gripper_window_after: int = DEFAULT_GRIPPER_WINDOW_AFTER,
    common_adapter_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, CombinedStandardizedDataset]:
    dataset_configs = (
        list(configs)
        if configs is not None
        else default_standardized_dataset_configs(dataset_ids=dataset_ids, root=root)
    )
    datasets = {
        config.dataset_id: build_standardized_dataset_from_config(
            config,
            horizon=horizon,
            normalize_actions=normalize_actions,
            filter_empty_language=filter_empty_language,
            max_samples_per_episode=max_samples_per_episode,
            gripper_window_before=gripper_window_before,
            gripper_window_after=gripper_window_after,
            common_adapter_kwargs=common_adapter_kwargs,
        )
        for config in dataset_configs
    }
    return split_standardized_datasets(
        datasets,
        split_ratios=split_ratios,
        seed=seed,
    )


def split_standardized_datasets(
    datasets: Mapping[str, Any] | Sequence[tuple[str, Any]],
    *,
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 0,
) -> dict[str, CombinedStandardizedDataset]:
    items = list(datasets.items() if isinstance(datasets, Mapping) else datasets)
    views_by_split: dict[str, list[tuple[str, StandardizedDatasetView]]] = {
        split_name: [] for split_name in SPLIT_NAMES
    }

    for dataset_id, dataset in items:
        for split_name in SPLIT_NAMES:
            indices = split_indices_for_dataset(
                dataset,
                split_name,
                dataset_id=dataset_id,
                split_ratios=split_ratios,
                seed=seed,
            )
            view = StandardizedDatasetView(
                dataset,
                indices,
                dataset_id=dataset_id,
                split_name=split_name,
            )
            views_by_split[split_name].append((dataset_id, view))

    return {
        split_name: CombinedStandardizedDataset(views, split_name=split_name)
        for split_name, views in views_by_split.items()
    }


def split_indices_for_dataset(
    dataset: Any,
    split_name: str,
    *,
    dataset_id: str,
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 0,
) -> list[int]:
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"split_name must be one of {SPLIT_NAMES}, got {split_name}")

    groups = _episode_sample_index_groups(dataset)
    group_keys = list(groups)
    rng = np.random.default_rng(_stable_seed(seed, dataset_id))
    shuffled_keys = group_keys[:]
    rng.shuffle(shuffled_keys)

    counts = _split_counts(len(shuffled_keys), split_ratios)
    split_index = SPLIT_NAMES.index(split_name)
    start = sum(counts[:split_index])
    stop = start + counts[split_index]
    selected_keys = set(shuffled_keys[start:stop])

    selected_indices: list[int] = []
    for group_key in group_keys:
        if group_key in selected_keys:
            selected_indices.extend(groups[group_key])
    return selected_indices


def _episode_sample_index_groups(dataset: Any) -> dict[Any, list[int]]:
    dataset_index = getattr(dataset, "index", None)
    if dataset_index is None:
        return {sample_index: [sample_index] for sample_index in range(len(dataset))}

    groups: dict[Any, list[int]] = {}
    for sample_index, item in enumerate(dataset_index):
        episode_key = item[0] if isinstance(item, tuple) and item else sample_index
        groups.setdefault(episode_key, []).append(sample_index)
    return groups


def _split_counts(total: int, split_ratios: Sequence[float]) -> list[int]:
    if len(split_ratios) != len(SPLIT_NAMES):
        raise ValueError(f"Expected {len(SPLIT_NAMES)} split ratios")
    if total < 0:
        raise ValueError("total must be non-negative")
    ratios = np.asarray(split_ratios, dtype=np.float64)
    if np.any(ratios < 0.0) or float(ratios.sum()) <= 0.0:
        raise ValueError("split ratios must be non-negative and sum to a positive value")
    if total == 0:
        return [0] * len(SPLIT_NAMES)

    ratios = ratios / ratios.sum()
    raw_counts = ratios * total
    counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(total - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw_counts - counts))
        for index in order[:remainder]:
            counts[int(index)] += 1

    positive_splits = [index for index, ratio in enumerate(ratios) if ratio > 0.0]
    if total >= len(positive_splits):
        for index in positive_splits:
            if counts[index] > 0:
                continue
            donor_candidates = [candidate for candidate in positive_splits if counts[candidate] > 1]
            if not donor_candidates:
                break
            donor = max(donor_candidates, key=lambda candidate: counts[candidate])
            counts[donor] -= 1
            counts[index] += 1

    return counts.astype(int).tolist()


def _stable_seed(seed: int, dataset_id: str) -> int:
    digest = hashlib.blake2b(dataset_id.encode("utf-8"), digest_size=4).digest()
    dataset_offset = int.from_bytes(digest, byteorder="little", signed=False)
    return (int(seed) + dataset_offset) % (2**32)


def _first_existing(preferred: Path, fallback: Path) -> Path:
    return preferred if preferred.exists() else fallback
