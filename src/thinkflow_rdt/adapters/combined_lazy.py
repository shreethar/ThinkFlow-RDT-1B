from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from .action_stats import (
    ActionNormalizationStats,
    normalize_action_horizon,
    resolve_action_stats,
)
from .bc_z import (
    bcz_episode_id,
    bcz_sample_from_episode,
    convert_bcz_episode,
    iter_bcz_raw_episodes,
)
from .bridge import (
    bridge_episode_id,
    bridge_sample_from_episode,
    convert_bridge_episode,
    iter_bridge_raw_episodes,
)
from .droid import DroidStandardizedDataset, pil_or_none as droid_pil_or_none
from .fractal import (
    ACTION_DIM,
    DEFAULT_HORIZON,
    STATE_DIM,
    FractalStandardizedDataset,
    pad_action_horizon,
)
from .kuka import KukaStandardizedDataset
from .sample_filtering import (
    DEFAULT_GRIPPER_WINDOW_AFTER,
    DEFAULT_GRIPPER_WINDOW_BEFORE,
    DEFAULT_MAX_SAMPLES_PER_EPISODE,
    build_episode_sample_indices,
)


try:
    from torch.utils.data import IterableDataset as TorchIterableDataset
    from torch.utils.data import get_worker_info
except ImportError:  # pragma: no cover - torch is a project dependency.
    class TorchIterableDataset:  # type: ignore[no-redef]
        pass

    def get_worker_info() -> Any:  # type: ignore[no-redef]
        return None


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)
SPLIT_NAMES = ("train", "validation", "test")


@dataclass
class LazyStandardizedDatasetConfig:
    dataset_id: str
    data_dir: str | Path
    source_split: str = "train"
    max_episodes: int | None = None
    shard_pattern: str | None = None
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LazyCombinedMember:
    dataset_id: str
    dataset: Iterable[dict[str, Any]]


class LazyStandardizedDataset(TorchIterableDataset):
    """
    Stream standardized samples from one dataset without loading all episodes.

    Splits are deterministic hash partitions over episode ids. This gives an
    approximate 80/10/10 split without a full pre-scan, and prevents an episode
    from appearing in multiple train/validation/test splits. Optional curriculum
    stages are deterministic hash partitions over sampled timesteps.
    """

    def __init__(
        self,
        config: LazyStandardizedDatasetConfig,
        *,
        split_name: str,
        split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
        seed: int = 0,
        stage: int | None = None,
        stage_count: int = 3,
        droid_stage_count: int = 2,
        horizon: int = DEFAULT_HORIZON,
        normalize_actions: bool = True,
        filter_empty_language: bool = True,
        max_samples_per_episode: int | None = DEFAULT_MAX_SAMPLES_PER_EPISODE,
        gripper_window_before: int = DEFAULT_GRIPPER_WINDOW_BEFORE,
        gripper_window_after: int = DEFAULT_GRIPPER_WINDOW_AFTER,
    ) -> None:
        if split_name not in SPLIT_NAMES:
            raise ValueError(f"split_name must be one of {SPLIT_NAMES}, got {split_name}")

        self.config = config
        self.dataset_id = config.dataset_id
        self.data_dir = Path(config.data_dir).expanduser().resolve()
        self.source_split = config.source_split
        self.split_name = split_name
        self.split_ratios = tuple(float(ratio) for ratio in split_ratios)
        self.seed = int(seed)
        self.stage = stage
        self.stage_count = int(stage_count)
        self.droid_stage_count = int(droid_stage_count)
        self.horizon = horizon
        self.normalize_actions = normalize_actions
        self.filter_empty_language = filter_empty_language
        self.max_samples_per_episode = max_samples_per_episode
        self.gripper_window_before = gripper_window_before
        self.gripper_window_after = gripper_window_after
        self.adapter_kwargs = dict(config.adapter_kwargs)
        self.action_stats = self._resolve_action_stats()

    def __iter__(self):
        eligible_seen = 0
        split_seen = 0
        worker = get_worker_info()
        raw_episode_index = 0
        raw_episode_iter = iter(self._iter_raw_episodes())

        while True:
            try:
                raw_episode = next(raw_episode_iter)
            except StopIteration:
                break
            except (FileNotFoundError, OSError, RuntimeError) as exc:
                if is_missing_local_shard_error(exc):
                    _print_missing_shard_warning(self.dataset_id, exc)
                    break
                raise

            current_raw_episode_index = raw_episode_index
            raw_episode_index += 1

            if self.dataset_id == "kuka" and self._kuka_only_success:
                if not _is_successful_episode(raw_episode):
                    continue

            if self.config.max_episodes is not None and eligible_seen >= self.config.max_episodes:
                break
            eligible_seen += 1

            episode_id = self._episode_id(raw_episode, current_raw_episode_index)
            assigned_split = episode_split_name(
                self.dataset_id,
                episode_id,
                split_ratios=self.split_ratios,
                seed=self.seed,
            )
            if assigned_split != self.split_name:
                continue
            if worker is not None:
                if split_seen % worker.num_workers != worker.id:
                    split_seen += 1
                    continue
            split_seen += 1

            steps = list(raw_episode["steps"])
            if not steps:
                continue

            episode = self._convert_episode(current_raw_episode_index, episode_id, steps)
            step_indices = build_episode_sample_indices(
                episode.instructions,
                episode.actions,
                max_samples_per_episode=self.max_samples_per_episode,
                filter_empty_language=self.filter_empty_language,
                gripper_window_before=self.gripper_window_before,
                gripper_window_after=self.gripper_window_after,
            )
            for step_index in step_indices:
                if self.stage is not None and not sample_belongs_to_stage(
                    self.dataset_id,
                    episode_id,
                    step_index,
                    self.stage,
                    stage_count=self.stage_count,
                    droid_stage_count=self.droid_stage_count,
                    seed=self.seed,
                ):
                    continue
                yield self._sample_from_episode(episode, step_index)

    @property
    def _kuka_only_success(self) -> bool:
        return bool(self.adapter_kwargs.get("only_success", True))

    def _resolve_action_stats(self) -> ActionNormalizationStats | None:
        return resolve_action_stats(
            normalize_actions=self.normalize_actions,
            action_stats=self.adapter_kwargs.get("action_stats"),
            action_stats_path=self.adapter_kwargs.get("action_stats_path"),
            search_dir=self.data_dir,
        )

    def _iter_raw_episodes(self):
        if self.dataset_id == "bc_z":
            yield from iter_bcz_raw_episodes(
                self.data_dir,
                split=self.source_split,
                shard_pattern=self.config.shard_pattern,
            )
        elif self.dataset_id == "bridge":
            yield from iter_bridge_raw_episodes(
                self.data_dir,
                split=self.source_split,
                shard_pattern=self.config.shard_pattern,
            )
        elif self.dataset_id in {"droid", "fractal", "kuka"}:
            yield from iter_tfds_raw_episodes(
                self.data_dir,
                split=self.source_split,
                shard_pattern=self.config.shard_pattern,
            )
        else:
            raise KeyError(f"Unknown dataset_id: {self.dataset_id}")

    def _episode_id(self, raw_episode: Any, raw_episode_index: int) -> str:
        if self.dataset_id == "bc_z":
            return bcz_episode_id(raw_episode, raw_episode_index, split=self.source_split)
        if self.dataset_id == "bridge":
            return bridge_episode_id(raw_episode, raw_episode_index, split=self.source_split)
        return f"{self.source_split}_{raw_episode_index:06d}"

    def _convert_episode(self, raw_episode_index: int, episode_id: str, steps: list[Any]) -> Any:
        if self.dataset_id == "bc_z":
            return convert_bcz_episode(
                episode_id,
                steps,
                sensed_close_threshold=float(
                    self.adapter_kwargs.get("sensed_close_threshold", 0.5)
                ),
            )
        if self.dataset_id == "bridge":
            return convert_bridge_episode(
                episode_id,
                steps,
                gripper_open_threshold=float(
                    self.adapter_kwargs.get("gripper_open_threshold", 0.5)
                ),
            )
        if self.dataset_id == "droid":
            converter = DroidStandardizedDataset.__new__(DroidStandardizedDataset)
            converter.split = self.source_split
            converter.gripper_closed_threshold = float(
                self.adapter_kwargs.get("gripper_closed_threshold", 0.5)
            )
            return converter._convert_episode(raw_episode_index, steps)
        if self.dataset_id == "fractal":
            converter = FractalStandardizedDataset.__new__(FractalStandardizedDataset)
            converter.split = self.source_split
            converter.gripper_command_threshold = float(
                self.adapter_kwargs.get("gripper_command_threshold", 0.05)
            )
            converter.gripper_state_threshold = float(
                self.adapter_kwargs.get("gripper_state_threshold", 0.5)
            )
            return converter._convert_episode(raw_episode_index, steps)
        if self.dataset_id == "kuka":
            converter = KukaStandardizedDataset.__new__(KukaStandardizedDataset)
            converter.split = self.source_split
            converter.gripper_command_threshold = float(
                self.adapter_kwargs.get("gripper_command_threshold", 0.05)
            )
            converter.gripper_state_threshold = float(
                self.adapter_kwargs.get("gripper_state_threshold", 0.5)
            )
            return converter._convert_episode(raw_episode_index, steps)
        raise KeyError(f"Unknown dataset_id: {self.dataset_id}")

    def _sample_from_episode(self, episode: Any, step_index: int) -> dict[str, Any]:
        if self.dataset_id == "bc_z":
            sample = bcz_sample_from_episode(
                episode,
                step_index,
                dataset_id=self.dataset_id,
                horizon=self.horizon,
                action_stats=self.action_stats,
            )
            return add_image_history(sample, episode, step_index)
        if self.dataset_id == "bridge":
            sample = bridge_sample_from_episode(
                episode,
                step_index,
                dataset_id=self.dataset_id,
                horizon=self.horizon,
                action_stats=self.action_stats,
            )
            return add_image_history(sample, episode, step_index)
        if self.dataset_id in {"fractal", "kuka"}:
            sample = _single_camera_sample_from_episode(
                episode,
                step_index,
                dataset_id=self.dataset_id,
                horizon=self.horizon,
                action_stats=self.action_stats,
            )
            return add_image_history(sample, episode, step_index)
        if self.dataset_id == "droid":
            sample = _droid_sample_from_episode(
                episode,
                step_index,
                dataset_id=self.dataset_id,
                horizon=self.horizon,
                action_stats=self.action_stats,
            )
            return add_image_history(sample, episode, step_index)
        raise KeyError(f"Unknown dataset_id: {self.dataset_id}")


class LazyCombinedStandardizedDataset(TorchIterableDataset):
    """Stream multiple lazy standardized datasets in sequence."""

    def __init__(
        self,
        datasets: Mapping[str, Iterable[dict[str, Any]]] | Sequence[tuple[str, Iterable[dict[str, Any]]]],
        *,
        split_name: str | None = None,
    ) -> None:
        items = datasets.items() if isinstance(datasets, Mapping) else datasets
        self.members = [
            LazyCombinedMember(dataset_id=dataset_id, dataset=dataset)
            for dataset_id, dataset in items
        ]
        self.split_name = split_name

    @property
    def dataset_ids(self) -> list[str]:
        return [member.dataset_id for member in self.members]

    def __iter__(self):
        for member in self.members:
            try:
                yield from member.dataset
            except (FileNotFoundError, OSError, RuntimeError) as exc:
                if is_missing_local_shard_error(exc):
                    _print_missing_shard_warning(member.dataset_id, exc)
                    continue
                raise


def build_lazy_combined_standardized_splits(
    configs: Sequence[LazyStandardizedDatasetConfig] | None = None,
    *,
    dataset_ids: Sequence[str] | None = None,
    root: str | Path | None = None,
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 0,
    stage: int | None = None,
    stage_count: int = 3,
    droid_stage_count: int = 2,
    horizon: int = DEFAULT_HORIZON,
    normalize_actions: bool = True,
    filter_empty_language: bool = True,
    max_samples_per_episode: int | None = DEFAULT_MAX_SAMPLES_PER_EPISODE,
    gripper_window_before: int = DEFAULT_GRIPPER_WINDOW_BEFORE,
    gripper_window_after: int = DEFAULT_GRIPPER_WINDOW_AFTER,
) -> dict[str, LazyCombinedStandardizedDataset]:
    dataset_configs = (
        list(configs)
        if configs is not None
        else default_lazy_standardized_dataset_configs(dataset_ids=dataset_ids, root=root)
    )

    split_datasets: dict[str, list[tuple[str, LazyStandardizedDataset]]] = {
        split_name: [] for split_name in SPLIT_NAMES
    }
    for split_name in SPLIT_NAMES:
        for config in dataset_configs:
            split_datasets[split_name].append(
                (
                    config.dataset_id,
                    LazyStandardizedDataset(
                        config,
                        split_name=split_name,
                        split_ratios=split_ratios,
                        seed=seed,
                        stage=stage,
                        stage_count=stage_count,
                        droid_stage_count=droid_stage_count,
                        horizon=horizon,
                        normalize_actions=normalize_actions,
                        filter_empty_language=filter_empty_language,
                        max_samples_per_episode=max_samples_per_episode,
                        gripper_window_before=gripper_window_before,
                        gripper_window_after=gripper_window_after,
                    ),
                )
            )

    return {
        split_name: LazyCombinedStandardizedDataset(datasets, split_name=split_name)
        for split_name, datasets in split_datasets.items()
    }


def build_combined_standardized_splits(*args: Any, **kwargs: Any) -> dict[str, LazyCombinedStandardizedDataset]:
    """Alias with the eager builder's name, scoped to this lazy module."""
    return build_lazy_combined_standardized_splits(*args, **kwargs)


def add_image_history(
    sample: dict[str, Any],
    episode: Any,
    step_index: int,
    *,
    history_size: int = 2,
) -> dict[str, Any]:
    """
    Add official-RDT-style image history while preserving the current `images` field.

    The order is previous/current timesteps, each with primary/wrist/secondary views.
    For the first step we duplicate the current image payload for shape stability, but
    mark the previous timestep invalid in `image_history_mask`.
    """
    if history_size != 2:
        raise ValueError("Only history_size=2 is currently supported")

    previous_index = max(0, int(step_index) - 1)
    timestep_indices = (previous_index, int(step_index))
    history: list[dict[str, Image.Image | None]] = []
    history_mask: list[dict[str, int]] = []
    for history_pos, image_index in enumerate(timestep_indices):
        is_valid_timestep = history_pos > 0 or step_index > 0
        frame = {
            "primary": _episode_image_at(episode, "primary", image_index),
            "wrist": _episode_image_at(episode, "wrist", image_index),
            "secondary": _episode_image_at(episode, "secondary", image_index),
        }
        history.append(frame)
        history_mask.append(
            {
                key: int(is_valid_timestep and value is not None)
                for key, value in frame.items()
            }
        )

    sample["image_history"] = history
    sample["image_history_mask"] = history_mask
    return sample


def _episode_image_at(episode: Any, key: str, index: int) -> Image.Image | None:
    if key == "primary":
        if hasattr(episode, "primary_images"):
            return _image_value_to_pil(episode.primary_images[index])
        if hasattr(episode, "images"):
            return _image_value_to_pil(episode.images[index])
        return None
    if key == "wrist":
        return _image_value_to_pil(episode.wrist_images[index]) if hasattr(episode, "wrist_images") else None
    if key == "secondary":
        return (
            _image_value_to_pil(episode.secondary_images[index])
            if hasattr(episode, "secondary_images")
            else None
        )
    raise KeyError(key)


def _image_value_to_pil(image: Any) -> Image.Image | None:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.size == 0 or not np.any(array):
        return None
    return Image.fromarray(array.astype(np.uint8)).convert("RGB")


def default_lazy_standardized_dataset_configs(
    dataset_ids: Sequence[str] | None = None,
    *,
    root: str | Path | None = None,
) -> list[LazyStandardizedDatasetConfig]:
    root_path = Path(root).expanduser().resolve() if root is not None else REPO_ROOT
    nested_mock_root = root_path / "dataset" / "mock_dataset"

    if _looks_like_mock_dataset_root(root_path):
        configs = _mock_layout_configs(root_path)
    elif nested_mock_root.exists():
        configs = _mock_layout_configs(nested_mock_root)
    else:
        configs = _hf_layout_configs(root_path)

    selected = list(configs) if dataset_ids is None else list(dataset_ids)
    return [configs[dataset_id] for dataset_id in selected]


def episode_split_name(
    dataset_id: str,
    episode_id: str,
    *,
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 0,
) -> str:
    if len(split_ratios) != len(SPLIT_NAMES):
        raise ValueError(f"Expected {len(SPLIT_NAMES)} split ratios")
    ratios = np.asarray(split_ratios, dtype=np.float64)
    if np.any(ratios < 0.0) or float(ratios.sum()) <= 0.0:
        raise ValueError("split ratios must be non-negative and sum to a positive value")
    ratios = ratios / ratios.sum()

    key = f"{seed}:{dataset_id}:{episode_id}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    bucket = int.from_bytes(digest, byteorder="little", signed=False) / float(2**64)

    cumulative = 0.0
    for split_name, ratio in zip(SPLIT_NAMES, ratios):
        cumulative += float(ratio)
        if bucket < cumulative:
            return split_name
    return SPLIT_NAMES[-1]


def episode_belongs_to_split(
    dataset_id: str,
    episode_id: str,
    split_name: str,
    *,
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 0,
) -> bool:
    return (
        episode_split_name(
            dataset_id,
            episode_id,
            split_ratios=split_ratios,
            seed=seed,
        )
        == split_name
    )


def episode_stage_index(
    dataset_id: str,
    episode_id: str,
    *,
    stage_count: int = 3,
    droid_stage_count: int = 2,
    seed: int = 0,
) -> int:
    active_stage_count = droid_stage_count if dataset_id == "droid" else stage_count
    if active_stage_count <= 0:
        raise ValueError("active stage count must be positive")
    key = f"stage:{seed}:{dataset_id}:{episode_id}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    bucket = int.from_bytes(digest, byteorder="little", signed=False) / float(2**64)
    return min(int(bucket * active_stage_count), active_stage_count - 1) + 1


def episode_belongs_to_stage(
    dataset_id: str,
    episode_id: str,
    stage: int,
    *,
    stage_count: int = 3,
    droid_stage_count: int = 2,
    seed: int = 0,
) -> bool:
    if stage < 1 or stage > stage_count:
        raise ValueError(f"stage must be in [1, {stage_count}], got {stage}")
    if dataset_id == "droid" and stage > droid_stage_count:
        return False
    return (
        episode_stage_index(
            dataset_id,
            episode_id,
            stage_count=stage_count,
            droid_stage_count=droid_stage_count,
            seed=seed,
        )
        == stage
    )


def sample_stage_index(
    dataset_id: str,
    episode_id: str,
    step_index: int,
    *,
    stage_count: int = 3,
    droid_stage_count: int = 2,
    seed: int = 0,
) -> int:
    active_stage_count = droid_stage_count if dataset_id == "droid" else stage_count
    if active_stage_count <= 0:
        raise ValueError("active stage count must be positive")
    key = f"stage:{seed}:{dataset_id}:{episode_id}:{step_index}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    bucket = int.from_bytes(digest, byteorder="little", signed=False) / float(2**64)
    return min(int(bucket * active_stage_count), active_stage_count - 1) + 1


def sample_belongs_to_stage(
    dataset_id: str,
    episode_id: str,
    step_index: int,
    stage: int,
    *,
    stage_count: int = 3,
    droid_stage_count: int = 2,
    seed: int = 0,
) -> bool:
    if stage < 1 or stage > stage_count:
        raise ValueError(f"stage must be in [1, {stage_count}], got {stage}")
    if dataset_id == "droid" and stage > droid_stage_count:
        return False
    return (
        sample_stage_index(
            dataset_id,
            episode_id,
            step_index,
            stage_count=stage_count,
            droid_stage_count=droid_stage_count,
            seed=seed,
        )
        == stage
    )


def is_missing_local_shard_error(exc: BaseException) -> bool:
    message = str(exc)
    missing_file_markers = (
        "No such file or directory",
        "The system cannot find the file specified",
        "open() failed",
        "NewRandomAccessFile failed to Create/Open",
        "Failed to open",
    )
    shard_markers = (
        ".tfrecord",
        ".array_record",
        "array_record-",
        "-of-",
    )
    return any(marker in message for marker in missing_file_markers) and any(
        marker in message for marker in shard_markers
    )


def _print_missing_shard_warning(dataset_id: str, exc: BaseException) -> None:
    print(
        f"[{dataset_id}] stopping this dataset stream because a local shard is "
        f"missing. Continuing with the next dataset. Details: {exc}"
    )


def iter_tfds_raw_episodes(
    data_dir: Path,
    *,
    split: str,
    shard_pattern: str | None,
):
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise ImportError(
            "Lazy standardized adapters require tensorflow and tensorflow-datasets "
            "to read TFDS shards."
        ) from exc

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    builder = tfds.builder_from_directory(str(data_dir))
    shard_paths = find_local_shards(data_dir, split=split, shard_pattern=shard_pattern)
    if shard_paths:
        dataset = tf.data.TFRecordDataset([str(path) for path in shard_paths]).map(
            builder.info.features.deserialize_example,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        dataset = builder.as_dataset(split=split)

    yield from dataset


def find_local_shards(
    data_dir: Path,
    *,
    split: str,
    shard_pattern: str | None,
) -> list[Path]:
    if shard_pattern is not None:
        pattern_path = Path(shard_pattern)
        if pattern_path.is_absolute():
            return sorted(path for path in pattern_path.parent.glob(pattern_path.name))
        return sorted(data_dir.glob(shard_pattern))

    split_shards = sorted(data_dir.glob(f"*{split}*.tfrecord*"))
    if split_shards:
        return split_shards
    return sorted(data_dir.glob("*.tfrecord*"))


def _single_camera_sample_from_episode(
    episode: Any,
    step_index: int,
    *,
    dataset_id: str,
    horizon: int,
    action_stats: ActionNormalizationStats | None,
) -> dict[str, Any]:
    actions, actions_mask = pad_action_horizon(
        episode.actions,
        step_index,
        horizon=horizon,
        action_dim=ACTION_DIM,
    )
    if action_stats is not None:
        actions = normalize_action_horizon(actions, actions_mask, action_stats)

    return {
        "dataset_id": dataset_id,
        "episode_id": episode.episode_id,
        "step_idx": str(step_index),
        "instruction": episode.instructions[step_index],
        "images": {
            "primary": Image.fromarray(episode.images[step_index]).copy(),
            "wrist": None,
            "secondary": None,
        },
        "image_mask": {"primary": 1, "wrist": 0, "secondary": 0},
        "state": episode.states[step_index].astype(np.float32, copy=True),
        "state_mask": np.ones((STATE_DIM,), dtype=np.float32),
        "actions": actions,
        "actions_mask": actions_mask,
    }


def _droid_sample_from_episode(
    episode: Any,
    step_index: int,
    *,
    dataset_id: str,
    horizon: int,
    action_stats: ActionNormalizationStats | None,
) -> dict[str, Any]:
    actions, actions_mask = pad_action_horizon(
        episode.actions,
        step_index,
        horizon=horizon,
        action_dim=ACTION_DIM,
    )
    if action_stats is not None:
        actions = normalize_action_horizon(actions, actions_mask, action_stats)

    primary = droid_pil_or_none(episode.primary_images[step_index])
    wrist = droid_pil_or_none(episode.wrist_images[step_index])
    secondary = droid_pil_or_none(episode.secondary_images[step_index])

    return {
        "dataset_id": dataset_id,
        "episode_id": episode.episode_id,
        "step_idx": str(step_index),
        "instruction": episode.instructions[step_index],
        "images": {
            "primary": primary,
            "wrist": wrist,
            "secondary": secondary,
        },
        "image_mask": {
            "primary": int(primary is not None),
            "wrist": int(wrist is not None),
            "secondary": int(secondary is not None),
        },
        "state": episode.states[step_index].astype(np.float32, copy=True),
        "state_mask": np.ones((STATE_DIM,), dtype=np.float32),
        "actions": actions,
        "actions_mask": actions_mask,
    }


def _is_successful_episode(raw_episode: Any) -> bool:
    if "success" not in raw_episode:
        return True
    raw_success = raw_episode["success"].numpy() if hasattr(raw_episode["success"], "numpy") else raw_episode["success"]
    if isinstance(raw_success, np.ndarray):
        raw_success = raw_success.item()
    return bool(raw_success)


def _mock_layout_configs(mock_root: Path) -> dict[str, LazyStandardizedDatasetConfig]:
    return {
        "bc_z": LazyStandardizedDatasetConfig(
            dataset_id="bc_z",
            data_dir=mock_root / "bc_z_dataset" / "data",
        ),
        "bridge": LazyStandardizedDatasetConfig(
            dataset_id="bridge",
            data_dir=_first_existing(
                mock_root / "bridge_dataset" / "data",
                mock_root / "bridge_dataset" / "bridge_subset",
            ),
        ),
        "droid": LazyStandardizedDatasetConfig(
            dataset_id="droid",
            data_dir=_first_existing(
                mock_root / "droid_dataset" / "data",
                mock_root / "droid_dataset" / "droid_100" / "1.0.0",
            ),
        ),
        "fractal": LazyStandardizedDatasetConfig(
            dataset_id="fractal",
            data_dir=mock_root / "fractal_dataset" / "data",
        ),
        "kuka": LazyStandardizedDatasetConfig(
            dataset_id="kuka",
            data_dir=mock_root / "kuka_dataset" / "data",
        ),
    }


def _looks_like_mock_dataset_root(root: Path) -> bool:
    return any(
        (root / dataset_dir).exists()
        for dataset_dir in (
            "bc_z_dataset",
            "bridge_dataset",
            "droid_dataset",
            "fractal_dataset",
            "kuka_dataset",
        )
    )


def _hf_layout_configs(root: Path) -> dict[str, LazyStandardizedDatasetConfig]:
    return {
        "bc_z": LazyStandardizedDatasetConfig(
            dataset_id="bc_z",
            data_dir=root / "bc_z" / "data",
        ),
        "bridge": LazyStandardizedDatasetConfig(
            dataset_id="bridge",
            data_dir=root / "bridge" / "data",
        ),
        "droid": LazyStandardizedDatasetConfig(
            dataset_id="droid",
            data_dir=root / "droid" / "data",
        ),
        "fractal": LazyStandardizedDatasetConfig(
            dataset_id="fractal",
            data_dir=root / "fractal" / "data",
        ),
        "kuka": LazyStandardizedDatasetConfig(
            dataset_id="kuka",
            data_dir=root / "kuka" / "data",
        ),
    }


def _first_existing(preferred: Path, fallback: Path) -> Path:
    return preferred if preferred.exists() else fallback
