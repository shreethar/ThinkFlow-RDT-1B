from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .action_stats import (
    ActionNormalizationStats,
    normalize_action_horizon,
    resolve_action_stats,
)
from .fractal import ACTION_DIM, DEFAULT_HORIZON, STATE_DIM, pad_action_horizon
from .sample_filtering import (
    DEFAULT_GRIPPER_WINDOW_AFTER,
    DEFAULT_GRIPPER_WINDOW_BEFORE,
    DEFAULT_MAX_SAMPLES_PER_EPISODE,
    build_dataset_sample_index,
    build_episode_sample_indices,
)


@dataclass(frozen=True)
class BridgeEpisode:
    episode_id: str
    instructions: list[str]
    primary_images: list[np.ndarray | None]
    secondary_images: list[np.ndarray | None]
    states: np.ndarray
    actions: np.ndarray


def zero_image_to_none(image: np.ndarray) -> np.ndarray | None:
    """Return None for Bridge's all-zero placeholder images."""
    array = np.asarray(image, dtype=np.uint8)
    if array.size == 0 or not np.any(array):
        return None
    return array


def pil_or_none(image: np.ndarray | None) -> Image.Image | None:
    if image is None:
        return None
    return Image.fromarray(image).copy()


def standardize_bridge_gripper_open_to_closed(
    raw_gripper: np.ndarray,
    *,
    open_threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert Bridge's gripper convention to the project convention.

    Bridge raw convention:
      1 => open, 0 => closed

    Standardized convention:
      0 => open, 1 => closed
    """
    raw = np.asarray(raw_gripper, dtype=np.float32)
    return (raw <= open_threshold).astype(np.float32)


class BridgeStandardizedDataset:
    """
    Bridge TFDS adapter emitting one standardized sample per timestep.

    Output schema:
      dataset_id: str
      episode_id: str
      step_idx: str
      instruction: str
      images: {"primary": PIL.Image | None, "wrist": None, "secondary": PIL.Image | None}
      image_mask: {"primary": 0 | 1, "wrist": 0, "secondary": 0 | 1}
      state: np.ndarray [7]
      state_mask: np.ndarray [7]
      actions: np.ndarray [H, 7]
      actions_mask: np.ndarray [H]

    Bridge mapping:
      primary image: steps["observation"]["image_0"]
      secondary image: steps["observation"]["image_1"]
      wrist image: None
      state: steps["observation"]["state"], with gripper inverted to 0=open/1=closed
      action: steps["action"], with gripper inverted to 0=open/1=closed
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        split: str = "train",
        horizon: int = DEFAULT_HORIZON,
        dataset_id: str = "bridge",
        max_episodes: int | None = None,
        shard_pattern: str | None = None,
        gripper_open_threshold: float = 0.5,
        normalize_actions: bool = False,
        action_stats_path: str | Path | None = None,
        action_stats: ActionNormalizationStats | dict[str, Any] | None = None,
        filter_empty_language: bool = True,
        max_samples_per_episode: int | None = DEFAULT_MAX_SAMPLES_PER_EPISODE,
        gripper_window_before: int = DEFAULT_GRIPPER_WINDOW_BEFORE,
        gripper_window_after: int = DEFAULT_GRIPPER_WINDOW_AFTER,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.split = split
        self.horizon = horizon
        self.dataset_id = dataset_id
        self.gripper_open_threshold = gripper_open_threshold
        self.filter_empty_language = filter_empty_language
        self.max_samples_per_episode = max_samples_per_episode
        self.gripper_window_before = gripper_window_before
        self.gripper_window_after = gripper_window_after
        self.action_stats = resolve_action_stats(
            normalize_actions=normalize_actions,
            action_stats=action_stats,
            action_stats_path=action_stats_path,
            search_dir=self.data_dir,
        )

        self.episodes = self._load_episodes(
            max_episodes=max_episodes,
            shard_pattern=shard_pattern,
        )
        self.index = build_dataset_sample_index(
            self.episodes,
            max_samples_per_episode=max_samples_per_episode,
            filter_empty_language=filter_empty_language,
            gripper_window_before=gripper_window_before,
            gripper_window_after=gripper_window_after,
        )
        if not self.index:
            raise ValueError(f"No Bridge samples found in {self.data_dir}")

    @classmethod
    def from_episodes(
        cls,
        episodes: list[BridgeEpisode],
        *,
        horizon: int = DEFAULT_HORIZON,
        dataset_id: str = "bridge",
        split: str = "train",
        normalize_actions: bool = False,
        action_stats: ActionNormalizationStats | dict[str, Any] | None = None,
        filter_empty_language: bool = True,
        max_samples_per_episode: int | None = DEFAULT_MAX_SAMPLES_PER_EPISODE,
        gripper_window_before: int = DEFAULT_GRIPPER_WINDOW_BEFORE,
        gripper_window_after: int = DEFAULT_GRIPPER_WINDOW_AFTER,
    ) -> BridgeStandardizedDataset:
        obj = cls.__new__(cls)
        obj.data_dir = Path(".").resolve()
        obj.split = split
        obj.horizon = horizon
        obj.dataset_id = dataset_id
        obj.gripper_open_threshold = 0.5
        obj.filter_empty_language = filter_empty_language
        obj.max_samples_per_episode = max_samples_per_episode
        obj.gripper_window_before = gripper_window_before
        obj.gripper_window_after = gripper_window_after
        obj.action_stats = resolve_action_stats(
            normalize_actions=normalize_actions or action_stats is not None,
            action_stats=action_stats,
        )
        obj.episodes = episodes
        obj.index = build_dataset_sample_index(
            episodes,
            max_samples_per_episode=max_samples_per_episode,
            filter_empty_language=filter_empty_language,
            gripper_window_before=gripper_window_before,
            gripper_window_after=gripper_window_after,
        )
        return obj

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, step_index = self.index[index]
        episode = self.episodes[episode_index]
        return bridge_sample_from_episode(
            episode,
            step_index,
            dataset_id=self.dataset_id,
            horizon=self.horizon,
            action_stats=self.action_stats,
        )

    def _load_episodes(
        self,
        *,
        max_episodes: int | None,
        shard_pattern: str | None,
    ) -> list[BridgeEpisode]:
        if not self.data_dir.exists():
            raise FileNotFoundError(self.data_dir)

        episodes: list[BridgeEpisode] = []
        for episode_index, raw_episode in enumerate(
            iter_bridge_raw_episodes(
                self.data_dir,
                split=self.split,
                shard_pattern=shard_pattern,
            )
        ):
            if max_episodes is not None and episode_index >= max_episodes:
                break
            steps = list(raw_episode["steps"])
            if not steps:
                continue
            episode_id = self._episode_id(raw_episode, episode_index)
            episodes.append(self._convert_episode(episode_id, steps))
        return episodes

    def _find_local_shards(self, shard_pattern: str | None) -> list[Path]:
        return find_local_shards(self.data_dir, split=self.split, shard_pattern=shard_pattern)

    def _convert_episode(self, episode_id: str, steps: list[Any]) -> BridgeEpisode:
        return convert_bridge_episode(
            episode_id,
            steps,
            gripper_open_threshold=self.gripper_open_threshold,
        )

    def _episode_id(self, raw_episode: Any, episode_index: int) -> str:
        return bridge_episode_id(raw_episode, episode_index, split=self.split)

    @staticmethod
    def _decode_text(value: Any) -> str:
        return decode_text(value)


class BridgeStandardizedIterableDataset:
    """
    Lazy Bridge adapter for large local shard sets.

    This has the same sample schema as BridgeStandardizedDataset, but it does
    not build an in-memory episode list. It streams one TFDS episode at a time,
    samples up to max_samples_per_episode steps, yields those samples, then
    releases the episode.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        split: str = "train",
        horizon: int = DEFAULT_HORIZON,
        dataset_id: str = "bridge",
        max_episodes: int | None = None,
        shard_pattern: str | None = None,
        gripper_open_threshold: float = 0.5,
        normalize_actions: bool = False,
        action_stats_path: str | Path | None = None,
        action_stats: ActionNormalizationStats | dict[str, Any] | None = None,
        filter_empty_language: bool = True,
        max_samples_per_episode: int | None = DEFAULT_MAX_SAMPLES_PER_EPISODE,
        gripper_window_before: int = DEFAULT_GRIPPER_WINDOW_BEFORE,
        gripper_window_after: int = DEFAULT_GRIPPER_WINDOW_AFTER,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.split = split
        self.horizon = horizon
        self.dataset_id = dataset_id
        self.max_episodes = max_episodes
        self.shard_pattern = shard_pattern
        self.gripper_open_threshold = gripper_open_threshold
        self.filter_empty_language = filter_empty_language
        self.max_samples_per_episode = max_samples_per_episode
        self.gripper_window_before = gripper_window_before
        self.gripper_window_after = gripper_window_after
        self.action_stats = resolve_action_stats(
            normalize_actions=normalize_actions,
            action_stats=action_stats,
            action_stats_path=action_stats_path,
            search_dir=self.data_dir,
        )

    def __iter__(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(self.data_dir)

        yielded_any = False
        for episode_index, raw_episode in enumerate(
            iter_bridge_raw_episodes(
                self.data_dir,
                split=self.split,
                shard_pattern=self.shard_pattern,
            )
        ):
            if self.max_episodes is not None and episode_index >= self.max_episodes:
                break
            steps = list(raw_episode["steps"])
            if not steps:
                continue
            episode = convert_bridge_episode(
                bridge_episode_id(raw_episode, episode_index, split=self.split),
                steps,
                gripper_open_threshold=self.gripper_open_threshold,
            )
            step_indices = build_episode_sample_indices(
                episode.instructions,
                episode.actions,
                max_samples_per_episode=self.max_samples_per_episode,
                filter_empty_language=self.filter_empty_language,
                gripper_window_before=self.gripper_window_before,
                gripper_window_after=self.gripper_window_after,
            )
            for step_index in step_indices:
                yielded_any = True
                yield bridge_sample_from_episode(
                    episode,
                    step_index,
                    dataset_id=self.dataset_id,
                    horizon=self.horizon,
                    action_stats=self.action_stats,
                )
        if not yielded_any:
            raise ValueError(f"No Bridge samples found in {self.data_dir}")


def bridge_sample_from_episode(
    episode: BridgeEpisode,
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

    primary = pil_or_none(episode.primary_images[step_index])
    secondary = pil_or_none(episode.secondary_images[step_index])

    return {
        "dataset_id": dataset_id,
        "episode_id": episode.episode_id,
        "step_idx": str(step_index),
        "instruction": episode.instructions[step_index],
        "images": {
            "primary": primary,
            "wrist": None,
            "secondary": secondary,
        },
        "image_mask": {
            "primary": int(primary is not None),
            "wrist": 0,
            "secondary": int(secondary is not None),
        },
        "state": episode.states[step_index].astype(np.float32, copy=True),
        "state_mask": np.ones((STATE_DIM,), dtype=np.float32),
        "actions": actions,
        "actions_mask": actions_mask,
    }


def convert_bridge_episode(
    episode_id: str,
    steps: list[Any],
    *,
    gripper_open_threshold: float,
) -> BridgeEpisode:
    instructions: list[str] = []
    primary_images: list[np.ndarray | None] = []
    secondary_images: list[np.ndarray | None] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    for step in steps:
        observation = step["observation"]
        instructions.append(decode_text(step["language_instruction"]))
        primary_images.append(
            zero_image_to_none(np.asarray(observation["image_0"].numpy(), dtype=np.uint8))
        )
        secondary_images.append(
            zero_image_to_none(np.asarray(observation["image_1"].numpy(), dtype=np.uint8))
        )
        states.append(np.asarray(observation["state"].numpy(), dtype=np.float32))
        actions.append(np.asarray(step["action"].numpy(), dtype=np.float32))

    state_array = np.stack(states, axis=0).astype(np.float32)
    action_array = np.stack(actions, axis=0).astype(np.float32)

    state_array[:, 6] = standardize_bridge_gripper_open_to_closed(
        state_array[:, 6],
        open_threshold=gripper_open_threshold,
    )
    action_array[:, 6] = standardize_bridge_gripper_open_to_closed(
        action_array[:, 6],
        open_threshold=gripper_open_threshold,
    )

    return BridgeEpisode(
        episode_id=episode_id,
        instructions=instructions,
        primary_images=primary_images,
        secondary_images=secondary_images,
        states=state_array,
        actions=action_array,
    )


def iter_bridge_raw_episodes(
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
            "Bridge adapters require tensorflow and tensorflow-datasets to read TFDS shards."
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


def bridge_episode_id(raw_episode: Any, episode_index: int, *, split: str) -> str:
    metadata = raw_episode.get("episode_metadata", None)
    if metadata is None or "episode_id" not in metadata:
        return f"{split}_{episode_index:06d}"
    raw_episode_id = metadata["episode_id"].numpy()
    if isinstance(raw_episode_id, np.ndarray):
        raw_episode_id = raw_episode_id.item()
    return str(raw_episode_id)


def decode_text(value: Any) -> str:
    raw = value.numpy() if hasattr(value, "numpy") else value
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)
