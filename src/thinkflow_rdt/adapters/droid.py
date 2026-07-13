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
)


@dataclass(frozen=True)
class DroidEpisode:
    episode_id: str
    instructions: list[str]
    primary_images: list[np.ndarray | None]
    wrist_images: list[np.ndarray | None]
    secondary_images: list[np.ndarray | None]
    states: np.ndarray
    actions: np.ndarray


def pil_or_none(image: np.ndarray | None) -> Image.Image | None:
    if image is None:
        return None
    return Image.fromarray(image).copy()


def standardize_droid_state(
    cartesian_position: np.ndarray,
    gripper_position: np.ndarray,
    *,
    closed_threshold: float = 0.5,
) -> np.ndarray:
    """
    Build canonical 7D Droid state.

    Droid observation/cartesian_position is the 6D end-effector state.
    Droid gripper already uses 0=open, 1=closed, so we only threshold it.
    """
    cartesian = np.asarray(cartesian_position, dtype=np.float32)
    gripper = np.asarray(gripper_position, dtype=np.float32)
    if cartesian.shape[-1] != 6:
        raise ValueError(f"Expected cartesian_position last dim 6, got {cartesian.shape}")
    if gripper.ndim == cartesian.ndim - 1:
        gripper = np.expand_dims(gripper, axis=-1)
    if gripper.shape[-1] != 1:
        raise ValueError(f"Expected gripper_position last dim 1, got {gripper.shape}")

    closed = (gripper >= closed_threshold).astype(np.float32)
    return np.concatenate([cartesian, closed], axis=-1).astype(np.float32)


def standardize_droid_action(
    absolute_action: np.ndarray,
    cartesian_position: np.ndarray,
    *,
    closed_threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert Droid's absolute Cartesian action into canonical relative delta action.

    Droid steps/action stores [absolute_cartesian_position(6), gripper_position].
    The project action convention stores [cartesian_delta(6), gripper_closed].
    """
    action = np.asarray(absolute_action, dtype=np.float32)
    cartesian = np.asarray(cartesian_position, dtype=np.float32)
    if action.shape[-1] != 7:
        raise ValueError(f"Expected action last dim 7, got {action.shape}")
    if cartesian.shape[-1] != 6:
        raise ValueError(f"Expected cartesian_position last dim 6, got {cartesian.shape}")

    delta = action[..., :6] - cartesian
    gripper = (action[..., 6:7] >= closed_threshold).astype(np.float32)
    return np.concatenate([delta, gripper], axis=-1).astype(np.float32)


class DroidStandardizedDataset:
    """
    Droid TFDS adapter emitting one standardized sample per timestep.

    Output schema:
      dataset_id: str
      episode_id: str
      step_idx: str
      instruction: str
      images: {"primary": PIL.Image, "wrist": PIL.Image, "secondary": PIL.Image}
      image_mask: {"primary": 1, "wrist": 1, "secondary": 1}
      state: np.ndarray [7]
      state_mask: np.ndarray [7]
      actions: np.ndarray [H, 7]
      actions_mask: np.ndarray [H]

    Droid mapping:
      primary image: steps["observation"]["exterior_image_1_left"]
      wrist image: steps["observation"]["wrist_image_left"]
      secondary image: steps["observation"]["exterior_image_2_left"]
      state: [observation/cartesian_position(6), observation/gripper_position]
      action: [steps/action[:6] - observation/cartesian_position, steps/action[6]]
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        split: str = "train",
        horizon: int = DEFAULT_HORIZON,
        dataset_id: str = "droid",
        max_episodes: int | None = None,
        shard_pattern: str | None = None,
        gripper_closed_threshold: float = 0.5,
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
        self.gripper_closed_threshold = gripper_closed_threshold
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
            raise ValueError(f"No Droid samples found in {self.data_dir}")

    @classmethod
    def from_episodes(
        cls,
        episodes: list[DroidEpisode],
        *,
        horizon: int = DEFAULT_HORIZON,
        dataset_id: str = "droid",
        split: str = "train",
        normalize_actions: bool = False,
        action_stats: ActionNormalizationStats | dict[str, Any] | None = None,
        filter_empty_language: bool = True,
        max_samples_per_episode: int | None = DEFAULT_MAX_SAMPLES_PER_EPISODE,
        gripper_window_before: int = DEFAULT_GRIPPER_WINDOW_BEFORE,
        gripper_window_after: int = DEFAULT_GRIPPER_WINDOW_AFTER,
    ) -> DroidStandardizedDataset:
        obj = cls.__new__(cls)
        obj.data_dir = Path(".").resolve()
        obj.split = split
        obj.horizon = horizon
        obj.dataset_id = dataset_id
        obj.gripper_closed_threshold = 0.5
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
        actions, actions_mask = pad_action_horizon(
            episode.actions,
            step_index,
            horizon=self.horizon,
            action_dim=ACTION_DIM,
        )
        if self.action_stats is not None:
            actions = normalize_action_horizon(actions, actions_mask, self.action_stats)

        primary = pil_or_none(episode.primary_images[step_index])
        wrist = pil_or_none(episode.wrist_images[step_index])
        secondary = pil_or_none(episode.secondary_images[step_index])

        return {
            "dataset_id": self.dataset_id,
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

    def _load_episodes(
        self,
        *,
        max_episodes: int | None,
        shard_pattern: str | None,
    ) -> list[DroidEpisode]:
        if not self.data_dir.exists():
            raise FileNotFoundError(self.data_dir)

        try:
            import tensorflow as tf
            import tensorflow_datasets as tfds
        except ImportError as exc:
            raise ImportError(
                "DroidStandardizedDataset requires tensorflow and "
                "tensorflow-datasets to read TFDS shards."
            ) from exc

        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

        builder = tfds.builder_from_directory(str(self.data_dir))
        shard_paths = self._find_local_shards(shard_pattern)
        if shard_paths:
            dataset = tf.data.TFRecordDataset([str(path) for path in shard_paths]).map(
                builder.info.features.deserialize_example,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            dataset = builder.as_dataset(split=self.split)

        episodes: list[DroidEpisode] = []
        for episode_index, raw_episode in enumerate(dataset):
            if max_episodes is not None and episode_index >= max_episodes:
                break
            steps = list(raw_episode["steps"])
            if not steps:
                continue
            episodes.append(self._convert_episode(episode_index, steps))
        return episodes

    def _find_local_shards(self, shard_pattern: str | None) -> list[Path]:
        if shard_pattern is not None:
            pattern_path = Path(shard_pattern)
            if pattern_path.is_absolute():
                return sorted(path for path in pattern_path.parent.glob(pattern_path.name))
            return sorted(self.data_dir.glob(shard_pattern))

        split_shards = sorted(self.data_dir.glob(f"*{self.split}*.tfrecord*"))
        if split_shards:
            return split_shards
        return sorted(self.data_dir.glob("*.tfrecord*"))

    def _convert_episode(self, episode_index: int, steps: list[Any]) -> DroidEpisode:
        instructions: list[str] = []
        primary_images: list[np.ndarray | None] = []
        wrist_images: list[np.ndarray | None] = []
        secondary_images: list[np.ndarray | None] = []
        cartesian_positions: list[np.ndarray] = []
        gripper_positions: list[np.ndarray] = []
        absolute_actions: list[np.ndarray] = []

        for step in steps:
            observation = step["observation"]
            instructions.append(self._instruction(step))
            primary_images.append(
                np.asarray(observation["exterior_image_1_left"].numpy(), dtype=np.uint8)
            )
            wrist_images.append(
                np.asarray(observation["wrist_image_left"].numpy(), dtype=np.uint8)
            )
            secondary_images.append(
                np.asarray(observation["exterior_image_2_left"].numpy(), dtype=np.uint8)
            )
            cartesian_positions.append(
                np.asarray(observation["cartesian_position"].numpy(), dtype=np.float32)
            )
            gripper_positions.append(
                np.asarray(observation["gripper_position"].numpy(), dtype=np.float32)
            )
            absolute_actions.append(np.asarray(step["action"].numpy(), dtype=np.float32))

        cartesian_array = np.stack(cartesian_positions, axis=0).astype(np.float32)
        gripper_array = np.stack(gripper_positions, axis=0).astype(np.float32)
        absolute_action_array = np.stack(absolute_actions, axis=0).astype(np.float32)

        states = standardize_droid_state(
            cartesian_array,
            gripper_array,
            closed_threshold=self.gripper_closed_threshold,
        )
        actions = standardize_droid_action(
            absolute_action_array,
            cartesian_array,
            closed_threshold=self.gripper_closed_threshold,
        )

        return DroidEpisode(
            episode_id=f"{self.split}_{episode_index:06d}",
            instructions=instructions,
            primary_images=primary_images,
            wrist_images=wrist_images,
            secondary_images=secondary_images,
            states=states,
            actions=actions,
        )

    def _instruction(self, step: Any) -> str:
        for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
            if key not in step:
                continue
            text = self._decode_text(step[key]).strip()
            if text:
                return text
        return ""

    @staticmethod
    def _decode_text(value: Any) -> str:
        raw = value.numpy() if hasattr(value, "numpy") else value
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)
