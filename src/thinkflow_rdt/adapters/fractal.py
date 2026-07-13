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
from .sample_filtering import (
    DEFAULT_GRIPPER_WINDOW_AFTER,
    DEFAULT_GRIPPER_WINDOW_BEFORE,
    DEFAULT_MAX_SAMPLES_PER_EPISODE,
    build_dataset_sample_index,
)


DEFAULT_HORIZON = 32
STATE_DIM = 7
ACTION_DIM = 7


@dataclass(frozen=True)
class FractalEpisode:
    episode_id: str
    instructions: list[str]
    images: list[np.ndarray]
    states: np.ndarray
    actions: np.ndarray


def quat_xyzw_to_rpy_rad(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) in xyzw order to roll, pitch, yaw radians."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.where(
        np.abs(sin_pitch) >= 1.0,
        np.sign(sin_pitch) * np.pi / 2.0,
        np.arcsin(sin_pitch),
    )
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)


def standardize_binary_gripper_action(
    gripper_closedness_action: np.ndarray,
    gripper_closed: np.ndarray,
    *,
    command_threshold: float = 0.05,
    state_threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert Fractal's gripper command stream to a binary closed target.

    Fractal action semantics are command-like:
      positive => close, negative => open, near zero => no-op/hold.
    The standardized action dimension is absolute binary target:
      0 => open, 1 => closed.
    """
    raw_action = np.asarray(gripper_closedness_action, dtype=np.float32).reshape(-1)
    closed_state = np.asarray(gripper_closed, dtype=np.float32).reshape(-1)
    if raw_action.shape[0] != closed_state.shape[0]:
        raise ValueError(
            "gripper_closedness_action and gripper_closed must have the same length"
        )
    if raw_action.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    current_target = float(closed_state[0] >= state_threshold)
    binary_target = np.zeros_like(raw_action, dtype=np.float32)
    for index, command in enumerate(raw_action):
        if command > command_threshold:
            current_target = 1.0
        elif command < -command_threshold:
            current_target = 0.0
        binary_target[index] = current_target
    return binary_target


def pad_action_horizon(
    actions: np.ndarray,
    start_idx: int,
    *,
    horizon: int = DEFAULT_HORIZON,
    action_dim: int = ACTION_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != action_dim:
        raise ValueError(f"Expected actions [T, {action_dim}], got {actions.shape}")
    if start_idx < 0 or start_idx >= actions.shape[0]:
        raise IndexError(start_idx)

    valid = min(horizon, actions.shape[0] - start_idx)
    output = np.zeros((horizon, action_dim), dtype=np.float32)
    mask = np.zeros((horizon,), dtype=np.float32)
    output[:valid] = actions[start_idx : start_idx + valid]
    mask[:valid] = 1.0
    return output, mask


class FractalStandardizedDataset:
    """
    Fractal TFDS adapter emitting one standardized sample per timestep.

    Output schema:
      dataset_id: str
      episode_id: str
      step_idx: str
      instruction: str
      images: {"primary": PIL.Image, "wrist": None, "secondary": None}
      image_mask: {"primary": 1, "wrist": 0, "secondary": 0}
      state: np.ndarray [7]
      state_mask: np.ndarray [7]
      actions: np.ndarray [H, 7]
      actions_mask: np.ndarray [H]
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        split: str = "train",
        horizon: int = DEFAULT_HORIZON,
        dataset_id: str = "fractal",
        max_episodes: int | None = None,
        shard_pattern: str | None = None,
        gripper_command_threshold: float = 0.05,
        gripper_state_threshold: float = 0.5,
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
        self.gripper_command_threshold = gripper_command_threshold
        self.gripper_state_threshold = gripper_state_threshold
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
            raise ValueError(f"No Fractal samples found in {self.data_dir}")

    @classmethod
    def from_episodes(
        cls,
        episodes: list[FractalEpisode],
        *,
        horizon: int = DEFAULT_HORIZON,
        dataset_id: str = "fractal",
        split: str = "train",
        normalize_actions: bool = False,
        action_stats: ActionNormalizationStats | dict[str, Any] | None = None,
        filter_empty_language: bool = True,
        max_samples_per_episode: int | None = DEFAULT_MAX_SAMPLES_PER_EPISODE,
        gripper_window_before: int = DEFAULT_GRIPPER_WINDOW_BEFORE,
        gripper_window_after: int = DEFAULT_GRIPPER_WINDOW_AFTER,
    ) -> FractalStandardizedDataset:
        obj = cls.__new__(cls)
        obj.data_dir = Path(".").resolve()
        obj.split = split
        obj.horizon = horizon
        obj.dataset_id = dataset_id
        obj.gripper_command_threshold = 0.05
        obj.gripper_state_threshold = 0.5
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

        return {
            "dataset_id": self.dataset_id,
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

    def _load_episodes(
        self,
        *,
        max_episodes: int | None,
        shard_pattern: str | None,
    ) -> list[FractalEpisode]:
        if not self.data_dir.exists():
            raise FileNotFoundError(self.data_dir)

        try:
            import tensorflow as tf
            import tensorflow_datasets as tfds
        except ImportError as exc:
            raise ImportError(
                "FractalStandardizedDataset requires tensorflow and "
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

        episodes: list[FractalEpisode] = []
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

    def _convert_episode(self, episode_index: int, steps: list[Any]) -> FractalEpisode:
        instructions: list[str] = []
        images: list[np.ndarray] = []
        poses: list[np.ndarray] = []
        world_vectors: list[np.ndarray] = []
        rotation_deltas: list[np.ndarray] = []
        gripper_actions: list[float] = []
        gripper_closed: list[float] = []

        for step in steps:
            observation = step["observation"]
            action = step["action"]
            instructions.append(
                self._decode_text(observation["natural_language_instruction"])
            )
            images.append(np.asarray(observation["image"].numpy(), dtype=np.uint8))
            poses.append(
                np.asarray(observation["base_pose_tool_reached"].numpy(), dtype=np.float32)
            )
            world_vectors.append(
                np.asarray(action["world_vector"].numpy(), dtype=np.float32)
            )
            rotation_deltas.append(
                np.asarray(action["rotation_delta"].numpy(), dtype=np.float32)
            )
            gripper_actions.append(
                float(action["gripper_closedness_action"].numpy().reshape(-1)[0])
            )
            gripper_closed.append(
                float(observation["gripper_closed"].numpy().reshape(-1)[0])
            )

        pose = np.stack(poses, axis=0).astype(np.float32)
        rpy = quat_xyzw_to_rpy_rad(pose[:, 3:7])
        gripper_state = (
            np.asarray(gripper_closed, dtype=np.float32) >= self.gripper_state_threshold
        ).astype(np.float32)
        states = np.concatenate([pose[:, :3], rpy, gripper_state[:, None]], axis=1)

        gripper_target = standardize_binary_gripper_action(
            np.asarray(gripper_actions, dtype=np.float32),
            np.asarray(gripper_closed, dtype=np.float32),
            command_threshold=self.gripper_command_threshold,
            state_threshold=self.gripper_state_threshold,
        )
        actions = np.concatenate(
            [
                np.stack(world_vectors, axis=0).astype(np.float32),
                np.stack(rotation_deltas, axis=0).astype(np.float32),
                gripper_target[:, None],
            ],
            axis=1,
        )

        return FractalEpisode(
            episode_id=f"{self.split}_{episode_index:06d}",
            instructions=instructions,
            images=images,
            states=states.astype(np.float32),
            actions=actions.astype(np.float32),
        )

    @staticmethod
    def _decode_text(value: Any) -> str:
        raw = value.numpy() if hasattr(value, "numpy") else value
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)
