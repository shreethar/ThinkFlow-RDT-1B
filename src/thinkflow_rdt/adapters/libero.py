from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .action_stats import (
    ActionNormalizationStats,
    denormalize_action_array,
    normalize_action_horizon,
)
from .fractal import pad_action_horizon


@dataclass(frozen=True)
class LiberoEpisode:
    episode_id: str
    instruction: str
    primary_images: np.ndarray
    wrist_images: np.ndarray | None
    states: np.ndarray
    actions: np.ndarray
    joint_states: np.ndarray | None = None

    @property
    def instructions(self) -> list[str]:
        return [self.instruction] * len(self.actions)


def _dataset(group, *names: str):
    for name in names:
        if name in group:
            return np.asarray(group[name])
    raise KeyError(f"None of {names!r} found below {group.name}")


def _optional_dataset(group, *names: str) -> np.ndarray | None:
    for name in names:
        if name in group:
            return np.asarray(group[name])
    return None


def _libero_joint_positions(obs) -> np.ndarray | None:
    direct = _optional_dataset(
        obs,
        "joint_states",
        "robot0_joint_pos",
        "joint_pos",
        "joint_positions",
        "joint_qpos",
    )
    if direct is not None:
        values = np.asarray(direct, dtype=np.float32)
    elif "robot0_joint_pos_sin" in obs and "robot0_joint_pos_cos" in obs:
        # Some robomimic-style files store joints as sin/cos only. Recover the
        # wrapped angle so the cache still has a plain joint-position signal.
        values = np.arctan2(
            np.asarray(obs["robot0_joint_pos_sin"], dtype=np.float32),
            np.asarray(obs["robot0_joint_pos_cos"], dtype=np.float32),
        )
    else:
        return None
    if values.ndim == 1:
        values = values[:, None]
    return values.astype(np.float32, copy=False)


def _instruction(group, fallback: str) -> str:
    for owner in (group, group.parent):
        for key in ("language_instruction", "problem_info", "task_name"):
            value = owner.attrs.get(key)
            if value is not None:
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                if key == "problem_info":
                    try:
                        parsed = json.loads(str(value))
                        value = parsed.get("language_instruction", value)
                    except (json.JSONDecodeError, TypeError):
                        pass
                return str(value)
    return fallback.replace("_", " ")


def libero_gripper_closed(raw: np.ndarray) -> np.ndarray:
    """LIBERO/robosuite uses +1=open and -1=close."""
    values = np.asarray(raw, dtype=np.float32)
    return (values < 0.0).astype(np.float32)


def libero_orientation_to_rpy(orientation: np.ndarray) -> np.ndarray:
    """Convert LIBERO absolute orientation (rotvec or xyzw quaternion) to XYZ RPY."""
    values = np.asarray(orientation, dtype=np.float64)
    if values.shape[-1] == 3:
        rotation = Rotation.from_rotvec(values.reshape(-1, 3))
    elif values.shape[-1] == 4:
        rotation = Rotation.from_quat(values.reshape(-1, 4))
    else:
        raise ValueError(
            "LIBERO orientation must be rotation-vector [...,3] or xyzw "
            f"quaternion [...,4], got {values.shape}"
        )
    return rotation.as_euler("xyz", degrees=False).reshape(
        *values.shape[:-1], 3
    ).astype(np.float32)


def libero_gripper_state_to_closed(gripper_state: np.ndarray) -> np.ndarray:
    """Binarize Panda finger joint positions: near zero is closed."""
    values = np.asarray(gripper_state, dtype=np.float32)
    if values.shape[-1] == 0:
        raise ValueError("Empty LIBERO gripper state")
    opening = np.mean(np.abs(values), axis=-1)
    return (opening < 0.035).astype(np.float32)


def libero_observation_to_rdt(
    observation: dict,
    *,
    gripper_closed: float | None = None,
) -> dict:
    """Convert one live LIBERO observation to the exact raw RDT cache convention."""
    position = _dataset(observation, "ee_pos", "robot0_eef_pos").reshape(-1)[:3]
    orientation = _dataset(observation, "ee_ori", "robot0_eef_quat").reshape(-1)
    if gripper_closed is None:
        gripper = _dataset(observation, "gripper_states", "robot0_gripper_qpos")
        gripper_closed = float(libero_gripper_state_to_closed(gripper))
    state = np.concatenate(
        [position, libero_orientation_to_rpy(orientation), [float(gripper_closed)]],
    ).astype(np.float32)
    primary = _dataset(observation, "agentview_rgb", "agentview_image")
    wrist = None
    for key in ("eye_in_hand_rgb", "robot0_eye_in_hand_image"):
        if key in observation:
            wrist = np.asarray(observation[key])
            break
    return {"state": state, "primary": primary, "wrist": wrist}


def rdt_action_to_libero(
    normalized_action: np.ndarray,
    stats: ActionNormalizationStats,
) -> np.ndarray:
    """Denormalize one RDT action and restore LIBERO's +1 open/-1 close command."""
    action = denormalize_action_array(np.asarray(normalized_action, dtype=np.float32), stats)
    result = action[..., :7].copy()
    result[..., 6] = np.where(action[..., 6] >= 0.5, -1.0, 1.0)
    return result.astype(np.float32)


def convert_libero_demo(group, *, episode_id: str) -> LiberoEpisode:
    obs = group["obs"]
    actions = np.asarray(group["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f"{group.name}/actions must be [T, >=7], got {actions.shape}")
    actions = actions[:, :7].copy()
    actions[:, 6] = libero_gripper_closed(actions[:, 6])

    position = _dataset(obs, "ee_pos", "robot0_eef_pos")
    orientation = _dataset(obs, "ee_ori", "robot0_eef_quat")
    rpy = libero_orientation_to_rpy(orientation)

    if "gripper_states" in obs:
        state_closed = libero_gripper_state_to_closed(obs["gripper_states"])
    else:
        # Compatibility with datasets that omit finger observations.
        state_closed = np.empty((actions.shape[0],), dtype=np.float32)
        state_closed[0] = actions[0, 6]
        state_closed[1:] = actions[:-1, 6]
    states = np.concatenate([position[:, :3], rpy, state_closed[:, None]], axis=-1)

    primary = _dataset(obs, "agentview_rgb", "agentview_image")
    wrist = None
    for key in ("eye_in_hand_rgb", "robot0_eye_in_hand_image"):
        if key in obs:
            wrist = np.asarray(obs[key])
            break
    joint_states = _libero_joint_positions(obs)
    length = min(len(actions), len(states), len(primary))
    if wrist is not None:
        length = min(length, len(wrist))
    if joint_states is not None:
        length = min(length, len(joint_states))
    return LiberoEpisode(
        episode_id=episode_id,
        instruction=_instruction(group, episode_id),
        primary_images=primary[:length],
        wrist_images=None if wrist is None else wrist[:length],
        states=states[:length].astype(np.float32),
        actions=actions[:length].astype(np.float32),
        joint_states=None if joint_states is None else joint_states[:length].astype(np.float32),
    )


def iter_libero_episodes(data_dir: str | Path) -> Iterator[LiberoEpisode]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Reading LIBERO demonstrations requires h5py") from exc
    root = Path(data_dir).expanduser().resolve()
    files = [root] if root.is_file() else sorted(root.rglob("*.hdf5")) + sorted(root.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5/.h5 LIBERO demonstrations below {root}")
    for path in files:
        with h5py.File(path, "r") as handle:
            demos = handle["data"] if "data" in handle else handle
            for name in sorted(demos):
                group = demos[name]
                if hasattr(group, "keys") and "obs" in group and "actions" in group:
                    yield convert_libero_demo(group, episode_id=f"{path.stem}:{name}")


def libero_sample_from_episode(
    episode: LiberoEpisode,
    step_index: int,
    *,
    dataset_id: str = "libero_object",
    horizon: int,
    action_stats: ActionNormalizationStats | None,
    action_target_mode: str = "delta",
) -> dict:
    if action_target_mode == "delta":
        target_sequence = episode.actions
    elif action_target_mode == "absolute_state":
        # For target-state RDT fine-tuning, the supervised "action" chunk is a
        # horizon of absolute observed EEF states, not LIBERO controller deltas.
        # Rollout converts each predicted absolute target back to a LIBERO
        # delta-controller command using current_state -> target_state error.
        target_sequence = episode.states
    else:
        raise ValueError(f"Unsupported action_target_mode: {action_target_mode}")

    actions, mask = pad_action_horizon(target_sequence, step_index, horizon=horizon)
    if action_stats is not None and action_target_mode == "delta":
        actions = normalize_action_horizon(actions, mask, action_stats)
    wrist = None if episode.wrist_images is None else Image.fromarray(episode.wrist_images[step_index]).copy()
    sample = {
        "dataset_id": dataset_id,
        "episode_id": episode.episode_id,
        "step_idx": str(step_index),
        "instruction": episode.instruction,
        "images": {
            "primary": Image.fromarray(episode.primary_images[step_index]).copy(),
            "wrist": wrist,
            "secondary": None,
        },
        "image_mask": {"primary": 1, "wrist": int(wrist is not None), "secondary": 0},
        "state": episode.states[step_index].copy(),
        "state_mask": np.ones((7,), dtype=np.float32),
        "actions": actions,
        "actions_mask": mask,
        "ctrl_freq": 20.0,
    }
    if episode.joint_states is not None:
        joint_sequence = episode.joint_states
        joint_horizon, joint_horizon_mask = pad_action_horizon(
            joint_sequence,
            step_index,
            horizon=horizon,
            action_dim=int(joint_sequence.shape[1]),
        )
        sample["joint_state"] = joint_sequence[step_index].copy()
        sample["joint_states"] = joint_horizon
        sample["joint_states_mask"] = joint_horizon_mask
    return sample
