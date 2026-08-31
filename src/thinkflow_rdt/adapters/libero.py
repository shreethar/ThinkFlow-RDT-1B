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


LIBERO_STATE_DIM = 11
LIBERO_ACTION_DIM = 10
LIBERO_ROT_COMMAND_SCALE = 0.5
LIBERO_STATE_NATIVE_INDICES = (30, 31, 32, 33, 34, 35, 36, 37, 38, 10, 11)
LIBERO_ACTION_NATIVE_INDICES = (30, 31, 32, 33, 34, 35, 36, 37, 38, 10)


@dataclass(frozen=True)
class LiberoEpisode:
    episode_id: str
    instruction: str
    primary_images: np.ndarray
    wrist_images: np.ndarray | None
    states: np.ndarray
    actions: np.ndarray
    native_states: np.ndarray
    native_actions: np.ndarray
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
    """Legacy diagnostic helper that labels negative commands as closed.

    The current LIBERO cache and rollout path never calls this helper: raw
    action signs are preserved without assigning open/close semantics.
    """
    values = np.asarray(raw, dtype=np.float32)
    return (values < 0.0).astype(np.float32)


def libero_image_to_rgb(image: np.ndarray) -> np.ndarray:
    """Flip a raw Robosuite RGB observation into conventional top-left origin.

    Robosuite's MuJoCo camera arrays have an OpenGL-style bottom-left origin.
    LIBERO stores those arrays directly in its demonstration HDF5 files, while
    its own visualization utilities flip the image vertically. ``axis=-3`` is
    the height axis for both ``[H, W, C]`` and ``[T, H, W, C]`` arrays.
    """
    values = np.asarray(image)
    if values.ndim < 3 or values.shape[-1] not in (1, 3, 4):
        raise ValueError(
            "Expected a channel-last image [..., H, W, C], got "
            f"{values.shape}"
        )
    return np.flip(values, axis=-3).copy()


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


def rotation_matrix_to_ortho6d(matrix: np.ndarray) -> np.ndarray:
    """Encode a rotation matrix as its first two columns."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices [...,3,3], got {values.shape}")
    return np.concatenate(
        [values[..., :, 0], values[..., :, 1]],
        axis=-1,
    ).astype(np.float32)


def ortho6d_to_rotation_matrix(ortho6d: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Project a 6D rotation prediction onto SO(3) with Gram-Schmidt."""
    values = np.asarray(ortho6d, dtype=np.float64)
    if values.shape[-1] != 6:
        raise ValueError(f"Expected ortho6D values [...,6], got {values.shape}")

    first = values[..., :3]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    default_first = np.zeros_like(first)
    default_first[..., 0] = 1.0
    first = np.where(first_norm > eps, first, default_first)
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), eps)

    second = values[..., 3:6]
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)

    # A diffusion prediction can briefly contain parallel or zero columns.
    # Choose the canonical axis least aligned with the first column as a stable
    # fallback, then orthogonalize it in the same way.
    fallback_index = np.argmin(np.abs(first), axis=-1)
    fallback = np.eye(3, dtype=np.float64)[fallback_index]
    fallback = fallback - np.sum(first * fallback, axis=-1, keepdims=True) * first
    fallback /= np.maximum(np.linalg.norm(fallback, axis=-1, keepdims=True), eps)
    second = np.where(second_norm > eps, second, fallback)
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), eps)

    third = np.cross(first, second)
    return np.stack([first, second, third], axis=-1).astype(np.float32)


def libero_orientation_to_ortho6d(orientation: np.ndarray) -> np.ndarray:
    """Convert an absolute LIBERO rotvec/quaternion directly to ortho6D."""
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
    matrices = rotation.as_matrix().reshape(*values.shape[:-1], 3, 3)
    return rotation_matrix_to_ortho6d(matrices)


def ortho6d_to_libero_orientation(ortho6d: np.ndarray) -> np.ndarray:
    """Convert an absolute ortho6D pose back to LIBERO's rotvec form."""
    matrices = ortho6d_to_rotation_matrix(ortho6d)
    flat = matrices.reshape(-1, 3, 3)
    return Rotation.from_matrix(flat).as_rotvec().reshape(
        *matrices.shape[:-2], 3
    ).astype(np.float32)


def libero_rot_command_to_ortho6d(command: np.ndarray) -> np.ndarray:
    """Encode a raw OSC_POSE rotation command as a relative ortho6D rotation.

    Robosuite scales each normalized rotation command by 0.5 radians before
    constructing the controller's relative rotation.
    """
    values = np.asarray(command, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError(f"Expected LIBERO rotation commands [...,3], got {values.shape}")
    delta_rotvec = values * LIBERO_ROT_COMMAND_SCALE
    matrices = Rotation.from_rotvec(delta_rotvec.reshape(-1, 3)).as_matrix()
    matrices = matrices.reshape(*values.shape[:-1], 3, 3)
    return rotation_matrix_to_ortho6d(matrices)


def ortho6d_to_libero_rot_command(ortho6d: np.ndarray) -> np.ndarray:
    """Decode relative ortho6D rotations to raw normalized OSC_POSE commands."""
    matrices = ortho6d_to_rotation_matrix(ortho6d)
    delta_rotvec = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec()
    command = delta_rotvec.reshape(*matrices.shape[:-2], 3) / LIBERO_ROT_COMMAND_SCALE
    return np.clip(command, -1.0, 1.0).astype(np.float32)


def libero_action_to_rdt(raw_action: np.ndarray) -> np.ndarray:
    """Convert raw 7D LIBERO commands to 10D command-space RDT targets.

    Translation and gripper values are copied unchanged. Only the three-axis
    normalized rotation command is represented as a relative ortho6D rotation.
    """
    values = np.asarray(raw_action, dtype=np.float32)
    if values.shape[-1] < 7:
        raise ValueError(f"Expected raw LIBERO actions [...,>=7], got {values.shape}")
    return np.concatenate(
        [
            values[..., :3],
            libero_rot_command_to_ortho6d(values[..., 3:6]),
            values[..., 6:7],
        ],
        axis=-1,
    ).astype(np.float32)


def libero_gripper_state_to_closed(gripper_state: np.ndarray) -> np.ndarray:
    """Legacy diagnostic binarization; the current state keeps both raw qpos."""
    values = np.asarray(gripper_state, dtype=np.float32)
    if values.shape[-1] == 0:
        raise ValueError("Empty LIBERO gripper state")
    opening = np.mean(np.abs(values), axis=-1)
    return (opening < 0.035).astype(np.float32)


def libero_observation_to_rdt(
    observation: dict,
) -> dict:
    """Convert a live observation to [xyz, absolute ortho6D, raw finger qpos]."""
    position = _dataset(observation, "ee_pos", "robot0_eef_pos").reshape(-1)[:3]
    orientation = _dataset(observation, "ee_ori", "robot0_eef_quat").reshape(-1)
    gripper = _dataset(observation, "gripper_states", "robot0_gripper_qpos").reshape(-1)
    joints = _optional_dataset(
        observation,
        "joint_states",
        "robot0_joint_pos",
        "joint_pos",
    )
    if joints is not None:
        joints = np.asarray(joints).reshape(-1)
        if joints.size < 7:
            raise ValueError(
                f"Expected seven LIBERO joint positions, got {joints.shape}"
            )
    if gripper.size < 2:
        raise ValueError(f"Expected two raw LIBERO finger positions, got {gripper.shape}")
    state = np.concatenate(
        [position, libero_orientation_to_ortho6d(orientation), gripper[:2]],
    ).astype(np.float32)
    primary = libero_image_to_rgb(
        _dataset(observation, "agentview_rgb", "agentview_image")
    )
    wrist = None
    for key in ("eye_in_hand_rgb", "robot0_eye_in_hand_image"):
        if key in observation:
            wrist = libero_image_to_rgb(observation[key])
            break
    result = {
        "state": state,
        "primary": primary,
        "wrist": wrist,
    }
    if joints is not None:
        result["joint_state"] = joints[:7].astype(np.float32)
    return result


def rdt_action_to_libero(
    encoded_action: np.ndarray,
    stats: ActionNormalizationStats | None = None,
) -> np.ndarray:
    """Decode a 10D RDT target back to a raw 7D LIBERO command."""
    values = np.asarray(encoded_action, dtype=np.float32)
    action = denormalize_action_array(values, stats) if stats is not None else values
    if action.shape[-1] != LIBERO_ACTION_DIM:
        raise ValueError(
            f"Expected encoded LIBERO actions [...,{LIBERO_ACTION_DIM}], got {action.shape}"
        )
    result = np.empty((*action.shape[:-1], 7), dtype=np.float32)
    result[..., :3] = np.clip(action[..., :3], -1.0, 1.0)
    result[..., 3:6] = ortho6d_to_libero_rot_command(action[..., 3:9])
    # Preserve the raw demonstrated command convention and merely respect the
    # environment's action bounds. There is no open/closed remapping here.
    result[..., 6] = np.clip(action[..., 9], -1.0, 1.0)
    return result.astype(np.float32)


def convert_libero_demo(group, *, episode_id: str) -> LiberoEpisode:
    obs = group["obs"]
    actions = np.asarray(group["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f"{group.name}/actions must be [T, >=7], got {actions.shape}")
    raw_actions = actions[:, :7].copy()
    actions = libero_action_to_rdt(raw_actions)

    position = _dataset(obs, "ee_pos", "robot0_eef_pos")
    orientation = _dataset(obs, "ee_ori", "robot0_eef_quat")
    if orientation.shape[-1] == 3:
        native_orientation = np.asarray(orientation, dtype=np.float32)
    elif orientation.shape[-1] == 4:
        native_orientation = Rotation.from_quat(
            np.asarray(orientation, dtype=np.float64).reshape(-1, 4)
        ).as_rotvec().reshape(*orientation.shape[:-1], 3).astype(np.float32)
    else:
        raise ValueError(
            "Expected LIBERO end-effector orientation [T,3] or [T,4], got "
            f"{orientation.shape}"
        )
    orientation_6d = libero_orientation_to_ortho6d(orientation)
    gripper_states = _dataset(obs, "gripper_states", "robot0_gripper_qpos")
    if gripper_states.ndim != 2 or gripper_states.shape[1] < 2:
        raise ValueError(
            f"Expected raw two-finger states [T,>=2], got {gripper_states.shape}"
        )
    states = np.concatenate(
        [position[:, :3], orientation_6d, gripper_states[:, :2]],
        axis=-1,
    )
    native_states = np.concatenate(
        [position[:, :3], native_orientation, gripper_states[:, :2]],
        axis=-1,
    )

    primary = libero_image_to_rgb(
        _dataset(obs, "agentview_rgb", "agentview_image")
    )
    wrist = None
    for key in ("eye_in_hand_rgb", "robot0_eye_in_hand_image"):
        if key in obs:
            wrist = libero_image_to_rgb(obs[key])
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
        native_states=native_states[:length].astype(np.float32),
        native_actions=raw_actions[:length].astype(np.float32),
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
        raise ValueError(
            "LIBERO absolute_state targets are incompatible with the current "
            "11-D observation / 10-D raw-command schema. Use "
            "action_target_mode='delta'."
        )
    else:
        raise ValueError(f"Unsupported action_target_mode: {action_target_mode}")

    actions, mask = pad_action_horizon(
        target_sequence,
        step_index,
        horizon=horizon,
        action_dim=int(target_sequence.shape[1]),
    )
    native_actions, native_mask = pad_action_horizon(
        episode.native_actions,
        step_index,
        horizon=horizon,
        action_dim=7,
    )
    if not np.array_equal(native_mask, mask):
        raise RuntimeError("Native and RDT LIBERO action horizon masks diverged")
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
        "state_mask": np.ones((episode.states.shape[1],), dtype=np.float32),
        "actions": actions,
        "actions_mask": mask,
        "action_dim_mask": np.ones((target_sequence.shape[1],), dtype=np.float32),
        # Keep the original LIBERO tensors available for feature-only caches.
        # This avoids a lossy 8D->11D->8D or 7D->10D->7D round trip when the
        # downstream policy consumes LIBERO's native observation/command schema.
        "libero_native_state": episode.native_states[step_index].copy(),
        "libero_native_actions": native_actions,
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
