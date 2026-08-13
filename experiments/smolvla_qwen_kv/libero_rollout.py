"""LIBERO rollout helpers with no dependency on the ThinkFlow RDT model stack."""

from __future__ import annotations

from typing import Any

import cv2
import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


LIBERO_STATE_DIM = 11
LIBERO_ACTION_DIM = 10


def install_robosuite_mujoco_compatibility() -> None:
    """Adapt robosuite 1.4's old-style MuJoCo mass-matrix call."""

    try:
        from robosuite.controllers.base_controller import Controller
    except ModuleNotFoundError:
        return

    def update(controller, force: bool = False) -> None:
        if not (controller.new_update or force):
            return
        sim = controller.sim
        sim.forward()
        site_id = sim.model.site_name2id(controller.eef_name)
        controller.ee_pos = np.asarray(sim.data.site_xpos[site_id]).copy()
        controller.ee_ori_mat = np.asarray(sim.data.site_xmat[site_id]).reshape(3, 3).copy()
        controller.ee_pos_vel = np.asarray(sim.data.get_site_xvelp(controller.eef_name)).copy()
        controller.ee_ori_vel = np.asarray(sim.data.get_site_xvelr(controller.eef_name)).copy()
        controller.joint_pos = np.asarray(sim.data.qpos[controller.qpos_index]).copy()
        controller.joint_vel = np.asarray(sim.data.qvel[controller.qvel_index]).copy()
        controller.J_pos = np.asarray(
            sim.data.get_site_jacp(controller.eef_name).reshape((3, -1))[
                :, controller.qvel_index
            ]
        ).copy()
        controller.J_ori = np.asarray(
            sim.data.get_site_jacr(controller.eef_name).reshape((3, -1))[
                :, controller.qvel_index
            ]
        ).copy()
        controller.J_full = np.vstack([controller.J_pos, controller.J_ori])
        mass_matrix = np.empty((sim.model.nv, sim.model.nv), dtype=np.float64, order="C")
        try:
            mujoco.mj_fullM(sim.model._model, sim.data._data, mass_matrix)
        except TypeError:
            mujoco.mj_fullM(sim.model._model, mass_matrix, sim.data.qM)
        controller.mass_matrix = mass_matrix[controller.qvel_index, :][
            :, controller.qvel_index
        ]
        controller.new_update = False

    Controller.update = update


def _observation_value(observation: dict[str, Any], *keys: str) -> np.ndarray:
    for key in keys:
        if key in observation:
            return np.asarray(observation[key])
    raise KeyError(f"None of {keys!r} found in LIBERO observation")


def _orientation_to_ortho6d(orientation: np.ndarray) -> np.ndarray:
    values = np.asarray(orientation, dtype=np.float64).reshape(-1)
    if values.size == 3:
        matrix = Rotation.from_rotvec(values).as_matrix()
    elif values.size == 4:
        matrix = Rotation.from_quat(values).as_matrix()
    else:
        raise ValueError(f"Expected rotvec or quaternion orientation, got {values.shape}")
    return np.concatenate([matrix[:, 0], matrix[:, 1]]).astype(np.float32)


def _image_to_rgb(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim != 3 or values.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected channel-last LIBERO image, got {values.shape}")
    return np.flip(values, axis=0).copy()


def _convert_observation(observation: dict[str, Any]) -> dict[str, Any]:
    position = _observation_value(observation, "ee_pos", "robot0_eef_pos").reshape(-1)[:3]
    orientation = _observation_value(
        observation, "ee_ori", "robot0_eef_quat"
    ).reshape(-1)
    fingers = _observation_value(
        observation, "gripper_states", "robot0_gripper_qpos"
    ).reshape(-1)
    if fingers.size < 2:
        raise ValueError(f"Expected two LIBERO finger positions, got {fingers.shape}")
    state = np.concatenate(
        [position, _orientation_to_ortho6d(orientation), fingers[:2]]
    ).astype(np.float32)
    primary = _image_to_rgb(
        _observation_value(observation, "agentview_rgb", "agentview_image")
    )
    wrist = None
    for key in ("eye_in_hand_rgb", "robot0_eye_in_hand_image"):
        if key in observation:
            wrist = _image_to_rgb(observation[key])
            break
    return {"state": state, "primary": primary, "wrist": wrist}


def rollout_sample(
    observation: dict[str, Any],
    previous_observation: dict[str, Any] | None,
    *,
    dataset_id: str,
    instruction: str,
    horizon: int,
) -> dict[str, Any]:
    converted = _convert_observation(observation)
    current = {
        "primary": Image.fromarray(converted["primary"]).convert("RGB"),
        "wrist": (
            None
            if converted["wrist"] is None
            else Image.fromarray(converted["wrist"]).convert("RGB")
        ),
        "secondary": None,
    }
    if previous_observation is None:
        previous = current
        previous_mask = {"primary": 0, "wrist": 0, "secondary": 0}
    else:
        old = _convert_observation(previous_observation)
        previous = {
            "primary": Image.fromarray(old["primary"]).convert("RGB"),
            "wrist": (
                None
                if old["wrist"] is None
                else Image.fromarray(old["wrist"]).convert("RGB")
            ),
            "secondary": None,
        }
        previous_mask = {
            "primary": 1,
            "wrist": int(previous["wrist"] is not None),
            "secondary": 0,
        }
    current_mask = {
        "primary": 1,
        "wrist": int(current["wrist"] is not None),
        "secondary": 0,
    }
    return {
        "dataset_id": dataset_id,
        "episode_id": "rollout",
        "step_idx": "0",
        "instruction": instruction,
        "images": current,
        "image_mask": current_mask,
        "image_history": [previous, current],
        "image_history_mask": [previous_mask, current_mask],
        "state": converted["state"].copy(),
        "state_mask": np.ones(LIBERO_STATE_DIM, dtype=np.float32),
        "actions": np.zeros((horizon, LIBERO_ACTION_DIM), dtype=np.float32),
        "actions_mask": np.ones(horizon, dtype=np.float32),
        "action_dim_mask": np.ones(LIBERO_ACTION_DIM, dtype=np.float32),
        "ctrl_freq": 20.0,
    }


def frame_for_video(frame: np.ndarray, text: str) -> np.ndarray:
    """Convert LIBERO's bottom-up frame and overlay a compact status label."""

    result = np.asarray(frame)[::-1].copy()
    width = int(result.shape[1])
    cv2.rectangle(result, (0, 0), (width, 34), (0, 0, 0), thickness=-1)
    cv2.putText(
        result,
        text,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result
