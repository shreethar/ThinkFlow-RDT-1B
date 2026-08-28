#!/usr/bin/env python3
"""Isolated SimplerEnv worker used by the B0-OXE evaluator.

This file deliberately has no ThinkFlow, PyTorch, or Transformers imports.  It
runs in SimplerEnv's Python 3.10/3.11 environment and communicates with the
policy process over a local authenticated Unix socket.
"""

from __future__ import annotations

import argparse
import os
import traceback
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--authkey-hex", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--renderer-offscreen", action="store_true")
    return parser.parse_args()


def serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def observation_packet(env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    from simpler_env.utils.env.observation_utils import (
        get_image_from_maniskill2_obs_dict,
    )
    from transforms3d.euler import quat2euler

    unwrapped = env.unwrapped
    robot = unwrapped.agent.robot
    tcp_pose_at_base = robot.pose.inv() * unwrapped.tcp.pose
    # SAPIEN uses scalar-first wxyz quaternions; transforms3d expects wxyz too.
    roll, pitch, yaw = quat2euler(tcp_pose_at_base.q, axes="sxyz")
    gripper_closed = float(unwrapped.agent.get_gripper_closedness())
    state = np.asarray(
        [
            *np.asarray(tcp_pose_at_base.p, dtype=np.float32).tolist(),
            roll,
            pitch,
            yaw,
            gripper_closed,
        ],
        dtype=np.float32,
    )

    object_states: dict[str, Any] = {}
    for name in (
        "obj",
        "episode_source_obj",
        "episode_target_obj",
        "source_obj",
        "target_obj",
    ):
        entity = getattr(unwrapped, name, None)
        pose = getattr(entity, "pose", None)
        if pose is not None:
            pose_at_base = robot.pose.inv() * pose
            object_states[name] = {
                "world_position": np.asarray(pose.p, dtype=np.float32),
                "base_position": np.asarray(pose_at_base.p, dtype=np.float32),
                "world_quaternion_wxyz": np.asarray(pose.q, dtype=np.float32),
            }

    image = np.asarray(
        get_image_from_maniskill2_obs_dict(env, observation), dtype=np.uint8
    ).copy()
    return {
        "image": image,
        "state_7d": state,
        "tcp_position_base": state[:3].copy(),
        "tcp_rpy_base": state[3:6].copy(),
        "gripper_closedness": gripper_closed,
        "robot_qpos": np.asarray(robot.get_qpos(), dtype=np.float32).copy(),
        "object_states": object_states,
        "instruction": str(unwrapped.get_language_instruction()),
    }


def main() -> None:
    args = parse_args()
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    if args.socket.exists():
        args.socket.unlink()
    listener = Listener(
        str(args.socket), family="AF_UNIX", authkey=bytes.fromhex(args.authkey_hex)
    )
    connection = listener.accept()
    env = None
    try:
        import simpler_env

        kwargs: dict[str, Any] = {}
        if args.renderer_offscreen:
            kwargs["renderer_kwargs"] = {"offscreen_only": True}
        env = simpler_env.make(args.task, **kwargs)
        observation, reset_info = env.reset(seed=args.seed)
        packet = observation_packet(env, observation)
        packet.update(
            {
                "kind": "ready",
                "reset_info": serializable(reset_info),
                "action_low": np.asarray(env.action_space.low, dtype=np.float32),
                "action_high": np.asarray(env.action_space.high, dtype=np.float32),
                "robot_uid": str(env.unwrapped.robot_uid),
                "control_frequency": float(env.unwrapped.control_freq),
            }
        )
        connection.send(packet)

        while True:
            request = connection.recv()
            command = request.get("command")
            if command == "close":
                connection.send({"kind": "closed"})
                break
            if command != "step":
                raise ValueError(f"Unknown worker command: {command!r}")
            requested_action = np.asarray(request["action"], dtype=np.float32)
            if requested_action.shape != env.action_space.shape:
                raise ValueError(
                    f"Action shape {requested_action.shape} != {env.action_space.shape}"
                )
            executed_action = np.clip(
                requested_action, env.action_space.low, env.action_space.high
            ).astype(np.float32)
            observation, reward, terminated, truncated, info = env.step(
                executed_action
            )
            packet = observation_packet(env, observation)
            packet.update(
                {
                    "kind": "step",
                    "requested_action": requested_action,
                    "executed_action": executed_action,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": serializable(info),
                }
            )
            connection.send(packet)
    except BaseException as error:
        try:
            connection.send(
                {
                    "kind": "error",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
        except BaseException:
            pass
        raise
    finally:
        if env is not None:
            env.close()
        connection.close()
        listener.close()
        try:
            args.socket.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    # GLFW should not be required for off-screen evaluation.
    os.environ.setdefault("DISPLAY", "")
    main()
