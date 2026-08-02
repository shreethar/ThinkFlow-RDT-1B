#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/thinkflow-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/thinkflow-matplotlib")

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inspect_libero_outputs import install_robosuite_mujoco_compatibility  # noqa: E402
from replay_libero_demo_actions import (  # noqa: E402
    LIBERO_BENCHMARK_CHOICES,
    LIBERO_DEFAULT_BENCHMARK,
    action_stats,
    find_demo_file,
    json_default,
    select_demo,
)
from thinkflow_rdt.adapters.libero import (  # noqa: E402
    libero_gripper_closed,
    libero_gripper_state_to_closed,
    libero_observation_to_rdt,
    libero_orientation_to_rpy,
)


def frame_for_video(frame: np.ndarray, text: str) -> np.ndarray:
    frame = np.asarray(frame)[::-1].copy()
    width = int(frame.shape[1])
    cv2.rectangle(frame, (0, 0), (width, 34), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def wrap_rpy_delta(delta: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(delta), np.cos(delta)).astype(np.float32)


def demo_states_7d_closed(demo: h5py.Group) -> np.ndarray:
    obs = demo["obs"]
    position = None
    for key in ("ee_pos", "robot0_eef_pos"):
        if key in obs:
            position = np.asarray(obs[key], dtype=np.float32)[..., :3]
            break
    orientation = None
    for key in ("ee_ori", "robot0_eef_quat"):
        if key in obs:
            orientation = np.asarray(obs[key])
            break
    if position is None or orientation is None:
        raise KeyError("Could not find EEF position/orientation in demo obs")
    rpy = libero_orientation_to_rpy(orientation)
    gripper_closed = None
    for key in ("gripper_states", "robot0_gripper_qpos"):
        if key in obs:
            gripper_closed = libero_gripper_state_to_closed(np.asarray(obs[key]))
            break
    if gripper_closed is None:
        actions = np.asarray(demo["actions"], dtype=np.float32)[:, :7]
        gripper_closed = libero_gripper_closed(actions[:, 6])
    return np.concatenate([position, rpy, gripper_closed[..., None]], axis=-1).astype(np.float32)


def target_state_to_libero_action(
    target_closed: np.ndarray,
    current_closed: np.ndarray,
    *,
    pos_scale: float,
    rot_scale: float,
    max_delta_pos: float | None,
    max_delta_rot: float | None,
    gripper_source: str,
    demo_action: np.ndarray,
) -> np.ndarray:
    target_open = np.asarray(target_closed, dtype=np.float32).copy()
    target_open[6] = 1.0 - target_open[6]
    current_open = np.asarray(current_closed, dtype=np.float32).copy()
    current_open[6] = 1.0 - current_open[6]
    action = np.zeros((7,), dtype=np.float32)
    action[:3] = (target_open[:3] - current_open[:3]) * float(pos_scale)
    action[3:6] = wrap_rpy_delta(target_open[3:6] - current_open[3:6]) * float(rot_scale)
    if max_delta_pos is not None:
        action[:3] = np.clip(action[:3], -float(max_delta_pos), float(max_delta_pos))
    if max_delta_rot is not None:
        action[3:6] = np.clip(action[3:6], -float(max_delta_rot), float(max_delta_rot))
    if gripper_source == "target_state":
        action[6] = 1.0 if float(target_open[6]) >= 0.5 else -1.0
    elif gripper_source == "demo_action":
        action[6] = float(demo_action[6])
    elif gripper_source == "zero":
        action[6] = 0.0
    else:
        raise ValueError(f"Unsupported gripper source: {gripper_source}")
    return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay LIBERO using recorded future demo states as oracle absolute "
            "target states, converted to controller actions. This isolates the "
            "absolute-target-state -> LIBERO-action conversion."
        )
    )
    parser.add_argument("--libero-root", type=Path, default=Path("/workspace/LIBERO"))
    parser.add_argument("--benchmark", choices=LIBERO_BENCHMARK_CHOICES, default=LIBERO_DEFAULT_BENCHMARK)
    parser.add_argument("--task-id", type=int, default=0, choices=range(10))
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/datasets/libero_object"))
    parser.add_argument("--demo-hdf5", type=Path)
    parser.add_argument("--demo-name")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--target-offset", type=int, default=1)
    parser.add_argument("--pos-scale", type=float, default=1.0)
    parser.add_argument("--rot-scale", type=float, default=1.0)
    parser.add_argument("--max-delta-pos", type=float, default=0.05, help="Set negative to disable clipping.")
    parser.add_argument("--max-delta-rot", type=float, default=0.25, help="Set negative to disable clipping.")
    parser.add_argument(
        "--gripper-source",
        choices=["target_state", "demo_action", "zero"],
        default="demo_action",
        help="Use demo_action first so xyz/rot conversion can be isolated from gripper thresholding.",
    )
    parser.add_argument("--video-resolution", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("outputs/libero_oracle_target_state_replay.mp4"))
    parser.add_argument("--per-step-jsonl", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.libero_root) not in sys.path:
        sys.path.insert(0, str(args.libero_root))
    if args.target_offset < 1:
        raise ValueError("--target-offset must be >= 1")
    max_delta_pos = None if args.max_delta_pos < 0 else args.max_delta_pos
    max_delta_rot = None if args.max_delta_rot < 0 else args.max_delta_rot

    install_robosuite_mujoco_compatibility()
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark(args.benchmark)(0)
    task = benchmark.get_task(args.task_id)
    demo_hdf5 = args.demo_hdf5 or find_demo_file(args.dataset_dir, task_name=task.name)
    start_time = time.perf_counter()

    with h5py.File(demo_hdf5, "r") as handle:
        demo_name, demo = select_demo(handle, args.demo_name)
        if "states" not in demo:
            raise KeyError(f"{demo.name} has no states dataset; cannot reset simulator to demo state")
        if "actions" not in demo:
            raise KeyError(f"{demo.name} has no actions dataset")
        initial_state = np.asarray(demo["states"][0], dtype=np.float64)
        demo_actions = np.asarray(demo["actions"], dtype=np.float32)[:, :7]
        states_closed = demo_states_7d_closed(demo)

    usable_steps = min(len(demo_actions), max(0, len(states_closed) - args.target_offset))
    replay_steps = usable_steps if args.max_steps is None else min(args.max_steps, usable_steps)
    if replay_steps <= 0:
        raise RuntimeError("No usable demo steps for oracle target-state replay")

    env = OffScreenRenderEnv(
        bddl_file_name=benchmark.get_task_bddl_file_path(args.task_id),
        camera_heights=128,
        camera_widths=128,
        horizon=replay_steps + 10,
    )
    observation = env.reset()
    observation = env.set_init_state(initial_state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.output,
        format="FFMPEG",
        fps=args.fps,
        codec="libx264",
        quality=8,
    )
    per_step_handle = None
    if args.per_step_jsonl is not None:
        args.per_step_jsonl.parent.mkdir(parents=True, exist_ok=True)
        per_step_handle = args.per_step_jsonl.open("w", encoding="utf-8")

    success = False
    done = False
    executed_steps = 0
    eef_position_errors: list[float] = []
    generated_actions: list[np.ndarray] = []
    final_observation = observation
    try:
        for step in range(replay_steps):
            current_closed = libero_observation_to_rdt(observation)["state"]
            target_index = min(step + args.target_offset, len(states_closed) - 1)
            target_closed = states_closed[target_index]
            action = target_state_to_libero_action(
                target_closed,
                current_closed,
                pos_scale=args.pos_scale,
                rot_scale=args.rot_scale,
                max_delta_pos=max_delta_pos,
                max_delta_rot=max_delta_rot,
                gripper_source=args.gripper_source,
                demo_action=demo_actions[min(step, len(demo_actions) - 1)],
            )
            generated_actions.append(action.copy())
            observation, reward, done, _ = env.step(action)
            final_observation = observation
            executed_steps = step + 1
            success = bool(done) or bool(env.check_success())

            live_state = libero_observation_to_rdt(observation)["state"]
            compare_index = min(step + 1, len(states_closed) - 1)
            recorded_pos = states_closed[compare_index, :3]
            eef_error = float(np.linalg.norm(live_state[:3] - recorded_pos))
            eef_position_errors.append(eef_error)

            if per_step_handle is not None:
                per_step_handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "target_index": target_index,
                            "oracle_target_state_7d_closed": target_closed,
                            "current_state_before_step_7d_closed": current_closed,
                            "generated_action": action,
                            "demo_action_same_step": demo_actions[min(step, len(demo_actions) - 1)],
                            "reward": float(reward),
                            "done": bool(done),
                            "success": bool(success),
                            "live_state_after_step_7d_closed": live_state,
                            "recorded_eef_pos_next": recorded_pos,
                            "live_vs_recorded_eef_l2": eef_error,
                        },
                        default=json_default,
                    )
                    + "\n"
                )
                per_step_handle.flush()

            label = f"oracle step={step + 1}/{replay_steps} err={eef_error:.3f} success={int(success)}"
            video_frame = env.env.sim.render(
                width=args.video_resolution,
                height=args.video_resolution,
                camera_name="agentview",
            )
            writer.append_data(frame_for_video(video_frame, label))
            if success:
                break
    finally:
        if per_step_handle is not None:
            per_step_handle.close()
        writer.close()
        env.close()

    generated = np.asarray(generated_actions, dtype=np.float32)
    error_array = np.asarray(eef_position_errors, dtype=np.float32)
    final_state = libero_observation_to_rdt(final_observation)["state"]
    summary: dict[str, Any] = {
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "task_name": task.name,
        "instruction": task.language,
        "demo_hdf5": demo_hdf5.resolve(),
        "demo_name": demo_name,
        "target_offset": int(args.target_offset),
        "pos_scale": float(args.pos_scale),
        "rot_scale": float(args.rot_scale),
        "max_delta_pos": max_delta_pos,
        "max_delta_rot": max_delta_rot,
        "gripper_source": args.gripper_source,
        "actions_replayed": int(executed_steps),
        "requested_replay_steps": int(replay_steps),
        "success": bool(success),
        "done": bool(done),
        "final_state_7d_closed": final_state,
        "generated_action_stats": action_stats(generated) if len(generated) else {},
        "demo_action_stats_same_span": action_stats(demo_actions[:replay_steps]),
        "video": args.output.resolve(),
        "elapsed_sec": time.perf_counter() - start_time,
    }
    if error_array.size:
        summary["eef_position_replay_error"] = {
            "count": int(error_array.size),
            "mean_l2": float(error_array.mean()),
            "max_l2": float(error_array.max()),
            "final_l2": float(error_array[-1]),
        }
    if args.per_step_jsonl is not None:
        summary["per_step_jsonl"] = args.per_step_jsonl.resolve()
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
