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
from thinkflow_rdt.adapters.libero import (  # noqa: E402
    libero_action_to_rdt,
    libero_observation_to_rdt,
    rdt_action_to_libero,
)

LIBERO_BENCHMARK_CHOICES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)
LIBERO_DEFAULT_BENCHMARK = "libero_spatial"


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def frame_for_video(frame: np.ndarray, text: str) -> np.ndarray:
    frame = np.asarray(frame)[::-1].copy()
    width = int(frame.shape[1])
    cv2.rectangle(frame, (0, 0), (width, 34), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def action_stats(actions: np.ndarray) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float32)
    return {
        "shape": list(actions.shape),
        "min": actions.min(axis=0).astype(float).tolist(),
        "max": actions.max(axis=0).astype(float).tolist(),
        "mean": actions.mean(axis=0).astype(float).tolist(),
        "std": actions.std(axis=0).astype(float).tolist(),
        "gripper_command_counts": {
            # Keep the dataset/controller convention raw. Its physical meaning
            # is environment/controller-specific and is not remapped here.
            "positive": int(np.sum(actions[:, 6] > 0.0)),
            "negative": int(np.sum(actions[:, 6] < 0.0)),
            "zero": int(np.sum(actions[:, 6] == 0.0)),
        },
    }


def codec_roundtrip(raw_actions: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Exercise the exact LIBERO 7D -> RDT 10D -> LIBERO 7D codec."""
    raw = np.asarray(raw_actions, dtype=np.float32)[..., :7]
    encoded = libero_action_to_rdt(raw)
    decoded = rdt_action_to_libero(encoded)
    error = decoded.astype(np.float64) - raw.astype(np.float64)
    absolute_error = np.abs(error)
    translation_l2 = np.linalg.norm(error[..., :3], axis=-1)
    rotation_l2 = np.linalg.norm(error[..., 3:6], axis=-1)
    report = {
        "raw_shape": list(raw.shape),
        "encoded_shape": list(encoded.shape),
        "decoded_shape": list(decoded.shape),
        "allclose_atol_1e-6": bool(np.allclose(decoded, raw, rtol=0.0, atol=1e-6)),
        "max_abs_error": float(absolute_error.max(initial=0.0)),
        "mean_abs_error": float(absolute_error.mean()) if absolute_error.size else 0.0,
        "per_dimension_max_abs_error": absolute_error.max(axis=0, initial=0.0).tolist(),
        "translation_max_l2": float(translation_l2.max(initial=0.0)),
        "rotation_command_max_l2": float(rotation_l2.max(initial=0.0)),
        "gripper_max_abs_error": float(absolute_error[..., 6].max(initial=0.0)),
    }
    return decoded, report


def find_demo_file(dataset_dir: Path, *, task_name: str | None) -> Path:
    files = sorted(dataset_dir.rglob("*.hdf5")) + sorted(dataset_dir.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5/.h5 files found below {dataset_dir}")
    if task_name:
        normalized_task = task_name.lower().replace(" ", "_")
        for path in files:
            if normalized_task in path.stem.lower():
                return path
    return files[0]


def select_demo(handle: h5py.File, demo_name: str | None) -> tuple[str, h5py.Group]:
    root = handle["data"] if "data" in handle else handle
    names = sorted(name for name in root.keys() if hasattr(root[name], "keys"))
    if not names:
        raise RuntimeError("No demo groups found")
    selected = demo_name or names[0]
    if selected not in root:
        raise KeyError(f"Demo {selected!r} not found. Available examples: {names[:10]}")
    return selected, root[selected]


def recorded_eef_positions(demo: h5py.Group) -> np.ndarray | None:
    if "obs" not in demo:
        return None
    obs = demo["obs"]
    for key in ("ee_pos", "robot0_eef_pos"):
        if key in obs:
            return np.asarray(obs[key], dtype=np.float32)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay raw LIBERO demonstration actions in the LIBERO simulator. "
            "This tests whether the downloaded demo actions and simulator/controller "
            "setup work before any RDT policy is involved."
        )
    )
    parser.add_argument("--libero-root", type=Path, default=Path("/workspace/LIBERO"))
    parser.add_argument("--benchmark", choices=LIBERO_BENCHMARK_CHOICES, default=LIBERO_DEFAULT_BENCHMARK)
    parser.add_argument("--task-id", type=int, default=0, choices=range(10))
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/datasets/libero_object"))
    parser.add_argument("--demo-hdf5", type=Path)
    parser.add_argument("--demo-name")
    parser.add_argument("--max-steps", type=int, help="Limit replay length. Defaults to full demo action length.")
    parser.add_argument(
        "--action-source",
        choices=("raw", "codec_roundtrip"),
        default="raw",
        help=(
            "raw replays the HDF5 commands directly; codec_roundtrip first applies "
            "the production 7D -> 10D ortho6D -> 7D conversion and replays its output."
        ),
    )
    parser.add_argument("--settle-steps", type=int, default=0, help="Optional zero-action steps after setting demo state.")
    parser.add_argument("--video-resolution", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("outputs/libero_demo_replay.mp4"))
    parser.add_argument("--per-step-jsonl", type=Path, help="Optional per-step replay diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.libero_root) not in sys.path:
        sys.path.insert(0, str(args.libero_root))
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
        raw_actions = np.asarray(demo["actions"], dtype=np.float32)[:, :7]
        recorded_positions = recorded_eef_positions(demo)

    decoded_actions, codec_report = codec_roundtrip(raw_actions)
    actions = raw_actions if args.action_source == "raw" else decoded_actions

    replay_steps = len(actions) if args.max_steps is None else min(args.max_steps, len(actions))
    env = OffScreenRenderEnv(
        bddl_file_name=benchmark.get_task_bddl_file_path(args.task_id),
        camera_heights=128,
        camera_widths=128,
        horizon=replay_steps + args.settle_steps + 10,
    )
    observation = env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(args.settle_steps):
        observation, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

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
    final_observation = observation
    try:
        for step in range(replay_steps):
            action = actions[step]
            observation, reward, done, _ = env.step(action)
            final_observation = observation
            executed_steps = step + 1
            success = bool(done) or bool(env.check_success())

            converted = libero_observation_to_rdt(observation)
            live_pos = converted["state"][:3].astype(np.float32)
            recorded_pos = None
            eef_error = None
            if recorded_positions is not None:
                compare_index = min(step + 1, len(recorded_positions) - 1)
                recorded_pos = recorded_positions[compare_index, :3]
                eef_error = float(np.linalg.norm(live_pos - recorded_pos))
                eef_position_errors.append(eef_error)

            if per_step_handle is not None:
                per_step_handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "raw_demo_action": raw_actions[step],
                            "action": action,
                            "action_source": args.action_source,
                            "reward": float(reward),
                            "done": bool(done),
                            "success": bool(success),
                            "live_state_7d_closed": converted["state"],
                            "recorded_eef_pos": recorded_pos,
                            "live_vs_recorded_eef_l2": eef_error,
                        },
                        default=json_default,
                    )
                    + "\n"
                )
                per_step_handle.flush()

            label = f"demo={demo_name} step={step + 1}/{replay_steps} success={int(success)}"
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

    final_state = libero_observation_to_rdt(final_observation)["state"]
    error_array = np.asarray(eef_position_errors, dtype=np.float32)
    summary: dict[str, Any] = {
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "task_name": task.name,
        "instruction": task.language,
        "demo_hdf5": demo_hdf5.resolve(),
        "demo_name": demo_name,
        "action_source": args.action_source,
        "actions_replayed": int(executed_steps),
        "requested_replay_steps": int(replay_steps),
        "success": bool(success),
        "done": bool(done),
        "final_state_7d_closed": final_state,
        "action_stats": action_stats(actions[:replay_steps]),
        "codec_roundtrip": codec_report,
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
