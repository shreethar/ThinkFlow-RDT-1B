#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/thinkflow-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/thinkflow-matplotlib")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from thinkflow_rdt.adapters.libero import (  # noqa: E402
    libero_action_to_rdt,
    libero_observation_to_rdt,
    libero_orientation_to_ortho6d,
)

RDT_VALUE_DIM = 128
RDT_TOTAL_DIM = 256


def install_robosuite_mujoco_compatibility() -> None:
    """Patch robosuite 1.4's old-style MuJoCo mass-matrix call for live env inspection."""
    try:
        import mujoco
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
            sim.data.get_site_jacp(controller.eef_name).reshape((3, -1))[:, controller.qvel_index]
        ).copy()
        controller.J_ori = np.asarray(
            sim.data.get_site_jacr(controller.eef_name).reshape((3, -1))[:, controller.qvel_index]
        ).copy()
        controller.J_full = np.vstack([controller.J_pos, controller.J_ori])
        mass_matrix = np.empty((sim.model.nv, sim.model.nv), dtype=np.float64, order="C")
        try:
            mujoco.mj_fullM(sim.model._model, sim.data._data, mass_matrix)
        except TypeError:
            mujoco.mj_fullM(sim.model._model, mass_matrix, sim.data.qM)
        controller.mass_matrix = mass_matrix[controller.qvel_index, :][:, controller.qvel_index]
        controller.new_update = False

    Controller.update = update


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def summarize_array(values: Any, *, max_values: int = 8) -> dict[str, Any]:
    array = np.asarray(values)
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size == 0:
        return summary
    flat = array.reshape(-1)
    if np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        numeric = flat.astype(np.float64)
        finite = numeric[np.isfinite(numeric)]
        if finite.size:
            summary.update(
                {
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                    "mean": float(finite.mean()),
                    "std": float(finite.std()),
                }
            )
        summary["first_values"] = flat[:max_values].astype(float).tolist()
        unique = np.unique(flat[: min(flat.size, 10000)])
        if unique.size <= 16:
            summary["unique_values_sample"] = unique.astype(float).tolist()
    else:
        summary["first_values"] = [str(item) for item in flat[:max_values]]
    return summary


def summarize_mapping(mapping: Any) -> dict[str, Any]:
    return {
        str(key): summarize_array(value)
        for key, value in sorted(mapping.items(), key=lambda item: str(item[0]))
        if not hasattr(value, "keys")
    }


def hdf5_tree(group: Any, *, depth: int, max_items: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    count = 0
    for key in sorted(group.keys()):
        if count >= max_items:
            result["..."] = f"truncated after {max_items} keys"
            break
        count += 1
        value = group[key]
        if hasattr(value, "keys") and depth > 0:
            result[key] = {
                "type": "group",
                "attrs": {name: _json_default(attr) for name, attr in value.attrs.items()},
                "children": hdf5_tree(value, depth=depth - 1, max_items=max_items),
            }
        elif hasattr(value, "shape"):
            result[key] = {
                "type": "dataset",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        else:
            result[key] = {"type": type(value).__name__}
    return result


def rdt_256_from_libero_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros((RDT_VALUE_DIM,), dtype=np.float32)
    masks = np.zeros((RDT_VALUE_DIM,), dtype=np.float32)
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] != 11:
        raise ValueError(
            f"Expected 11D [xyz,ortho6D,finger0,finger1], got {state.shape}"
        )
    values[10:12] = state[9:11]
    masks[10:12] = 1.0
    values[30:33] = state[:3]
    masks[30:33] = 1.0
    values[33:39] = state[3:9]
    masks[33:39] = 1.0
    packed = np.concatenate([values, masks], axis=0)
    return values, masks, packed


def euler_xyz_to_ortho6d_numpy(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(euler, dtype=np.float64).reshape(3)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    first_column = np.array([cy * cp, sy * cp, -sp], dtype=np.float32)
    second_column = np.array(
        [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr],
        dtype=np.float32,
    )
    return np.concatenate([first_column, second_column], axis=0)


def rdt_conversion_report(state: np.ndarray) -> dict[str, Any]:
    state = np.asarray(state, dtype=np.float32).reshape(11)
    values, masks, packed = rdt_256_from_libero_state(state)
    nonzero_value_indices = np.flatnonzero(values).astype(int).tolist()
    nonzero_mask_indices = np.flatnonzero(masks).astype(int).tolist()
    return {
        "libero_11d_state": {
            "names": [
                "x", "y", "z",
                "rot6d_0", "rot6d_1", "rot6d_2",
                "rot6d_3", "rot6d_4", "rot6d_5",
                "finger_0", "finger_1",
            ],
            "values": state,
        },
        "rdt_128_values_nonzero_indices": nonzero_value_indices,
        "rdt_128_mask_indices": nonzero_mask_indices,
        "rdt_slots": {
            "raw_fingers_indices_10_11": values[10:12],
            "xyz_indices_30_32": values[30:33],
            "ortho6d_indices_33_38": values[33:39],
        },
        "packed_256_shape": list(packed.shape),
        "packed_256_nonzero_count": int(np.count_nonzero(packed)),
    }


def inspect_demo_file(path: Path, *, demo_name: str | None, step: int) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        root = handle["data"] if "data" in handle else handle
        names = sorted(name for name in root.keys() if hasattr(root[name], "keys"))
        if not names:
            raise RuntimeError(f"No demo groups found in {path}")
        selected = demo_name or names[0]
        if selected not in root:
            raise KeyError(f"Demo {selected!r} not found. Available examples: {names[:10]}")
        demo = root[selected]
        obs = demo["obs"]
        actions = np.asarray(demo["actions"], dtype=np.float32)
        step_index = min(max(step, 0), len(actions) - 1)

        position = None
        for key in ("ee_pos", "robot0_eef_pos"):
            if key in obs:
                position = np.asarray(obs[key])[step_index, :3]
                break
        orientation = None
        for key in ("ee_ori", "robot0_eef_quat"):
            if key in obs:
                orientation = np.asarray(obs[key])[step_index]
                break
        gripper_state = None
        for key in ("gripper_states", "robot0_gripper_qpos"):
            if key in obs:
                gripper_state = np.asarray(obs[key])[step_index]
                break
        if position is None or orientation is None:
            raise KeyError("Could not find LIBERO EEF position/orientation keys in demo obs")
        if gripper_state is None or np.asarray(gripper_state).size < 2:
            raise KeyError("Could not find two raw LIBERO finger positions")
        state = np.concatenate(
            [
                np.asarray(position, dtype=np.float32).reshape(3),
                libero_orientation_to_ortho6d(np.asarray(orientation)).reshape(6),
                np.asarray(gripper_state, dtype=np.float32).reshape(-1)[:2],
            ],
            axis=0,
        )
        raw_action = actions[step_index, :7].copy()
        encoded_action = libero_action_to_rdt(raw_action)
        return {
            "file": path,
            "root_attrs": {name: _json_default(value) for name, value in handle.attrs.items()},
            "tree": hdf5_tree(handle, depth=3, max_items=30),
            "available_demo_count": len(names),
            "selected_demo": selected,
            "selected_step": step_index,
            "demo_attrs": {name: _json_default(value) for name, value in demo.attrs.items()},
            "obs_keys_summary": summarize_mapping(obs),
            "raw_action_at_step": {
                "names": [
                    "delta_x",
                    "delta_y",
                    "delta_z",
                    "delta_rx",
                    "delta_ry",
                    "delta_rz",
                    "libero_gripper_command",
                ],
                "values": raw_action,
                "gripper_convention": "raw HDF5 value preserved without remapping",
            },
            "adapter_10d_action_at_step": {
                "names": [
                    "delta_x",
                    "delta_y",
                    "delta_z",
                    "delta_rot6d_0", "delta_rot6d_1", "delta_rot6d_2",
                    "delta_rot6d_3", "delta_rot6d_4", "delta_rot6d_5",
                    "raw_gripper_command",
                ],
                "values": encoded_action,
            },
            "state_conversion": rdt_conversion_report(state),
        }


def find_first_demo_file(dataset_dir: Path) -> Path:
    files = sorted(dataset_dir.rglob("*.hdf5")) + sorted(dataset_dir.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5/.h5 files found below {dataset_dir}")
    return files[0]


def inspect_live_env(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.libero_root) not in sys.path:
        sys.path.insert(0, str(args.libero_root))
    install_robosuite_mujoco_compatibility()
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark(args.benchmark)(0)
    task = benchmark.get_task(args.task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=benchmark.get_task_bddl_file_path(args.task_id),
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
        horizon=50,
    )
    observation = env.reset()
    init_states = None
    init_file = (
        args.libero_root
        / "libero"
        / "libero"
        / "init_files"
        / task.problem_folder
        / task.init_states_file
    )
    if init_file.exists():
        import torch

        init_states = torch.load(init_file, map_location="cpu", weights_only=False)
        state_index = args.init_state_index % len(init_states)
        observation = env.set_init_state(init_states[state_index])
    else:
        state_index = None
    converted = libero_observation_to_rdt(observation)
    env.close()
    return {
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "task_name": task.name,
        "instruction": task.language,
        "init_state_index": state_index,
        "raw_observation_keys_summary": summarize_mapping(observation),
        "adapter_primary_image": summarize_array(converted["primary"]),
        "adapter_wrist_image": None if converted["wrist"] is None else summarize_array(converted["wrist"]),
        "state_conversion": rdt_conversion_report(converted["state"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect exactly what LIBERO observations/demos output and how they map into RDT state slots."
    )
    parser.add_argument("--libero-root", type=Path, default=Path("/workspace/LIBERO"))
    parser.add_argument("--benchmark", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-index", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/datasets/libero_object"),
        help="Directory containing downloaded LIBERO .hdf5/.h5 demos.",
    )
    parser.add_argument("--demo-file", type=Path)
    parser.add_argument("--demo-name")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--live-env", action="store_true", help="Also reset a live LIBERO env and inspect obs keys.")
    parser.add_argument("--output", type=Path, help="Write JSON report here.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "conventions": {
            "gripper_action": "raw HDF5 command, no remapping",
            "adapter_state": "[xyz,absolute_ortho6d,finger0,finger1] (11D)",
            "model_action": "[dxyz,relative_ortho6d,raw_gripper_command] (10D)",
            "rdt_native_state": "256D = 128 values + 128 masks",
            "rdt_slots": {
                "state values[10:12]": "raw finger0/finger1",
                "action values[10]": "raw gripper command",
                "values[30:33]": "xyz position",
                "values[33:39]": "ortho6d rotation",
            },
        }
    }
    demo_file = args.demo_file
    if demo_file is None and args.dataset_dir.exists():
        demo_file = find_first_demo_file(args.dataset_dir)
    if demo_file is not None:
        report["downloaded_demo_file"] = inspect_demo_file(
            demo_file,
            demo_name=args.demo_name,
            step=args.step,
        )
    else:
        report["downloaded_demo_file"] = {
            "skipped": f"No demo file provided and dataset dir not found: {args.dataset_dir}"
        }
    if args.live_env:
        report["live_env_observation"] = inspect_live_env(args)

    text = json.dumps(report, indent=2, default=_json_default)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    print(text)


if __name__ == "__main__":
    main()
