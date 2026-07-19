from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thinkflow_rdt.adapters.action_stats import (  # noqa: E402
    ACTION_DIM_NAMES,
    compute_action_quantile_stats,
    write_action_stats_to_audit,
)
from thinkflow_rdt.adapters.bridge import (  # noqa: E402
    standardize_bridge_gripper_open_to_closed,
)
from thinkflow_rdt.adapters.droid import standardize_droid_action  # noqa: E402
from thinkflow_rdt.adapters.fractal import standardize_binary_gripper_action  # noqa: E402


DATASET_CONFIGS: dict[str, dict[str, Path]] = {
    "fractal": {
        "data_dir": REPO_ROOT / "dataset" / "mock_dataset" / "fractal_dataset" / "data",
        "audit_json": REPO_ROOT / "dataset" / "mock_dataset" / "fractal_dataset" / "audit.json",
    },
    "kuka": {
        "data_dir": REPO_ROOT / "dataset" / "mock_dataset" / "kuka_dataset" / "data",
        "audit_json": REPO_ROOT / "dataset" / "mock_dataset" / "kuka_dataset" / "audit.json",
    },
    "bridge": {
        "data_dir": REPO_ROOT
        / "dataset"
        / "mock_dataset"
        / "bridge_dataset"
        / "data",
        "audit_json": REPO_ROOT / "dataset" / "mock_dataset" / "bridge_dataset" / "audit.json",
    },
    "droid": {
        "data_dir": REPO_ROOT
        / "dataset"
        / "mock_dataset"
        / "droid_dataset"
        / "data",
        "audit_json": REPO_ROOT / "dataset" / "mock_dataset" / "droid_dataset" / "audit.json",
    },
    "bc_z": {
        "data_dir": REPO_ROOT / "dataset" / "mock_dataset" / "bc_z_dataset" / "data",
        "audit_json": REPO_ROOT / "dataset" / "mock_dataset" / "bc_z_dataset" / "audit.json",
    },
}


def tensor_numpy(value: Any, dtype: Any = np.float32) -> np.ndarray:
    raw = value.numpy() if hasattr(value, "numpy") else value
    return np.asarray(raw, dtype=dtype)


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


def iter_tfds_episodes(
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
            "compute_action_stats.py requires tensorflow and tensorflow-datasets."
        ) from exc

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    builder = tfds.builder_from_directory(str(data_dir))
    shard_paths = find_local_shards(
        data_dir,
        split=split,
        shard_pattern=shard_pattern,
    )
    if shard_paths:
        dataset = tf.data.TFRecordDataset([str(path) for path in shard_paths]).map(
            builder.info.features.deserialize_example,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        try:
            dataset = builder.as_dataset(split=split)
        except NotImplementedError:
            dataset = builder.as_data_source(split=split)
            
    try:
        try:
            yield from dataset
        except TypeError:
            for episode in dataset:
                yield episode
    except RuntimeError as e:
        if "No such file or directory" in str(e):
            print(f"Stopping early due to missing shard: {e}")
        else:
            raise


def fractal_actions_from_steps(steps: list[Any]) -> np.ndarray:
    world_vectors: list[np.ndarray] = []
    rotation_deltas: list[np.ndarray] = []
    gripper_actions: list[float] = []
    gripper_closed: list[float] = []

    for step in steps:
        action = step["action"]
        observation = step["observation"]
        world_vectors.append(tensor_numpy(action["world_vector"]))
        rotation_deltas.append(tensor_numpy(action["rotation_delta"]))
        gripper_actions.append(
            float(tensor_numpy(action["gripper_closedness_action"]).reshape(-1)[0])
        )
        gripper_closed.append(float(tensor_numpy(observation["gripper_closed"]).reshape(-1)[0]))

    gripper_target = standardize_binary_gripper_action(
        np.asarray(gripper_actions, dtype=np.float32),
        np.asarray(gripper_closed, dtype=np.float32),
    )
    return np.concatenate(
        [
            np.stack(world_vectors, axis=0).astype(np.float32),
            np.stack(rotation_deltas, axis=0).astype(np.float32),
            gripper_target[:, None],
        ],
        axis=1,
    )


def kuka_actions_from_steps(steps: list[Any]) -> np.ndarray:
    return fractal_actions_from_steps(steps)


def bridge_actions_from_steps(steps: list[Any]) -> np.ndarray:
    actions = np.stack(
        [tensor_numpy(step["action"]) for step in steps],
        axis=0,
    ).astype(np.float32)
    actions[:, 6] = standardize_bridge_gripper_open_to_closed(actions[:, 6])
    return actions


def droid_actions_from_steps(steps: list[Any]) -> np.ndarray:
    absolute_actions = np.stack(
        [tensor_numpy(step["action"]) for step in steps],
        axis=0,
    ).astype(np.float32)
    cartesian_positions = np.stack(
        [tensor_numpy(step["observation"]["cartesian_position"]) for step in steps],
        axis=0,
    ).astype(np.float32)
    return standardize_droid_action(absolute_actions, cartesian_positions)


def bc_z_actions_from_steps(steps: list[Any]) -> np.ndarray:
    actions: list[np.ndarray] = []
    for step in steps:
        action = step["action"]
        xyz_res = tensor_numpy(action["future/xyz_residual"])
        xyz_next = xyz_res.reshape(10, 3)[0]
        
        aa_res = tensor_numpy(action["future/axis_angle_residual"])
        aa_next = aa_res.reshape(10, 3)[0]
        
        target_close = tensor_numpy(action["future/target_close"])
        target_close_next = np.array([target_close[0]], dtype=np.float32)
        
        action_7d = np.concatenate([xyz_next, aa_next, target_close_next], axis=-1)
        actions.append(action_7d)
    
    return np.stack(actions, axis=0).astype(np.float32)


ACTION_EXTRACTORS: dict[str, Callable[[list[Any]], np.ndarray]] = {
    "fractal": fractal_actions_from_steps,
    "kuka": kuka_actions_from_steps,
    "bridge": bridge_actions_from_steps,
    "droid": droid_actions_from_steps,
    "bc_z": bc_z_actions_from_steps,
}


def collect_actions(
    dataset_id: str,
    data_dir: Path,
    *,
    split: str,
    shard_pattern: str | None,
    max_episodes: int | None,
) -> tuple[np.ndarray, int, int]:
    extractor = ACTION_EXTRACTORS[dataset_id]
    action_parts: list[np.ndarray] = []
    episode_count = 0
    step_count = 0

    for raw_episode in iter_tfds_episodes(
        data_dir,
        split=split,
        shard_pattern=shard_pattern,
    ):
        if max_episodes is not None and episode_count >= max_episodes:
            break
        steps = list(raw_episode["steps"])
        if not steps:
            continue
        actions = extractor(steps)
        action_parts.append(actions.astype(np.float32))
        episode_count += 1
        step_count += int(actions.shape[0])
        if episode_count % 1000 == 0:
            print(f"  processed {episode_count} episodes / {step_count} steps")

    if not action_parts:
        raise ValueError(f"No actions found for {dataset_id} in {data_dir}")
    return np.concatenate(action_parts, axis=0), episode_count, step_count


def compute_for_dataset(
    dataset_id: str,
    *,
    data_dir: Path,
    audit_json: Path,
    split: str,
    shard_pattern: str | None,
    max_episodes: int | None,
    write: bool,
) -> None:
    print(f"\n[{dataset_id}] reading {data_dir}")
    actions, episode_count, step_count = collect_actions(
        dataset_id,
        data_dir,
        split=split,
        shard_pattern=shard_pattern,
        max_episodes=max_episodes,
    )
    stats = compute_action_quantile_stats(actions)
    source = {
        "dataset_id": dataset_id,
        "data_dir": str(data_dir),
        "split": split,
        "shard_pattern": shard_pattern,
        "max_episodes": max_episodes,
        "num_episodes": episode_count,
        "num_steps": step_count,
        "q_low": 0.01,
        "q_high": 0.99,
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Stats are computed from locally available shards only; rerun this script after downloading a larger subset.",
    }

    print(f"  episodes: {episode_count}")
    print(f"  steps: {step_count}")
    print("  dim_names:", json.dumps(ACTION_DIM_NAMES))
    print("  q01:", json.dumps(stats.q01.astype(float).tolist()))
    print("  q99:", json.dumps(stats.q99.astype(float).tolist()))

    if write:
        write_action_stats_to_audit(audit_json, stats, source=source)
        print(f"  wrote {audit_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-dataset q01/q99 stats for standardized 7D actions."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASET_CONFIGS.keys()],
        default="all",
        help="Dataset to process. Defaults to all local standardized adapters.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--shard-pattern", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write = not args.no_write

    if args.dataset == "all":
        if args.data_dir is not None or args.audit_json is not None:
            raise ValueError("--data-dir/--audit-json overrides are only valid for one dataset")
        dataset_ids = list(DATASET_CONFIGS)
    else:
        dataset_ids = [args.dataset]

    for dataset_id in dataset_ids:
        config = DATASET_CONFIGS[dataset_id]
        data_dir = (args.data_dir or config["data_dir"]).expanduser().resolve()
        audit_json = (args.audit_json or config["audit_json"]).expanduser().resolve()
        compute_for_dataset(
            dataset_id,
            data_dir=data_dir,
            audit_json=audit_json,
            split=args.split,
            shard_pattern=args.shard_pattern,
            max_episodes=args.max_episodes,
            write=write,
        )


if __name__ == "__main__":
    main()
