from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
from thinkflow_rdt.adapters.fractal import (  # noqa: E402
    DEFAULT_HORIZON,
    standardize_binary_gripper_action,
)
from thinkflow_rdt.adapters.sample_filtering import (  # noqa: E402
    DEFAULT_GRIPPER_WINDOW_AFTER,
    DEFAULT_GRIPPER_WINDOW_BEFORE,
    DEFAULT_MAX_SAMPLES_PER_EPISODE,
    build_episode_sample_indices,
)


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


@dataclass
class EpisodeFields:
    instructions: list[str]
    actions: np.ndarray


@dataclass
class CollectionCounts:
    episodes_seen: int = 0
    episodes_after_episode_filter: int = 0
    episodes_used: int = 0
    episodes_skipped_failed_success: int = 0
    episodes_skipped_no_steps: int = 0
    episodes_skipped_no_sampled_steps: int = 0
    steps_after_episode_filter: int = 0
    sampled_start_steps: int = 0
    action_rows_used_for_stats: int = 0


def tensor_numpy(value: Any, dtype: Any = np.float32) -> np.ndarray:
    raw = value.numpy() if hasattr(value, "numpy") else value
    return np.asarray(raw, dtype=dtype)


def decode_text(value: Any) -> str:
    raw = value.numpy() if hasattr(value, "numpy") else value
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


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
            "compute_sampled_action_stats.py requires tensorflow and "
            "tensorflow-datasets."
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
        yield from dataset
    except TypeError:
        for episode in dataset:
            yield episode
    except RuntimeError as exc:
        if "No such file or directory" in str(exc):
            print(f"Stopping early due to missing shard: {exc}")
        else:
            raise


def fractal_episode_fields_from_steps(steps: list[Any]) -> EpisodeFields:
    instructions: list[str] = []
    world_vectors: list[np.ndarray] = []
    rotation_deltas: list[np.ndarray] = []
    gripper_actions: list[float] = []
    gripper_closed: list[float] = []

    for step in steps:
        action = step["action"]
        observation = step["observation"]
        instructions.append(decode_text(observation["natural_language_instruction"]))
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
    actions = np.concatenate(
        [
            np.stack(world_vectors, axis=0).astype(np.float32),
            np.stack(rotation_deltas, axis=0).astype(np.float32),
            gripper_target[:, None],
        ],
        axis=1,
    )
    return EpisodeFields(instructions=instructions, actions=actions)


def kuka_episode_fields_from_steps(steps: list[Any]) -> EpisodeFields:
    return fractal_episode_fields_from_steps(steps)


def bridge_episode_fields_from_steps(steps: list[Any]) -> EpisodeFields:
    instructions: list[str] = []
    actions: list[np.ndarray] = []

    for step in steps:
        instructions.append(decode_text(step["language_instruction"]))
        actions.append(tensor_numpy(step["action"]))

    action_array = np.stack(actions, axis=0).astype(np.float32)
    action_array[:, 6] = standardize_bridge_gripper_open_to_closed(action_array[:, 6])
    return EpisodeFields(instructions=instructions, actions=action_array)


def droid_instruction(step: Any) -> str:
    for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        if key not in step:
            continue
        text = decode_text(step[key]).strip()
        if text:
            return text
    return ""


def droid_episode_fields_from_steps(steps: list[Any]) -> EpisodeFields:
    instructions: list[str] = []
    absolute_actions: list[np.ndarray] = []
    cartesian_positions: list[np.ndarray] = []

    for step in steps:
        observation = step["observation"]
        instructions.append(droid_instruction(step))
        absolute_actions.append(tensor_numpy(step["action"]))
        cartesian_positions.append(tensor_numpy(observation["cartesian_position"]))

    actions = standardize_droid_action(
        np.stack(absolute_actions, axis=0).astype(np.float32),
        np.stack(cartesian_positions, axis=0).astype(np.float32),
    )
    return EpisodeFields(instructions=instructions, actions=actions)


def bc_z_episode_fields_from_steps(steps: list[Any]) -> EpisodeFields:
    instructions: list[str] = []
    actions: list[np.ndarray] = []

    for step in steps:
        observation = step["observation"]
        action = step["action"]
        instructions.append(decode_text(observation["natural_language_instruction"]))

        xyz_res = tensor_numpy(action["future/xyz_residual"])
        xyz_next = xyz_res.reshape(10, 3)[0]

        axis_angle_res = tensor_numpy(action["future/axis_angle_residual"])
        axis_angle_next = axis_angle_res.reshape(10, 3)[0]

        target_close = tensor_numpy(action["future/target_close"])
        target_close_next = np.asarray([target_close[0]], dtype=np.float32)

        actions.append(
            np.concatenate(
                [xyz_next, axis_angle_next, target_close_next],
                axis=-1,
            )
        )

    return EpisodeFields(
        instructions=instructions,
        actions=np.stack(actions, axis=0).astype(np.float32),
    )


EPISODE_FIELD_EXTRACTORS: dict[str, Callable[[list[Any]], EpisodeFields]] = {
    "fractal": fractal_episode_fields_from_steps,
    "kuka": kuka_episode_fields_from_steps,
    "bridge": bridge_episode_fields_from_steps,
    "droid": droid_episode_fields_from_steps,
    "bc_z": bc_z_episode_fields_from_steps,
}


def is_successful_episode(raw_episode: Any) -> bool:
    if "success" not in raw_episode:
        return True
    raw_success = raw_episode["success"].numpy() if hasattr(raw_episode["success"], "numpy") else raw_episode["success"]
    if isinstance(raw_success, np.ndarray):
        raw_success = raw_success.item()
    return bool(raw_success)


def actions_for_scope(
    actions: np.ndarray,
    sampled_indices: list[int],
    *,
    scope: str,
    horizon: int,
) -> np.ndarray:
    if scope == "sampled-starts":
        return actions[np.asarray(sampled_indices, dtype=np.int64)]

    action_parts: list[np.ndarray] = []
    for step_index in sampled_indices:
        valid = min(horizon, actions.shape[0] - step_index)
        if valid > 0:
            action_parts.append(actions[step_index : step_index + valid])
    if not action_parts:
        return np.zeros((0, actions.shape[1]), dtype=np.float32)
    return np.concatenate(action_parts, axis=0).astype(np.float32)


def collect_sampled_actions(
    dataset_id: str,
    data_dir: Path,
    *,
    split: str,
    shard_pattern: str | None,
    max_episodes: int | None,
    filter_empty_language: bool,
    max_samples_per_episode: int | None,
    gripper_window_before: int,
    gripper_window_after: int,
    scope: str,
    horizon: int,
    kuka_only_success: bool,
) -> tuple[np.ndarray, CollectionCounts]:
    extractor = EPISODE_FIELD_EXTRACTORS[dataset_id]
    action_parts: list[np.ndarray] = []
    counts = CollectionCounts()

    for raw_index, raw_episode in enumerate(
        iter_tfds_episodes(
            data_dir,
            split=split,
            shard_pattern=shard_pattern,
        )
    ):
        if dataset_id != "kuka" and max_episodes is not None and raw_index >= max_episodes:
            break
        if (
            dataset_id == "kuka"
            and max_episodes is not None
            and counts.episodes_after_episode_filter >= max_episodes
        ):
            break

        counts.episodes_seen += 1

        if dataset_id == "kuka" and kuka_only_success and not is_successful_episode(raw_episode):
            counts.episodes_skipped_failed_success += 1
            continue

        steps = list(raw_episode["steps"])
        if not steps:
            counts.episodes_skipped_no_steps += 1
            continue

        counts.episodes_after_episode_filter += 1
        counts.steps_after_episode_filter += len(steps)

        fields = extractor(steps)
        sampled_indices = build_episode_sample_indices(
            fields.instructions,
            fields.actions,
            max_samples_per_episode=max_samples_per_episode,
            filter_empty_language=filter_empty_language,
            gripper_window_before=gripper_window_before,
            gripper_window_after=gripper_window_after,
        )
        if not sampled_indices:
            counts.episodes_skipped_no_sampled_steps += 1
            continue

        selected_actions = actions_for_scope(
            fields.actions,
            sampled_indices,
            scope=scope,
            horizon=horizon,
        )
        if selected_actions.shape[0] == 0:
            counts.episodes_skipped_no_sampled_steps += 1
            continue

        action_parts.append(selected_actions)
        counts.episodes_used += 1
        counts.sampled_start_steps += len(sampled_indices)
        counts.action_rows_used_for_stats += int(selected_actions.shape[0])

        if counts.episodes_used % 1000 == 0:
            print(
                "  processed "
                f"{counts.episodes_used} used episodes / "
                f"{counts.sampled_start_steps} sampled starts / "
                f"{counts.action_rows_used_for_stats} action rows"
            )

    if not action_parts:
        raise ValueError(f"No sampled actions found for {dataset_id} in {data_dir}")
    return np.concatenate(action_parts, axis=0).astype(np.float32), counts


def compute_for_dataset(
    dataset_id: str,
    *,
    data_dir: Path,
    audit_json: Path,
    split: str,
    shard_pattern: str | None,
    max_episodes: int | None,
    filter_empty_language: bool,
    max_samples_per_episode: int | None,
    gripper_window_before: int,
    gripper_window_after: int,
    scope: str,
    horizon: int,
    q_low: float,
    q_high: float,
    kuka_only_success: bool,
    write: bool,
) -> None:
    print(f"\n[{dataset_id}] reading {data_dir}")
    actions, counts = collect_sampled_actions(
        dataset_id,
        data_dir,
        split=split,
        shard_pattern=shard_pattern,
        max_episodes=max_episodes,
        filter_empty_language=filter_empty_language,
        max_samples_per_episode=max_samples_per_episode,
        gripper_window_before=gripper_window_before,
        gripper_window_after=gripper_window_after,
        scope=scope,
        horizon=horizon,
        kuka_only_success=kuka_only_success,
    )
    stats = compute_action_quantile_stats(actions, q_low=q_low, q_high=q_high)
    source = {
        "dataset_id": dataset_id,
        "data_dir": str(data_dir),
        "split": split,
        "shard_pattern": shard_pattern,
        "max_episodes": max_episodes,
        "stats_scope": scope,
        "horizon": horizon if scope == "sampled-horizons" else None,
        "filter_empty_language": filter_empty_language,
        "max_samples_per_episode": max_samples_per_episode,
        "gripper_window_before": gripper_window_before,
        "gripper_window_after": gripper_window_after,
        "kuka_only_success": bool(dataset_id == "kuka" and kuka_only_success),
        "num_episodes_seen": counts.episodes_seen,
        "num_episodes_after_episode_filter": counts.episodes_after_episode_filter,
        "num_episodes_used": counts.episodes_used,
        "num_episodes_skipped_failed_success": counts.episodes_skipped_failed_success,
        "num_episodes_skipped_no_steps": counts.episodes_skipped_no_steps,
        "num_episodes_skipped_no_sampled_steps": counts.episodes_skipped_no_sampled_steps,
        "num_steps_after_episode_filter": counts.steps_after_episode_filter,
        "num_sampled_start_steps": counts.sampled_start_steps,
        "num_action_rows_used_for_stats": counts.action_rows_used_for_stats,
        "q_low": q_low,
        "q_high": q_high,
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Stats are computed from locally available shards after applying the "
            "same language, Kuka success, max-64, and gripper-change sampling "
            "policy used by the standardized loaders. Rerun after downloading a "
            "larger subset."
        ),
    }

    print(f"  episodes seen: {counts.episodes_seen}")
    print(f"  episodes after episode filter: {counts.episodes_after_episode_filter}")
    print(f"  episodes used: {counts.episodes_used}")
    print(f"  steps after episode filter: {counts.steps_after_episode_filter}")
    print(f"  sampled start steps: {counts.sampled_start_steps}")
    print(f"  action rows used for stats: {counts.action_rows_used_for_stats}")
    if counts.episodes_skipped_failed_success:
        print(f"  skipped failed Kuka episodes: {counts.episodes_skipped_failed_success}")
    if counts.episodes_skipped_no_sampled_steps:
        print(f"  skipped no sampled steps: {counts.episodes_skipped_no_sampled_steps}")
    print("  dim_names:", json.dumps(ACTION_DIM_NAMES))
    print("  q01:", json.dumps(stats.q01.astype(float).tolist()))
    print("  q99:", json.dumps(stats.q99.astype(float).tolist()))

    if write:
        write_action_stats_to_audit(audit_json, stats, source=source)
        print(f"  wrote {audit_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute q01/q99 stats from the same sampled timesteps used by the "
            "standardized dataset loaders."
        )
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
    parser.add_argument(
        "--include-empty-language",
        action="store_true",
        help="Disable the loader-default empty-language filter.",
    )
    parser.add_argument(
        "--max-samples-per-episode",
        type=int,
        default=DEFAULT_MAX_SAMPLES_PER_EPISODE,
        help="Loader sampling cap. Use -1 to keep all valid steps.",
    )
    parser.add_argument(
        "--gripper-window-before",
        type=int,
        default=DEFAULT_GRIPPER_WINDOW_BEFORE,
    )
    parser.add_argument(
        "--gripper-window-after",
        type=int,
        default=DEFAULT_GRIPPER_WINDOW_AFTER,
    )
    parser.add_argument(
        "--scope",
        choices=["sampled-horizons", "sampled-starts"],
        default="sampled-horizons",
        help=(
            "sampled-horizons uses every valid row from each sampled [H, 7] "
            "target; sampled-starts uses only actions[step_idx]."
        ),
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--q-low", type=float, default=0.01)
    parser.add_argument("--q-high", type=float, default=0.99)
    parser.add_argument(
        "--kuka-include-failures",
        action="store_true",
        help="Disable Kuka's loader-default episode['success'] filter.",
    )
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write = not args.no_write
    max_samples_per_episode = (
        None if args.max_samples_per_episode < 0 else args.max_samples_per_episode
    )

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
            filter_empty_language=not args.include_empty_language,
            max_samples_per_episode=max_samples_per_episode,
            gripper_window_before=args.gripper_window_before,
            gripper_window_after=args.gripper_window_after,
            scope=args.scope,
            horizon=args.horizon,
            q_low=args.q_low,
            q_high=args.q_high,
            kuka_only_success=not args.kuka_include_failures,
            write=write,
        )


if __name__ == "__main__":
    main()
