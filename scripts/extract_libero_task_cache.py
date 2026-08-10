#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


SAMPLE_LEVEL_KEYS = {
    "qwen_kv",
    "state",
    "state_dim_mask",
    "actions",
    "action_time_mask",
    "action_dim_mask",
    "ctrl_freq",
    "metadata",
    "sample_lang_index",
    "joint_state",
    "joint_states",
    "joint_states_mask",
    "sample_image_indices",
    "sample_image_mask",
    "image_slot_jpegs",
    "image_slot_mask",
    "latent_waypoints",
}


def numeric_demo_key(episode_id: str) -> tuple[int, str]:
    match = re.search(r":demo_(\d+)$", episode_id)
    return (int(match.group(1)), episode_id) if match else (sys.maxsize, episode_id)


def slice_value(value: Any, indices: list[int]) -> Any:
    if torch.is_tensor(value):
        return value[torch.as_tensor(indices, dtype=torch.long)]
    if isinstance(value, tuple):
        return tuple(value[index] for index in indices)
    if isinstance(value, list):
        return [value[index] for index in indices]
    return value


def subset_sample_shard(
    pack: dict[str, Any],
    indices: list[int],
    *,
    sample_start_index: int,
) -> dict[str, Any]:
    if pack.get("cache_layout") != "sample_shard":
        raise ValueError(f"Expected sample_shard, got {pack.get('cache_layout')!r}")
    original_count = int(pack["num_samples"])
    if not indices or min(indices) < 0 or max(indices) >= original_count:
        raise IndexError(f"Invalid subset {indices} for {original_count} samples")
    result = dict(pack)
    for key in SAMPLE_LEVEL_KEYS.intersection(pack):
        value = pack[key]
        if key == "ctrl_freq" and not (
            torch.is_tensor(value) and value.ndim > 0
        ):
            continue
        result[key] = slice_value(value, indices)
    result["num_samples"] = len(indices)
    result["sample_start_index"] = sample_start_index
    result["sample_stop_index"] = sample_start_index + len(indices)
    return result


def resolve_task_name(args: argparse.Namespace) -> str:
    if args.task_name:
        return args.task_name
    libero_root = args.libero_root.expanduser().resolve()
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))
    from libero.libero.benchmark import get_benchmark

    benchmark = get_benchmark(args.suite)(0)
    if not 0 <= args.task_id < benchmark.n_tasks:
        raise ValueError(f"Task ID {args.task_id} is outside [0,{benchmark.n_tasks})")
    return benchmark.get_task(args.task_id).name


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("path"):
                raise ValueError(f"Invalid manifest row {path}:{line_number}")
            source_path = Path(str(row["path"]))
            if not source_path.is_absolute():
                source_path = (path.parent / source_path).resolve()
            row = dict(row)
            row["_source_path"] = source_path
            rows.append(row)
    return rows


def candidate_episode_ids(
    rows: list[dict[str, Any]],
    *,
    episode_prefix: str,
) -> list[str]:
    episodes = {
        str(row[key])
        for row in rows
        for key in ("first_episode_id", "last_episode_id")
        if row.get(key) and str(row[key]).startswith(episode_prefix)
    }
    return sorted(episodes, key=numeric_demo_key)


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def manifest_row(
    source_row: dict[str, Any],
    *,
    destination_name: str,
    metadata: list[dict[str, Any]],
    sample_start_index: int,
) -> dict[str, Any]:
    row = {key: value for key, value in source_row.items() if key != "_source_path"}
    row.update(
        {
            "path": destination_name,
            "num_samples": len(metadata),
            "sample_start_index": sample_start_index,
            "sample_stop_index": sample_start_index + len(metadata),
            "first_dataset_id": metadata[0].get("dataset_id"),
            "first_episode_id": metadata[0].get("episode_id"),
            "first_step_idx": str(metadata[0].get("step_idx")),
            "last_dataset_id": metadata[-1].get("dataset_id"),
            "last_episode_id": metadata[-1].get("episode_id"),
            "last_step_idx": str(metadata[-1].get("step_idx")),
        }
    )
    return row


def extract(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_cache_root.expanduser().resolve()
    output_root = args.output_cache_root.expanduser().resolve()
    source_manifest = source_root / args.source_split / "manifest.jsonl"
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output cache is not empty: {output_root}. Choose a new directory."
        )

    task_name = resolve_task_name(args)
    episode_prefix = f"{task_name}_demo:"
    rows = load_manifest(source_manifest)
    available = candidate_episode_ids(rows, episode_prefix=episode_prefix)
    if len(available) < args.num_demos:
        raise ValueError(
            f"Requested {args.num_demos} demos, but {args.source_split} exposes "
            f"only {len(available)} for {task_name}"
        )
    selected = available[: args.num_demos]
    selected_set = set(selected)

    train_dir = output_root / "train"
    validation_dir = output_root / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    steps_by_episode: dict[str, list[int]] = defaultdict(list)
    storage_counts: dict[str, int] = defaultdict(int)
    cursor = 0

    candidate_rows = [
        row
        for row in rows
        if row.get("first_episode_id") in selected_set
        or row.get("last_episode_id") in selected_set
    ]
    for source_row in candidate_rows:
        source_path = Path(source_row["_source_path"])
        pack = torch.load(source_path, map_location="cpu", weights_only=True)
        if pack.get("cache_layout") != "sample_shard":
            raise ValueError(
                f"{source_path} uses {pack.get('cache_layout')!r}; only sample_shard is supported"
            )
        metadata = list(pack.get("metadata", []))
        keep = [
            index
            for index, item in enumerate(metadata)
            if item.get("episode_id") in selected_set
        ]
        if not keep:
            continue
        selected_metadata = [metadata[index] for index in keep]
        destination_name = f"shard_{len(output_rows):09d}.pt"
        destination = train_dir / destination_name
        if len(keep) == int(pack["num_samples"]):
            storage_counts[link_or_copy(source_path, destination)] += 1
        else:
            filtered = subset_sample_shard(
                pack,
                keep,
                sample_start_index=cursor,
            )
            torch.save(filtered, destination)
            storage_counts["repacked"] += 1
        output_rows.append(
            manifest_row(
                source_row,
                destination_name=destination_name,
                metadata=selected_metadata,
                sample_start_index=cursor,
            )
        )
        for item in selected_metadata:
            steps_by_episode[str(item["episode_id"])].append(int(item["step_idx"]))
        cursor += len(keep)
        del pack

    missing_episodes = selected_set.difference(steps_by_episode)
    if missing_episodes:
        raise RuntimeError(f"Selected episodes had no extracted samples: {sorted(missing_episodes)}")
    episode_audit = []
    for episode_id in selected:
        steps = sorted(steps_by_episode[episode_id])
        expected = list(range(steps[-1] + 1))
        if steps != expected:
            missing = sorted(set(expected).difference(steps))
            raise RuntimeError(
                f"Non-contiguous cache for {episode_id}; first missing steps: {missing[:20]}"
            )
        episode_audit.append(
            {
                "episode_id": episode_id,
                "samples": len(steps),
                "first_step": steps[0],
                "last_step": steps[-1],
            }
        )

    train_manifest = train_dir / "manifest.jsonl"
    with train_manifest.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    validation_manifest = validation_dir / "manifest.jsonl"
    with validation_manifest.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            validation_row = dict(row)
            validation_row["path"] = f"../train/{row['path']}"
            handle.write(json.dumps(validation_row) + "\n")

    source_metadata = source_root / "precompute_metadata.json"
    metadata = (
        json.loads(source_metadata.read_text(encoding="utf-8"))
        if source_metadata.exists()
        else {}
    )
    metadata["subset"] = {
        "source_cache_root": str(source_root),
        "source_split": args.source_split,
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": task_name,
        "episode_ids": selected,
        "sample_count": cursor,
        "validation_mirrors_training": True,
    }
    (output_root / "precompute_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    audit = {
        "source_cache_root": str(source_root),
        "output_cache_root": str(output_root),
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": task_name,
        "episodes": episode_audit,
        "episode_count": len(episode_audit),
        "sample_count": cursor,
        "shard_count": len(output_rows),
        "storage": dict(storage_counts),
        "validation_mirrors_training": True,
    }
    (output_root / "selection.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract complete demos for one LIBERO task from a cached suite."
    )
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--output-cache-root", type=Path, required=True)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--num-demos", type=int, default=10)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--libero-root", type=Path, default=Path("/home/ubuntu/LIBERO"))
    args = parser.parse_args()
    if args.num_demos <= 0:
        parser.error("--num-demos must be positive")
    return args


def main() -> None:
    audit = extract(parse_args())
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
