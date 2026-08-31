#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a B2/B3 hidden+waypoint cache before training."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-variant", choices=("b2", "b3"), required=True)
    parser.add_argument("--expected-dataset", required=True)
    return parser.parse_args()


def first_manifest_entry(path: Path) -> tuple[dict[str, Any], Path]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, str):
            payload = {"path": payload}
        shard = Path(str(payload["path"]))
        if not shard.is_absolute():
            shard = (path.parent / shard).resolve()
        return payload, shard
    raise ValueError(f"Manifest is empty: {path}")


def main() -> None:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    entry, shard_path = first_manifest_entry(manifest)
    if not shard_path.is_file():
        raise FileNotFoundError(shard_path)
    shard = torch.load(shard_path, map_location="cpu", weights_only=True)
    if shard.get("cache_layout") not in {"episode_pack", "sample_shard"}:
        raise ValueError(f"Unsupported cache layout in {shard_path}")
    first_metadata = (
        shard.get("metadata", [{}])[0]
        if shard.get("metadata")
        else {}
    )
    dataset_id = str(
        shard.get("dataset_id")
        or entry.get("dataset_id")
        or entry.get("first_dataset_id")
        or first_metadata.get("dataset_id")
        or ""
    )
    if dataset_id != args.expected_dataset:
        raise ValueError(
            f"Cache dataset {dataset_id!r} does not match {args.expected_dataset!r}"
        )
    variant = str(
        shard.get("conditioning_variant")
        or entry.get("conditioning_variant")
        or ""
    ).lower()
    if variant != args.expected_variant:
        raise ValueError(
            f"Cache variant {variant!r} does not match {args.expected_variant!r}. "
            "Re-extract with --feature-variant."
        )
    if shard["cache_layout"] == "episode_pack":
        required = {
            "qwen_anchor_kv": (5, 2048),
            "qwen_anchor_hidden_states": (5, 2560),
            "latent_waypoints": (5, 2),
        }
    else:
        required = {
            "qwen_kv": (5, 2048),
            "qwen_hidden_states": (5, 2560),
            "latent_waypoints": (5, 2),
        }
    sample_count = int(shard["num_samples"])
    for key, trailing_shape in required.items():
        if key not in shard:
            raise KeyError(f"{shard_path} lacks required feature {key!r}")
        tensor = torch.as_tensor(shard[key])
        if tuple(tensor.shape) != (sample_count, *trailing_shape):
            raise ValueError(
                f"{key} has {tuple(tensor.shape)}, expected "
                f"{(sample_count, *trailing_shape)}"
            )
        if not bool(torch.isfinite(tensor.float()).all()):
            raise ValueError(f"{key} contains NaN/Inf")
    for key, width in (("state", 9), ("actions", 7)):
        if key not in shard:
            raise KeyError(f"{shard_path} lacks Libero_RDT field {key!r}")
        tensor = torch.as_tensor(shard[key])
        if tensor.shape[0] != sample_count or tensor.shape[-1] != width:
            raise ValueError(
                f"{key} has {tuple(tensor.shape)}; expected sample axis "
                f"{sample_count} and final width {width}"
            )
    actions = torch.as_tensor(shard["actions"])
    if tuple(actions.shape) != (sample_count, 64, 7):
        raise ValueError(
            f"actions has {tuple(actions.shape)}, expected "
            f"{(sample_count, 64, 7)}"
        )
    state = torch.as_tensor(shard["state"], dtype=torch.float32)
    if not bool(((state[:, 7:9] >= 0) & (state[:, 7:9] <= 1)).all()):
        raise ValueError("Cached gripper state is not normalized to [0,1]")
    print(
        json.dumps(
            {
                "status": "ok",
                "manifest": str(manifest),
                "first_shard": str(shard_path),
                "dataset": dataset_id,
                "variant": variant,
                "samples_in_first_shard": sample_count,
                "training_condition": "hidden[5,2560]+waypoint[5,2]",
                "retained_unused_feature": "kv[5,2048]",
                "cached_state": "joint7+normalized_gripper2",
                "cached_actions": "[samples,64,7] raw LIBERO commands",
                "native_state_slots": {"joints": "0:7", "gripper": "10:12"},
                "native_action_slots": {"eef_delta": "39:45", "gripper": 10},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
