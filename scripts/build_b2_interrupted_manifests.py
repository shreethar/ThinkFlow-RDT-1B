#!/usr/bin/env python
"""Build episode-safe train/validation manifests from interrupted B2 caches.

The extractor writes one physical episode per ``.pt`` file but only publishes
``manifest.jsonl`` after a split completes. This utility recovers completed
episode packs without moving or rewriting them. It never splits one episode
between train and validation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


FEATURE_TYPE = "latent_student_spatial_kv"


@dataclass(frozen=True)
class EpisodePack:
    root: Path
    path: Path
    dataset_id: str
    episode_id: str
    num_samples: int
    manifest_fields: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        action="append",
        required=True,
        help="B2 cache root containing interrupted train/episode_*.pt packs.",
    )
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--train-samples", type=int, default=640_000)
    parser.add_argument(
        "--validation-fraction-if-short",
        type=float,
        default=0.10,
        help=(
            "Validation fraction used only when fewer than --train-samples "
            "were extracted."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def stable_order_key(pack: EpisodePack, seed: int) -> bytes:
    payload = (
        f"{seed}\0{pack.dataset_id}\0{pack.episode_id}\0{pack.path.as_posix()}"
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def inspect_pack(root: Path, path: Path) -> EpisodePack:
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record.get("cache_layout") != "episode_pack":
        raise ValueError(f"{path} is not an episode_pack")
    if record.get("feature_type") != FEATURE_TYPE:
        raise ValueError(
            f"{path} has feature_type={record.get('feature_type')!r}, "
            f"expected {FEATURE_TYPE!r}"
        )

    num_samples = int(record["num_samples"])
    qwen_kv = torch.as_tensor(record["qwen_anchor_kv"])
    if tuple(qwen_kv.shape) != (num_samples, 5, 2048):
        raise ValueError(
            f"{path} has Qwen KV shape {tuple(qwen_kv.shape)}, expected "
            f"({num_samples}, 5, 2048)"
        )
    if qwen_kv.dtype != torch.bfloat16:
        raise ValueError(f"{path} has Qwen KV dtype {qwen_kv.dtype}, expected BF16")
    if not bool(torch.isfinite(qwen_kv.float()).all()):
        raise ValueError(f"{path} contains non-finite Qwen KV values")

    required_shapes = {
        "latent_waypoints": (num_samples, 5, 2),
        "state": (num_samples, 7),
        "actions": (num_samples, 64, 7),
        "action_time_mask": (num_samples, 64),
        "sample_anchor_index": (num_samples,),
    }
    for name, expected_shape in required_shapes.items():
        actual_shape = tuple(torch.as_tensor(record[name]).shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"{path} has {name} shape {actual_shape}, expected {expected_shape}"
            )

    dataset_id = str(record["dataset_id"])
    episode_id = str(record["episode_id"])
    step_indices = list(record.get("sample_step_idx", []))
    if len(step_indices) != num_samples:
        raise ValueError(
            f"{path} has {len(step_indices)} step indices for {num_samples} samples"
        )

    fields: dict[str, Any] = {
        "cache_layout": "episode_pack",
        "feature_type": FEATURE_TYPE,
        "dataset_id": dataset_id,
        "episode_id": episode_id,
        "num_samples": num_samples,
        "sample_start_index": int(record.get("sample_start_index", 0)),
        "sample_stop_index": int(record.get("sample_start_index", 0))
        + num_samples,
        "sample_step_start": str(step_indices[0]),
        "sample_step_stop": str(step_indices[-1]),
        "lang_token_count": int(torch.as_tensor(record["lang_tokens"]).shape[0]),
        "qwen_anchor_count": int(qwen_kv.shape[0]),
        "qwen_token_count": int(qwen_kv.shape[1]),
        "qwen_kv_dim": int(qwen_kv.shape[2]),
        "image_pool_count": len(record.get("image_jpegs", [])),
        "image_slot_count": int(record.get("image_slot_count", 0)),
        "has_img_tokens": False,
        "has_image_slots": True,
        "has_latent_waypoints": True,
        "qwen_cache_scope": str(record.get("qwen_cache_scope", "per_sample")),
        "actions_normalized": bool(record.get("actions_normalized", False)),
        "instruction": record.get("instruction"),
    }
    return EpisodePack(
        root=root,
        path=path,
        dataset_id=dataset_id,
        episode_id=episode_id,
        num_samples=num_samples,
        manifest_fields=fields,
    )


def inventory(cache_roots: list[Path], source_split: str) -> list[EpisodePack]:
    packs: list[EpisodePack] = []
    episode_keys: dict[tuple[str, str], Path] = {}
    for raw_root in cache_roots:
        root = raw_root.expanduser().resolve()
        source_dir = root / source_split
        paths = sorted(source_dir.glob("episodes_*/*.pt"))
        if not paths:
            raise FileNotFoundError(f"No episode packs found under {source_dir}")
        for path in paths:
            pack = inspect_pack(root, path.resolve())
            episode_key = (pack.dataset_id, pack.episode_id)
            previous = episode_keys.get(episode_key)
            if previous is not None:
                raise ValueError(
                    "Duplicate physical episode across cache roots: "
                    f"{episode_key} appears in {previous} and {path}"
                )
            episode_keys[episode_key] = path
            packs.append(pack)
    return packs


def select_train_packs(
    packs: list[EpisodePack],
    *,
    target_samples: int,
    seed: int,
) -> tuple[list[EpisodePack], list[EpisodePack]]:
    ordered = sorted(packs, key=lambda pack: stable_order_key(pack, seed))
    selected: list[EpisodePack] = []
    held_out: list[EpisodePack] = []
    remaining = target_samples
    for pack in ordered:
        if pack.num_samples <= remaining:
            selected.append(pack)
            remaining -= pack.num_samples
        else:
            held_out.append(pack)

    # Never return a zero-episode validation split. This matters when recovery
    # is run on a cache whose sample count exactly matches the requested target.
    if not held_out and len(selected) > 1:
        returned = selected.pop()
        held_out.append(returned)
    if not selected or not held_out:
        raise ValueError("At least two complete episode packs are required")
    return selected, held_out


def manifest_row(pack: EpisodePack, manifest_dir: Path) -> dict[str, Any]:
    row = dict(pack.manifest_fields)
    row["path"] = Path(os.path.relpath(pack.path, manifest_dir)).as_posix()
    return row


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def summarize(packs: list[EpisodePack]) -> dict[str, Any]:
    return {
        "episodes": len(packs),
        "samples": sum(pack.num_samples for pack in packs),
        "samples_by_dataset": dict(
            sorted(
                Counter(
                    {
                        dataset_id: sum(
                            pack.num_samples
                            for pack in packs
                            if pack.dataset_id == dataset_id
                        )
                        for dataset_id in {pack.dataset_id for pack in packs}
                    }
                ).items()
            )
        ),
    }


def main() -> None:
    args = parse_args()
    if args.train_samples <= 0:
        raise ValueError("--train-samples must be positive")
    if not 0.0 < args.validation_fraction_if_short < 1.0:
        raise ValueError("--validation-fraction-if-short must be in (0, 1)")

    roots = [path.expanduser().resolve() for path in args.cache_root]
    packs = inventory(roots, args.source_split)
    total_samples = sum(pack.num_samples for pack in packs)
    if total_samples > args.train_samples:
        effective_target = args.train_samples
        split_reason = "requested_train_sample_target"
    else:
        validation_samples = max(
            1,
            round(total_samples * args.validation_fraction_if_short),
        )
        effective_target = total_samples - validation_samples
        split_reason = "insufficient_samples_reserved_validation_fraction"

    train_packs, validation_packs = select_train_packs(
        packs,
        target_samples=effective_target,
        seed=args.seed,
    )
    train_summary = summarize(train_packs)
    validation_summary = summarize(validation_packs)
    report = {
        "feature_type": FEATURE_TYPE,
        "source_split": args.source_split,
        "cache_roots": [str(path) for path in roots],
        "seed": args.seed,
        "requested_train_samples": args.train_samples,
        "available_samples": total_samples,
        "requested_train_shortfall": max(0, args.train_samples - total_samples),
        "effective_train_target": effective_target,
        "split_reason": split_reason,
        "episode_leakage": False,
        "train": train_summary,
        "validation": validation_summary,
    }
    print(json.dumps(report, indent=2))
    if args.dry_run:
        return

    for root in roots:
        train_manifest = root / "train" / "manifest.jsonl"
        validation_manifest = root / "validation" / "manifest.jsonl"
        report_path = root / "recovered_manifest_split.json"
        existing = [
            path
            for path in (train_manifest, validation_manifest, report_path)
            if path.exists()
        ]
        if existing and not args.overwrite:
            raise FileExistsError(
                "Refusing to overwrite recovered manifests; pass --overwrite: "
                + ", ".join(str(path) for path in existing)
            )

        root_train = [pack for pack in train_packs if pack.root == root]
        root_validation = [
            pack for pack in validation_packs if pack.root == root
        ]
        write_jsonl_atomic(
            train_manifest,
            [manifest_row(pack, train_manifest.parent) for pack in root_train],
        )
        write_jsonl_atomic(
            validation_manifest,
            [
                manifest_row(pack, validation_manifest.parent)
                for pack in root_validation
            ],
        )
        root_report = dict(report)
        root_report["root"] = str(root)
        root_report["root_train"] = summarize(root_train)
        root_report["root_validation"] = summarize(root_validation)
        report_path.write_text(
            json.dumps(root_report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
