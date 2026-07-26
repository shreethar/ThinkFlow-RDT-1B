#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


SPLIT_NAMES = ("train", "validation", "test")
PREFERRED_KINDS = ("first_step", "first_gripper_change")
torch: Any = None


def require_torch() -> Any:
    global torch
    if torch is None:
        import torch as torch_module

        torch = torch_module
    return torch


def parse_step_idx(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return None


def choose_anchor_indices(
    *,
    anchor_kinds: list[str],
    anchor_step_idx: list[Any],
    target_anchors: int,
) -> list[int]:
    anchor_count = len(anchor_kinds)
    if anchor_count <= target_anchors:
        return list(range(anchor_count))

    chosen: list[int] = []
    for preferred_kind in PREFERRED_KINDS:
        for index, kind in enumerate(anchor_kinds):
            if kind == preferred_kind and index not in chosen:
                chosen.append(index)
                break
        if len(chosen) >= target_anchors:
            break

    if len(chosen) < target_anchors:
        parsed_steps = [parse_step_idx(value) for value in anchor_step_idx]
        ranked = sorted(
            range(anchor_count),
            key=lambda index: (
                index in chosen,
                parsed_steps[index] is None,
                parsed_steps[index] if parsed_steps[index] is not None else index,
                index,
            ),
        )
        for index in ranked:
            if index not in chosen:
                chosen.append(index)
            if len(chosen) >= target_anchors:
                break

    return sorted(chosen)


def remap_sample_anchor_indices(
    *,
    sample_step_idx: list[Any],
    old_sample_anchor_index: torch.Tensor,
    old_anchor_step_idx: list[Any],
    kept_old_indices: list[int],
) -> torch.Tensor:
    old_to_new = {old_index: new_index for new_index, old_index in enumerate(kept_old_indices)}
    kept_steps = [parse_step_idx(old_anchor_step_idx[index]) for index in kept_old_indices]
    new_indices: list[int] = []

    for sample_index, old_anchor_raw in enumerate(old_sample_anchor_index.flatten().tolist()):
        old_anchor = int(old_anchor_raw)
        if old_anchor in old_to_new:
            new_indices.append(old_to_new[old_anchor])
            continue

        sample_step = parse_step_idx(sample_step_idx[sample_index]) if sample_index < len(sample_step_idx) else None
        if sample_step is not None and any(step is not None for step in kept_steps):
            best_new_index = min(
                range(len(kept_old_indices)),
                key=lambda index: (
                    abs(sample_step - kept_steps[index])
                    if kept_steps[index] is not None
                    else float("inf"),
                    index,
                ),
            )
        else:
            best_new_index = min(
                range(len(kept_old_indices)),
                key=lambda index: (abs(old_anchor - kept_old_indices[index]), index),
            )
        new_indices.append(best_new_index)

    return torch.as_tensor(new_indices, dtype=torch.long)


def downsample_pack(pack: dict[str, Any], *, target_anchors: int) -> tuple[dict[str, Any], bool]:
    if pack.get("cache_layout") != "episode_pack":
        return pack, False

    qwen_anchor_kv = torch.as_tensor(pack["qwen_anchor_kv"])
    anchor_count = int(qwen_anchor_kv.shape[0])
    if anchor_count <= target_anchors:
        return pack, False

    anchor_kinds = [str(kind) for kind in pack["qwen_anchor_kind"]]
    anchor_step_idx = list(pack["qwen_anchor_step_idx"])
    kept_old_indices = choose_anchor_indices(
        anchor_kinds=anchor_kinds,
        anchor_step_idx=anchor_step_idx,
        target_anchors=target_anchors,
    )

    output = dict(pack)
    output["qwen_anchor_kv"] = qwen_anchor_kv[kept_old_indices].clone()
    output["qwen_anchor_step_idx"] = [str(anchor_step_idx[index]) for index in kept_old_indices]
    output["qwen_anchor_kind"] = [str(anchor_kinds[index]) for index in kept_old_indices]
    output["sample_anchor_index"] = remap_sample_anchor_indices(
        sample_step_idx=list(pack.get("sample_step_idx", [])),
        old_sample_anchor_index=torch.as_tensor(pack["sample_anchor_index"], dtype=torch.long),
        old_anchor_step_idx=anchor_step_idx,
        kept_old_indices=kept_old_indices,
    )
    return output, True


def atomic_torch_save(record: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(record, tmp_path)
    os.replace(tmp_path, path)


def process_manifest(
    *,
    input_manifest: Path,
    output_manifest: Path,
    input_split_dir: Path,
    output_split_dir: Path,
    target_anchors: int,
    in_place: bool,
    dry_run: bool,
) -> tuple[int, int]:
    processed = 0
    changed = 0
    output_lines: list[str] = []

    with input_manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            path_value = item if isinstance(item, str) else item.get("path")
            if not path_value:
                raise ValueError(f"Manifest line {line_number} has no path: {input_manifest}")

            input_path = Path(path_value)
            if not input_path.is_absolute():
                input_path = input_split_dir / input_path

            output_path = output_split_dir / input_path.name
            if not in_place and not dry_run:
                output_split_dir.mkdir(parents=True, exist_ok=True)

            if isinstance(item, dict) and item.get("cache_layout") == "episode_pack":
                pack = torch.load(input_path, map_location="cpu", weights_only=False)
                new_pack, did_change = downsample_pack(pack, target_anchors=target_anchors)
                if did_change:
                    changed += 1
                if not dry_run:
                    target_path = input_path if in_place else output_path
                    atomic_torch_save(new_pack, target_path)
                if isinstance(item, dict):
                    item = dict(item)
                    item["qwen_anchor_count"] = int(torch.as_tensor(new_pack["qwen_anchor_kv"]).shape[0])
                    item["path"] = input_path.name if in_place else output_path.name
            else:
                if not in_place and not dry_run:
                    shutil.copy2(input_path, output_path)
                if isinstance(item, dict):
                    item = dict(item)
                    item["path"] = input_path.name if in_place else output_path.name

            output_lines.append(json.dumps(item) + "\n")
            processed += 1

    if not dry_run:
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        tmp_manifest = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
        tmp_manifest.write_text("".join(output_lines), encoding="utf-8")
        os.replace(tmp_manifest, output_manifest)

    return processed, changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Downsample episode_pack Qwen anchors, usually from 8 anchors to 2, "
            "without rerunning Qwen."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", action="append", choices=SPLIT_NAMES)
    parser.add_argument("--target-anchors", type=int, default=2)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite packs and manifests under --input-dir. Saves storage but is destructive.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_torch()
    input_dir = args.input_dir.expanduser().resolve()
    if args.target_anchors <= 0:
        raise ValueError("--target-anchors must be positive")
    if args.in_place:
        output_dir = input_dir
    else:
        if args.output_dir is None:
            raise ValueError("--output-dir is required unless --in-place is set")
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(f"{output_dir} is not empty; pass --overwrite")

    splits = args.split if args.split is not None else list(SPLIT_NAMES)
    total_processed = 0
    total_changed = 0
    for split_name in splits:
        input_split_dir = input_dir / split_name
        input_manifest = input_split_dir / "manifest.jsonl"
        if not input_manifest.exists():
            print(f"[{split_name}] skipped missing manifest: {input_manifest}")
            continue

        output_split_dir = output_dir / split_name
        output_manifest = output_split_dir / "manifest.jsonl"
        processed, changed = process_manifest(
            input_manifest=input_manifest,
            output_manifest=output_manifest,
            input_split_dir=input_split_dir,
            output_split_dir=output_split_dir,
            target_anchors=args.target_anchors,
            in_place=args.in_place,
            dry_run=args.dry_run,
        )
        total_processed += processed
        total_changed += changed
        print(f"[{split_name}] processed {processed} packs, downsampled {changed}")

    print(
        f"Done. Processed {total_processed} packs; downsampled {total_changed} "
        f"to at most {args.target_anchors} anchors."
    )


if __name__ == "__main__":
    main()
