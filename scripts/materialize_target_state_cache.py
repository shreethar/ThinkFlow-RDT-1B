#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch


def read_manifest(manifest: Path) -> list[dict[str, Any]]:
    base = manifest.parent
    items: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if isinstance(raw, str):
                item = {"path": raw}
            elif isinstance(raw, dict):
                item = dict(raw)
            else:
                raise TypeError(f"{manifest}:{line_number} is not a string or object")
            path = Path(str(item["path"]))
            item["_resolved_path"] = path if path.is_absolute() else (base / path).resolve()
            item["_line_number"] = line_number
            items.append(item)
    return items


def parse_step_indices(values: Any, count: int) -> list[int] | None:
    if values is None:
        return list(range(count))
    output: list[int] = []
    for value in list(values):
        try:
            output.append(int(value))
        except (TypeError, ValueError):
            return None
    return output if len(output) == count else None


def first_index_by_step(step_indices: list[int]) -> tuple[dict[int, int], int]:
    mapping: dict[int, int] = {}
    duplicates = 0
    for index, step in enumerate(step_indices):
        if step in mapping:
            duplicates += 1
            continue
        mapping[step] = index
    return mapping, duplicates


def build_target_state_horizon(
    states: torch.Tensor,
    old_time_mask: torch.Tensor,
    *,
    step_indices: list[int] | None,
    target_offset: int,
    preserve_original_time_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    if states.ndim != 2:
        raise ValueError(f"Expected states [N,D], got {tuple(states.shape)}")
    if old_time_mask.ndim != 2:
        raise ValueError(f"Expected action_time_mask [N,H], got {tuple(old_time_mask.shape)}")

    sample_count, action_dim = int(states.shape[0]), int(states.shape[1])
    horizon = int(old_time_mask.shape[1])
    targets = torch.zeros(sample_count, horizon, action_dim, dtype=torch.float32)
    target_mask = torch.zeros(sample_count, horizon, dtype=torch.bool)

    index_by_step: dict[int, int] | None = None
    duplicate_steps = 0
    if step_indices is not None:
        index_by_step, duplicate_steps = first_index_by_step(step_indices)

    missing = 0
    filled = 0
    for sample_index in range(sample_count):
        source_step = step_indices[sample_index] if step_indices is not None else sample_index
        for offset in range(horizon):
            if preserve_original_time_mask and not bool(old_time_mask[sample_index, offset].item()):
                continue
            target_index: int | None
            if index_by_step is None:
                target_index = sample_index + offset + target_offset
            else:
                target_index = index_by_step.get(source_step + offset + target_offset)
            if target_index is None or target_index < 0 or target_index >= sample_count:
                missing += 1
                continue
            targets[sample_index, offset] = states[target_index]
            target_mask[sample_index, offset] = True
            filled += 1

    return targets, target_mask, {
        "filled": filled,
        "missing": missing,
        "duplicate_steps": duplicate_steps,
    }


def output_name_for_item(item: dict[str, Any], used_names: set[str]) -> str:
    source = Path(str(item["_resolved_path"]))
    name = source.name
    if name not in used_names:
        used_names.add(name)
        return name
    stem = source.stem
    suffix = source.suffix
    line_number = int(item["_line_number"])
    name = f"{stem}_line{line_number}{suffix}"
    used_names.add(name)
    return name


def convert_pack(
    pack: dict[str, Any],
    *,
    target_offset: int,
    preserve_original_time_mask: bool,
) -> tuple[dict[str, Any], dict[str, int]]:
    states = torch.as_tensor(pack["state"], dtype=torch.float32)
    if "action_time_mask" in pack:
        old_time_mask = torch.as_tensor(pack["action_time_mask"], dtype=torch.bool)
    elif "actions" in pack:
        old_actions = torch.as_tensor(pack["actions"])
        old_time_mask = torch.ones(old_actions.shape[:2], dtype=torch.bool)
    else:
        raise KeyError("Pack has neither action_time_mask nor actions")

    step_indices = parse_step_indices(pack.get("sample_step_idx"), int(states.shape[0]))
    targets, target_mask, counts = build_target_state_horizon(
        states,
        old_time_mask,
        step_indices=step_indices,
        target_offset=target_offset,
        preserve_original_time_mask=preserve_original_time_mask,
    )
    converted = dict(pack)
    converted["actions"] = targets
    converted["action_time_mask"] = target_mask
    converted["action_target_layout"] = "absolute_target_state"
    converted["target_state_offset"] = int(target_offset)
    converted["target_state_preserved_original_time_mask"] = bool(
        preserve_original_time_mask
    )
    return converted, counts


def convert_manifest(
    input_manifest: Path,
    output_manifest: Path,
    *,
    target_offset: int,
    preserve_original_time_mask: bool,
    copy_unsupported: bool,
) -> dict[str, int]:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    counts = {
        "rows": 0,
        "converted": 0,
        "copied_unsupported": 0,
        "filled_targets": 0,
        "missing_targets": 0,
        "duplicate_step_rows": 0,
    }

    with output_manifest.open("w", encoding="utf-8") as out:
        for item in read_manifest(input_manifest):
            counts["rows"] += 1
            layout = str(item.get("cache_layout", "sample"))
            source_path = Path(item["_resolved_path"])
            output_name = output_name_for_item(item, used_names)
            output_path = output_manifest.parent / output_name
            if output_path.resolve() == source_path.resolve():
                raise ValueError(
                    f"Refusing to overwrite source cache file: {source_path}"
                )
            manifest_item = {
                key: value
                for key, value in item.items()
                if not key.startswith("_")
            }
            manifest_item["path"] = output_name

            if layout not in {"episode_pack", "sample_shard"}:
                if not copy_unsupported:
                    raise ValueError(
                        f"Unsupported cache_layout={layout!r} at {input_manifest}:"
                        f"{item['_line_number']}; use --copy-unsupported to pass it through"
                    )
                shutil.copy2(source_path, output_path)
                counts["copied_unsupported"] += 1
            else:
                pack = torch.load(source_path, map_location="cpu", weights_only=True)
                converted, pack_counts = convert_pack(
                    pack,
                    target_offset=target_offset,
                    preserve_original_time_mask=preserve_original_time_mask,
                )
                converted["source_action_cache_path"] = str(source_path)
                torch.save(converted, output_path)
                manifest_item["action_target_layout"] = "absolute_target_state"
                manifest_item["target_state_offset"] = int(target_offset)
                counts["converted"] += 1
                counts["filled_targets"] += pack_counts["filled"]
                counts["missing_targets"] += pack_counts["missing"]
                counts["duplicate_step_rows"] += pack_counts["duplicate_steps"]

            out.write(json.dumps(manifest_item) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite cached action horizons to absolute target-state horizons "
            "using the cached per-sample state sequence."
        )
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--target-offset",
        type=int,
        default=1,
        help="Use state[t + horizon_index + target_offset] as the target.",
    )
    parser.add_argument(
        "--ignore-original-time-mask",
        action="store_true",
        help="Use every reconstructable target state even if the old action mask was false.",
    )
    parser.add_argument(
        "--copy-unsupported",
        action="store_true",
        help="Copy sample-record rows through instead of failing.",
    )
    args = parser.parse_args()

    counts = convert_manifest(
        args.input_manifest.expanduser().resolve(),
        args.output_manifest.expanduser().resolve(),
        target_offset=args.target_offset,
        preserve_original_time_mask=not args.ignore_original_time_mask,
        copy_unsupported=args.copy_unsupported,
    )
    print("Target-state cache materialized")
    print(f"  input:  {args.input_manifest.expanduser().resolve()}")
    print(f"  output: {args.output_manifest.expanduser().resolve()}")
    for key, value in counts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
