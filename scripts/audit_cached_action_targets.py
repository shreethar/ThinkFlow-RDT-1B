#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thinkflow_rdt.adapters.action_stats import (  # noqa: E402
    ActionNormalizationStats,
    denormalize_action_array,
)


@dataclass
class RunningMetric:
    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        if math.isfinite(value):
            self.total += float(value)
            self.count += 1

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")


@dataclass
class AuditStats:
    samples: int = 0
    compared_rows: int = 0
    absolute_same_l1: RunningMetric = field(default_factory=RunningMetric)
    absolute_next_l1: RunningMetric = field(default_factory=RunningMetric)
    delta_from_current_l1: RunningMetric = field(default_factory=RunningMetric)
    action_abs_mean: RunningMetric = field(default_factory=RunningMetric)
    state_abs_mean: RunningMetric = field(default_factory=RunningMetric)


def manifest_items(manifest: Path) -> list[dict[str, Any]]:
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


def numeric_step_indices(values: Any, count: int) -> list[int] | None:
    if values is None:
        return list(range(count))
    output: list[int] = []
    for value in list(values):
        try:
            output.append(int(value))
        except (TypeError, ValueError):
            return None
    if len(output) != count:
        return None
    return output


def contiguous_index_map(step_indices: list[int]) -> dict[int, int] | None:
    mapping = {step: index for index, step in enumerate(step_indices)}
    if len(mapping) != len(step_indices):
        return None
    return mapping


def maybe_denormalize(
    actions: torch.Tensor,
    stats: ActionNormalizationStats | None,
) -> torch.Tensor:
    if stats is None:
        return actions.float()
    array = denormalize_action_array(actions.float().cpu().numpy(), stats)
    return torch.as_tensor(array, dtype=torch.float32)


def audit_pack(
    pack: dict[str, Any],
    stats: AuditStats,
    *,
    action_stats: ActionNormalizationStats | None,
    max_samples: int | None,
    pose_dims: int,
) -> None:
    state = torch.as_tensor(pack["state"], dtype=torch.float32)
    actions = maybe_denormalize(torch.as_tensor(pack["actions"]), action_stats)
    time_mask = torch.as_tensor(
        pack.get("action_time_mask", torch.ones(actions.shape[:2])),
        dtype=torch.bool,
    )
    if state.ndim != 2 or actions.ndim != 3:
        raise ValueError(f"Expected pack state [N,D] and actions [N,H,D], got {state.shape} and {actions.shape}")
    if state.shape[0] != actions.shape[0] or state.shape[1] != actions.shape[2]:
        raise ValueError(f"State/action shape mismatch: {state.shape} vs {actions.shape}")

    sample_count = int(state.shape[0])
    step_indices = numeric_step_indices(pack.get("sample_step_idx"), sample_count)
    index_by_step = contiguous_index_map(step_indices) if step_indices is not None else None
    limit = sample_count if max_samples is None else min(sample_count, max_samples)
    dims = min(pose_dims, int(state.shape[1]))

    for sample_index in range(limit):
        stats.samples += 1
        current = state[sample_index, :dims]
        first_action = actions[sample_index, 0, :dims]
        stats.action_abs_mean.add(float(first_action.abs().mean().item()))
        stats.state_abs_mean.add(float(current.abs().mean().item()))
        horizon = int(actions.shape[1])
        for offset in range(horizon):
            if not bool(time_mask[sample_index, offset].item()):
                continue
            action_row = actions[sample_index, offset, :dims]
            same_index = sample_index + offset
            next_index = sample_index + offset + 1
            if same_index < sample_count:
                stats.absolute_same_l1.add(
                    float((action_row - state[same_index, :dims]).abs().mean().item())
                )
            if next_index < sample_count:
                stats.absolute_next_l1.add(
                    float((action_row - state[next_index, :dims]).abs().mean().item())
                )
            if index_by_step is not None and step_indices is not None:
                target_step = step_indices[sample_index] + offset + 1
                mapped = index_by_step.get(target_step)
                if mapped is not None:
                    predicted_next = current + action_row
                    stats.delta_from_current_l1.add(
                        float((predicted_next - state[mapped, :dims]).abs().mean().item())
                    )
            stats.compared_rows += 1


def load_action_stats(path: Path | None) -> ActionNormalizationStats | None:
    if path is None:
        return None
    with path.expanduser().open("r", encoding="utf-8") as handle:
        return ActionNormalizationStats.from_mapping(json.load(handle))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit cached RDT actions against cached state sequences. Lower "
            "absolute_* means actions look like absolute target states; lower "
            "delta_from_current means they look like one-step deltas."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--action-stats", type=Path)
    parser.add_argument("--max-packs", type=int, default=None)
    parser.add_argument("--max-samples-per-pack", type=int, default=None)
    parser.add_argument(
        "--pose-dims",
        type=int,
        default=6,
        help="Compare only pose dimensions by default; gripper conventions can dominate L1.",
    )
    args = parser.parse_args()

    action_stats = load_action_stats(args.action_stats)
    stats = AuditStats()
    layouts: dict[str, int] = {}
    packs_seen = 0
    unsupported = 0
    for item in manifest_items(args.manifest):
        if args.max_packs is not None and packs_seen >= args.max_packs:
            break
        layout = str(item.get("cache_layout", "sample"))
        layouts[layout] = layouts.get(layout, 0) + 1
        if layout not in {"episode_pack", "sample_shard"}:
            unsupported += 1
            continue
        pack = torch.load(item["_resolved_path"], map_location="cpu", weights_only=True)
        audit_pack(
            pack,
            stats,
            action_stats=action_stats,
            max_samples=args.max_samples_per_pack,
            pose_dims=args.pose_dims,
        )
        packs_seen += 1

    print("Cached action target audit")
    print(f"  manifest: {args.manifest.expanduser().resolve()}")
    print(f"  layouts: {layouts}")
    print(f"  packs audited: {packs_seen}")
    print(f"  unsupported manifest rows skipped: {unsupported}")
    print(f"  samples audited: {stats.samples}")
    print(f"  compared horizon rows: {stats.compared_rows}")
    print(f"  mean |action|: {stats.action_abs_mean.mean:.6f}")
    print(f"  mean |state|: {stats.state_abs_mean.mean:.6f}")
    print(f"  L1 action vs state[t+h]: {stats.absolute_same_l1.mean:.6f}")
    print(f"  L1 action vs state[t+h+1]: {stats.absolute_next_l1.mean:.6f}")
    print(f"  L1 state[t] + action vs state[t+h+1]: {stats.delta_from_current_l1.mean:.6f}")
    print()
    print("Interpretation:")
    print("  smallest absolute_* score => actions resemble absolute target states")
    print("  smallest delta_from_current score => actions resemble relative deltas")
    if action_stats is None:
        print("  note: no --action-stats was provided; normalized caches may need denormalization")


if __name__ == "__main__":
    main()
