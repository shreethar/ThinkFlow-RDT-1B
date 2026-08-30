#!/usr/bin/env python3
"""Diagnose Fractal translation targets and SimplerEnv rollout execution.

The cache mode needs no simulator. The rollout mode needs only completed
trajectory.jsonl files, not model checkpoints or feature caches.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def load_action_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    normalization = audit["action_normalization"]
    return (
        np.asarray(normalization["q01"], dtype=np.float64),
        np.asarray(normalization["q99"], dtype=np.float64),
    )


def denormalize(values: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    return (values + 1.0) * 0.5 * (q99 - q01) + q01


def cache_paths(manifest: Path, max_packs: int, seed: int) -> list[Path]:
    paths: list[Path] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("dataset_id") == "fractal":
                paths.append(Path(row["path"]))
    if max_packs > 0 and len(paths) > max_packs:
        random.Random(seed).shuffle(paths)
        paths = paths[:max_packs]
    return paths


def analyze_cache(args: argparse.Namespace) -> dict[str, Any]:
    q01, q99 = load_action_stats(args.action_stats)
    paths = cache_paths(args.manifest, args.max_packs, args.seed)
    normalized_rows: list[np.ndarray] = []
    decoded_rows: list[np.ndarray] = []
    first10_normalized: list[np.ndarray] = []
    invalid_abs_max = 0.0
    sample_count = 0
    valid_count = 0
    invalid_count = 0
    for index, path in enumerate(paths, start=1):
        pack = torch.load(path, map_location="cpu", weights_only=False)
        if pack.get("dataset_id") != "fractal":
            continue
        actions = np.asarray(pack["actions"], dtype=np.float64)
        mask = np.asarray(pack["action_time_mask"], dtype=bool)
        if actions.ndim != 3 or actions.shape[-1] != 7:
            raise ValueError(f"Unexpected actions shape {actions.shape} in {path}")
        sample_count += actions.shape[0]
        valid = actions[mask]
        invalid = actions[~mask]
        valid_count += valid.shape[0]
        invalid_count += invalid.shape[0]
        if invalid.size:
            invalid_abs_max = max(invalid_abs_max, float(np.abs(invalid).max()))
        normalized_rows.append(valid[:, :3])
        decoded_rows.append(denormalize(valid, q01, q99)[:, :3])
        horizon_mask = mask.copy()
        horizon_mask[:, 10:] = False
        first10_normalized.append(actions[:, :, :3][horizon_mask])
        if index % 250 == 0:
            print(f"[cache] loaded {index}/{len(paths)} packs", flush=True)

    normalized_xyz = np.concatenate(normalized_rows, axis=0)
    decoded_xyz = np.concatenate(decoded_rows, axis=0)
    first10_xyz = np.concatenate(first10_normalized, axis=0)
    decoded_first10 = denormalize(
        np.pad(first10_xyz, ((0, 0), (0, 4))), q01, q99
    )[:, :3]
    renormalized = 2.0 * (decoded_xyz - q01[:3]) / (q99[:3] - q01[:3]) - 1.0
    roundtrip_error = renormalized - normalized_xyz
    magnitude = np.linalg.norm(decoded_xyz, axis=1)
    first10_magnitude = np.linalg.norm(decoded_first10, axis=1)
    covariance = np.cov(decoded_xyz, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    result = {
        "mode": "cache",
        "manifest": args.manifest,
        "action_stats": args.action_stats,
        "packs_analyzed": len(paths),
        "samples": sample_count,
        "valid_action_rows": valid_count,
        "padded_action_rows": invalid_count,
        "padded_action_abs_max": invalid_abs_max,
        "q01_xyz": q01[:3],
        "q99_xyz": q99[:3],
        "normalized_xyz_per_axis": {
            axis: distribution(normalized_xyz[:, i])
            for i, axis in enumerate(("x", "y", "z"))
        },
        "decoded_xyz_per_axis": {
            axis: distribution(decoded_xyz[:, i])
            for i, axis in enumerate(("x", "y", "z"))
        },
        "decoded_translation_magnitude": distribution(magnitude),
        "decoded_first10_translation_magnitude": distribution(first10_magnitude),
        "normalized_saturation_fraction_per_axis": {
            axis: float(np.mean(np.abs(normalized_xyz[:, i]) >= 0.999))
            for i, axis in enumerate(("x", "y", "z"))
        },
        "near_zero_decoded_translation_fraction": float(np.mean(magnitude < 1e-4)),
        "normalization_roundtrip_max_abs_error": float(np.abs(roundtrip_error).max()),
        "xyz_covariance": covariance,
        "xyz_covariance_eigenvalues": eigenvalues,
        "warning": (
            "Decoded values are Fractal controller commands, not metres of achieved "
            "TCP motion. Compare them with achieved_tcp_delta_xyz_rpy in rollout mode."
        ),
    }
    write_json(args.output, result)
    print(json.dumps(jsonable(result), indent=2))
    return result


def choose_object_position(
    row: dict[str, Any], *, before: bool = False
) -> np.ndarray | None:
    key_name = "object_states_before" if before else "object_states_after"
    states = row.get(key_name) or row.get("object_states_after") or {}
    for key in ("episode_source_obj", "source_obj", "obj", "episode_target_obj", "target_obj"):
        item = states.get(key)
        if item and "base_position" in item:
            return np.asarray(item["base_position"], dtype=np.float64)
    return None


def safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def linear_tracking(requested: np.ndarray, achieved: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for index, axis in enumerate(("x", "y", "z")):
        x = requested[:, index]
        y = achieved[:, index]
        denominator = float(np.dot(x, x))
        slope = float(np.dot(x, y) / denominator) if denominator > 1e-12 else None
        residual = y - x
        output[axis] = {
            "correlation": safe_corr(x, y),
            "achieved_per_requested_slope_through_origin": slope,
            "mean_tracking_error": float(residual.mean()),
            "tracking_rmse": float(np.sqrt(np.mean(residual**2))),
        }
    return output


def analyze_rollouts(args: argparse.Namespace) -> dict[str, Any]:
    trajectory_paths = sorted(args.rollout_root.rglob("trajectory.jsonl"))
    requested: list[np.ndarray] = []
    executed: list[np.ndarray] = []
    achieved: list[np.ndarray] = []
    direction_cosines: list[float] = []
    distance_progress: list[float] = []
    per_episode: list[dict[str, Any]] = []
    for path in trajectory_paths:
        episode_requested: list[np.ndarray] = []
        episode_achieved: list[np.ndarray] = []
        episode_cosines: list[float] = []
        episode_distances: list[float] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                req = np.asarray(row["requested_action"], dtype=np.float64)[:3]
                exe = np.asarray(row["executed_action"], dtype=np.float64)[:3]
                ach = np.asarray(row["achieved_tcp_delta_xyz_rpy"], dtype=np.float64)[:3]
                requested.append(req)
                executed.append(exe)
                achieved.append(ach)
                episode_requested.append(req)
                episode_achieved.append(ach)
                object_before = choose_object_position(row, before=True)
                object_after = choose_object_position(row)
                if object_before is None or object_after is None:
                    continue
                tcp_before = np.asarray(row["state_before_7d"], dtype=np.float64)[:3]
                tcp_after = np.asarray(row["state_after_7d"], dtype=np.float64)[:3]
                target = object_before - tcp_before
                denominator = float(np.linalg.norm(req) * np.linalg.norm(target))
                if denominator > 1e-12:
                    cosine = float(np.dot(req, target) / denominator)
                    direction_cosines.append(cosine)
                    episode_cosines.append(cosine)
                before_distance = float(np.linalg.norm(object_before - tcp_before))
                after_distance = float(np.linalg.norm(object_after - tcp_after))
                progress = before_distance - after_distance
                distance_progress.append(progress)
                episode_distances.append(after_distance)
        summary_path = path.with_name("summary.json")
        summary = {}
        if summary_path.is_file():
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
        per_episode.append(
            {
                "trajectory": path,
                "task": summary.get("task"),
                "seed": summary.get("seed"),
                "success": bool(summary.get("success", False)),
                "steps": len(episode_requested),
                "mean_requested_magnitude": float(
                    np.linalg.norm(np.asarray(episode_requested), axis=1).mean()
                ) if episode_requested else None,
                "mean_achieved_magnitude": float(
                    np.linalg.norm(np.asarray(episode_achieved), axis=1).mean()
                ) if episode_achieved else None,
                "mean_object_direction_cosine": float(np.mean(episode_cosines))
                if episode_cosines else None,
                "minimum_tcp_object_distance": float(min(episode_distances))
                if episode_distances else None,
            }
        )

    if not requested:
        raise ValueError(f"No trajectory rows found beneath {args.rollout_root}")
    requested_array = np.stack(requested)
    executed_array = np.stack(executed)
    achieved_array = np.stack(achieved)
    result = {
        "mode": "rollouts",
        "rollout_root": args.rollout_root,
        "episodes": len(per_episode),
        "action_rows": len(requested),
        "requested_vs_executed_max_abs_difference": float(
            np.abs(requested_array - executed_array).max()
        ),
        "clipped_action_row_fraction": float(
            np.mean(np.any(np.abs(requested_array - executed_array) > 1e-7, axis=1))
        ),
        "requested_vs_achieved": linear_tracking(requested_array, achieved_array),
        "requested_translation_magnitude": distribution(
            np.linalg.norm(requested_array, axis=1)
        ),
        "achieved_translation_magnitude": distribution(
            np.linalg.norm(achieved_array, axis=1)
        ),
        "object_direction_cosine": distribution(np.asarray(direction_cosines)),
        "tcp_object_distance_progress_per_step": distribution(
            np.asarray(distance_progress)
        ),
        "per_episode": per_episode,
    }
    write_json(args.output, result)
    print(json.dumps(jsonable(result), indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cache", "rollouts"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--action-stats", type=Path)
    parser.add_argument("--max-packs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout-root", type=Path)
    args = parser.parse_args()
    if args.mode == "cache":
        if args.manifest is None or args.action_stats is None:
            parser.error("cache mode requires --manifest and --action-stats")
    elif args.rollout_root is None:
        parser.error("rollouts mode requires --rollout-root")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "cache":
        analyze_cache(args)
    else:
        analyze_rollouts(args)


if __name__ == "__main__":
    main()
