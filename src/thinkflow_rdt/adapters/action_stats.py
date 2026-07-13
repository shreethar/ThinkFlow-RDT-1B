from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ACTION_DIM_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_rx",
    "delta_ry",
    "delta_rz",
    "gripper_closed",
]


@dataclass(frozen=True)
class ActionNormalizationStats:
    q01: np.ndarray
    q99: np.ndarray
    eps: float = 1e-6

    def __post_init__(self) -> None:
        q01 = np.asarray(self.q01, dtype=np.float32)
        q99 = np.asarray(self.q99, dtype=np.float32)
        if q01.shape != q99.shape:
            raise ValueError(f"q01 and q99 shapes differ: {q01.shape} vs {q99.shape}")
        if q01.ndim != 1:
            raise ValueError(f"Expected 1D stats arrays, got {q01.shape}")
        object.__setattr__(self, "q01", q01)
        object.__setattr__(self, "q99", q99)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> ActionNormalizationStats:
        block = mapping.get("action_normalization", mapping)
        if "q01" not in block or "q99" not in block:
            raise KeyError("Action stats mapping must contain q01 and q99")
        return cls(
            q01=np.asarray(block["q01"], dtype=np.float32),
            q99=np.asarray(block["q99"], dtype=np.float32),
            eps=float(block.get("eps", 1e-6)),
        )

    def to_audit_block(
        self,
        *,
        source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        block: dict[str, Any] = {
            "method": "clip each standardized action dimension to q01/q99, then linearly map to [-1, 1]",
            "dim_names": ACTION_DIM_NAMES[: len(self.q01)],
            "q01": self.q01.astype(float).tolist(),
            "q99": self.q99.astype(float).tolist(),
            "eps": self.eps,
        }
        if source is not None:
            block["source"] = source
        return block


def compute_action_quantile_stats(
    actions: np.ndarray,
    *,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> ActionNormalizationStats:
    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.ndim != 2:
        raise ValueError(f"Expected actions [N, D], got {action_array.shape}")
    if action_array.shape[0] == 0:
        raise ValueError("Cannot compute action stats from zero actions")
    q01 = np.quantile(action_array, q_low, axis=0).astype(np.float32)
    q99 = np.quantile(action_array, q_high, axis=0).astype(np.float32)
    return ActionNormalizationStats(q01=q01, q99=q99)


def normalize_action_array(
    actions: np.ndarray,
    stats: ActionNormalizationStats,
) -> np.ndarray:
    action_array = np.asarray(actions, dtype=np.float32)
    scale = stats.q99 - stats.q01
    clipped = np.clip(action_array, stats.q01, stats.q99)
    normalized = np.zeros_like(clipped, dtype=np.float32)
    valid_dims = np.abs(scale) > stats.eps
    normalized[..., valid_dims] = (
        2.0 * (clipped[..., valid_dims] - stats.q01[valid_dims]) / scale[valid_dims]
        - 1.0
    )
    return normalized.astype(np.float32)


def denormalize_action_array(
    normalized_actions: np.ndarray,
    stats: ActionNormalizationStats,
) -> np.ndarray:
    normalized = np.asarray(normalized_actions, dtype=np.float32)
    clipped = np.clip(normalized, -1.0, 1.0)
    scale = stats.q99 - stats.q01
    actions = np.broadcast_to(stats.q01, clipped.shape).astype(np.float32).copy()
    valid_dims = np.abs(scale) > stats.eps
    actions[..., valid_dims] = (
        (clipped[..., valid_dims] + 1.0) * 0.5 * scale[valid_dims]
        + stats.q01[valid_dims]
    )
    return actions.astype(np.float32)


def normalize_action_horizon(
    actions: np.ndarray,
    actions_mask: np.ndarray,
    stats: ActionNormalizationStats,
) -> np.ndarray:
    action_array = np.asarray(actions, dtype=np.float32).copy()
    mask = np.asarray(actions_mask, dtype=np.float32) > 0.0
    if action_array.shape[0] != mask.shape[0]:
        raise ValueError(
            f"actions and mask horizon differ: {action_array.shape[0]} vs {mask.shape[0]}"
        )
    if np.any(mask):
        action_array[mask] = normalize_action_array(action_array[mask], stats)
    return action_array


def load_action_stats(path: str | Path) -> ActionNormalizationStats:
    stats_path = Path(path).expanduser().resolve()
    with stats_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return ActionNormalizationStats.from_mapping(data)


def write_action_stats_to_audit(
    audit_path: str | Path,
    stats: ActionNormalizationStats,
    *,
    source: dict[str, Any] | None = None,
) -> None:
    path = Path(audit_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    audit["action_normalization"] = stats.to_audit_block(source=source)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=4)
        handle.write("\n")


def find_audit_json(start: str | Path) -> Path | None:
    current = Path(start).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        candidate = directory / "audit.json"
        if candidate.exists():
            return candidate
    return None


def resolve_action_stats(
    *,
    normalize_actions: bool,
    action_stats: ActionNormalizationStats | dict[str, Any] | None = None,
    action_stats_path: str | Path | None = None,
    search_dir: str | Path | None = None,
) -> ActionNormalizationStats | None:
    if not normalize_actions:
        return None
    if action_stats is not None:
        if isinstance(action_stats, ActionNormalizationStats):
            return action_stats
        return ActionNormalizationStats.from_mapping(action_stats)
    if action_stats_path is None and search_dir is not None:
        action_stats_path = find_audit_json(search_dir)
    if action_stats_path is None:
        raise ValueError(
            "normalize_actions=True requires action_stats, action_stats_path, "
            "or an audit.json discoverable from data_dir"
        )
    return load_action_stats(action_stats_path)
