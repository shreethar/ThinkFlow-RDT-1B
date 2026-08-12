"""Streaming mean/std statistics for cached LIBERO state and action tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_STATE

from .cached_libero import add_native_libero_tensors


class _Moments:
    def __init__(self, width: int) -> None:
        self.total = torch.zeros(width, dtype=torch.float64)
        self.total_squared = torch.zeros(width, dtype=torch.float64)
        self.count = torch.zeros(width, dtype=torch.float64)

    def update(self, values: Tensor, valid: Tensor | None = None) -> None:
        values = values.detach().double().reshape(-1, values.shape[-1])
        if valid is None:
            valid_flat = torch.ones(values.shape[0], dtype=torch.bool)
        else:
            valid_flat = valid.detach().bool().reshape(-1)
            if valid_flat.shape[0] != values.shape[0]:
                raise ValueError("Moment values and validity mask have different lengths")
        selected = values[valid_flat]
        if selected.numel() == 0:
            return
        self.total += selected.sum(dim=0)
        self.total_squared += selected.square().sum(dim=0)
        self.count += selected.shape[0]

    def finish(self) -> dict[str, Tensor]:
        denominator = self.count.clamp_min(1)
        mean = self.total / denominator
        variance = self.total_squared / denominator - mean.square()
        std = variance.clamp_min(1e-12).sqrt()
        return {"mean": mean.float(), "std": std.float()}


def compute_cache_stats(
    shard_paths: Sequence[str | Path],
    *,
    chunk_size: int = 50,
) -> tuple[dict[str, dict[str, Tensor]], int]:
    """Compute the exact statistics used by SmolVLA's MEAN_STD processor."""

    state_moments = _Moments(8)
    action_moments = _Moments(7)
    num_samples = 0
    for path in shard_paths:
        pack = torch.load(path, map_location="cpu", weights_only=False)
        add_native_libero_tensors(pack)
        states = torch.as_tensor(pack["_smolvla_state"], dtype=torch.float32)
        actions = torch.as_tensor(pack["_smolvla_actions"], dtype=torch.float32)[:, :chunk_size]
        valid = torch.as_tensor(pack["action_time_mask"], dtype=torch.bool)[:, :chunk_size]
        if states.shape[-1] != 8 or actions.shape[-1] != 7:
            raise ValueError(
                f"Unexpected dimensions in {path}: state={states.shape}, actions={actions.shape}"
            )
        state_moments.update(states)
        action_moments.update(actions, valid)
        num_samples += states.shape[0]
        del pack
    return (
        {
            OBS_STATE: state_moments.finish(),
            ACTION: action_moments.finish(),
        },
        num_samples,
    )


def load_or_compute_cache_stats(
    stats_path: str | Path,
    shard_paths: Sequence[str | Path],
    *,
    chunk_size: int = 50,
) -> tuple[dict[str, dict[str, Tensor]], int]:
    path = Path(stats_path)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("schema") != "libero_native_state8_action7_v1"
            or "stats" not in payload
            or "num_samples" not in payload
        ):
            raise ValueError(
                f"Legacy or invalid stats file {path}; remove it once so the new cache summary can be computed"
            )
        stats = payload["stats"]
        num_samples = int(payload["num_samples"])
    else:
        stats, num_samples = compute_cache_stats(shard_paths, chunk_size=chunk_size)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "libero_native_state8_action7_v1",
                "stats": stats,
                "num_samples": num_samples,
            },
            path,
        )
    return stats, num_samples
