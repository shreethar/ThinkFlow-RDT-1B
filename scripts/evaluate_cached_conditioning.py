#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thinkflow_rdt.checkpoint import load_trainable_artifact  # noqa: E402
from thinkflow_rdt.config import ExperimentConfig, load_config  # noqa: E402
from thinkflow_rdt.data import (  # noqa: E402
    ONLINE_SIGLIP_REQUIRED_KEYS,
    CachedFeatureDataset,
    RDTBatchCollator,
    RDTOnlineSiglipBatchCollator,
)
from thinkflow_rdt.model import SFTConditionedRDT  # noqa: E402
from thinkflow_rdt.train import add_online_siglip_features, load_online_siglip  # noqa: E402

DEFAULT_LOSS_VARIANTS = (
    "baseline",
    "zero_qwen",
    "shuffle_qwen",
    "cross_dataset_qwen",
    "zero_lang",
    "shuffle_lang",
    "zero_image",
    "shuffle_image",
    "zero_state",
    "zero_ctrl_freq",
)
DEFAULT_SAMPLE_VARIANTS = (
    "baseline",
    "zero_qwen",
    "shuffle_qwen",
    "cross_dataset_qwen",
)


def json_default(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def resolve_manifest_line_path(
    item: object,
    *,
    manifest_dir: Path,
) -> tuple[object, Path]:
    if isinstance(item, str):
        path = Path(item)
        resolved = path if path.is_absolute() else (manifest_dir / path).resolve()
        return str(resolved), resolved
    if not isinstance(item, dict):
        raise TypeError(f"Manifest line must be a JSON string/object, got {type(item)}")
    path_value = item.get("path")
    if not path_value:
        raise ValueError(f"Manifest object has no path: {item}")
    path = Path(str(path_value))
    resolved = path if path.is_absolute() else (manifest_dir / path).resolve()
    rewritten = dict(item)
    rewritten["path"] = str(resolved)
    return rewritten, resolved


def merge_manifests(input_manifests: list[Path], output_manifest: Path) -> int:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_manifest.open("w", encoding="utf-8") as out:
        for manifest in input_manifests:
            manifest = manifest.expanduser().resolve()
            if not manifest.exists():
                raise FileNotFoundError(manifest)
            with manifest.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    item = json.loads(stripped)
                    rewritten, resolved = resolve_manifest_line_path(
                        item,
                        manifest_dir=manifest.parent,
                    )
                    if not resolved.exists():
                        raise FileNotFoundError(
                            f"{manifest}:{line_number} points to missing cache file {resolved}"
                        )
                    out.write(json.dumps(rewritten) + "\n")
                    count += 1
    return count


def manifests_from_cache_roots(cache_roots: list[Path], *, split: str) -> list[Path]:
    manifests = []
    for root in cache_roots:
        manifest = root.expanduser().resolve() / split / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        manifests.append(manifest)
    return manifests


def manifests_from_cache_parts_root(parts_root: Path, *, parts: list[int], split: str) -> list[Path]:
    root = parts_root.expanduser().resolve()
    manifests = []
    for part in parts:
        manifest = root / f"part_{int(part)}" / split / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        manifests.append(manifest)
    return manifests


def selected_indices(dataset_size: int, *, count: int, seed: int, mode: str) -> list[int]:
    count = min(max(0, count), dataset_size)
    if mode == "first":
        return list(range(count))
    if mode == "even":
        if count <= 1:
            return [0] if count == 1 else []
        return [round(i * (dataset_size - 1) / (count - 1)) for i in range(count)]
    rng = random.Random(seed)
    return sorted(rng.sample(range(dataset_size), count))


def move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def roll_or_zero(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] <= 1:
        return torch.zeros_like(tensor)
    return torch.roll(tensor, shifts=1, dims=0)


def sample_dataset_id(sample: dict[str, Any]) -> str:
    dataset_id = sample.get("dataset_id")
    if dataset_id is None:
        return "<unknown>"
    return str(dataset_id)


def build_cross_dataset_qwen_replacements(
    dataset: CachedFeatureDataset,
    indices: list[int],
    *,
    seed: int,
) -> dict[str, Any]:
    """Preselect mismatched Qwen tensors from a different dataset for each index."""
    qwen_by_dataset: dict[str, list[tuple[int, torch.Tensor]]] = {}
    source_dataset_by_index: dict[int, str] = {}
    for index in indices:
        sample = dataset[index]
        dataset_id = sample_dataset_id(sample)
        qwen_kv = torch.as_tensor(sample["qwen_kv"]).detach().cpu()
        if qwen_kv.ndim == 1:
            qwen_kv = qwen_kv.unsqueeze(0)
        if qwen_kv.ndim != 2:
            raise ValueError(
                f"Expected qwen_kv [tokens, dim] for index {index}, got {tuple(qwen_kv.shape)}"
            )
        qwen_by_dataset.setdefault(dataset_id, []).append((index, qwen_kv.clone()))
        source_dataset_by_index[index] = dataset_id

    rng = random.Random(seed)
    replacements: dict[int, torch.Tensor] = {}
    replacement_dataset_by_index: dict[int, str] = {}
    pair_counts: dict[str, int] = {}
    dataset_ids = sorted(qwen_by_dataset)
    for index in indices:
        source_dataset_id = source_dataset_by_index[index]
        candidate_datasets = [
            dataset_id for dataset_id in dataset_ids if dataset_id != source_dataset_id
        ]
        if not candidate_datasets:
            continue
        replacement_dataset_id = rng.choice(candidate_datasets)
        _, replacement_qwen = rng.choice(qwen_by_dataset[replacement_dataset_id])
        replacements[index] = replacement_qwen.clone()
        replacement_dataset_by_index[index] = replacement_dataset_id
        pair_key = f"{source_dataset_id}->{replacement_dataset_id}"
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

    return {
        "replacements": replacements,
        "source_dataset_by_index": source_dataset_by_index,
        "replacement_dataset_by_index": replacement_dataset_by_index,
        "summary": {
            "selected_indices": len(indices),
            "replaced_indices": len(replacements),
            "fallback_indices": len(indices) - len(replacements),
            "source_dataset_counts": {
                dataset_id: len(values)
                for dataset_id, values in sorted(qwen_by_dataset.items())
            },
            "replacement_pair_counts": dict(sorted(pair_counts.items())),
        },
    }


def cross_dataset_qwen_for_batch(
    cross_dataset_qwen: dict[str, Any],
    batch_indices: list[int],
    reference: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    tensors = []
    missing = 0
    replacements: dict[int, torch.Tensor] = cross_dataset_qwen["replacements"]
    for row, index in enumerate(batch_indices):
        replacement = replacements.get(index)
        if replacement is None:
            tensors.append(torch.zeros_like(reference[row]))
            missing += 1
        else:
            tensors.append(
                replacement.to(device=reference.device, dtype=reference.dtype)
            )
    return torch.stack(tensors, dim=0), missing


def apply_variant(
    batch: dict[str, Any],
    variant: str,
    *,
    cross_dataset_qwen_batch: torch.Tensor | None = None,
) -> dict[str, Any]:
    out = clone_batch(batch)
    if variant == "baseline":
        return out
    if variant == "zero_qwen":
        out["qwen_kv"] = torch.zeros_like(out["qwen_kv"])
        return out
    if variant == "shuffle_qwen":
        out["qwen_kv"] = roll_or_zero(out["qwen_kv"])
        return out
    if variant == "cross_dataset_qwen":
        if cross_dataset_qwen_batch is None:
            out["qwen_kv"] = torch.zeros_like(out["qwen_kv"])
            return out
        if cross_dataset_qwen_batch.shape != out["qwen_kv"].shape:
            raise ValueError(
                "cross_dataset_qwen replacement shape "
                f"{tuple(cross_dataset_qwen_batch.shape)} does not match "
                f"batch qwen_kv shape {tuple(out['qwen_kv'].shape)}"
            )
        out["qwen_kv"] = cross_dataset_qwen_batch
        return out
    if variant == "zero_lang":
        out["lang_tokens"] = torch.zeros_like(out["lang_tokens"])
        return out
    if variant == "shuffle_lang":
        out["lang_tokens"] = roll_or_zero(out["lang_tokens"])
        out["lang_mask"] = roll_or_zero(out["lang_mask"])
        return out
    if variant == "zero_image":
        out["img_tokens"] = torch.zeros_like(out["img_tokens"])
        return out
    if variant == "shuffle_image":
        out["img_tokens"] = roll_or_zero(out["img_tokens"])
        out["img_mask"] = roll_or_zero(out["img_mask"])
        return out
    if variant == "zero_state":
        out["state"] = torch.zeros_like(out["state"])
        return out
    if variant == "zero_ctrl_freq":
        out["ctrl_freq"] = torch.zeros_like(out["ctrl_freq"])
        return out
    if variant == "shuffle_all_context":
        out["qwen_kv"] = roll_or_zero(out["qwen_kv"])
        out["lang_tokens"] = roll_or_zero(out["lang_tokens"])
        out["lang_mask"] = roll_or_zero(out["lang_mask"])
        out["img_tokens"] = roll_or_zero(out["img_tokens"])
        out["img_mask"] = roll_or_zero(out["img_mask"])
        return out
    raise ValueError(f"Unknown variant: {variant}")


def timestep_ranges(num_timesteps: int, buckets: int) -> list[tuple[int, int]]:
    if buckets <= 0:
        raise ValueError("--timestep-buckets must be positive")
    ranges = []
    for index in range(buckets):
        start = round(index * num_timesteps / buckets)
        stop = round((index + 1) * num_timesteps / buckets)
        ranges.append((start, max(start + 1, stop)))
    return ranges


def sample_timesteps(
    *,
    batch_size: int,
    low: int,
    high: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randint(
        low,
        high,
        (batch_size,),
        device=device,
        dtype=torch.long,
        generator=generator,
    )


def metric_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
    time_mask: torch.Tensor,
    dim_mask: torch.Tensor,
    *,
    action_layout: str,
) -> dict[str, torch.Tensor]:
    valid = time_mask.unsqueeze(-1).to(prediction.dtype) * dim_mask.unsqueeze(1).to(
        prediction.dtype
    )
    raw_diff = prediction - target
    if action_layout == "libero_ortho6d":
        # The six rotation values are a continuous matrix representation, not
        # angles. They must be compared directly rather than wrapped at pi.
        diff = raw_diff
        rotation_slice = slice(3, 9)
        gripper_slice = slice(9, 10)
    else:
        diff = torch.cat(
            [
                raw_diff[..., :3],
                torch.atan2(
                    torch.sin(raw_diff[..., 3:6]),
                    torch.cos(raw_diff[..., 3:6]),
                ),
                raw_diff[..., 6:],
            ],
            dim=-1,
        )
        rotation_slice = slice(3, 6)
        gripper_slice = slice(6, 7)
    squared = diff.pow(2) * valid
    absolute = diff.abs() * valid
    return {
        "loss_sum": squared.sum(),
        "mae_sum": absolute.sum(),
        "valid_count": valid.sum(),
        "xyz_loss_sum": squared[..., :3].sum(),
        "xyz_valid_count": valid[..., :3].sum(),
        "rot_loss_sum": squared[..., rotation_slice].sum(),
        "rot_valid_count": valid[..., rotation_slice].sum(),
        "gripper_loss_sum": squared[..., gripper_slice].sum(),
        "gripper_valid_count": valid[..., gripper_slice].sum(),
    }


@torch.no_grad()
def loss_at_timesteps(
    model: SFTConditionedRDT,
    batch: dict[str, torch.Tensor],
    timesteps: torch.Tensor,
    *,
    noise_seed: int,
) -> dict[str, torch.Tensor]:
    batch = model.cast_batch(batch)
    states = batch["state"].unsqueeze(1)
    actions = batch["actions"]
    time_mask = batch["action_time_mask"].bool()
    action_dim_mask = batch["action_dim_mask"].to(actions.dtype).unsqueeze(1)
    state_dim_mask = batch.get(
        "state_dim_mask",
        torch.ones_like(batch["state"]),
    ).to(states.dtype).unsqueeze(1)
    generator = torch.Generator(device=actions.device)
    generator.manual_seed(noise_seed)
    noise = torch.randn(
        actions.shape,
        device=actions.device,
        dtype=actions.dtype,
        generator=generator,
    )
    noisy_actions = model.runner.noise_scheduler.add_noise(actions, noise, timesteps)
    action_token_mask = (
        action_dim_mask.expand(-1, actions.shape[1], -1)
        * time_mask.unsqueeze(-1).to(actions.dtype)
    )
    noisy_actions = noisy_actions * action_token_mask
    state_input = model._state_encoder_input(states, state_dim_mask)
    state_cond = model.runner.state_adaptor(state_input)
    action_cond = model._adapt_actions(noisy_actions, action_token_mask)
    state_action_cond = torch.cat([state_cond, action_cond], dim=1)
    (
        lang_cond,
        img_cond,
        lang_mask,
        external_kv,
        external_cross_kv,
        extra_cross_cond,
        extra_cross_mask,
    ) = model._adapt_static_conditions(batch, state_cond=state_cond)
    prediction = model.runner.model(
        state_action_cond,
        batch["ctrl_freq"],
        timesteps,
        lang_cond,
        img_cond,
        lang_mask=lang_mask,
        img_mask=batch["img_mask"].bool(),
        external_kv=external_kv,
        external_cross_kv=external_cross_kv,
        extra_cross_cond=extra_cross_cond,
        extra_cross_mask=extra_cross_mask,
        unified_cross_attention=(
            model.cfg.model.qwen_fusion == "unified_cross_attention"
        ),
    )
    return metric_sums(
        prediction.float(),
        actions.float(),
        batch["action_time_mask"],
        batch["action_dim_mask"],
        action_layout=model.cfg.model.action_encoder_layout,
    )


def add_sums(target: dict[str, float], sums: dict[str, torch.Tensor], *, prefix: str = "") -> None:
    for key, value in sums.items():
        target[prefix + key] = target.get(prefix + key, 0.0) + float(value.detach().double().cpu())


def finalize_loss_sums(sums: dict[str, float]) -> dict[str, float]:
    valid = max(sums.get("valid_count", 0.0), 1.0)
    xyz_valid = max(sums.get("xyz_valid_count", 0.0), 1.0)
    rot_valid = max(sums.get("rot_valid_count", 0.0), 1.0)
    gripper_valid = max(sums.get("gripper_valid_count", 0.0), 1.0)
    return {
        "loss": sums.get("loss_sum", 0.0) / valid,
        "target_mae": sums.get("mae_sum", 0.0) / valid,
        "loss_xyz": sums.get("xyz_loss_sum", 0.0) / xyz_valid,
        "loss_rot": sums.get("rot_loss_sum", 0.0) / rot_valid,
        "loss_gripper": sums.get("gripper_loss_sum", 0.0) / gripper_valid,
        "valid_values": sums.get("valid_count", 0.0),
    }


def initialize_horizon_store(horizon: int) -> dict[str, torch.Tensor]:
    return {
        "all_error": torch.zeros(horizon, dtype=torch.float64),
        "all_count": torch.zeros(horizon, dtype=torch.float64),
        "xyz_error": torch.zeros(horizon, dtype=torch.float64),
        "xyz_count": torch.zeros(horizon, dtype=torch.float64),
        "rot_error": torch.zeros(horizon, dtype=torch.float64),
        "rot_count": torch.zeros(horizon, dtype=torch.float64),
        "gripper_error": torch.zeros(horizon, dtype=torch.float64),
        "gripper_count": torch.zeros(horizon, dtype=torch.float64),
    }


def add_horizon_metrics(
    store: dict[str, torch.Tensor],
    prediction: torch.Tensor,
    target: torch.Tensor,
    time_mask: torch.Tensor,
    dim_mask: torch.Tensor,
    *,
    action_layout: str,
) -> None:
    prediction = prediction.detach().float().cpu()
    target = target.detach().float().cpu()
    time_mask = time_mask.detach().bool().cpu()
    dim_mask = dim_mask.detach().float().cpu()
    valid = time_mask.unsqueeze(-1).float() * dim_mask.unsqueeze(1)
    raw_diff = prediction - target
    if action_layout == "libero_ortho6d":
        diff = raw_diff
        rotation_slice = slice(3, 9)
        gripper_slice = slice(9, 10)
    else:
        diff = torch.cat(
            [
                raw_diff[..., :3],
                torch.atan2(
                    torch.sin(raw_diff[..., 3:6]),
                    torch.cos(raw_diff[..., 3:6]),
                ),
                raw_diff[..., 6:],
            ],
            dim=-1,
        )
        rotation_slice = slice(3, 6)
        gripper_slice = slice(6, 7)
    squared = diff.pow(2) * valid
    store["all_error"] += squared.sum(dim=(0, 2)).double()
    store["all_count"] += valid.sum(dim=(0, 2)).double()
    store["xyz_error"] += squared[..., :3].sum(dim=(0, 2)).double()
    store["xyz_count"] += valid[..., :3].sum(dim=(0, 2)).double()
    store["rot_error"] += squared[..., rotation_slice].sum(dim=(0, 2)).double()
    store["rot_count"] += valid[..., rotation_slice].sum(dim=(0, 2)).double()
    store["gripper_error"] += squared[..., gripper_slice].sum(dim=(0, 2)).double()
    store["gripper_count"] += valid[..., gripper_slice].sum(dim=(0, 2)).double()


def finalize_horizon_metrics(store: dict[str, torch.Tensor]) -> dict[str, list[float]]:
    def div(error_key: str, count_key: str) -> list[float]:
        values = store[error_key] / store[count_key].clamp_min(1.0)
        return values.tolist()

    return {
        "mse": div("all_error", "all_count"),
        "mse_xyz": div("xyz_error", "xyz_count"),
        "mse_rot": div("rot_error", "rot_count"),
        "mse_gripper": div("gripper_error", "gripper_count"),
        "valid_values": store["all_count"].tolist(),
    }


def initialize_gripper_store() -> dict[str, float]:
    return {
        "valid": 0.0,
        "correct": 0.0,
        "tp_positive": 0.0,
        "fp_positive": 0.0,
        "fn_positive": 0.0,
        "tp_negative": 0.0,
        "fp_negative": 0.0,
        "fn_negative": 0.0,
        "transition_valid": 0.0,
        "transition_correct": 0.0,
        "target_positive": 0.0,
        "pred_positive": 0.0,
    }


def add_gripper_metrics(
    store: dict[str, float],
    prediction: torch.Tensor,
    target: torch.Tensor,
    time_mask: torch.Tensor,
    dim_mask: torch.Tensor,
    *,
    action_layout: str,
) -> None:
    if action_layout == "libero_ortho6d":
        gripper_index = 9
        threshold = 0.0
    else:
        gripper_index = 6
        threshold = 0.5
    pred = prediction.detach().float().cpu()[..., gripper_index] >= threshold
    true = target.detach().float().cpu()[..., gripper_index] >= threshold
    valid = time_mask.detach().bool().cpu()
    valid &= dim_mask.detach().float().cpu()[:, gripper_index].unsqueeze(1) > 0
    store["valid"] += float(valid.sum())
    store["correct"] += float(((pred == true) & valid).sum())
    store["target_positive"] += float((true & valid).sum())
    store["pred_positive"] += float((pred & valid).sum())
    store["tp_positive"] += float((pred & true & valid).sum())
    store["fp_positive"] += float((pred & ~true & valid).sum())
    store["fn_positive"] += float((~pred & true & valid).sum())
    store["tp_negative"] += float((~pred & ~true & valid).sum())
    store["fp_negative"] += float((~pred & true & valid).sum())
    store["fn_negative"] += float((pred & ~true & valid).sum())
    transition_valid = valid[:, 1:] & valid[:, :-1] & (true[:, 1:] != true[:, :-1])
    store["transition_valid"] += float(transition_valid.sum())
    store["transition_correct"] += float(((pred[:, 1:] == true[:, 1:]) & transition_valid).sum())


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def finalize_gripper_metrics(
    store: dict[str, float],
    *,
    action_layout: str,
) -> dict[str, float | str]:
    return {
        "accuracy": safe_ratio(store["correct"], store["valid"]),
        "valid": store["valid"],
        "positive_label": (
            "raw_action_ge_0"
            if action_layout == "libero_ortho6d"
            else "legacy_gripper_open_ge_0.5"
        ),
        "target_positive_rate": safe_ratio(store["target_positive"], store["valid"]),
        "pred_positive_rate": safe_ratio(store["pred_positive"], store["valid"]),
        "positive_precision": safe_ratio(
            store["tp_positive"],
            store["tp_positive"] + store["fp_positive"],
        ),
        "positive_recall": safe_ratio(
            store["tp_positive"],
            store["tp_positive"] + store["fn_positive"],
        ),
        "negative_precision": safe_ratio(
            store["tp_negative"],
            store["tp_negative"] + store["fp_negative"],
        ),
        "negative_recall": safe_ratio(
            store["tp_negative"],
            store["tp_negative"] + store["fn_negative"],
        ),
        "transition_accuracy": safe_ratio(
            store["transition_correct"],
            store["transition_valid"],
        ),
        "transition_valid": store["transition_valid"],
    }


def build_collator(cfg: ExperimentConfig, *, online_siglip: bool):
    if online_siglip:
        return RDTOnlineSiglipBatchCollator(
            max_lang_tokens=cfg.model.max_lang_tokens,
            pred_horizon=cfg.model.pred_horizon,
            feature_dim=cfg.model.qwen_hidden_size,
            state_dim=cfg.model.state_dim,
            action_dim=cfg.model.action_dim,
            lang_token_dim=cfg.model.lang_token_dim,
            qwen_kv_dim=cfg.model.qwen_kv_dim,
            convert_cached_gripper_closed_to_open=(
                cfg.model.convert_cached_gripper_closed_to_open
            ),
        )
    return RDTBatchCollator(
        max_lang_tokens=cfg.model.max_lang_tokens,
        image_tokens=cfg.model.image_tokens,
        pred_horizon=cfg.model.pred_horizon,
        feature_dim=cfg.model.qwen_hidden_size,
        state_dim=cfg.model.state_dim,
        action_dim=cfg.model.action_dim,
        lang_token_dim=cfg.model.lang_token_dim,
        img_token_dim=cfg.model.img_token_dim,
        qwen_kv_dim=cfg.model.qwen_kv_dim,
        convert_cached_gripper_closed_to_open=(
            cfg.model.convert_cached_gripper_closed_to_open
        ),
    )


def load_eval_model(
    cfg: ExperimentConfig,
    checkpoint: Path | None,
    *,
    device: torch.device,
) -> SFTConditionedRDT:
    model = SFTConditionedRDT(cfg, load_pretrained=True)
    if checkpoint is not None:
        load_trainable_artifact(model, checkpoint, trainable=False)
    model.to(device).eval()
    return model


def evaluate_model(
    *,
    model_name: str,
    model: SFTConditionedRDT,
    cfg: ExperimentConfig,
    dataset: CachedFeatureDataset,
    indices: list[int],
    collator: Any,
    device: torch.device,
    online_siglip: tuple[Any, Any] | None,
    loss_variants: list[str],
    sample_variants: list[str],
    cross_dataset_qwen: dict[str, Any] | None,
    sample_action_batches: int,
    timestep_bucket_count: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    ranges = timestep_ranges(model.runner.num_train_timesteps, timestep_bucket_count)
    loss_accumulators: dict[str, dict[str, dict[str, float]]] = {
        variant: {
            f"{start}-{stop - 1}": {}
            for start, stop in ranges
        }
        for variant in loss_variants
    }
    sample_loss_accumulators: dict[str, dict[str, float]] = {
        variant: {} for variant in sample_variants
    }
    horizon_accumulators = {
        variant: initialize_horizon_store(cfg.model.pred_horizon)
        for variant in sample_variants
    }
    gripper_accumulators = {
        variant: initialize_gripper_store()
        for variant in sample_variants
    }
    variant_notes: dict[str, list[str]] = {variant: [] for variant in set(loss_variants + sample_variants)}

    batches_seen = 0
    examples_seen = 0
    for start_index in range(0, len(indices), batch_size):
        batch_indices = indices[start_index : start_index + batch_size]
        samples = [dataset[index] for index in batch_indices]
        batch = move_tensor_batch(collator(samples), device)
        if online_siglip is not None:
            batch = add_online_siglip_features(
                batch,
                processor=online_siglip[0],
                encoder=online_siglip[1],
                cfg=cfg,
                device=device,
            )
        if len(batch_indices) <= 1:
            for variant in ("shuffle_qwen", "shuffle_lang", "shuffle_image", "shuffle_all_context"):
                if variant in variant_notes:
                    variant_notes[variant].append(
                        "Batch size is 1 for at least one batch; shuffle variant falls back to zeros."
                    )
        cross_dataset_qwen_batch = None
        if "cross_dataset_qwen" in set(loss_variants + sample_variants):
            if cross_dataset_qwen is None:
                variant_notes["cross_dataset_qwen"].append(
                    "No cross-dataset Qwen replacement pool was provided; falling back to zeros."
                )
            else:
                cross_dataset_qwen_batch, missing = cross_dataset_qwen_for_batch(
                    cross_dataset_qwen,
                    batch_indices,
                    batch["qwen_kv"],
                )
                if missing:
                    variant_notes["cross_dataset_qwen"].append(
                        f"{missing} sample(s) in one batch had no different-dataset Qwen replacement; "
                        "those rows fall back to zeros."
                    )

        for variant in loss_variants:
            variant_batch = apply_variant(
                batch,
                variant,
                cross_dataset_qwen_batch=cross_dataset_qwen_batch,
            )
            for bucket_index, (low, high) in enumerate(ranges):
                timesteps = sample_timesteps(
                    batch_size=len(batch_indices),
                    low=low,
                    high=high,
                    device=device,
                    seed=seed + batches_seen * 1009 + bucket_index,
                )
                sums = loss_at_timesteps(
                    model,
                    variant_batch,
                    timesteps,
                    noise_seed=seed + batches_seen * 9173 + bucket_index,
                )
                add_sums(
                    loss_accumulators[variant][f"{low}-{high - 1}"],
                    sums,
                )

        if batches_seen < sample_action_batches:
            for variant in sample_variants:
                variant_batch = apply_variant(
                    batch,
                    variant,
                    cross_dataset_qwen_batch=cross_dataset_qwen_batch,
                )
                torch.manual_seed(seed + batches_seen)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed + batches_seen)
                prediction = model.sample_actions(variant_batch).float()
                target = variant_batch["actions"].to(prediction.dtype)
                sums = metric_sums(
                    prediction,
                    target,
                    variant_batch["action_time_mask"],
                    variant_batch["action_dim_mask"],
                    action_layout=cfg.model.action_encoder_layout,
                )
                add_sums(sample_loss_accumulators[variant], sums)
                add_horizon_metrics(
                    horizon_accumulators[variant],
                    prediction,
                    target,
                    variant_batch["action_time_mask"],
                    variant_batch["action_dim_mask"],
                    action_layout=cfg.model.action_encoder_layout,
                )
                add_gripper_metrics(
                    gripper_accumulators[variant],
                    prediction,
                    target,
                    variant_batch["action_time_mask"],
                    variant_batch["action_dim_mask"],
                    action_layout=cfg.model.action_encoder_layout,
                )

        batches_seen += 1
        examples_seen += len(batch_indices)

    loss_by_variant = {}
    for variant, buckets in loss_accumulators.items():
        finalized = {
            bucket: finalize_loss_sums(sums)
            for bucket, sums in buckets.items()
        }
        average_sums: dict[str, float] = {}
        for sums in buckets.values():
            for key, value in sums.items():
                average_sums[key] = average_sums.get(key, 0.0) + value
        finalized["all_buckets_average"] = finalize_loss_sums(average_sums)
        loss_by_variant[variant] = finalized

    sample_by_variant = {}
    for variant in sample_variants:
        sample_by_variant[variant] = {
            "sample_metrics": finalize_loss_sums(sample_loss_accumulators[variant]),
            "horizon": finalize_horizon_metrics(horizon_accumulators[variant]),
            "gripper": finalize_gripper_metrics(
                gripper_accumulators[variant],
                action_layout=cfg.model.action_encoder_layout,
            ),
        }

    return {
        "model_name": model_name,
        "examples_seen": examples_seen,
        "batches_seen": batches_seen,
        "loss_by_timestep_bucket": loss_by_variant,
        "sample_action_metrics": sample_by_variant,
        "variant_notes": {
            key: sorted(set(value))
            for key, value in variant_notes.items()
            if value
        },
    }


def parse_list(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None:
        return list(default)
    if value.strip().lower() == "all":
        return list(DEFAULT_LOSS_VARIANTS) + ["shuffle_all_context"]
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cached RDT validation behavior with condition ablations, "
            "diffusion-timestep buckets, horizon metrics, and gripper diagnostics."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--cache-root", type=Path, action="append", default=[])
    parser.add_argument("--cache-parts-root", type=Path)
    parser.add_argument("--cache-parts", type=int, nargs="+", choices=[1, 2, 3], default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-mode", choices=["random", "first", "even"], default="random")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--timestep-buckets", type=int, default=10)
    parser.add_argument(
        "--loss-variants",
        default=None,
        help=(
            "Comma-separated variants. Default: baseline,zero_qwen,shuffle_qwen,cross_dataset_qwen,"
            "zero_lang,shuffle_lang,zero_image,shuffle_image,zero_state,zero_ctrl_freq. "
            "Use 'all' to include shuffle_all_context."
        ),
    )
    parser.add_argument(
        "--sample-variants",
        default=None,
        help=(
            "Comma-separated variants for diffusion sample_actions diagnostics. "
            "Default: baseline,zero_qwen,shuffle_qwen,cross_dataset_qwen."
        ),
    )
    parser.add_argument(
        "--sample-action-batches",
        type=int,
        default=4,
        help="Number of batches to run full diffusion sampling on. Set 0 to skip.",
    )
    parser.add_argument("--online-siglip", action="store_true")
    parser.add_argument(
        "--siglip-model-id",
        default="/home/ubuntu/RoboticsDiffusionTransformer/google/siglip-so400m-patch14-384",
    )
    parser.add_argument("--siglip-fallback-model-id", default="google/siglip-so400m-patch14-384")
    parser.add_argument(
        "--include-no-qwen-fusion-model",
        action="store_true",
        help="Also evaluate the trained checkpoint with cfg.model.qwen_fusion forced to 'none'.",
    )
    parser.add_argument(
        "--compare-pretrained-only",
        action="store_true",
        help=(
            "Also evaluate a pretrained-only no-Qwen baseline. This is a rough "
            "preservation check because the wrapper output head differs from official RDT."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.sample_action_batches < 0:
        raise ValueError("--sample-action-batches must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    manifests = [path.expanduser().resolve() for path in args.manifest]
    manifests.extend(manifests_from_cache_roots(args.cache_root, split=args.split))
    if args.cache_parts_root is not None:
        manifests.extend(
            manifests_from_cache_parts_root(
                args.cache_parts_root,
                parts=args.cache_parts or [1, 2, 3],
                split=args.split,
            )
        )
    if not manifests:
        raise ValueError("Provide --manifest, --cache-root, or --cache-parts-root")
    merged_manifest = args.output_dir / f"merged_{args.split}_manifest.jsonl"
    merged_rows = merge_manifests(manifests, merged_manifest)

    required_keys = ONLINE_SIGLIP_REQUIRED_KEYS if args.online_siglip else None
    dataset = CachedFeatureDataset(merged_manifest, required_keys=required_keys)
    indices = selected_indices(
        len(dataset),
        count=args.num_samples,
        seed=args.seed,
        mode=args.sample_mode,
    )
    collator = build_collator(cfg, online_siglip=args.online_siglip)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    online_siglip = None
    if args.online_siglip:
        online_siglip = load_online_siglip(
            model_id=args.siglip_model_id,
            fallback_model_id=args.siglip_fallback_model_id,
            cfg=cfg,
            device=device,
        )

    loss_variants = parse_list(args.loss_variants, DEFAULT_LOSS_VARIANTS)
    sample_variants = parse_list(args.sample_variants, DEFAULT_SAMPLE_VARIANTS)
    needs_cross_dataset_qwen = "cross_dataset_qwen" in set(loss_variants + sample_variants)
    cross_dataset_qwen = (
        build_cross_dataset_qwen_replacements(
            dataset,
            indices,
            seed=args.seed + 314159,
        )
        if needs_cross_dataset_qwen
        else None
    )
    report: dict[str, Any] = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "manifests": [str(path) for path in manifests],
        "merged_manifest": str(merged_manifest.resolve()),
        "merged_manifest_rows": merged_rows,
        "dataset_samples": len(dataset),
        "selected_samples": len(indices),
        "selected_indices_preview": indices[:20],
        "batch_size": args.batch_size,
        "online_siglip": bool(args.online_siglip),
        "loss_variants": loss_variants,
        "sample_variants": sample_variants,
        "sample_action_batches": args.sample_action_batches,
        "timestep_buckets": args.timestep_buckets,
        "cross_dataset_qwen": (
            cross_dataset_qwen["summary"]
            if cross_dataset_qwen is not None
            else None
        ),
        "models": {},
        "notes": [
            "zero_* variants remove a condition; shuffle_* variants swap conditions within the batch.",
            "cross_dataset_qwen replaces only qwen_kv with a qwen_kv tensor sampled from a different dataset id among the selected eval samples.",
            "shuffle variants need batch_size > 1; singleton batches fall back to zeros.",
            "sample_action_metrics use full diffusion sampling and are usually more expensive than loss buckets.",
            "pretrained-only comparison is approximate because this project wrapper changes the official RDT output interface.",
        ],
    }

    model_specs: list[tuple[str, ExperimentConfig, Path | None, list[str], list[str]]] = [
        ("trained", cfg, args.checkpoint, loss_variants, sample_variants)
    ]
    if args.include_no_qwen_fusion_model:
        no_qwen_cfg = replace(cfg, model=replace(cfg.model, qwen_fusion="none"))
        model_specs.append(
            (
                "trained_no_qwen_fusion",
                no_qwen_cfg,
                args.checkpoint,
                ["baseline"],
                ["baseline"] if args.sample_action_batches > 0 else [],
            )
        )
    if args.compare_pretrained_only:
        pretrained_cfg = replace(cfg, model=replace(cfg.model, qwen_fusion="none"))
        model_specs.append(
            (
                "pretrained_only_no_qwen",
                pretrained_cfg,
                None,
                ["baseline"],
                ["baseline"] if args.sample_action_batches > 0 else [],
            )
        )

    for model_name, model_cfg, checkpoint, model_loss_variants, model_sample_variants in model_specs:
        print(f"Evaluating {model_name}...", flush=True)
        model = load_eval_model(model_cfg, checkpoint, device=device)
        report["models"][model_name] = evaluate_model(
            model_name=model_name,
            model=model,
            cfg=model_cfg,
            dataset=dataset,
            indices=indices,
            collator=build_collator(model_cfg, online_siglip=args.online_siglip),
            device=device,
            online_siglip=online_siglip,
            loss_variants=model_loss_variants,
            sample_variants=model_sample_variants,
            cross_dataset_qwen=cross_dataset_qwen,
            sample_action_batches=args.sample_action_batches,
            timestep_bucket_count=args.timestep_buckets,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report_path = args.output_dir / "cached_conditioning_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=json_default) + "\n", encoding="utf-8")
    print(f"Wrote {report_path.resolve()}")
    for model_name, model_report in report["models"].items():
        print(f"\n{model_name}")
        for variant, buckets in model_report["loss_by_timestep_bucket"].items():
            avg = buckets["all_buckets_average"]
            print(
                f"  loss {variant}: {avg['loss']:.6f} "
                f"xyz={avg['loss_xyz']:.6f} rot={avg['loss_rot']:.6f} "
                f"grip={avg['loss_gripper']:.6f}"
            )
        for variant, metrics in model_report["sample_action_metrics"].items():
            sample = metrics["sample_metrics"]
            grip = metrics["gripper"]
            print(
                f"  sample {variant}: mse={sample['loss']:.6f} "
                f"xyz={sample['loss_xyz']:.6f} rot={sample['loss_rot']:.6f} "
                f"grip_mse={sample['loss_gripper']:.6f} "
                f"grip_acc={grip['accuracy']:.3f}"
            )


if __name__ == "__main__":
    main()
