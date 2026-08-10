#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from thinkflow_rdt.adapters.libero import rdt_action_to_libero  # noqa: E402
from thinkflow_rdt.checkpoint import (  # noqa: E402
    load_trainable_artifact,
    save_trainable_artifact,
)
from thinkflow_rdt.config import ExperimentConfig, load_config  # noqa: E402
from thinkflow_rdt.data import (  # noqa: E402
    ONLINE_SIGLIP_REQUIRED_KEYS,
    CachedFeatureDataset,
    RDTOnlineSiglipBatchCollator,
)
from thinkflow_rdt.model import SFTConditionedRDT  # noqa: E402
from thinkflow_rdt.train import (  # noqa: E402
    add_online_siglip_features,
    create_optimizer,
    load_online_siglip,
    seed_everything,
)


RAW_ACTION_NAMES = ("dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper")
EVAL_HORIZONS = (1, 4, 8, 64)
DENOISING_TIMESTEP_BUCKETS = (
    (0, 199),
    (200, 399),
    (400, 599),
    (600, 799),
    (800, 999),
)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    return str(value)


def wandb_sampling_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    """Flatten decoded real-sampling diagnostics into stable W&B keys."""
    result: dict[str, float | int] = {}
    for horizon, values in metrics["horizon_rmse"].items():
        if values is None:
            continue
        for name, value in values.items():
            result[f"sampling/horizon_{horizon}/{name}"] = value
    for group, source_key in (
        ("correlation", "per_dimension_correlation"),
        ("sign_agreement", "per_dimension_sign_agreement"),
        ("saturation_fraction", "per_dimension_saturation_fraction"),
    ):
        for dimension, value in metrics[source_key].items():
            if value is not None:
                result[f"sampling/{group}/{dimension}"] = value
    result["sampling/sign_agreement/overall"] = metrics["overall_sign_agreement"]
    result["sampling/saturation_fraction/overall"] = metrics[
        "overall_saturation_fraction"
    ]
    for name, value in metrics["gripper"].items():
        if isinstance(value, (int, float)):
            result[f"sampling/gripper/{name}"] = value
    for phase, values in metrics.get("gripper_phase", {}).items():
        for name, value in values.items():
            if isinstance(value, (int, float)):
                result[f"sampling/gripper_phase/{phase}/{name}"] = value
    for bucket, values in metrics.get("gripper_denoising_by_timestep", {}).items():
        result[f"denoising/{bucket}/gripper_accuracy"] = values["gripper"][
            "accuracy"
        ]
        for phase, phase_values in values.get("gripper_phase", {}).items():
            for name, value in phase_values.items():
                if isinstance(value, (int, float)):
                    result[f"denoising/{bucket}/{phase}/{name}"] = value
    result["sampling/trajectory_count"] = metrics["sampled_trajectories"]
    result["sampling/diffusion_repeats"] = metrics["diffusion_sampling_repeats"]
    return result


def horizon_weight_vector(horizon: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return weights for one-indexed horizons 1-4, 5-8, 9-16, and 17-64."""
    if horizon != 64:
        raise ValueError(f"This experiment expects a 64-step horizon, got {horizon}")
    weights = torch.ones(horizon, dtype=torch.float32, device=device)
    weights[:4] = 5.0
    weights[4:8] = 3.0
    weights[8:16] = 2.0
    return weights


def move_tensors(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def model_tensor_batch(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    keys = (
        "qwen_kv",
        "lang_tokens",
        "lang_mask",
        "img_tokens",
        "img_mask",
        "state",
        "state_dim_mask",
        "actions",
        "action_time_mask",
        "action_dim_mask",
        "ctrl_freq",
    )
    return {key: batch[key].detach().cpu() for key in keys}


def build_gripper_release_masks(
    samples: list[dict[str, Any]],
    *,
    horizon: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Mark final +1 -> -1 release targets using complete episode timelines."""
    by_episode: dict[str, list[tuple[int, float]]] = {}
    for sample in samples:
        episode_id = str(sample["episode_id"])
        step = int(sample["step_idx"])
        command = float(torch.as_tensor(sample["actions"])[0, 9])
        by_episode.setdefault(episode_id, []).append((step, command))

    release_start: dict[str, int] = {}
    for episode_id, timeline in by_episode.items():
        ordered = sorted(timeline)
        transitions = [
            step
            for (previous_step, previous), (step, current) in zip(ordered, ordered[1:])
            if step == previous_step + 1 and previous >= 0.0 and current < 0.0
        ]
        if transitions:
            release_start[episode_id] = transitions[-1]

    masks = torch.zeros(len(samples), horizon, dtype=torch.bool)
    for sample_index, sample in enumerate(samples):
        start = release_start.get(str(sample["episode_id"]))
        if start is None:
            continue
        step = int(sample["step_idx"])
        valid = torch.as_tensor(sample["action_time_mask"], dtype=torch.bool)[:horizon]
        offsets = torch.arange(horizon)
        masks[sample_index] = valid & (step + offsets >= start)
    return masks, {
        "episodes": len(by_episode),
        "episodes_with_release": len(release_start),
        "release_targets": int(masks.sum()),
    }


def attach_gripper_training_weights(
    batches: list[dict[str, torch.Tensor]],
    release_masks: torch.Tensor,
    *,
    release_weight: float,
) -> None:
    """Attach phase labels and per-target gripper weights to encoded batches."""
    cursor = 0
    for batch in batches:
        count = int(batch["state"].shape[0])
        mask = release_masks[cursor : cursor + count].clone()
        batch["gripper_release_mask"] = mask
        batch["gripper_loss_weights"] = torch.where(
            mask,
            torch.full(mask.shape, release_weight, dtype=torch.float32),
            torch.ones(mask.shape, dtype=torch.float32),
        )
        cursor += count
    if cursor != len(release_masks):
        raise ValueError(
            f"Release-mask count {len(release_masks)} does not match encoded samples {cursor}"
        )


def build_release_group_sampling_probabilities(
    release_masks: torch.Tensor,
    *,
    batch_size: int,
    oversample_factor: float,
    oversample_horizon: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Weight fixed micro-batches by nearby release-window prevalence.

    Retained SigLIP features are stored in fixed contiguous micro-batches.  We
    therefore sample those batches with replacement, weighting each batch by
    the mean per-sample weight.  A sample is release-relevant when at least one
    of its first ``oversample_horizon`` targets belongs to the final open phase.
    """
    if release_masks.ndim != 2:
        raise ValueError("release_masks must have shape [samples, horizon]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if oversample_factor < 1.0:
        raise ValueError("oversample_factor must be at least 1")
    if not 1 <= oversample_horizon <= release_masks.shape[1]:
        raise ValueError(
            "oversample_horizon must be between 1 and the prediction horizon"
        )

    release_samples = release_masks[:, :oversample_horizon].any(dim=1)
    sample_weights = torch.where(
        release_samples,
        torch.full(release_samples.shape, oversample_factor, dtype=torch.float64),
        torch.ones(release_samples.shape, dtype=torch.float64),
    )
    group_weights = []
    group_release_fractions = []
    for start in range(0, len(release_samples), batch_size):
        stop = min(start + batch_size, len(release_samples))
        group_weights.append(float(sample_weights[start:stop].mean()))
        group_release_fractions.append(
            float(release_samples[start:stop].float().mean())
        )

    probabilities = np.asarray(group_weights, dtype=np.float64)
    probabilities /= probabilities.sum()
    release_fractions = np.asarray(group_release_fractions, dtype=np.float64)
    natural_group_probabilities = np.full(
        len(group_weights), 1.0 / len(group_weights), dtype=np.float64
    )
    return probabilities, {
        "oversample_factor": float(oversample_factor),
        "oversample_horizon": int(oversample_horizon),
        "release_relevant_samples": int(release_samples.sum()),
        "total_samples": int(len(release_samples)),
        "natural_release_window_fraction": float(release_samples.float().mean()),
        "natural_group_sampled_release_fraction": float(
            np.dot(natural_group_probabilities, release_fractions)
        ),
        "expected_oversampled_release_fraction": float(
            np.dot(probabilities, release_fractions)
        ),
    }


def build_online_siglip_collator(cfg: ExperimentConfig) -> RDTOnlineSiglipBatchCollator:
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


@torch.no_grad()
def encode_fixed_feature_batches(
    samples: list[dict[str, Any]],
    *,
    cfg: ExperimentConfig,
    batch_size: int,
    device: torch.device,
    collator: RDTOnlineSiglipBatchCollator,
    processor: Any,
    encoder: Any,
) -> list[dict[str, torch.Tensor]]:
    fixed_batches: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(samples), batch_size):
        batch = move_tensors(collator(samples[start : start + batch_size]), device)
        batch = add_online_siglip_features(
            batch,
            processor=processor,
            encoder=encoder,
            cfg=cfg,
            device=device,
        )
        fixed_batches.append(model_tensor_batch(batch))
    return fixed_batches


def select_fixed_feature_batches(
    source_batches: list[dict[str, torch.Tensor]],
    positions: list[int],
    *,
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    """Select ordered samples from already encoded CPU batches without SigLIP."""
    if not positions:
        raise ValueError("At least one fixed-feature position is required")
    offsets = []
    cursor = 0
    for batch_index, batch in enumerate(source_batches):
        count = int(batch["state"].shape[0])
        offsets.extend((batch_index, local_index) for local_index in range(count))
        cursor += count
    if min(positions) < 0 or max(positions) >= cursor:
        raise IndexError(f"Fixed-feature positions exceed {cursor} encoded samples")
    selected = [offsets[position] for position in positions]
    result = []
    for start in range(0, len(selected), batch_size):
        rows = selected[start : start + batch_size]
        tensor_batch = {
            key: torch.stack(
                [source_batches[batch_index][key][local_index] for batch_index, local_index in rows]
            )
            for key in source_batches[0]
        }
        result.append(tensor_batch)
    return result


@torch.no_grad()
def prepare_fixed_feature_batches(
    samples: list[dict[str, Any]],
    *,
    cfg: ExperimentConfig,
    batch_size: int,
    device: torch.device,
    siglip_model_id: str,
    siglip_fallback_model_id: str | None,
) -> list[dict[str, torch.Tensor]]:
    """Compute SigLIP once, then retain fixed CPU tensors for repeated overfitting."""
    collator = build_online_siglip_collator(cfg)
    processor, encoder = load_online_siglip(
        model_id=siglip_model_id,
        fallback_model_id=siglip_fallback_model_id,
        cfg=cfg,
        device=device,
    )
    fixed_batches = encode_fixed_feature_batches(
        samples,
        cfg=cfg,
        batch_size=batch_size,
        device=device,
        collator=collator,
        processor=processor,
        encoder=encoder,
    )
    del encoder, processor
    gc.collect()
    torch.cuda.empty_cache()
    return fixed_batches


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def signs(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.where(values > eps, 1, np.where(values < -eps, -1, 0))


def decoded_command_metrics(
    prediction_10d: np.ndarray,
    target_10d: np.ndarray,
    time_mask: np.ndarray,
    gripper_release_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure actual decoded 7D commands produced by full diffusion sampling."""
    predicted = rdt_action_to_libero(prediction_10d)
    target = rdt_action_to_libero(target_10d)
    result: dict[str, Any] = {"horizon_rmse": {}}
    for horizon in EVAL_HORIZONS:
        clipped_horizon = min(horizon, predicted.shape[1])
        valid = time_mask[:, :clipped_horizon]
        pred_values = predicted[:, :clipped_horizon][valid]
        target_values = target[:, :clipped_horizon][valid]
        if pred_values.size == 0:
            result["horizon_rmse"][str(horizon)] = None
            continue
        error = pred_values - target_values
        result["horizon_rmse"][str(horizon)] = {
            "command_rmse_7d": float(np.sqrt(np.mean(error**2))),
            "motion_rmse_6d": float(np.sqrt(np.mean(error[:, :6] ** 2))),
            "gripper_rmse": float(np.sqrt(np.mean(error[:, 6] ** 2))),
            "valid_commands": int(len(pred_values)),
        }

    valid_prediction = predicted[time_mask]
    valid_target = target[time_mask]
    correlations = {}
    sign_agreement = {}
    saturation = {}
    for dimension, name in enumerate(RAW_ACTION_NAMES):
        correlations[name] = correlation(
            valid_prediction[:, dimension],
            valid_target[:, dimension],
        )
        sign_agreement[name] = float(
            np.mean(
                signs(valid_prediction[:, dimension])
                == signs(valid_target[:, dimension])
            )
        )
        saturation[name] = float(
            np.mean(np.abs(valid_prediction[:, dimension]) >= 1.0 - 1e-6)
        )

    pred_gripper = valid_prediction[:, 6] >= 0.0
    true_gripper = valid_target[:, 6] >= 0.0
    tp = int(np.sum(pred_gripper & true_gripper))
    fp = int(np.sum(pred_gripper & ~true_gripper))
    fn = int(np.sum(~pred_gripper & true_gripper))
    tn = int(np.sum(~pred_gripper & ~true_gripper))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    result.update(
        {
            "per_dimension_correlation": correlations,
            "per_dimension_sign_agreement": sign_agreement,
            "overall_sign_agreement": float(
                np.mean(signs(valid_prediction) == signs(valid_target))
            ),
            "per_dimension_saturation_fraction": saturation,
            "overall_saturation_fraction": float(
                np.mean(np.abs(valid_prediction) >= 1.0 - 1e-6)
            ),
            "gripper": {
                "positive_class": "raw_command_ge_0",
                "accuracy": float(np.mean(pred_gripper == true_gripper)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            },
        }
    )
    if gripper_release_mask is not None:
        release_mask = np.asarray(gripper_release_mask, dtype=bool) & time_mask
        hold_mask = time_mask & (target[..., 6] >= 0.0)
        approach_mask = time_mask & (target[..., 6] < 0.0) & ~release_mask

        def phase_accuracy(mask: np.ndarray) -> dict[str, float | int | None]:
            count = int(mask.sum())
            return {
                "accuracy": (
                    None
                    if count == 0
                    else float(np.mean((predicted[..., 6] >= 0.0)[mask] == (target[..., 6] >= 0.0)[mask]))
                ),
                "commands": count,
            }

        transition_errors: list[int] = []
        transition_opportunities = 0
        missed_transitions = 0
        for row in range(len(predicted)):
            valid_count = int(time_mask[row].sum())
            if valid_count < 2:
                continue
            release = release_mask[row, :valid_count]
            starts = np.flatnonzero(release[1:] & ~release[:-1]) + 1
            if not len(starts):
                continue
            target_start = int(starts[0])
            transition_opportunities += 1
            predicted_positive = predicted[row, :valid_count, 6] >= 0.0
            predicted_starts = np.flatnonzero(
                predicted_positive[:-1] & ~predicted_positive[1:]
            ) + 1
            if not len(predicted_starts):
                missed_transitions += 1
                continue
            closest = int(predicted_starts[np.argmin(np.abs(predicted_starts - target_start))])
            transition_errors.append(closest - target_start)

        result["gripper_phase"] = {
            "approach_open": phase_accuracy(approach_mask),
            "close_hold": phase_accuracy(hold_mask),
            "release_open": phase_accuracy(release_mask),
            "release_transition_timing": {
                "opportunities": transition_opportunities,
                "detected": len(transition_errors),
                "missed": missed_transitions,
                "mean_signed_error_steps": (
                    None if not transition_errors else float(np.mean(transition_errors))
                ),
                "mae_steps": (
                    None if not transition_errors else float(np.mean(np.abs(transition_errors)))
                ),
            },
        }
    return result


@torch.no_grad()
def evaluate_real_sampling(
    model: SFTConditionedRDT,
    fixed_batches: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    repeats: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()
    predictions = []
    targets = []
    masks = []
    release_masks = []
    for repeat in range(repeats):
        for batch_index, cpu_batch in enumerate(fixed_batches):
            batch = move_tensors(cpu_batch, device)
            sample_seed = seed + repeat * 100_003 + batch_index
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)
            predictions.append(model.sample_actions(batch).float().cpu())
            targets.append(cpu_batch["actions"].float().cpu())
            masks.append(cpu_batch["action_time_mask"].bool().cpu())
            if "gripper_release_mask" in cpu_batch:
                release_masks.append(cpu_batch["gripper_release_mask"].bool().cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    mask = torch.cat(masks)
    release_mask = torch.cat(release_masks) if release_masks else None
    metrics = decoded_command_metrics(
        prediction.numpy(),
        target.numpy(),
        mask.numpy(),
        None if release_mask is None else release_mask.numpy(),
    )
    metrics["diffusion_sampling_repeats"] = repeats
    metrics["sampled_trajectories"] = int(prediction.shape[0])
    model.train()
    tensors = {"prediction_10d": prediction, "target_10d": target, "time_mask": mask}
    if release_mask is not None:
        tensors["gripper_release_mask"] = release_mask
    return metrics, tensors


@torch.no_grad()
def evaluate_gripper_denoising_by_timestep(
    model: SFTConditionedRDT,
    fixed_batches: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Measure one-call x0 gripper prediction from low through high noise."""
    was_training = model.training
    model.eval()
    result: dict[str, dict[str, Any]] = {}
    for bucket_index, (low, high) in enumerate(DENOISING_TIMESTEP_BUCKETS):
        predictions = []
        targets = []
        masks = []
        release_masks = []
        for batch_index, cpu_batch in enumerate(fixed_batches):
            batch = move_tensors(cpu_batch, device)
            batch_seed = seed + bucket_index * 100_003 + batch_index
            generator = torch.Generator(device="cpu").manual_seed(batch_seed)
            count = int(batch["actions"].shape[0])
            batch["diffusion_timesteps"] = torch.randint(
                low,
                high + 1,
                (count,),
                generator=generator,
                dtype=torch.long,
            ).to(device)
            batch["diffusion_noise"] = torch.randn(
                batch["actions"].shape,
                generator=generator,
                dtype=torch.float32,
            ).to(device)
            batch["return_denoising_prediction"] = True
            denoising = model(batch)
            predictions.append(denoising["denoising_prediction"].float().cpu())
            targets.append(cpu_batch["actions"].float().cpu())
            masks.append(cpu_batch["action_time_mask"].bool().cpu())
            if "gripper_release_mask" in cpu_batch:
                release_masks.append(
                    cpu_batch["gripper_release_mask"].bool().cpu()
                )

        prediction = torch.cat(predictions)
        target = torch.cat(targets)
        mask = torch.cat(masks)
        release_mask = torch.cat(release_masks) if release_masks else None
        decoded = decoded_command_metrics(
            prediction.numpy(),
            target.numpy(),
            mask.numpy(),
            None if release_mask is None else release_mask.numpy(),
        )
        bucket_name = f"t{low:03d}_{high:03d}"
        result[bucket_name] = {
            "timestep_min": low,
            "timestep_max": high,
            "gripper": decoded["gripper"],
            "gripper_phase": decoded.get("gripper_phase", {}),
        }
    model.train(was_training)
    return result


def report_line(
    step: int,
    total: float,
    imitation: float,
    bce: float,
    rotation_geodesic: float,
    unweighted: float,
    elapsed: float,
) -> None:
    print(
        f"step={step:05d} total_loss={total:.7f} "
        f"weighted_imitation_loss={imitation:.7f} gripper_bce_loss={bce:.7f} "
        f"rotation_geodesic_loss={rotation_geodesic:.7f} "
        f"unweighted_denoising_loss={unweighted:.7f} elapsed={elapsed:.1f}s",
        flush=True,
    )


def run_rollout(
    args: argparse.Namespace,
    *,
    artifact_dir: Path,
    cache_suite_root: Path,
) -> dict[str, Any]:
    output_dir = args.output_dir / f"rollout_{time.strftime('%Y%m%d_%H%M%S')}"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/evaluate_libero_rdt.py"),
        "--config",
        str(args.config),
        "--benchmark",
        args.suite,
        "--checkpoint",
        str(artifact_dir),
        "--cache-root",
        str(cache_suite_root),
        "--output-dir",
        str(output_dir),
        "--libero-root",
        str(args.libero_root),
        "--episodes-per-task",
        str(args.rollout_episodes_per_task),
        "--env-batch-size",
        str(args.rollout_env_batch_size),
        "--action-chunk",
        str(args.rollout_action_chunk),
        "--max-steps",
        str(args.rollout_max_steps),
        "--seed",
        str(args.seed),
    ]
    if args.rollout_demo_hdf5 is not None:
        command.extend(["--demo-hdf5", str(args.rollout_demo_hdf5)])
        for demo_name in args.rollout_demo_name:
            command.extend(["--demo-name", demo_name])
    for task_id in args.rollout_task_id:
        command.extend(["--task-id", str(task_id)])
    if args.base_artifact is not None:
        command.extend(["--base-artifact", str(args.base_artifact)])
    if args.rollout_save_videos:
        command.extend(
            [
                "--save-videos",
                "--video-resolution",
                str(args.rollout_video_resolution),
            ]
        )
    for option, value in (
        ("--qwen-model-id", args.qwen_model_id),
        ("--qwen-processor-id", args.qwen_processor_id),
        ("--t5-model-id", args.t5_model_id),
        ("--siglip-model-id", args.siglip_model_id),
    ):
        if value:
            command.extend([option, str(value)])
    print("Running rollout:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    summary_path = output_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    summary["output_dir"] = str(output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit a tiny cached LIBERO subset and evaluate real diffusion samples."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/b0_rdt1b_lora.yaml"))
    parser.add_argument("--cache-root", type=Path, default=Path("cache_features_libero_b0_raw_ortho6d"))
    parser.add_argument(
        "--suite",
        choices=("libero_10", "libero_spatial", "libero_goal", "libero_object"),
        default="libero_object",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/libero_cached_overfit"))
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Train on every sample in the selected manifest.",
    )
    parser.add_argument(
        "--streaming-online-siglip",
        action="store_true",
        help=(
            "Compute SigLIP per training batch instead of retaining all image "
            "tokens in RAM. Use this only when host RAM is limited."
        ),
    )
    parser.add_argument(
        "--sampling-num-samples",
        type=int,
        default=32,
        help=(
            "Number of evenly spaced training conditions used for expensive "
            "real-diffusion sampling metrics."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help=(
            "Effective batch per optimizer step. Must be divisible by "
            "--batch-size; implemented with gradient accumulation."
        ),
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate-lora", type=float, default=1e-4)
    parser.add_argument("--learning-rate-interfaces", type=float, default=1e-4)
    parser.add_argument(
        "--gripper-bce-weight",
        type=float,
        default=0.0,
        help="Weight of auxiliary BCE on the existing clean gripper output.",
    )
    parser.add_argument(
        "--gripper-bce-logit-scale",
        type=float,
        default=5.0,
        help="Multiply the raw gripper output before BCE; adds no parameters.",
    )
    parser.add_argument(
        "--mask-noisy-gripper-input",
        action="store_true",
        help=(
            "Zero the noisy gripper input during training and diffusion sampling, "
            "forcing the existing output to use task conditions instead of "
            "copying the forward-noised sign."
        ),
    )
    parser.add_argument(
        "--release-gripper-weight",
        type=float,
        default=1.0,
        help="Extra MSE/BCE weight for final +1 to -1 release targets.",
    )
    parser.add_argument(
        "--release-oversample-factor",
        type=float,
        default=1.0,
        help=(
            "Relative sampling weight for windows with a nearby final release; "
            "1 disables oversampling."
        ),
    )
    parser.add_argument(
        "--release-oversample-horizon",
        type=int,
        default=8,
        help=(
            "A window is release-relevant when its first N targets include the "
            "final open phase."
        ),
    )
    parser.add_argument(
        "--rotation-geodesic-weight",
        type=float,
        default=0.0,
        help="Weight of auxiliary SO(3) loss on decoded ortho6D rotations.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--sampling-repeats", type=int, default=3)
    parser.add_argument("--inference-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-artifact", type=Path)
    parser.add_argument("--init-artifact", type=Path)
    parser.add_argument("--siglip-model-id", default=None)
    parser.add_argument("--siglip-fallback-model-id", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--qwen-model-id", default=None)
    parser.add_argument("--qwen-processor-id", default=None)
    parser.add_argument("--t5-model-id", default=None)
    parser.add_argument("--report-to", choices=("none", "wandb"), default="none")
    parser.add_argument("--wandb-project", default="thinkflow-rdt-b0-libero")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default="libero-cached-overfit-test")
    parser.add_argument("--run-rollout", action="store_true")
    parser.add_argument("--libero-root", type=Path, default=Path("/home/ubuntu/LIBERO"))
    parser.add_argument("--rollout-task-id", type=int, action="append", default=None)
    parser.add_argument("--rollout-episodes-per-task", type=int, default=2)
    parser.add_argument("--rollout-env-batch-size", type=int, default=2)
    parser.add_argument("--rollout-action-chunk", type=int, default=1)
    parser.add_argument("--rollout-max-steps", type=int, default=600)
    parser.add_argument("--rollout-save-videos", action="store_true")
    parser.add_argument("--rollout-video-resolution", type=int, default=512)
    parser.add_argument(
        "--rollout-demo-hdf5",
        type=Path,
        default=None,
        help="Evaluate from exact demonstration simulator states instead of official init states.",
    )
    parser.add_argument(
        "--rollout-demo-name",
        action="append",
        default=[],
        help="Demo group to evaluate when --rollout-demo-hdf5 is set; repeatable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wandb_run = None
    if args.rollout_task_id is None:
        args.rollout_task_id = [0]
    if args.rollout_demo_name and args.rollout_demo_hdf5 is None:
        raise ValueError("--rollout-demo-name requires --rollout-demo-hdf5")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the RDT-1B overfitting experiment")
    if (not args.all_samples and args.num_samples <= 0) or args.batch_size <= 0 or args.steps <= 0:
        raise ValueError("--num-samples, --batch-size, and --steps must be positive")
    if args.sampling_repeats <= 0 or args.inference_steps <= 0:
        raise ValueError("--sampling-repeats and --inference-steps must be positive")
    if args.sampling_num_samples <= 0:
        raise ValueError("--sampling-num-samples must be positive")
    if args.gripper_bce_weight < 0:
        raise ValueError("--gripper-bce-weight must be non-negative")
    if args.gripper_bce_logit_scale <= 0:
        raise ValueError("--gripper-bce-logit-scale must be positive")
    if args.release_gripper_weight < 1:
        raise ValueError("--release-gripper-weight must be at least 1")
    if args.release_oversample_factor < 1:
        raise ValueError("--release-oversample-factor must be at least 1")
    if args.release_oversample_horizon <= 0:
        raise ValueError("--release-oversample-horizon must be positive")
    if args.rotation_geodesic_weight < 0:
        raise ValueError("--rotation-geodesic-weight must be non-negative")
    if args.streaming_online_siglip and (
        args.release_gripper_weight != 1 or args.release_oversample_factor != 1
    ):
        raise ValueError(
            "Release-phase weighting/oversampling requires retained samples; "
            "disable --streaming-online-siglip"
        )
    if args.global_batch_size is not None:
        if args.global_batch_size < args.batch_size:
            raise ValueError("--global-batch-size cannot be smaller than --batch-size")
        if args.global_batch_size % args.batch_size:
            raise ValueError("--global-batch-size must be divisible by --batch-size")
    seed_everything(args.seed)
    device = torch.device("cuda")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested_cache_root = args.cache_root.expanduser().resolve()
    cache_suite_root = (
        requested_cache_root
        if (requested_cache_root / args.split / "manifest.jsonl").exists()
        else requested_cache_root / args.suite
    )
    manifest = cache_suite_root / args.split / "manifest.jsonl"
    metadata_path = cache_suite_root / "precompute_metadata.json"
    dataset = CachedFeatureDataset(manifest, required_keys=ONLINE_SIGLIP_REQUIRED_KEYS)
    if args.all_samples:
        indices = list(range(args.sample_start, len(dataset)))
    else:
        stop = min(args.sample_start + args.num_samples, len(dataset))
        indices = list(range(args.sample_start, stop))
        if len(indices) != args.num_samples:
            raise ValueError(
                f"Requested {args.num_samples} samples but only found {len(indices)}"
            )
    with metadata_path.open("r", encoding="utf-8") as handle:
        cache_metadata = json.load(handle)

    cfg = load_config(args.config)
    cfg = replace(
        cfg,
        noise_scheduler=replace(
            cfg.noise_scheduler,
            num_inference_timesteps=args.inference_steps,
        ),
        training=replace(
            cfg.training,
            learning_rate_lora=args.learning_rate_lora,
            learning_rate_interfaces=args.learning_rate_interfaces,
            max_grad_norm=args.max_grad_norm,
        ),
    )
    if cfg.model.pred_horizon != 64:
        raise ValueError(f"Expected pred_horizon=64, got {cfg.model.pred_horizon}")
    if args.release_oversample_horizon > cfg.model.pred_horizon:
        raise ValueError(
            "--release-oversample-horizon cannot exceed the prediction horizon"
        )
    if cfg.model.action_encoder_layout != "libero_ortho6d":
        raise ValueError("This experiment requires action_encoder_layout=libero_ortho6d")

    if args.report_to == "wandb":
        import wandb

        wandb_config = {
            key: json_default(value)
            for key, value in vars(args).items()
        }
        wandb_config.update(
            {
                "resolved_cache_root": str(cache_suite_root),
                "training_sample_count": len(indices),
                "horizon_weights": {
                    "1-4": 5.0,
                    "5-8": 3.0,
                    "9-16": 2.0,
                    "17-64": 1.0,
                },
            }
        )
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=wandb_config,
        )

    siglip_id = args.siglip_model_id or cache_metadata.get(
        "siglip_model_id", "google/siglip-so400m-patch14-384"
    )
    streaming_collator = None
    siglip_processor = None
    siglip_encoder = None
    training_fixed_batches = None
    release_group_probabilities = None
    release_sampling_audit: dict[str, float | int] | None = None
    if args.streaming_online_siglip:
        audit_count = min(args.sampling_num_samples, len(indices))
        audit_positions = np.linspace(0, len(indices) - 1, audit_count, dtype=int)
        audit_indices = [indices[position] for position in np.unique(audit_positions)]
        audit_samples = [dataset[index] for index in audit_indices]
        print(
            f"Loading streaming SigLIP {siglip_id}; real-sampling audit uses "
            f"{len(audit_indices)}/{len(indices)} evenly spaced conditions"
        )
        streaming_collator = build_online_siglip_collator(cfg)
        siglip_processor, siglip_encoder = load_online_siglip(
            model_id=siglip_id,
            fallback_model_id=args.siglip_fallback_model_id,
            cfg=cfg,
            device=device,
        )
        fixed_batches = encode_fixed_feature_batches(
            audit_samples,
            cfg=cfg,
            batch_size=args.batch_size,
            device=device,
            collator=streaming_collator,
            processor=siglip_processor,
            encoder=siglip_encoder,
        )
        del audit_samples
    else:
        samples = [dataset[index] for index in indices]
        print(
            f"Precomputing and retaining SigLIP tokens for {len(indices)} "
            f"training samples using {siglip_id}"
        )
        training_fixed_batches = prepare_fixed_feature_batches(
            samples,
            cfg=cfg,
            batch_size=args.batch_size,
            device=device,
            siglip_model_id=siglip_id,
            siglip_fallback_model_id=args.siglip_fallback_model_id,
        )
        release_masks, release_audit = build_gripper_release_masks(
            samples,
            horizon=cfg.model.pred_horizon,
        )
        attach_gripper_training_weights(
            training_fixed_batches,
            release_masks,
            release_weight=args.release_gripper_weight,
        )
        print(
            "Gripper release weighting: "
            f"weight={args.release_gripper_weight:g} audit={release_audit}"
        )
        if args.release_oversample_factor > 1.0:
            (
                release_group_probabilities,
                release_sampling_audit,
            ) = build_release_group_sampling_probabilities(
                release_masks,
                batch_size=args.batch_size,
                oversample_factor=args.release_oversample_factor,
                oversample_horizon=args.release_oversample_horizon,
            )
            print(
                "Gripper release oversampling: "
                f"{json.dumps(release_sampling_audit, default=json_default)}"
            )
        audit_count = min(args.sampling_num_samples, len(indices))
        audit_positions_array = np.unique(
            np.linspace(0, len(indices) - 1, audit_count, dtype=int)
        )
        audit_positions = audit_positions_array.tolist()
        audit_indices = [indices[position] for position in audit_positions]
        fixed_batches = select_fixed_feature_batches(
            training_fixed_batches,
            audit_positions,
            batch_size=args.batch_size,
        )
        del samples
        gc.collect()

    model = SFTConditionedRDT(
        cfg,
        load_pretrained=True,
        base_artifact=None if args.base_artifact is None else str(args.base_artifact),
    )
    if args.init_artifact is not None:
        load_trainable_artifact(model, args.init_artifact, trainable=True)
        # Loading a trainable artifact makes every saved interface trainable.
        # Reapply this experiment's configuration before building the optimizer.
        model.runner.state_adaptor.requires_grad_(not cfg.model.freeze_state_adaptor)
        model.runner.lang_adaptor.requires_grad_(not cfg.model.freeze_condition_adaptors)
        model.runner.img_adaptor.requires_grad_(not cfg.model.freeze_condition_adaptors)
    # Explicit experiment configuration takes precedence over initialization
    # artifact metadata. The value is saved and restored for rollout.
    model.mask_noisy_gripper_input = bool(args.mask_noisy_gripper_input)
    model.to(device).train()
    optimizer = create_optimizer(model, cfg)
    weights = horizon_weight_vector(cfg.model.pred_horizon, device=device)
    print("Horizon weights:", weights.cpu().tolist())
    print(json.dumps(model.trainable_parameter_report(), indent=2, default=json_default))

    history: list[dict[str, Any]] = []
    sample_snapshots: dict[str, dict[str, torch.Tensor]] = {}
    started = time.perf_counter()
    sampling_seed = args.seed + 1_000_000
    training_groups = [
        indices[start : start + args.batch_size]
        for start in range(0, len(indices), args.batch_size)
    ]
    training_rng = np.random.default_rng(args.seed)
    if release_group_probabilities is None:
        training_order = training_rng.permutation(len(training_groups)).tolist()
    else:
        training_order = training_rng.choice(
            len(training_groups),
            size=len(training_groups),
            replace=True,
            p=release_group_probabilities,
        ).tolist()
    training_group_position = 0
    full_group_indices = [
        index for index, group in enumerate(training_groups)
        if len(group) == args.batch_size
    ]
    accumulation_steps = (
        1
        if args.global_batch_size is None
        else args.global_batch_size // args.batch_size
    )
    effective_global_batch = args.batch_size * accumulation_steps
    approximate_epochs = args.steps * effective_global_batch / max(len(indices), 1)
    print(
        f"Training samples={len(indices)} groups={len(training_groups)} "
        f"micro_batch={args.batch_size} accumulation={accumulation_steps} "
        f"effective_global_batch={effective_global_batch} steps={args.steps} "
        f"approximate_sample_epochs={approximate_epochs:.2f}"
    )

    baseline_metrics, baseline_tensors = evaluate_real_sampling(
        model,
        fixed_batches,
        device=device,
        repeats=args.sampling_repeats,
        seed=sampling_seed,
    )
    baseline_metrics["gripper_denoising_by_timestep"] = (
        evaluate_gripper_denoising_by_timestep(
            model,
            fixed_batches,
            device=device,
            seed=sampling_seed + 500_000,
        )
    )
    history.append({"step": 0, "real_sampling": baseline_metrics})
    sample_snapshots["step_000000"] = baseline_tensors
    print("Baseline real-sampling metrics:")
    print(json.dumps(baseline_metrics, indent=2, default=json_default))
    if wandb_run is not None:
        wandb_run.log(wandb_sampling_metrics(baseline_metrics), step=0)

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss_sum = 0.0
        imitation_loss_sum = 0.0
        gripper_bce_loss_sum = 0.0
        rotation_geodesic_loss_sum = 0.0
        unweighted_loss_sum = 0.0
        for _micro_step in range(accumulation_steps):
            if training_group_position >= len(training_order):
                if release_group_probabilities is None:
                    training_order = training_rng.permutation(
                        len(training_groups)
                    ).tolist()
                else:
                    training_order = training_rng.choice(
                        len(training_groups),
                        size=len(training_groups),
                        replace=True,
                        p=release_group_probabilities,
                    ).tolist()
                training_group_position = 0
            group_index = training_order[training_group_position]
            group = list(training_groups[group_index])
            training_group_position += 1
            padding_count = args.batch_size - len(group)
            if args.streaming_online_siglip:
                assert streaming_collator is not None
                assert siglip_processor is not None and siglip_encoder is not None
                if padding_count:
                    padding = training_rng.choice(indices, size=padding_count, replace=True)
                    group.extend(int(index) for index in padding)
                batch_samples = [dataset[index] for index in group]
                batch = move_tensors(streaming_collator(batch_samples), device)
                batch = add_online_siglip_features(
                    batch,
                    processor=siglip_processor,
                    encoder=siglip_encoder,
                    cfg=cfg,
                    device=device,
                )
                del batch_samples
            else:
                assert training_fixed_batches is not None
                cpu_batch = training_fixed_batches[group_index]
                if padding_count:
                    if not full_group_indices:
                        raise ValueError(
                            "The selected dataset has fewer samples than --batch-size"
                        )
                    filler_index = int(training_rng.choice(full_group_indices))
                    filler = training_fixed_batches[filler_index]
                    cpu_batch = {
                        key: torch.cat([value, filler[key][:padding_count]], dim=0)
                        for key, value in cpu_batch.items()
                    }
                batch = move_tensors(cpu_batch, device)
            batch["horizon_loss_weights"] = weights
            batch["gripper_bce_weight"] = torch.tensor(
                args.gripper_bce_weight,
                device=device,
            )
            batch["gripper_bce_logit_scale"] = torch.tensor(
                args.gripper_bce_logit_scale,
                device=device,
            )
            batch["rotation_geodesic_weight"] = torch.tensor(
                args.rotation_geodesic_weight,
                device=device,
            )
            metrics = model(batch)
            loss = metrics["loss"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss}")
            (loss / accumulation_steps).backward()
            total_loss_sum += float(loss.detach().float().cpu())
            imitation_loss_sum += float(metrics["imitation_loss"].float().cpu())
            gripper_bce_loss_sum += float(metrics["gripper_bce_loss"].float().cpu())
            rotation_geodesic_loss_sum += float(
                metrics["rotation_geodesic_loss"].float().cpu()
            )
            unweighted_loss_sum += float(
                metrics["sample_unweighted_imitation_loss"].mean().float().cpu()
            )
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        total_loss = total_loss_sum / accumulation_steps
        imitation_loss = imitation_loss_sum / accumulation_steps
        gripper_bce_loss = gripper_bce_loss_sum / accumulation_steps
        rotation_geodesic_loss = rotation_geodesic_loss_sum / accumulation_steps
        unweighted_loss = unweighted_loss_sum / accumulation_steps
        if step % args.log_every == 0 or step == 1:
            report_line(
                step,
                total_loss,
                imitation_loss,
                gripper_bce_loss,
                rotation_geodesic_loss,
                unweighted_loss,
                time.perf_counter() - started,
            )
            history.append(
                {
                    "step": step,
                    "total_loss": total_loss,
                    "weighted_imitation_loss": imitation_loss,
                    "gripper_bce_loss": gripper_bce_loss,
                    "rotation_geodesic_loss": rotation_geodesic_loss,
                    "unweighted_denoising_loss": unweighted_loss,
                    "grad_norm": float(torch.as_tensor(grad_norm).float().cpu()),
                }
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/total_loss": total_loss,
                        "train/weighted_imitation_loss": imitation_loss,
                        "train/gripper_bce_loss": gripper_bce_loss,
                        "train/rotation_geodesic_loss": rotation_geodesic_loss,
                        "train/unweighted_denoising_loss": unweighted_loss,
                        "train/grad_norm": float(
                            torch.as_tensor(grad_norm).float().cpu()
                        ),
                        "train/elapsed_seconds": time.perf_counter() - started,
                    },
                    step=step,
                )

        should_sample = step == args.steps or (
            args.sample_every > 0 and step % args.sample_every == 0
        )
        if should_sample:
            sampling_metrics, sampling_tensors = evaluate_real_sampling(
                model,
                fixed_batches,
                device=device,
                repeats=args.sampling_repeats,
                # Reuse the same initial diffusion noise at every checkpoint,
                # so sampling progress is an apples-to-apples comparison.
                seed=sampling_seed,
            )
            sampling_metrics["gripper_denoising_by_timestep"] = (
                evaluate_gripper_denoising_by_timestep(
                    model,
                    fixed_batches,
                    device=device,
                    seed=sampling_seed + 500_000,
                )
            )
            history.append({"step": step, "real_sampling": sampling_metrics})
            sample_snapshots[f"step_{step:06d}"] = sampling_tensors
            print(f"Real-sampling metrics at step {step}:")
            print(json.dumps(sampling_metrics, indent=2, default=json_default))
            if wandb_run is not None:
                wandb_run.log(wandb_sampling_metrics(sampling_metrics), step=step)

    artifact_dir = args.output_dir / "artifact"
    artifact_metadata = {
        "experiment": "cached_libero_overfit",
        "suite": args.suite,
        "split": args.split,
        "sample_indices": indices,
        "training_sample_count": len(indices),
        "streaming_online_siglip": bool(args.streaming_online_siglip),
        "real_sampling_audit_indices": audit_indices,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_global_batch": effective_global_batch,
        "approximate_sample_epochs": approximate_epochs,
        "horizon_weights_one_indexed": {
            "1-4": 5.0,
            "5-8": 3.0,
            "9-16": 2.0,
            "17-64": 1.0,
        },
        "inference_steps": args.inference_steps,
        "sampling_repeats": args.sampling_repeats,
        "sampling_seed": sampling_seed,
        "gripper_bce_weight": args.gripper_bce_weight,
        "gripper_bce_logit_scale": args.gripper_bce_logit_scale,
        "mask_noisy_gripper_input": bool(args.mask_noisy_gripper_input),
        "release_gripper_weight": args.release_gripper_weight,
        "release_oversample_factor": args.release_oversample_factor,
        "release_oversample_horizon": args.release_oversample_horizon,
        "release_sampling_audit": release_sampling_audit,
        "rotation_geodesic_weight": args.rotation_geodesic_weight,
    }
    save_trainable_artifact(model, artifact_dir, artifact_metadata)
    torch.save(sample_snapshots, args.output_dir / "sampled_predictions.pt")

    report: dict[str, Any] = {
        "configuration": artifact_metadata,
        "cache_root": cache_suite_root,
        "manifest": manifest,
        "artifact": artifact_dir,
        "history": history,
        "rollout": {"requested": bool(args.run_rollout)},
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )

    if args.run_rollout:
        del optimizer, model, fixed_batches
        if training_fixed_batches is not None:
            del training_fixed_batches
        if siglip_encoder is not None:
            del siglip_encoder
        if siglip_processor is not None:
            del siglip_processor
        gc.collect()
        torch.cuda.empty_cache()
        report["rollout"] = run_rollout(
            args,
            artifact_dir=artifact_dir,
            cache_suite_root=cache_suite_root,
        )
        report_path.write_text(
            json.dumps(report, indent=2, default=json_default) + "\n",
            encoding="utf-8",
        )
        if wandb_run is not None:
            rollout = report["rollout"]
            wandb_run.log(
                {
                    "rollout/episodes": rollout["episodes"],
                    "rollout/successes": rollout["successes"],
                    "rollout/success_rate": rollout["success_rate"],
                },
                step=args.steps,
            )
            episodes_path = Path(rollout["output_dir"]) / "episodes.jsonl"
            if episodes_path.exists():
                import wandb

                video_metrics = {}
                for line in episodes_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    episode = json.loads(line)
                    video_path = episode.get("video")
                    if video_path and Path(video_path).exists():
                        key = (
                            f"rollout/video_task{int(episode['task_id']):02d}_"
                            f"init{int(episode['init_state_index']):02d}"
                        )
                        video_metrics[key] = wandb.Video(
                            video_path,
                            fps=20,
                            format="mp4",
                        )
                if video_metrics:
                    wandb_run.log(video_metrics, step=args.steps)

    print(f"Saved artifact: {artifact_dir}")
    print(f"Saved report:   {report_path}")
    if wandb_run is not None:
        wandb_run.summary["artifact_path"] = str(artifact_dir)
        wandb_run.summary["report_path"] = str(report_path)
        wandb_run.finish()


if __name__ == "__main__":
    main()
