#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thinkflow_rdt.checkpoint import load_trainable_artifact  # noqa: E402
from thinkflow_rdt.config import load_config  # noqa: E402
from thinkflow_rdt.data import (  # noqa: E402
    CachedFeatureDataset,
    RDTBatchCollator,
    RDTOnlineSiglipBatchCollator,
)
from thinkflow_rdt.model import SFTConditionedRDT  # noqa: E402
from thinkflow_rdt.train import add_online_siglip_features, load_online_siglip  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a lightweight cached-sample sanity check against a trained RDT "
            "artifact. This is not a rollout; it checks whether cached inputs can "
            "be loaded and whether predicted chunks are numerically sane."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-mode",
        choices=["random", "first", "even"],
        default="random",
    )
    parser.add_argument(
        "--skip-diffusion-sampling",
        action="store_true",
        help="Only run the one-step imitation loss forward; skip sample_actions().",
    )
    parser.add_argument(
        "--loss-repeats",
        type=int,
        default=3,
        help=(
            "Repeat the random-timestep imitation-loss forward N times. The "
            "diffusion timestep/noise are random, so a few repeats make the "
            "sanity number less jumpy."
        ),
    )
    parser.add_argument(
        "--online-siglip",
        action="store_true",
        help="Use cached JPEG image slots and compute SigLIP features online.",
    )
    parser.add_argument(
        "--siglip-model-id",
        default="/home/ubuntu/RoboticsDiffusionTransformer/google/siglip-so400m-patch14-384",
    )
    parser.add_argument(
        "--siglip-fallback-model-id",
        default="google/siglip-so400m-patch14-384",
    )
    return parser.parse_args()


def selected_indices(
    dataset_size: int,
    *,
    count: int,
    seed: int,
    mode: str,
) -> list[int]:
    count = min(max(0, count), dataset_size)
    if mode == "first":
        return list(range(count))
    if mode == "even":
        if count <= 1:
            return [0] if count == 1 else []
        return [
            round(index * (dataset_size - 1) / (count - 1))
            for index in range(count)
        ]
    rng = random.Random(seed)
    return sorted(rng.sample(range(dataset_size), count))


def move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def masked_component_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    time_mask: torch.Tensor,
    dim_mask: torch.Tensor,
    start: int,
    stop: int,
) -> float:
    valid = time_mask.unsqueeze(-1).to(prediction.dtype) * dim_mask.unsqueeze(1).to(
        prediction.dtype
    )
    error = (prediction - target).pow(2) * valid
    denom = valid[..., start:stop].sum().clamp_min(1.0)
    return float((error[..., start:stop].sum() / denom).detach().cpu())


def tensor_stats(name: str, tensor: torch.Tensor) -> dict[str, float | str]:
    tensor = tensor.detach().float().cpu()
    return {
        f"{name}/min": float(tensor.min()),
        f"{name}/max": float(tensor.max()),
        f"{name}/mean": float(tensor.mean()),
        f"{name}/std": float(tensor.std(unbiased=False)),
    }


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.loss_repeats <= 0:
        raise ValueError("--loss-repeats must be positive")

    cfg = load_config(args.config)
    dataset = CachedFeatureDataset(
        args.manifest,
        required_keys=(
            {
                "qwen_kv",
                "lang_tokens",
                "image_slot_jpegs",
                "image_slot_mask",
                "state",
                "actions",
                "ctrl_freq",
            }
            if args.online_siglip
            else None
        ),
    )
    indices = selected_indices(
        len(dataset),
        count=args.num_samples,
        seed=args.seed,
        mode=args.sample_mode,
    )

    if args.online_siglip:
        collator = RDTOnlineSiglipBatchCollator(
            max_lang_tokens=cfg.model.max_lang_tokens,
            pred_horizon=cfg.model.pred_horizon,
            feature_dim=cfg.model.qwen_hidden_size,
            state_dim=cfg.model.state_dim,
            action_dim=cfg.model.action_dim,
            lang_token_dim=cfg.model.lang_token_dim,
            qwen_kv_dim=cfg.model.qwen_kv_dim,
        )
    else:
        collator = RDTBatchCollator(
            max_lang_tokens=cfg.model.max_lang_tokens,
            image_tokens=cfg.model.image_tokens,
            pred_horizon=cfg.model.pred_horizon,
            feature_dim=cfg.model.qwen_hidden_size,
            state_dim=cfg.model.state_dim,
            action_dim=cfg.model.action_dim,
            lang_token_dim=cfg.model.lang_token_dim,
            img_token_dim=cfg.model.img_token_dim,
            qwen_kv_dim=cfg.model.qwen_kv_dim,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    online_siglip = None
    if args.online_siglip:
        online_siglip = load_online_siglip(
            model_id=args.siglip_model_id,
            fallback_model_id=args.siglip_fallback_model_id,
            cfg=cfg,
            device=device,
        )

    model = SFTConditionedRDT(cfg, load_pretrained=True)
    load_trainable_artifact(model, args.checkpoint, trainable=False)
    model.to(device).eval()

    summaries: list[dict[str, Any]] = []
    saved_batches: list[dict[str, torch.Tensor]] = []
    saved_predictions: list[torch.Tensor] = []
    saved_targets: list[torch.Tensor] = []

    for start in range(0, len(indices), args.batch_size):
        batch_indices = indices[start : start + args.batch_size]
        samples = [dataset[index] for index in batch_indices]
        batch = collator(samples)
        batch = move_tensor_batch(batch, device)
        if online_siglip is not None:
            batch = add_online_siglip_features(
                batch,
                processor=online_siglip[0],
                encoder=online_siglip[1],
                cfg=cfg,
                device=device,
            )

        loss_metrics = []
        with torch.no_grad():
            for _ in range(args.loss_repeats):
                metrics = model(batch)
                loss_metrics.append(
                    {
                        "loss": float(metrics["loss"].detach().cpu()),
                        "loss_xyz": float(
                            (
                                metrics["xyz_loss_sum"]
                                / metrics["xyz_valid_count"].clamp_min(1.0)
                            )
                            .detach()
                            .cpu()
                        ),
                        "loss_rot": float(
                            (
                                metrics["rot_loss_sum"]
                                / metrics["rot_valid_count"].clamp_min(1.0)
                            )
                            .detach()
                            .cpu()
                        ),
                        "loss_gripper": float(
                            (
                                metrics["gripper_loss_sum"]
                                / metrics["gripper_valid_count"].clamp_min(1.0)
                            )
                            .detach()
                            .cpu()
                        ),
                    }
                )

            prediction = None
            sample_metrics: dict[str, float] = {}
            if not args.skip_diffusion_sampling:
                prediction = model.sample_actions(batch).float()
                target = batch["actions"].to(prediction.dtype)
                sample_metrics = {
                    "sample_mse": masked_component_mse(
                        prediction,
                        target,
                        batch["action_time_mask"],
                        batch["action_dim_mask"],
                        0,
                        cfg.model.action_dim,
                    ),
                    "sample_mse_xyz": masked_component_mse(
                        prediction,
                        target,
                        batch["action_time_mask"],
                        batch["action_dim_mask"],
                        0,
                        3,
                    ),
                    "sample_mse_rot": masked_component_mse(
                        prediction,
                        target,
                        batch["action_time_mask"],
                        batch["action_dim_mask"],
                        3,
                        6,
                    ),
                    "sample_mse_gripper": masked_component_mse(
                        prediction,
                        target,
                        batch["action_time_mask"],
                        batch["action_dim_mask"],
                        6,
                        7,
                    ),
                }
                sample_metrics.update(tensor_stats("prediction", prediction))
                sample_metrics.update(tensor_stats("target", target))
                saved_predictions.append(prediction.detach().cpu())
                saved_targets.append(target.detach().cpu())

        averaged_loss = {
            key: sum(item[key] for item in loss_metrics) / len(loss_metrics)
            for key in loss_metrics[0]
        }
        target_gripper_open = batch["actions"][..., 6].detach().float().cpu()
        summaries.append(
            {
                "indices": batch_indices,
                "metadata": [sample.get("metadata", {}) for sample in samples],
                "loss_forward": averaged_loss,
                "target_gripper_open_mean": float(target_gripper_open.mean()),
                "valid_action_tokens": int(batch["action_time_mask"].sum().item()),
                **sample_metrics,
            }
        )
        saved_batches.append(
            {
                key: value.detach().cpu()
                for key, value in batch.items()
                if torch.is_tensor(value)
            }
        )

    print(json.dumps({"checkpoint": str(args.checkpoint), "indices": indices}, indent=2))
    for item in summaries:
        print(json.dumps(item, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": str(Path(args.config).resolve()),
                "checkpoint": str(args.checkpoint.resolve()),
                "manifest": str(args.manifest.resolve()),
                "indices": indices,
                "summaries": summaries,
                "batches": saved_batches,
                "predictions": saved_predictions,
                "targets": saved_targets,
            },
            args.output,
        )
        print(f"Saved sanity output to {args.output.resolve()}")


if __name__ == "__main__":
    main()
