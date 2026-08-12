"""Fine-tune Qwen-KV SmolVLA directly from the existing LIBERO cache shards.

Run from the repository root:

    .venv/bin/python -m experiments.smolvla_qwen_kv.train_cached --help
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

from .cached_libero import CachedLiberoIterableDataset, list_shards
from .configuration import make_libero_kv_config
from .modeling import KVSmolVLAPolicy
from .stats import load_or_compute_cache_stats


LIBERO_SUITES = ("libero_10", "libero_spatial", "libero_goal", "libero_object")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", default="lerobot/smolvla_base")
    parser.add_argument("--cache-root", default="cache_features_libero_b0_raw_ortho6d")
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=LIBERO_SUITES,
        default=list(LIBERO_SUITES),
        help="Suites combined into one training stream (default: all four standard suites)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-10)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-vlm", action="store_true")
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--external-logit-bias-init", type=float, default=-4.0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_multiplier(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(step, 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def prepare_batch(raw_batch: dict, preprocessor, *, device: torch.device) -> dict:
    # LeRobot's canonical batch converter intentionally preserves only registered
    # observation/action/metadata keys. Keep custom Qwen KV out of that conversion
    # and attach it again after normalization/tokenization/device transfer.
    qwen_kv = raw_batch.pop("qwen_kv")
    batch = preprocessor(raw_batch)
    batch["qwen_kv"] = qwen_kv.to(device=device, non_blocking=True)
    return batch


@torch.no_grad()
def sampled_metrics(policy: KVSmolVLAPolicy, batch: dict) -> dict[str, float]:
    was_training = policy.training
    policy.eval()
    predicted = policy.predict_action_chunk(dict(batch))
    target = batch["action"][:, : predicted.shape[1], : predicted.shape[2]]
    valid = ~batch.get(
        "action_is_pad",
        torch.zeros(target.shape[:2], dtype=torch.bool, device=target.device),
    )[:, : predicted.shape[1]]
    error = (predicted - target).square()
    denominator = (valid.sum() * predicted.shape[-1]).clamp_min(1)
    rmse = torch.sqrt((error * valid.unsqueeze(-1)).sum() / denominator)
    motion_width = min(6, predicted.shape[-1])
    motion_denominator = (valid.sum() * motion_width).clamp_min(1)
    motion_rmse = torch.sqrt(
        (error[..., :motion_width] * valid.unsqueeze(-1)).sum() / motion_denominator
    )
    metrics = {
        "sampled_rmse_7d": float(rmse),
        "sampled_motion_rmse_6d": float(motion_rmse),
    }
    if predicted.shape[-1] >= 7:
        predicted_gripper = predicted[..., 6] >= 0
        target_gripper = target[..., 6] >= 0
        metrics["sampled_gripper_accuracy"] = float(
            ((predicted_gripper == target_gripper) & valid).sum() / valid.sum().clamp_min(1)
        )
    if was_training:
        policy.train()
    return metrics


def save_checkpoint(
    policy: KVSmolVLAPolicy,
    preprocessor,
    postprocessor,
    optimizer: torch.optim.Optimizer,
    scheduler,
    output_dir: Path,
    step: int,
) -> Path:
    checkpoint = output_dir / f"checkpoint-{step:06d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint)
    preprocessor.save_pretrained(checkpoint)
    postprocessor.save_pretrained(checkpoint)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint / "training_state.pt",
    )
    return checkpoint


def main() -> None:
    args = parse_args()
    if args.local_files_only:
        # Upstream SmolVLA constructs its nested AutoProcessor without forwarding
        # local_files_only. Offline mode makes that nested load obey this CLI flag.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.steps <= 0 or args.batch_size <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("steps, batch-size, and gradient-accumulation must be positive")
    if args.chunk_size > 64:
        raise ValueError("The current LIBERO cache contains at most 64 action targets")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    seed_everything(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the user-provided order but reject duplicates so a suite cannot be
    # accidentally oversampled merely by being listed twice.
    suites = list(dict.fromkeys(args.suites))
    suite_shards = {
        suite: list_shards(args.cache_root, suite, "train") for suite in suites
    }
    shard_paths = [path for suite in suites for path in suite_shards[suite]]
    stats, num_samples = load_or_compute_cache_stats(
        output_dir / "cache_stats.pt",
        shard_paths,
        chunk_size=args.chunk_size,
    )
    config = make_libero_kv_config(
        args.pretrained,
        device=str(device),
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        train_expert_only=not args.train_vlm,
        freeze_vision_encoder=not args.train_vlm,
        external_kv_logit_bias_init=args.external_logit_bias_init,
        local_files_only=args.local_files_only,
    )
    policy = KVSmolVLAPolicy.from_pretrained(
        args.pretrained,
        config=config,
        local_files_only=args.local_files_only,
        strict=False,
    )
    policy.train()
    preprocessor, postprocessor = make_smolvla_pre_post_processors(config, dataset_stats=stats)

    dataset = CachedLiberoIterableDataset(
        shard_paths,
        chunk_size=args.chunk_size,
        seed=args.seed,
        repeat=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    iterator = iter(loader)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda current: cosine_multiplier(
            current,
            warmup_steps=args.warmup_steps,
            total_steps=args.steps,
        ),
    )

    run_config = vars(args) | {
        "suites": suites,
        "shards_per_suite": {
            suite: len(paths) for suite, paths in suite_shards.items()
        },
        "shards": len(shard_paths),
        "samples": num_samples,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in policy.parameters()),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    print(json.dumps(run_config, indent=2))

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or "smolvla-base-qwen-kv-all-suites",
            config=run_config,
        )

    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    last_batch = None
    for step in range(1, args.steps + 1):
        accumulated_loss = 0.0
        for _ in range(args.gradient_accumulation):
            raw_batch = next(iterator)
            batch = prepare_batch(raw_batch, preprocessor, device=device)
            last_batch = batch
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=args.bf16 and device.type == "cuda",
            ):
                loss, _ = policy.forward(batch)
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            accumulated_loss += float(loss.detach()) / args.gradient_accumulation

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        metrics = {
            "step": step,
            "train/loss": accumulated_loss,
            "train/grad_norm": float(grad_norm),
            "train/lr": scheduler.get_last_lr()[0],
            "train/elapsed_sec": time.perf_counter() - started,
        }
        if step == 1 or step % args.log_every == 0:
            biases = policy.model.vlm_with_expert.external_logit_biases
            metrics["train/external_logit_bias_mean"] = float(
                torch.stack([value.detach().float().mean() for value in biases.values()]).mean()
            )
            print(json.dumps(metrics))
        if args.sample_every > 0 and step % args.sample_every == 0 and last_batch is not None:
            sample_values = sampled_metrics(policy, last_batch)
            metrics.update({f"validation/{key}": value for key, value in sample_values.items()})
            print(json.dumps({"step": step, **sample_values}, indent=2))
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)
        if args.save_every > 0 and step % args.save_every == 0:
            checkpoint = save_checkpoint(
                policy,
                preprocessor,
                postprocessor,
                optimizer,
                scheduler,
                output_dir,
                step,
            )
            print(f"Saved {checkpoint}")

    if args.save_every <= 0 or args.steps % args.save_every != 0:
        save_checkpoint(
            policy,
            preprocessor,
            postprocessor,
            optimizer,
            scheduler,
            output_dir,
            args.steps,
        )
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
