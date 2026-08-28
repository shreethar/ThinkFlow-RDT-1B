#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

from smoke_native128_full_finetune import GPUMonitor, move_batch
from thinkflow_rdt.config import load_config
from thinkflow_rdt.model import SFTConditionedRDT
from thinkflow_rdt.train import (
    add_online_siglip_features,
    attach_training_objective,
    create_dataloader,
    create_optimizer,
    load_online_siglip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark microbatch/gradient-accumulation combinations."
    )
    parser.add_argument("--config", default="configs/part3_rdt1b.yaml")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("cache_features/part_1_32frame_per_sample_qwen"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/batch_accum_sweep"),
    )
    parser.add_argument(
        "--micro-batch-sizes",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16, 24],
    )
    parser.add_argument(
        "--gradient-accumulations",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--timed-updates", type=int, default=3)
    parser.add_argument("--projected-training-steps", type=int, default=20_000)
    parser.add_argument(
        "--siglip-model-id",
        default="/home/ubuntu/models/siglip-so400m-patch14-384",
    )
    parser.add_argument(
        "--siglip-fallback-model-id",
        default="google/siglip-so400m-patch14-384",
    )
    return parser.parse_args()


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The RDT-1B sweep requires CUDA")
    if args.warmup_updates < 1 or args.timed_updates < 1:
        raise ValueError("warmup-updates and timed-updates must both be positive")
    if min(args.micro_batch_sizes) < 1:
        raise ValueError("micro-batch-sizes must be positive")
    if min(args.gradient_accumulations) < 1:
        raise ValueError("gradient-accumulations must be positive")

    device = torch.device("cuda")
    cache_root = args.cache_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(args.config)
    base_cfg = replace(
        base_cfg,
        output_dir=str(output_dir),
        data=replace(
            base_cfg.data,
            train_manifest=str(cache_root / "train" / "manifest.jsonl"),
            val_manifest=str(cache_root / "validation" / "manifest.jsonl"),
            num_workers=0,
            persistent_workers=False,
        ),
        training=replace(base_cfg.training, report_to="none"),
    )
    base_cfg.validate()

    model = SFTConditionedRDT(base_cfg, load_pretrained=True).to(device)
    optimizer = create_optimizer(model, base_cfg)
    siglip = load_online_siglip(
        model_id=args.siglip_model_id,
        fallback_model_id=args.siglip_fallback_model_id,
        cfg=base_cfg,
        device=device,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)

    results: list[dict[str, float | int | list[float]]] = []
    for micro_batch_size in args.micro_batch_sizes:
        loader_cfg = replace(
            base_cfg,
            training=replace(
                base_cfg.training,
                micro_batch_size=micro_batch_size,
            ),
        )
        loader = create_dataloader(
            loader_cfg.data.train_manifest,
            loader_cfg,
            shuffle=True,
            online_siglip=True,
        )
        iterator = iter(loader)

        for accumulation in args.gradient_accumulations:
            total_updates = args.warmup_updates + args.timed_updates
            update_times: list[float] = []
            losses: list[float] = []
            torch.cuda.reset_peak_memory_stats(device)
            monitor = GPUMonitor(device.index or 0)
            monitor.start()

            for _ in range(total_updates):
                update_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                for _ in range(accumulation):
                    batch, iterator = next_batch(iterator, loader)
                    batch = move_batch(batch, device)
                    batch = add_online_siglip_features(
                        batch,
                        processor=siglip[0],
                        encoder=siglip[1],
                        cfg=loader_cfg,
                        device=device,
                    )
                    attach_training_objective(
                        batch,
                        horizon_loss_weights=None,
                        xyz_loss_weight=0.0,
                        gripper_bce_weight=0.0,
                        gripper_bce_logit_scale=1.0,
                        rotation_geodesic_weight=0.0,
                    )
                    metrics = model(batch)
                    loss = metrics["loss"]
                    if not bool(torch.isfinite(loss).all()):
                        raise FloatingPointError(
                            "Non-finite loss for "
                            f"microbatch={micro_batch_size}, accumulation={accumulation}"
                        )
                    (loss / accumulation).backward()
                    losses.append(float(loss.detach().cpu()))

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), loader_cfg.training.max_grad_norm
                )
                if not bool(torch.isfinite(torch.as_tensor(grad_norm)).all()):
                    raise FloatingPointError(
                        "Non-finite gradient norm for "
                        f"microbatch={micro_batch_size}, accumulation={accumulation}"
                    )
                optimizer.step()
                torch.cuda.synchronize(device)
                update_times.append(time.perf_counter() - update_started)

            gpu_statistics = monitor.stop()
            timed_step_times = update_times[args.warmup_updates :]
            mean_step_seconds = statistics.fmean(timed_step_times)
            effective_batch_size = micro_batch_size * accumulation
            result: dict[str, float | int | list[float]] = {
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": accumulation,
                "effective_batch_size": effective_batch_size,
                "warmup_updates": args.warmup_updates,
                "timed_updates": args.timed_updates,
                "timed_step_times_sec": timed_step_times,
                "mean_step_time_sec": mean_step_seconds,
                "samples_per_sec": effective_batch_size / mean_step_seconds,
                "projected_training_steps": args.projected_training_steps,
                "projected_training_hours": (
                    mean_step_seconds * args.projected_training_steps / 3600.0
                ),
                "mean_loss": statistics.fmean(losses),
                "peak_cuda_memory_gib": (
                    torch.cuda.max_memory_allocated(device) / 1024.0**3
                ),
                **gpu_statistics,
            }
            results.append(result)
            print(json.dumps(result), flush=True)

    finite_results = [
        result
        for result in results
        if math.isfinite(float(result["mean_step_time_sec"]))
    ]
    report = {
        "device": torch.cuda.get_device_name(device),
        "config": str(Path(args.config).resolve()),
        "cache_root": str(cache_root),
        "methodology": {
            "warmup_updates_excluded": args.warmup_updates,
            "timed_updates": args.timed_updates,
            "timing_includes": [
                "data_loading",
                "online_frozen_siglip",
                "rdt_forward",
                "masked_diffusion_loss",
                "backward",
                "gradient_clipping",
                "optimizer_step",
            ],
        },
        "fastest_20k_optimizer_steps": min(
            finite_results,
            key=lambda result: float(result["projected_training_hours"]),
        ),
        "highest_sample_throughput": max(
            finite_results,
            key=lambda result: float(result["samples_per_sec"]),
        ),
        "results": results,
    }
    report_path = output_dir / "batch_accum_sweep.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved sweep report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
