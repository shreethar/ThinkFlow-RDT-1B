#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from thinkflow_rdt.checkpoint import (
    load_trainable_artifact,
    save_trainable_artifact,
)
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
        description=(
            "Run a requested number of full-finetune backward microsteps, time "
            "complete four-microbatch updates, then checkpoint/reload/sample "
            "validation."
        )
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
        default=Path("outputs/smoke_native128_full_finetune"),
    )
    parser.add_argument("--backpasses", type=int, default=10)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Skip checkpoint/reload/generation after timing the backward passes.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable RDT block checkpointing for the smoke run or benchmark.",
    )
    parser.add_argument(
        "--siglip-model-id",
        default="/home/ubuntu/models/siglip-so400m-patch14-384",
    )
    parser.add_argument(
        "--siglip-fallback-model-id",
        default="google/siglip-so400m-patch14-384",
    )
    return parser.parse_args()


class GPUMonitor:
    """Sample NVML utilization, device memory, and power during a benchmark."""

    def __init__(self, device_index: int, interval_seconds: float = 0.1):
        import pynvml

        self.pynvml = pynvml
        self.pynvml.nvmlInit()
        self.handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.samples: list[tuple[float, float, float]] = []
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            utilization = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            memory = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            try:
                power_watts = (
                    self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                )
            except self.pynvml.NVMLError:
                power_watts = math.nan
            self.samples.append(
                (
                    float(utilization.gpu),
                    float(memory.used) / 1024.0**3,
                    power_watts,
                )
            )
            self.stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> dict[str, float]:
        self.stop_event.set()
        self.thread.join()
        self.pynvml.nvmlShutdown()
        if not self.samples:
            return {}
        utilizations = [sample[0] for sample in self.samples]
        memory_used = [sample[1] for sample in self.samples]
        power = [sample[2] for sample in self.samples if math.isfinite(sample[2])]
        sorted_utilization = sorted(utilizations)
        p95_index = min(
            len(sorted_utilization) - 1,
            math.ceil(0.95 * len(sorted_utilization)) - 1,
        )
        return {
            "gpu_monitor_samples": float(len(self.samples)),
            "gpu_utilization_mean_percent": statistics.fmean(utilizations),
            "gpu_utilization_p95_percent": sorted_utilization[p95_index],
            "gpu_utilization_max_percent": max(utilizations),
            "gpu_device_memory_used_max_gib": max(memory_used),
            "gpu_power_mean_watts": (
                statistics.fmean(power) if power else math.nan
            ),
            "gpu_power_max_watts": max(power) if power else math.nan,
        }


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def main() -> None:
    args = parse_args()
    if args.backpasses < args.gradient_accumulation_steps:
        raise ValueError("backpasses must include at least one complete update")
    if not torch.cuda.is_available():
        raise RuntimeError("This full RDT-1B benchmark requires CUDA")
    device = torch.device("cuda")
    cache_root = args.cache_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoint-after-smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg = replace(
        cfg,
        output_dir=str(output_dir),
        model=replace(
            cfg.model,
            gradient_checkpointing=not args.no_gradient_checkpointing,
        ),
        data=replace(
            cfg.data,
            train_manifest=str(cache_root / "train" / "manifest.jsonl"),
            val_manifest=str(cache_root / "validation" / "manifest.jsonl"),
            num_workers=0,
            persistent_workers=False,
        ),
        training=replace(
            cfg.training,
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            global_batch_size=(
                args.micro_batch_size * args.gradient_accumulation_steps
            ),
            report_to="none",
        ),
    )
    cfg.validate()

    train_loader = create_dataloader(
        cfg.data.train_manifest,
        cfg,
        shuffle=True,
        online_siglip=True,
    )
    val_loader = create_dataloader(
        cfg.data.val_manifest,
        cfg,
        shuffle=False,
        online_siglip=True,
        stratified=True,
    )
    model = SFTConditionedRDT(cfg, load_pretrained=True).to(device)
    optimizer = create_optimizer(model, cfg)
    siglip = load_online_siglip(
        model_id=args.siglip_model_id,
        fallback_model_id=args.siglip_fallback_model_id,
        cfg=cfg,
        device=device,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    gpu_monitor = GPUMonitor(device.index or 0)
    gpu_monitor.start()

    iterator = iter(train_loader)
    update_times: list[float] = []
    backward_times: list[float] = []
    losses: list[float] = []
    update_started = time.perf_counter()
    for microstep in range(1, args.backpasses + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        batch = add_online_siglip_features(
            batch,
            processor=siglip[0],
            encoder=siglip[1],
            cfg=cfg,
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
        backward_started = time.perf_counter()
        metrics = model(batch)
        loss = metrics["loss"]
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError(f"Non-finite loss at microstep {microstep}")
        (loss / args.gradient_accumulation_steps).backward()
        torch.cuda.synchronize(device)
        backward_times.append(time.perf_counter() - backward_started)
        losses.append(float(loss.detach().cpu()))

        if microstep % args.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.max_grad_norm
            )
            if not bool(torch.isfinite(torch.as_tensor(grad_norm)).all()):
                raise FloatingPointError(
                    f"Non-finite gradient norm at microstep {microstep}"
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            update_times.append(time.perf_counter() - update_started)
            update_started = time.perf_counter()

    # The last two backward passes in the requested ten do not form a complete
    # accumulation window. Discard them rather than performing a half-sized step.
    optimizer.zero_grad(set_to_none=True)
    gpu_statistics = gpu_monitor.stop()
    effective_batch_size = (
        args.micro_batch_size * args.gradient_accumulation_steps
    )
    timing_report = {
        "backpasses": args.backpasses,
        "complete_optimizer_steps": len(update_times),
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": cfg.model.gradient_checkpointing,
        "effective_batch_size": effective_batch_size,
        "full_step_times_sec": update_times,
        "mean_full_step_time_sec": (
            sum(update_times) / len(update_times) if update_times else math.nan
        ),
        "steady_full_step_time_sec": (
            statistics.fmean(update_times[1:])
            if len(update_times) > 1
            else (update_times[0] if update_times else math.nan)
        ),
        "steady_samples_per_sec": (
            effective_batch_size / statistics.fmean(update_times[1:])
            if len(update_times) > 1
            else (
                effective_batch_size / update_times[0]
                if update_times
                else math.nan
            )
        ),
        "mean_backward_microstep_time_sec": (
            sum(backward_times) / len(backward_times)
        ),
        "mean_training_loss": sum(losses) / len(losses),
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated(device) / 1024.0**3
        ),
        **gpu_statistics,
    }
    if args.benchmark_only:
        report_path = output_dir / "benchmark_report.json"
        report_path.write_text(json.dumps(timing_report, indent=2) + "\n")
        print(json.dumps(timing_report, indent=2))
        return

    save_trainable_artifact(
        model,
        checkpoint_dir,
        {
            "smoke_test": True,
            "backpasses": args.backpasses,
            "complete_optimizer_steps": len(update_times),
            "config": asdict(cfg),
        },
    )

    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()

    reload_cfg = replace(
        cfg,
        model=replace(cfg.model, allow_random_frozen_state_adaptor=True),
    )
    reloaded = SFTConditionedRDT(reload_cfg, load_pretrained=False)
    load_trainable_artifact(reloaded, checkpoint_dir, trainable=False)
    reloaded.to(device).eval()
    validation_batch = move_batch(next(iter(val_loader)), device)
    validation_batch = add_online_siglip_features(
        validation_batch,
        processor=siglip[0],
        encoder=siglip[1],
        cfg=cfg,
        device=device,
    )
    with torch.no_grad():
        validation_metrics = reloaded(validation_batch)
        generated = reloaded(validation_batch, sample=True)
    target = validation_batch["actions"].to(generated)
    valid = (
        validation_batch["action_time_mask"].unsqueeze(-1).to(generated)
        * validation_batch["action_dim_mask"].unsqueeze(1).to(generated)
    )
    generated_mse = float(
        (((generated - target).square() * valid).sum() / valid.sum()).cpu()
    )

    report = {
        **timing_report,
        "reloaded_validation_loss": float(
            validation_metrics["loss"].detach().cpu()
        ),
        "reloaded_validation_generation_mse": generated_mse,
        "generated_shape": list(generated.shape),
        "active_action_indices": torch.nonzero(
            validation_batch["action_dim_mask"][0]
        ).flatten().cpu().tolist(),
        "checkpoint": str(checkpoint_dir),
    }
    report_path = output_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
