#!/usr/bin/env python
"""Run the production validation path on a small real cached subset."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from accelerate import Accelerator

from thinkflow_rdt.checkpoint import load_trainable_artifact
from thinkflow_rdt.config import load_config
from thinkflow_rdt.model import SFTConditionedRDT
from thinkflow_rdt.train import create_dataloader, load_online_siglip, validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/part3_rdt1b.yaml")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("cache_features/part_1_32frame_per_sample_qwen"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--qualitative-examples", type=int, default=1)
    parser.add_argument(
        "--siglip-model-id",
        default="/home/ubuntu/models/siglip-so400m-patch14-384",
    )
    parser.add_argument(
        "--siglip-fallback-model-id",
        default="google/siglip-so400m-patch14-384",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    accelerator = Accelerator(mixed_precision="bf16")
    cfg = load_config(args.config)
    cfg = replace(
        cfg,
        data=replace(
            cfg.data,
            val_manifest=str(
                args.cache_root.expanduser().resolve()
                / "validation"
                / "manifest.jsonl"
            ),
            num_workers=0,
            persistent_workers=False,
            shuffle_validation=False,
        ),
        training=replace(
            cfg.training,
            micro_batch_size=args.samples,
            gradient_accumulation_steps=1,
            global_batch_size=args.samples * accelerator.num_processes,
            validation_samples=args.samples * accelerator.num_processes,
            validation_batches=1,
            validation_batch_size=args.samples,
            sample_validation_batches=1,
            qualitative_validation_examples=args.qualitative_examples,
            # This constructs the exact wandb.Table object but does not create
            # or upload a W&B run.
            report_to="wandb",
        ),
    )
    cfg.validate()

    loader = create_dataloader(
        cfg.data.val_manifest,
        cfg,
        shuffle=False,
        online_siglip=True,
        stratified=cfg.data.stratified_validation,
    )
    model = SFTConditionedRDT(cfg, load_pretrained=True)
    if args.checkpoint is not None:
        load_trainable_artifact(model, args.checkpoint, trainable=False)
    model, loader = accelerator.prepare(model, loader)
    online_siglip = load_online_siglip(
        model_id=args.siglip_model_id,
        fallback_model_id=args.siglip_fallback_model_id,
        cfg=cfg,
        device=accelerator.device,
    )

    metrics = validate(
        model,
        loader,
        accelerator,
        cfg,
        online_siglip=online_siglip,
    )
    if not accelerator.is_main_process:
        return

    table = metrics.pop("val/qualitative_trajectories", None)
    report = {
        "status": "VALIDATE_OK",
        "scalar_metric_count": len(metrics),
        "sample_mse": metrics.get("val/sample_mse"),
        "gripper_accuracy": metrics.get(
            "val/sampled_native10/horizon_10/gripper_command/accuracy"
        ),
        "gripper_f1": metrics.get(
            "val/sampled_native10/horizon_10/gripper_command/f1"
        ),
        "gripper_transition_f1": metrics.get(
            "val/sampled_native10/horizon_10/gripper_transition/f1"
        ),
        "qualitative_table_type": (
            type(table).__name__ if table is not None else None
        ),
        "qualitative_table_columns": (
            list(table.columns) if table is not None else []
        ),
        "qualitative_table_rows": (
            len(table.data) if table is not None else 0
        ),
        "qwen_ablation_metric_count": sum(
            key.startswith("val/qwen_ablation/") for key in metrics
        ),
    }
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
