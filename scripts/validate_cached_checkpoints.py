#!/usr/bin/env python
"""Post-hoc validation for one or more cached-feature RDT checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import torch
from accelerate import Accelerator

from thinkflow_rdt.checkpoint import load_trainable_artifact
from thinkflow_rdt.config import load_config
from thinkflow_rdt.model import SFTConditionedRDT
from thinkflow_rdt.train import (
    create_dataloader,
    load_online_siglip,
    unwrap_model_without_optional_deepspeed,
    validate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/part3_rdt1b.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Checkpoint directory. Repeat in the desired comparison order.",
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=Path("output_2/manifests/val_manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_2/posthoc_validation_5k_20k"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--sample-validation-batches", type=int, default=1)
    parser.add_argument("--qualitative-examples", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--wandb-mode",
        choices=("offline", "online", "disabled"),
        default="offline",
    )
    parser.add_argument("--wandb-project", default="ThinkLite B0 OXE")
    parser.add_argument(
        "--wandb-run-name",
        default="posthoc-validation-steps-5k-10k-15k-20k",
    )
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--siglip-model-id",
        default="/home/ubuntu/models/siglip-so400m-patch14-384",
    )
    parser.add_argument(
        "--siglip-fallback-model-id",
        default="google/siglip-so400m-patch14-384",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Hugging Face to access the network instead of cached files only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate checkpoints whose scalar JSON report already exists.",
    )
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    step = int(metadata.get("global_step", -1))
    if step < 0:
        raise ValueError(f"No valid global_step in {metadata_path}")
    for required in ("rdt_full.pt", "interfaces.pt"):
        if not (path / required).is_file():
            raise FileNotFoundError(path / required)
    return step


def scalar_metrics(metrics: dict[str, object]) -> dict[str, float | int]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def save_qualitative_table(table, output_dir: Path) -> dict[str, object]:
    """Copy W&B table media to ordinary local files as an offline fallback."""
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(table.columns)
    rows: list[dict[str, object]] = []
    for row_index, values in enumerate(table.data):
        record: dict[str, object] = {}
        for column, value in zip(columns, values):
            media_path = getattr(value, "_path", None)
            if media_path is None:
                record[column] = value
                continue
            source = Path(media_path)
            suffix = source.suffix or ".png"
            destination = output_dir / f"{row_index:03d}_{column}{suffix}"
            shutil.copy2(source, destination)
            record[column] = destination.name
        rows.append(record)
    payload = {"columns": columns, "rows": rows}
    with (output_dir / "rows.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return {
        "columns": columns,
        "rows": len(rows),
        "directory": str(output_dir.resolve()),
    }


def write_comparison_csv(reports: list[dict[str, object]], path: Path) -> None:
    rows = []
    for report in reports:
        row = {
            "checkpoint_step": report["checkpoint_step"],
            "checkpoint": report["checkpoint"],
            **report["metrics"],
        }
        rows.append(row)
    columns = ["checkpoint_step", "checkpoint"] + sorted(
        set().union(*(set(row) for row in rows))
        - {"checkpoint_step", "checkpoint"}
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def report_matches_request(
    report: dict[str, object],
    *,
    batch_size: int,
    validation_samples: int,
    sample_validation_batches: int,
    qualitative_examples: int,
) -> bool:
    qualitative = report.get("qualitative") or {}
    return (
        int(report.get("batch_size", -1)) == batch_size
        and int(report.get("validation_samples", -1)) == validation_samples
        and int(report.get("sample_validation_batches", -1))
        == sample_validation_batches
        and int(qualitative.get("rows", -1)) == qualitative_examples
    )


def main() -> None:
    args = parse_args()
    checkpoints = [path.expanduser().resolve() for path in args.checkpoint]
    steps = [checkpoint_step(path) for path in checkpoints]
    if len(set(steps)) != len(steps):
        raise ValueError(f"Checkpoint steps must be unique, got {steps}")
    if args.batch_size <= 0 or args.validation_samples <= 0:
        raise ValueError("Batch size and validation sample count must be positive")
    if args.qualitative_examples < 0:
        raise ValueError("Qualitative example count cannot be negative")
    qualitative_capacity = (
        args.batch_size * args.sample_validation_batches
    )
    if args.qualitative_examples > qualitative_capacity:
        raise ValueError(
            "qualitative-examples exceeds sampled validation capacity: "
            f"{args.qualitative_examples} > {qualitative_capacity}"
        )

    if not args.allow_network:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_reports: list[dict[str, object]] = []
    if not args.force:
        for step in steps:
            report_path = output_dir / f"checkpoint_{step}_metrics.json"
            if not report_path.is_file():
                break
            with report_path.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
            if not report_matches_request(
                report,
                batch_size=args.batch_size,
                validation_samples=args.validation_samples,
                sample_validation_batches=args.sample_validation_batches,
                qualitative_examples=args.qualitative_examples,
            ):
                break
            completed_reports.append(report)
    if len(completed_reports) == len(checkpoints):
        completed_reports.sort(
            key=lambda report: int(report["checkpoint_step"])
        )
        write_comparison_csv(
            completed_reports,
            output_dir / "checkpoint_comparison.csv",
        )
        with (output_dir / "summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(completed_reports, handle, indent=2, allow_nan=True)
        print(
            "All requested checkpoint reports already exist with matching "
            f"settings in {output_dir}. Use --force to re-evaluate."
        )
        return
    if args.wandb_mode != "disabled":
        os.environ["WANDB_MODE"] = args.wandb_mode
        os.environ["WANDB_DIR"] = str(output_dir)

    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with=("wandb" if args.wandb_mode != "disabled" else None),
        project_dir=str(output_dir / "logs"),
    )
    if args.validation_samples % (
        args.batch_size * accelerator.num_processes
    ):
        raise ValueError(
            "validation-samples must be divisible by batch-size * world-size"
        )

    cfg = load_config(args.config)
    validation_manifest = args.validation_manifest.expanduser().resolve()
    if not validation_manifest.is_file():
        raise FileNotFoundError(validation_manifest)
    cfg = replace(
        cfg,
        model=replace(
            cfg.model,
            # Every frozen component is immediately restored from the artifact;
            # this avoids requiring the pretrained hub model while offline.
            allow_random_frozen_state_adaptor=True,
        ),
        data=replace(
            cfg.data,
            val_manifest=str(validation_manifest),
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            shuffle_validation=False,
            stratified_validation=True,
        ),
        training=replace(
            cfg.training,
            micro_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            global_batch_size=args.batch_size * accelerator.num_processes,
            validation_samples=args.validation_samples,
            validation_batches=(
                args.validation_samples
                // (args.batch_size * accelerator.num_processes)
            ),
            sample_validation_batches=args.sample_validation_batches,
            qualitative_validation_examples=args.qualitative_examples,
            report_to=(
                "wandb" if args.wandb_mode != "disabled" else "none"
            ),
        ),
    )
    cfg.validate()

    if args.wandb_mode != "disabled":
        wandb_kwargs: dict[str, object] = {
            "name": args.wandb_run_name,
            "mode": args.wandb_mode,
            "group": "posthoc-checkpoint-validation",
            "job_type": "validation",
        }
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        accelerator.init_trackers(
            args.wandb_project,
            config={
                "checkpoints": [str(path) for path in checkpoints],
                "checkpoint_steps": steps,
                "validation_config": asdict(cfg),
                "posthoc": {
                    "batch_size": args.batch_size,
                    "validation_samples": args.validation_samples,
                    "qualitative_examples": args.qualitative_examples,
                    "fixed_validation_seed": cfg.training.validation_seed,
                },
            },
            init_kwargs={"wandb": wandb_kwargs},
        )

    loader = create_dataloader(
        cfg.data.val_manifest,
        cfg,
        shuffle=False,
        online_siglip=True,
        stratified=True,
    )
    model = SFTConditionedRDT(cfg, load_pretrained=False)
    load_trainable_artifact(model, checkpoints[0], trainable=False)
    model, loader = accelerator.prepare(model, loader)
    online_siglip = load_online_siglip(
        model_id=args.siglip_model_id,
        fallback_model_id=args.siglip_fallback_model_id,
        cfg=cfg,
        device=accelerator.device,
    )

    reports: list[dict[str, object]] = []
    for checkpoint, step in zip(checkpoints, steps):
        report_path = output_dir / f"checkpoint_{step}_metrics.json"
        if report_path.exists() and not args.force:
            with report_path.open("r", encoding="utf-8") as handle:
                existing_report = json.load(handle)
            if report_matches_request(
                existing_report,
                batch_size=args.batch_size,
                validation_samples=args.validation_samples,
                sample_validation_batches=args.sample_validation_batches,
                qualitative_examples=args.qualitative_examples,
            ):
                reports.append(existing_report)
                if accelerator.is_main_process:
                    print(f"Skipping completed report for step {step}")
                continue

        accelerator.wait_for_everyone()
        unwrapped = unwrap_model_without_optional_deepspeed(
            accelerator,
            model,
        )
        load_trainable_artifact(unwrapped, checkpoint, trainable=False)
        torch.cuda.empty_cache()
        if accelerator.is_main_process:
            print(
                f"Validating checkpoint step {step}: batch={args.batch_size}, "
                f"samples={args.validation_samples}, "
                f"qualitative={args.qualitative_examples}"
            )
        metrics = validate(
            model,
            loader,
            accelerator,
            cfg,
            online_siglip=online_siglip,
        )

        table = metrics.get("val/qualitative_trajectories")
        qualitative = None
        if table is not None and accelerator.is_main_process:
            qualitative = save_qualitative_table(
                table,
                output_dir / f"checkpoint_{step}_qualitative",
            )
        if args.wandb_mode != "disabled" and accelerator.is_main_process:
            run = accelerator.get_tracker("wandb", unwrap=True)
            run.log(metrics, step=step, commit=True)

        if accelerator.is_main_process:
            report = {
                "checkpoint_step": step,
                "checkpoint": str(checkpoint),
                "batch_size": args.batch_size,
                "validation_samples": args.validation_samples,
                "sample_validation_batches": args.sample_validation_batches,
                "qualitative": qualitative,
                "metrics": scalar_metrics(metrics),
            }
            with report_path.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, allow_nan=True)
            reports.append(report)
            print(
                json.dumps(
                    {
                        "checkpoint_step": step,
                        "val/loss": report["metrics"].get("val/loss"),
                        "val/sample_mse": report["metrics"].get(
                            "val/sample_mse"
                        ),
                        "gripper_accuracy": report["metrics"].get(
                            "val/sampled_native10/gripper_open/accuracy"
                        ),
                        "gripper_f1": report["metrics"].get(
                            "val/sampled_native10/gripper_open/f1"
                        ),
                    },
                    indent=2,
                )
            )

    if accelerator.is_main_process:
        reports.sort(key=lambda report: int(report["checkpoint_step"]))
        write_comparison_csv(reports, output_dir / "checkpoint_comparison.csv")
        with (output_dir / "summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(reports, handle, indent=2, allow_nan=True)
        print(f"Reports written to {output_dir}")
        if args.wandb_mode == "offline":
            print(
                "W&B is offline. When networking returns, sync the run with: "
                f"wandb sync {output_dir}/wandb/offline-run-*"
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()
