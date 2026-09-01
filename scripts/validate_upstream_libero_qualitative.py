#!/usr/bin/env python
"""Validate upstream Libero_RDT EMA checkpoints on fixed cached examples."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import torch
from accelerate import Accelerator
from PIL import Image

from thinkflow_rdt.checkpoint import load_full_rdt_base
from thinkflow_rdt.config import load_config
from thinkflow_rdt.model import SFTConditionedRDT
from thinkflow_rdt.train import (
    create_dataloader,
    load_online_siglip,
    validate,
)


DEFAULT_EVALUATIONS = (
    (
        "libero_spatial",
        Path("output_3/checkpoints/RDT-1B-LIBERO-Spatial"),
        Path("cache_features_libero_b2_native/libero_spatial/validation/manifest.jsonl"),
    ),
    (
        "libero_10",
        Path("output_3/checkpoints/RDT-1B-LIBERO-Long"),
        Path("cache_features_libero_b2_native/libero_10/validation/manifest.jsonl"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/libero_b0_hidden_native128_full.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_3/upstream_libero_qualitative"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--siglip-model-id",
        default="/home/ubuntu/models/siglip-so400m-patch14-384",
    )
    parser.add_argument(
        "--siglip-fallback-model-id",
        default="google/siglip-so400m-patch14-384",
    )
    return parser.parse_args()


def scalar_metrics(metrics: dict[str, object]) -> dict[str, float | int]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (float, int)) and not isinstance(value, bool)
    }


def combine_images(observation: Path | None, trajectory: Path, output: Path) -> None:
    with Image.open(trajectory) as trajectory_image:
        trajectory_rgb = trajectory_image.convert("RGB")
        if observation is None:
            trajectory_rgb.save(output)
            return
        with Image.open(observation) as observation_image:
            observation_rgb = observation_image.convert("RGB")
            width = max(observation_rgb.width, trajectory_rgb.width)

            def resized(image: Image.Image) -> Image.Image:
                if image.width == width:
                    return image
                height = round(image.height * width / image.width)
                return image.resize((width, height), Image.Resampling.LANCZOS)

            observation_rgb = resized(observation_rgb)
            trajectory_rgb = resized(trajectory_rgb)
            combined = Image.new(
                "RGB",
                (width, observation_rgb.height + trajectory_rgb.height),
                "white",
            )
            combined.paste(observation_rgb, (0, 0))
            combined.paste(trajectory_rgb, (0, observation_rgb.height))
            combined.save(output)


def save_qualitative(table, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = list(table.columns)
    observation_index = columns.index("observation_images")
    trajectory_index = columns.index("target_vs_diffusion_sample")
    records: list[dict[str, object]] = []
    for row_index, row in enumerate(table.data):
        trajectory_source = Path(row[trajectory_index]._path)
        observation_value = row[observation_index]
        observation_source = (
            Path(observation_value._path)
            if observation_value is not None
            else None
        )
        trajectory_path = output_dir / f"{row_index:02d}_trajectory.png"
        shutil.copy2(trajectory_source, trajectory_path)
        observation_path = None
        if observation_source is not None:
            observation_path = output_dir / f"{row_index:02d}_observation.png"
            shutil.copy2(observation_source, observation_path)
        composite_path = output_dir / f"{row_index:02d}_qualitative.png"
        combine_images(observation_path, trajectory_path, composite_path)
        records.append(
            {
                "index": row_index,
                "dataset": row[columns.index("dataset")],
                "episode_id": row[columns.index("episode_id")],
                "step_idx": row[columns.index("step_idx")],
                "instruction": row[columns.index("instruction")],
                "observation": observation_path.name if observation_path else None,
                "trajectory": trajectory_path.name,
                "qualitative": composite_path.name,
            }
        )
    (output_dir / "rows.json").write_text(
        json.dumps(records, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return {"rows": len(records), "directory": str(output_dir.resolve())}


def main() -> None:
    args = parse_args()
    if args.batch_size < args.examples:
        raise ValueError("--batch-size must be at least --examples")
    os.environ.setdefault("WANDB_MODE", "disabled")
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for _, checkpoint, manifest in DEFAULT_EVALUATIONS:
        if not (checkpoint / "ema" / "model.safetensors").is_file():
            raise FileNotFoundError(checkpoint / "ema" / "model.safetensors")
        if not manifest.is_file():
            raise FileNotFoundError(manifest)

    accelerator = Accelerator(mixed_precision="bf16")
    base_cfg = load_config(args.config)
    # Upstream Libero_RDT has no Qwen branch. The B2 cache is used only because
    # it contains the required lossless images, T5 embeddings, native state,
    # and native action targets; its hidden/KV/waypoint fields are ignored.
    base_cfg = replace(
        base_cfg,
        model=replace(
            base_cfg.model,
            qwen_fusion="none",
            conditioning_variant="b0",
            gradient_checkpointing=False,
            allow_random_frozen_state_adaptor=True,
        ),
        training=replace(
            base_cfg.training,
            qwen_fusion_loss_weight=0.0,
            qwen_fusion_loss_margin=0.0,
            conditioning_warmup_steps=0,
            micro_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            global_batch_size=args.batch_size * accelerator.num_processes,
            validation_samples=args.batch_size * accelerator.num_processes,
            validation_batches=1,
            validation_batch_size=args.batch_size,
            sample_validation_batches=1,
            qualitative_validation_examples=args.examples,
            report_to="wandb",
        ),
    )

    first_checkpoint = DEFAULT_EVALUATIONS[0][1]
    model = SFTConditionedRDT(
        base_cfg,
        load_pretrained=True,
        base_artifact=str(first_checkpoint),
    )
    model.eval()
    model = accelerator.prepare(model)
    online_siglip = load_online_siglip(
        model_id=args.siglip_model_id,
        fallback_model_id=args.siglip_fallback_model_id,
        cfg=base_cfg,
        device=accelerator.device,
    )

    reports: list[dict[str, object]] = []
    for suite, checkpoint, manifest in DEFAULT_EVALUATIONS:
        cfg = replace(
            base_cfg,
            data=replace(
                base_cfg.data,
                val_manifest=str(manifest.resolve()),
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
                shuffle_validation=False,
                stratified_validation=True,
            ),
        )
        cfg.validate()
        # This is a single-process inference job. Calling Accelerator's generic
        # unwrap helper imports optional DeepSpeed even when the model is not a
        # DeepSpeed engine, which can fail if an unrelated installed DeepSpeed
        # build targets a different Torch release.
        unwrapped = getattr(model, "module", model)
        load_report = load_full_rdt_base(
            unwrapped,
            checkpoint,
            allow_language_position_mismatch=True,
        )
        torch.cuda.empty_cache()
        loader = create_dataloader(
            cfg.data.val_manifest,
            cfg,
            shuffle=False,
            online_siglip=True,
            stratified=True,
        )
        loader = accelerator.prepare(loader)
        print(
            f"Validating {suite} from {checkpoint}: "
            f"batch={args.batch_size}, qualitative={args.examples}"
        )
        metrics = validate(
            model,
            loader,
            accelerator,
            cfg,
            online_siglip=online_siglip,
        )
        qualitative_table = metrics.pop("val/qualitative_trajectories", None)
        if qualitative_table is None:
            raise RuntimeError("Validation did not produce a qualitative table")
        suite_output = output_root / suite
        qualitative = save_qualitative(
            qualitative_table,
            suite_output / "qualitative",
        )
        report = {
            "suite": suite,
            "checkpoint": str(checkpoint.resolve()),
            "manifest": str(manifest.resolve()),
            "fixed_validation_seed": cfg.training.validation_seed,
            "checkpoint_load": load_report,
            "qualitative": qualitative,
            "metrics": scalar_metrics(metrics),
        }
        (suite_output / "metrics.json").parent.mkdir(
            parents=True, exist_ok=True
        )
        (suite_output / "metrics.json").write_text(
            json.dumps(report, indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        reports.append(report)
        print(
            json.dumps(
                {
                    "suite": suite,
                    "sample_mse": report["metrics"].get("val/sample_mse"),
                    "horizon_10_rmse": report["metrics"].get(
                        "val/sampled_native7/horizon_10/rmse"
                    ),
                    "gripper_f1": report["metrics"].get(
                        "val/sampled_native7/horizon_10/gripper_command/f1"
                    ),
                    "qualitative_images": qualitative["rows"],
                },
                indent=2,
            )
        )
    (output_root / "summary.json").write_text(
        json.dumps(reports, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
