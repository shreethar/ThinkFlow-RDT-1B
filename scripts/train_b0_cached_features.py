#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thinkflow_rdt.config import load_config
from thinkflow_rdt.data import CachedFeatureDataset
from thinkflow_rdt.train import train


def manifest_sample_count(manifest_path: Path) -> int:
    dataset = CachedFeatureDataset(manifest_path)
    return len(dataset)


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
        raise TypeError(f"Manifest line must be a JSON string or object, got {type(item)}")

    path_value = item.get("path")
    if not path_value:
        raise ValueError(f"Manifest object has no path: {item}")
    path = Path(str(path_value))
    resolved = path if path.is_absolute() else (manifest_dir / path).resolve()
    rewritten = dict(item)
    rewritten["path"] = str(resolved)
    return rewritten, resolved


def merge_manifests(
    input_manifests: Iterable[Path],
    output_manifest: Path,
) -> int:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    line_count = 0
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
                    line_count += 1
    return line_count


def manifests_from_cache_roots(
    cache_roots: list[Path],
    *,
    split_name: str,
) -> list[Path]:
    manifests: list[Path] = []
    for root in cache_roots:
        manifest = root.expanduser().resolve() / split_name / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        manifests.append(manifest)
    return manifests


def manifests_from_cache_parts_root(
    parts_root: Path,
    *,
    parts: list[int],
    split_name: str,
) -> list[Path]:
    root = parts_root.expanduser().resolve()
    manifests: list[Path] = []
    for part in parts:
        manifest = root / f"part_{int(part)}" / split_name / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        manifests.append(manifest)
    return manifests


def optional_override(value: object | None, current: object) -> object:
    return current if value is None else value


def parse_horizon_loss_schedule(schedule: str | None, horizon: int) -> list[float] | None:
    if schedule is None:
        return None
    weights: list[float | None] = [None] * horizon
    for block in schedule.split(","):
        range_text, separator, weight_text = block.strip().partition(":")
        if not separator:
            raise ValueError(f"Invalid horizon weight block: {block!r}")
        start_text, range_separator, stop_text = range_text.partition("-")
        start = int(start_text)
        stop = int(stop_text) if range_separator else start
        weight = float(weight_text)
        if start < 1 or stop < start or stop > horizon:
            raise ValueError(
                f"Horizon range {start}-{stop} is outside one-indexed [1,{horizon}]"
            )
        for index in range(start - 1, stop):
            if weights[index] is not None:
                raise ValueError(f"Horizon step {index + 1} is assigned more than once")
            weights[index] = weight
    missing = [index + 1 for index, value in enumerate(weights) if value is None]
    if missing:
        raise ValueError(f"Horizon schedule does not cover steps: {missing}")
    return [float(value) for value in weights if value is not None]


def build_config(args: argparse.Namespace):
    cfg = load_config(args.config)
    output_dir = str(args.output_dir.expanduser().resolve())
    manifest_dir = Path(output_dir) / "manifests"

    train_manifests = [path.expanduser().resolve() for path in args.train_manifest]
    val_manifests = [path.expanduser().resolve() for path in args.val_manifest]
    if args.cache_root:
        cache_roots = [root.expanduser().resolve() for root in args.cache_root]
        train_manifests.extend(
            manifests_from_cache_roots(cache_roots, split_name=args.train_split)
        )
        val_manifests.extend(
            manifests_from_cache_roots(cache_roots, split_name=args.val_split)
        )
    if args.cache_parts_root:
        parts = args.cache_parts or [1, 2, 3]
        train_manifests.extend(
            manifests_from_cache_parts_root(
                args.cache_parts_root,
                parts=parts,
                split_name=args.train_split,
            )
        )
        val_manifests.extend(
            manifests_from_cache_parts_root(
                args.cache_parts_root,
                parts=parts,
                split_name=args.val_split,
            )
        )
    if not train_manifests:
        raise ValueError("Provide --cache-root or --train-manifest")
    if not val_manifests:
        raise ValueError("Provide --cache-root or --val-manifest")

    merged_train_manifest = manifest_dir / "train_manifest.jsonl"
    merged_val_manifest = manifest_dir / "val_manifest.jsonl"
    train_lines = merge_manifests(train_manifests, merged_train_manifest)
    val_lines = merge_manifests(val_manifests, merged_val_manifest)

    data_cfg = replace(
        cfg.data,
        train_manifest=str(merged_train_manifest),
        val_manifest=str(merged_val_manifest),
        num_workers=int(optional_override(args.num_workers, cfg.data.num_workers)),
        pin_memory=bool(optional_override(args.pin_memory, cfg.data.pin_memory)),
        persistent_workers=bool(
            optional_override(args.persistent_workers, cfg.data.persistent_workers)
        ),
    )
    training_cfg = replace(
        cfg.training,
        max_steps=int(optional_override(args.max_steps, cfg.training.max_steps)),
        micro_batch_size=int(
            optional_override(args.micro_batch_size, cfg.training.micro_batch_size)
        ),
        gradient_accumulation_steps=int(
            optional_override(
                args.gradient_accumulation_steps,
                cfg.training.gradient_accumulation_steps,
            )
        ),
        learning_rate_lora=float(
            optional_override(args.learning_rate_lora, cfg.training.learning_rate_lora)
        ),
        learning_rate_interfaces=float(
            optional_override(
                args.learning_rate_interfaces,
                cfg.training.learning_rate_interfaces,
            )
        ),
        warmup_steps=int(optional_override(args.warmup_steps, cfg.training.warmup_steps)),
        log_every=int(optional_override(args.log_every, cfg.training.log_every)),
        validate_every=int(
            optional_override(args.validate_every, cfg.training.validate_every)
        ),
        save_every=int(optional_override(args.save_every, cfg.training.save_every)),
        validation_batches=int(
            optional_override(args.validation_batches, cfg.training.validation_batches)
        ),
        sample_validation_batches=int(
            optional_override(
                args.sample_validation_batches,
                cfg.training.sample_validation_batches,
            )
        ),
        mixed_precision=str(
            optional_override(args.mixed_precision, cfg.training.mixed_precision)
        ),
        report_to=str(optional_override(args.report_to, cfg.training.report_to)),
        global_batch_size=optional_override(
            args.global_batch_size, cfg.training.global_batch_size
        ),
        wandb_project=str(
            optional_override(args.wandb_project, cfg.training.wandb_project)
        ),
        wandb_run_name=optional_override(
            args.wandb_run_name, cfg.training.wandb_run_name
        ),
    )
    model_cfg = cfg.model
    if args.no_gradient_checkpointing:
        model_cfg = replace(model_cfg, gradient_checkpointing=False)
    if args.image_tokens is not None:
        model_cfg = replace(model_cfg, image_tokens=int(args.image_tokens))
    if args.pred_horizon is not None:
        model_cfg = replace(model_cfg, pred_horizon=int(args.pred_horizon))
    cfg = replace(
        cfg,
        output_dir=output_dir,
        model=model_cfg,
        data=data_cfg,
        training=training_cfg,
    )

    print("Resolved cached-feature manifests:")
    print(f"  train manifests: {len(train_manifests)} sources, {train_lines} manifest rows")
    print(f"  val manifests:   {len(val_manifests)} sources, {val_lines} manifest rows")
    print(f"  merged train:    {merged_train_manifest}")
    print(f"  merged val:      {merged_val_manifest}")
    try:
        print(f"  train samples:   {manifest_sample_count(merged_train_manifest)}")
        print(f"  val samples:     {manifest_sample_count(merged_val_manifest)}")
    except Exception as exc:
        print(f"  sample count check skipped: {exc}")
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train B0 RDT from precomputed sample-by-sample cached features. "
            "Pass one or more cache roots such as cache_features/part_1, or "
            "use --cache-parts-root to train across part_1/part_2/part_3."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        action="append",
        help="Cache root containing train/ and validation/ manifest.jsonl. Repeat for parts.",
    )
    parser.add_argument(
        "--cache-parts-root",
        type=Path,
        help=(
            "Directory containing part_1/, part_2/, part_3/ cache roots. "
            "Defaults to all three parts unless --cache-parts is provided."
        ),
    )
    parser.add_argument(
        "--cache-parts",
        type=int,
        nargs="+",
        choices=[1, 2, 3],
        default=None,
        help="Part numbers to use under --cache-parts-root. Default: 1 2 3.",
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        action="append",
        default=[],
        help="Explicit train manifest. Repeat to merge multiple manifests.",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        action="append",
        default=[],
        help="Explicit validation manifest. Repeat to merge multiple manifests.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument(
        "--online-siglip",
        action="store_true",
        help=(
            "Use cached image slots and compute SigLIP online. Leave off for "
            "sample-by-sample caches that already contain img_tokens/img_mask."
        ),
    )
    parser.add_argument(
        "--siglip-model-id",
        default="/home/ubuntu/models/siglip-so400m-patch14-384",
    )
    parser.add_argument(
        "--siglip-fallback-model-id",
        default="google/siglip-so400m-patch14-384",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--init-artifact",
        type=Path,
        help="Initialize LoRA and trained interfaces from a prior checkpoint directory.",
    )
    parser.add_argument(
        "--base-artifact",
        type=Path,
        help=(
            "Initialize the base RDT transformer from a merged/full artifact "
            "before creating a fresh LoRA adapter. Use this after "
            "scripts/merge_lora_adapter.py when moving from OXE to LIBERO."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help=(
            "Effective global batch size. If set, accumulation is resolved as "
            "global_batch_size / (micro_batch_size * world_size)."
        ),
    )
    parser.add_argument("--learning-rate-lora", type=float, default=None)
    parser.add_argument("--learning-rate-interfaces", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--validation-batches", type=int, default=None)
    parser.add_argument("--sample-validation-batches", type=int, default=None)
    parser.add_argument("--mixed-precision", default=None)
    parser.add_argument("--report-to", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help=(
            "Disable RDT block gradient checkpointing. Faster if VRAM allows it, "
            "but memory use increases."
        ),
    )
    parser.add_argument(
        "--image-tokens",
        type=int,
        default=None,
        help=(
            "Override cfg.model.image_tokens. Cached img_tokens are truncated by "
            "the collator, which can speed training at some quality cost. Use only "
            "with fully precomputed image-token caches, not --online-siglip."
        ),
    )
    parser.add_argument(
        "--pred-horizon",
        type=int,
        default=None,
        help="Override cfg.model.pred_horizon; cached action horizons are truncated/padded.",
    )
    parser.add_argument(
        "--horizon-loss-schedule",
        default=None,
        help=(
            "One-indexed, non-overlapping horizon weights, for example "
            "'1-4:5,5-8:3,9-16:2,17-64:1'."
        ),
    )
    parser.add_argument(
        "--mask-noisy-gripper-input",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Hide the noised gripper channel from the action adaptor during "
            "training and decode gripper from the final clean-x0 estimate at "
            "inference. Recommended for discrete LIBERO gripper commands."
        ),
    )
    parser.add_argument("--gripper-bce-weight", type=float, default=0.0)
    parser.add_argument("--gripper-bce-logit-scale", type=float, default=1.0)
    parser.add_argument("--rotation-geodesic-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    horizon_loss_weights = parse_horizon_loss_schedule(
        args.horizon_loss_schedule,
        cfg.model.pred_horizon,
    )
    train(
        cfg,
        load_pretrained=not args.no_pretrained,
        online_siglip_model_id=args.siglip_model_id if args.online_siglip else None,
        online_siglip_fallback_model_id=args.siglip_fallback_model_id,
        base_artifact=args.base_artifact,
        init_artifact=args.init_artifact,
        horizon_loss_weights=horizon_loss_weights,
        mask_noisy_gripper_input=args.mask_noisy_gripper_input,
        gripper_bce_weight=args.gripper_bce_weight,
        gripper_bce_logit_scale=args.gripper_bce_logit_scale,
        rotation_geodesic_weight=args.rotation_geodesic_weight,
    )


if __name__ == "__main__":
    main()
