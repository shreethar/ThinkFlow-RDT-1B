#!/usr/bin/env python
"""Precompute up-to-32-sample OXE episode packs with LatentStudent spatial-token KV.

This is the B2 counterpart of ``run_precompute_32frame_episode_packs.sh``.
It keeps that launcher's episode sampling, action targets, T5 features, and
image packing, but replaces Qwen's single ``</think>`` KV token with the five
LatentStudent spatial-token KV pairs produced by
``precompute_latent_student_kv.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precompute_all_features import (  # noqa: E402
    QWEN_TRAJECTORY_PROMPT_TEMPLATE,
    SPLIT_NAMES,
    build_episode_image_pool,
    build_lazy_configs,
    compact_tokens,
    episode_pack_relative_path,
    iter_episode_sample_groups,
    prepare_split_output,
    resolve_model_id,
    standardized_collate_fn,
)
from precompute_latent_student_kv import (  # noqa: E402
    extract_latent_student_spatial_kv,
    load_student_and_processor,
    load_t5,
    precompute_t5_features_chunked,
)
from thinkflow_rdt.adapters.combined_lazy import (  # noqa: E402
    build_combined_standardized_splits,
)
from thinkflow_rdt.config import load_config  # noqa: E402


OXE_DATASETS = ("bc_z", "bridge", "droid", "fractal", "kuka")
FEATURE_TYPE = "latent_student_spatial_kv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute one up-to-32-sample episode pack per OXE episode using five "
            "LatentStudent spatial-token KV pairs per sample."
        )
    )
    parser.add_argument("--config", default="configs/part3_rdt1b_lora32.yaml")
    parser.add_argument("--root", type=Path, default=Path("dataset/hf_parts/part_1"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cache_features/part_1_32frame_latent_student_kv"),
    )
    parser.add_argument("--dataset", action="append", choices=OXE_DATASETS)
    parser.add_argument("--split", action="append", choices=SPLIT_NAMES)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--stage-count", type=int, default=3)
    parser.add_argument("--droid-stage-count", type=int, default=2)
    parser.add_argument("--no-stage-subdir", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=None)

    # These defaults intentionally match run_precompute_32frame_episode_packs.sh.
    parser.add_argument("--max-samples-per-episode", type=int, default=32)
    parser.add_argument(
        "--require-exact-samples-per-episode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip episodes that retain fewer than --max-samples-per-episode. "
            "Disabled by default so short Kuka episodes are saved in smaller packs."
        ),
    )
    parser.add_argument(
        "--gripper-change-scope",
        choices=["all", "first", "directional", "first_directional"],
        default="first_directional",
    )
    parser.add_argument("--open-to-close-before", type=int, default=4)
    parser.add_argument("--open-to-close-after", type=int, default=4)
    parser.add_argument("--close-to-open-before", type=int, default=4)
    parser.add_argument("--close-to-open-after", type=int, default=4)
    parser.add_argument(
        "--action-target-mode",
        choices=["delta", "absolute_state"],
        default="delta",
    )
    parser.add_argument("--no-normalize-actions", action="store_true")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--empty-cache-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--student-model-id", default="shreethar/LatentStudent-ckpt-240")
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--latent-student-code-dir", type=Path, default=None)
    parser.add_argument("--spatial-parameters-path", type=Path, default=None)
    parser.add_argument("--latent-count", type=int, default=6)
    parser.add_argument("--spatial-token-count", type=int, default=5)
    parser.add_argument("--layer-index", type=int, default=7)
    parser.add_argument("--prompt-template", default=QWEN_TRAJECTORY_PROMPT_TEMPLATE)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=None,
    )
    parser.add_argument(
        "--student-precision",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
    )

    parser.add_argument(
        "--t5-model-id",
        default="/home/ubuntu/RoboticsDiffusionTransformer/google/t5-v1_1-xxl",
    )
    parser.add_argument("--t5-fallback-model-id", default="google/t5-v1_1-xxl")
    parser.add_argument("--t5-precision", choices=["bf16", "8bit"], default="bf16")
    parser.add_argument("--t5-batch-size", type=int, default=32)
    parser.add_argument("--save-padded-features", action="store_true")

    parser.add_argument("--image-history-size", type=int, default=2)
    parser.add_argument("--max-images-per-sample", type=int, default=6)
    parser.add_argument("--image-codec", choices=["png", "jpeg"], default="jpeg")
    parser.add_argument("--image-jpeg-quality", type=int, default=90)
    parser.add_argument("--keep-no-image", action="store_true")
    parser.add_argument("--episode-shards-per-directory", type=int, default=500)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "max_samples_per_episode",
        "batch_size",
        "t5_batch_size",
        "spatial_token_count",
        "episode_shards_per_directory",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "open_to_close_before",
        "open_to_close_after",
        "close_to_open_before",
        "close_to_open_after",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    if args.spatial_token_count != 5:
        raise ValueError("B2 episode packs require --spatial-token-count 5")
    if not 1 <= args.image_jpeg_quality <= 100:
        raise ValueError("--image-jpeg-quality must be in [1, 100]")
    if "{task}" not in args.prompt_template:
        raise ValueError("--prompt-template must contain {task}")
    if args.max_samples_per_split is not None:
        if args.max_samples_per_split <= 0:
            raise ValueError("--max-samples-per-split must be positive")
        if (
            args.require_exact_samples_per_episode
            and args.max_samples_per_split % args.max_samples_per_episode != 0
        ):
            raise ValueError(
                "--max-samples-per-split must be divisible by "
                "--max-samples-per-episode when exact episode sizes are required"
            )


def compact_episode_language(
    lang_tokens: torch.Tensor,
    lang_mask: torch.Tensor,
    *,
    save_padded_features: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = lang_tokens[0]
    mask = lang_mask[0]
    if save_padded_features:
        return tokens.cpu(), mask.cpu()
    tokens = compact_tokens(tokens, mask)
    return tokens.cpu(), torch.ones(tokens.shape[0], dtype=torch.bool)


def save_episode_pack(
    *,
    split_dir: Path,
    episode_index: int,
    sample_start_index: int,
    batch: dict[str, Any],
    spatial_kv: torch.Tensor,
    latent_waypoints: torch.Tensor,
    lang_tokens: torch.Tensor,
    lang_mask: torch.Tensor,
    args: argparse.Namespace,
    actions_normalized: bool,
) -> tuple[int, str]:
    batch_size = len(batch["metadata"])
    if spatial_kv.shape[:2] != (batch_size, args.spatial_token_count):
        raise ValueError(
            "Expected spatial KV [samples, spatial_tokens, dim], got "
            f"{tuple(spatial_kv.shape)}"
        )
    if latent_waypoints.shape[:2] != (batch_size, args.spatial_token_count):
        raise ValueError(
            "Expected latent waypoints [samples, spatial_tokens, dim], got "
            f"{tuple(latent_waypoints.shape)}"
        )

    episode_lang_tokens, episode_lang_mask = compact_episode_language(
        lang_tokens,
        lang_mask,
        save_padded_features=args.save_padded_features,
    )
    image_pool, sample_image_indices = build_episode_image_pool(
        batch,
        image_history_size=args.image_history_size,
        image_jpeg_quality=args.image_jpeg_quality,
        image_codec=args.image_codec,
    )

    first_metadata = batch["metadata"][0]
    raw_instructions = [str(value) for value in batch["instructions"]]
    unique_instructions = list(dict.fromkeys(raw_instructions))
    step_indices = [str(value["step_idx"]) for value in batch["metadata"]]
    relative_path = episode_pack_relative_path(
        episode_index,
        shards_per_directory=args.episode_shards_per_directory,
    )
    path = split_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "cache_layout": "episode_pack",
        "feature_type": FEATURE_TYPE,
        "dataset_id": first_metadata["dataset_id"],
        "episode_id": first_metadata["episode_id"],
        "num_samples": batch_size,
        "sample_start_index": sample_start_index,
        "sample_step_idx": step_indices,
        "sample_anchor_index": torch.arange(batch_size, dtype=torch.long),
        "qwen_cache_scope": "per_sample",
        "qwen_anchor_kv": spatial_kv.cpu(),
        "qwen_anchor_step_idx": step_indices,
        "qwen_anchor_kind": ["per_sample"] * batch_size,
        "latent_waypoints": latent_waypoints.cpu(),
        "actions_normalized": bool(actions_normalized),
        "instruction": unique_instructions[0] if len(unique_instructions) == 1 else None,
        "instructions": raw_instructions,
        "lang_tokens": episode_lang_tokens,
        "lang_mask": episode_lang_mask,
        "state": batch["state"].cpu(),
        "state_dim_mask": batch["state_dim_mask"].cpu(),
        "actions": batch["actions"].cpu(),
        "action_time_mask": batch["action_time_mask"].cpu(),
        "action_dim_mask": batch["action_dim_mask"].cpu(),
        "ctrl_freq": batch["ctrl_freq"].cpu(),
        "image_jpegs": image_pool,
        "sample_image_indices": sample_image_indices.cpu(),
        "sample_image_mask": batch["siglip_slot_mask"].cpu(),
        "sample_image_count": torch.as_tensor(
            [int(value["image_count"]) for value in batch["metadata"]],
            dtype=torch.long,
        ),
        "image_slot_count": int(batch["siglip_slot_mask"].shape[1]),
    }
    for key in ("joint_state", "joint_states", "joint_states_mask"):
        if key in batch:
            record[key] = batch[key].cpu()

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(record, temporary_path)
    os.replace(temporary_path, path)

    manifest_line = json.dumps(
        {
            "path": relative_path.as_posix(),
            "cache_layout": "episode_pack",
            "feature_type": FEATURE_TYPE,
            "dataset_id": first_metadata["dataset_id"],
            "episode_id": first_metadata["episode_id"],
            "num_samples": batch_size,
            "sample_start_index": sample_start_index,
            "sample_stop_index": sample_start_index + batch_size,
            "sample_step_start": step_indices[0],
            "sample_step_stop": step_indices[-1],
            "lang_token_count": int(episode_lang_tokens.shape[0]),
            "qwen_anchor_count": batch_size,
            "qwen_token_count": int(spatial_kv.shape[1]),
            "qwen_kv_dim": int(spatial_kv.shape[2]),
            "image_pool_count": len(image_pool),
            "image_slot_count": int(batch["siglip_slot_mask"].shape[1]),
            "has_img_tokens": False,
            "has_image_slots": True,
            "has_latent_waypoints": True,
            "qwen_cache_scope": "per_sample",
            "actions_normalized": bool(actions_normalized),
            "instruction": unique_instructions[0] if len(unique_instructions) == 1 else None,
        }
    ) + "\n"
    return batch_size, manifest_line


def precompute_split(
    *,
    split_name: str,
    dataset: Any,
    output_dir: Path,
    cfg: Any,
    args: argparse.Namespace,
    student: Any,
    processor: Any,
    t5_tokenizer: Any,
    t5_encoder: Any,
    device: torch.device,
    actions_normalized: bool,
) -> None:
    split_dir = output_dir / split_name
    manifest_path = prepare_split_output(split_dir, overwrite=args.overwrite)
    temporary_manifest = split_dir / "manifest.jsonl.tmp"

    sample_count = 0
    episode_count = 0
    skipped_wrong_size = 0
    skipped_no_image = 0
    started = time.perf_counter()

    with temporary_manifest.open("w", encoding="utf-8") as manifest:
        progress = tqdm(
            iter_episode_sample_groups(dataset),
            desc=f"latent-episode-pack {split_name}",
            unit="episode",
        )
        for episode_samples in progress:
            if args.max_samples_per_split is not None:
                remaining = args.max_samples_per_split - sample_count
                if remaining <= 0:
                    break
            else:
                remaining = None

            if (
                args.require_exact_samples_per_episode
                and len(episode_samples) != args.max_samples_per_episode
            ):
                skipped_wrong_size += 1
                continue

            batch = standardized_collate_fn(
                episode_samples,
                max_images_per_sample=args.max_images_per_sample,
                image_history_size=args.image_history_size,
                image_jpeg_quality=args.image_jpeg_quality,
                skip_no_image=not args.keep_no_image,
                encode_image_slots=False,
            )
            if batch is None:
                skipped_no_image += len(episode_samples)
                continue
            skipped_no_image += int(batch.get("skipped_no_image", 0))

            retained = len(batch["metadata"])
            if (
                args.require_exact_samples_per_episode
                and retained != args.max_samples_per_episode
            ):
                skipped_wrong_size += 1
                continue
            if remaining is not None and retained > remaining:
                # Preserve the one-physical-episode-per-pack invariant.
                break

            spatial_kv, latent_waypoints = extract_latent_student_spatial_kv(
                batch,
                student=student,
                processor=processor,
                device=device,
                layer_index=args.layer_index,
                expected_dim=cfg.model.qwen_kv_dim,
                spatial_token_count=args.spatial_token_count,
                prompt_template=args.prompt_template,
            )

            # Match the B0 episode-pack layout: language is shared at episode level.
            lang_tokens, lang_mask = precompute_t5_features_chunked(
                [str(batch["instructions"][0])],
                tokenizer=t5_tokenizer,
                encoder=t5_encoder,
                max_lang_tokens=cfg.model.max_lang_tokens,
                expected_dim=cfg.model.lang_token_dim,
                device=device,
                batch_size=args.t5_batch_size,
            )
            saved, manifest_line = save_episode_pack(
                split_dir=split_dir,
                episode_index=episode_count,
                sample_start_index=sample_count,
                batch=batch,
                spatial_kv=spatial_kv,
                latent_waypoints=latent_waypoints,
                lang_tokens=lang_tokens,
                lang_mask=lang_mask,
                args=args,
                actions_normalized=actions_normalized,
            )
            manifest.write(manifest_line)
            sample_count += saved
            episode_count += 1
            progress.set_postfix(episodes=episode_count, samples=sample_count)

            if args.empty_cache_every > 0 and episode_count % args.empty_cache_every == 0:
                torch.cuda.empty_cache()

    shutil.move(str(temporary_manifest), str(manifest_path))
    elapsed = max(time.perf_counter() - started, 1e-9)
    print(
        f"[{split_name}] wrote {sample_count} samples in {episode_count} episode packs "
        f"to {split_dir} ({sample_count / elapsed:.1f} samples/s); "
        f"skipped_wrong_size={skipped_wrong_size}, skipped_no_image={skipped_no_image}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    normalize_actions = not args.no_normalize_actions
    if args.action_target_mode == "absolute_state":
        normalize_actions = False

    configs = build_lazy_configs(
        root=args.root.expanduser().resolve(),
        dataset_ids=args.dataset or list(OXE_DATASETS),
        max_episodes=args.max_episodes,
    )
    splits = build_combined_standardized_splits(
        configs=configs,
        seed=seed,
        stage=args.stage,
        stage_count=args.stage_count,
        droid_stage_count=args.droid_stage_count,
        horizon=cfg.model.pred_horizon,
        normalize_actions=normalize_actions,
        action_target_mode=args.action_target_mode,
        max_samples_per_episode=args.max_samples_per_episode,
        gripper_change_scope=args.gripper_change_scope,
        open_to_close_before=args.open_to_close_before,
        open_to_close_after=args.open_to_close_after,
        close_to_open_before=args.close_to_open_before,
        close_to_open_after=args.close_to_open_after,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device_map == "cuda" and device.type != "cuda":
        args.device_map = "cpu"
    print(f"Using latent-student extraction device: {device}")
    student, processor = load_student_and_processor(args, device)
    print("Loading T5 encoder...")
    t5_tokenizer, t5_encoder = load_t5(args, cfg)

    output_dir = args.output_dir
    if args.stage is not None and not args.no_stage_subdir:
        output_dir = output_dir / f"stage_{args.stage}"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_names = args.split or list(SPLIT_NAMES)

    metadata = {
        "feature_type": FEATURE_TYPE,
        "config": args.config,
        "root": str(args.root),
        "splits": split_names,
        "datasets": [config.dataset_id for config in configs],
        "seed": seed,
        "stage": args.stage,
        "stage_count": args.stage_count,
        "droid_stage_count": args.droid_stage_count,
        "normalize_actions": normalize_actions,
        "action_target_mode": args.action_target_mode,
        "state_dim": cfg.model.state_dim,
        "action_dim": cfg.model.action_dim,
        "state_encoder_layout": cfg.model.state_encoder_layout,
        "action_encoder_layout": cfg.model.action_encoder_layout,
        "max_samples_per_episode": args.max_samples_per_episode,
        "require_exact_samples_per_episode": args.require_exact_samples_per_episode,
        "gripper_change_scope": args.gripper_change_scope,
        "open_to_close_before": args.open_to_close_before,
        "open_to_close_after": args.open_to_close_after,
        "close_to_open_before": args.close_to_open_before,
        "close_to_open_after": args.close_to_open_after,
        "student_model_id": args.student_model_id,
        "processor_id": args.processor_id or args.student_model_id,
        "spatial_parameters_path": (
            str(args.spatial_parameters_path)
            if args.spatial_parameters_path is not None
            else None
        ),
        "latent_count": args.latent_count,
        "spatial_token_count": args.spatial_token_count,
        "layer_index": args.layer_index,
        "prompt_template": args.prompt_template,
        "qwen_kv_dim": cfg.model.qwen_kv_dim,
        "qwen_cache_scope": "per_sample",
        "qwen_kv_granularity": "five_spatial_tokens_per_sample_step",
        "include_t5": True,
        "t5_model_id": resolve_model_id(
            args.t5_model_id,
            args.t5_fallback_model_id,
        ),
        "t5_precision": args.t5_precision,
        "image_history_size": args.image_history_size,
        "max_images_per_sample": args.max_images_per_sample,
        "image_storage_codec": args.image_codec,
        "image_storage_lossless": args.image_codec == "png",
        "image_jpeg_quality": (
            args.image_jpeg_quality if args.image_codec == "jpeg" else None
        ),
        "cache_layout": "episode_pack",
        "episode_shards_per_directory": args.episode_shards_per_directory,
    }
    (output_dir / "precompute_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    for split_name in split_names:
        precompute_split(
            split_name=split_name,
            dataset=splits[split_name],
            output_dir=output_dir,
            cfg=cfg,
            args=args,
            student=student,
            processor=processor,
            t5_tokenizer=t5_tokenizer,
            t5_encoder=t5_encoder,
            device=device,
            actions_normalized=normalize_actions,
        )


if __name__ == "__main__":
    main()
