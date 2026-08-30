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
from collections import Counter, deque
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
SCRIPT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
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
    resolve_model_id,
    standardized_collate_fn,
)
from precompute_latent_student_kv import (  # noqa: E402
    extract_latent_student_spatial_kv,
    load_student_and_processor,
    load_t5,
    precompute_t5_features_chunked,
    tokenizer_end_think_id,
)
from thinkflow_rdt.adapters.combined_lazy import (  # noqa: E402
    build_combined_standardized_splits,
)
from thinkflow_rdt.adapters.action_stats import find_audit_json  # noqa: E402
from thinkflow_rdt.config import load_config  # noqa: E402


OXE_DATASETS = ("bc_z", "bridge", "droid", "fractal", "kuka")
FEATURE_TYPE = "latent_student_spatial_kv"


def first_existing_path(*candidates: Path, fallback: str | None = None) -> str | None:
    for candidate in candidates:
        if candidate.expanduser().exists():
            return str(candidate.expanduser().resolve())
    return fallback


DEFAULT_STUDENT_MODEL_ID = first_existing_path(
    REPO_ROOT / "model" / "LatentStudent-ckpt-400-fixed",
    Path("/workspace/model/LatentStudent-ckpt-400-fixed"),
    fallback="shreethar/LatentStudent-ckpt-240",
)
DEFAULT_PROCESSOR_ID = first_existing_path(
    REPO_ROOT / "model" / "model" / "stage1_unsloth",
    REPO_ROOT / "model" / "stage1_unsloth",
    Path("/workspace/model/stage1_unsloth"),
)
DEFAULT_LATENT_STUDENT_CODE_DIR = first_existing_path(
    Path("/home/ubuntu/VLA-FYP/train/stage2"),
    Path("/workspace/VLA-FYP/train/stage2"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute one up-to-32-sample episode pack per OXE episode using five "
            "LatentStudent spatial-token KV pairs per sample."
        )
    )
    parser.add_argument("--config", default="configs/part3_rdt1b.yaml")
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
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        default=(0.8, 0.1, 0.1),
        help="Deterministic episode-level source split ratios.",
    )
    parser.add_argument(
        "--dataset-schedule",
        choices=("round_robin", "sequential"),
        default="round_robin",
        help=(
            "round_robin alternates physical episodes across requested datasets, "
            "so an interrupted or capped run still contains every available dataset."
        ),
    )

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
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from a durable manifest left by an interrupted run. Existing "
            "episode occurrences are skipped before any model inference."
        ),
    )

    parser.add_argument("--student-model-id", default=DEFAULT_STUDENT_MODEL_ID)
    parser.add_argument("--processor-id", default=DEFAULT_PROCESSOR_ID)
    parser.add_argument(
        "--latent-student-code-dir",
        type=Path,
        default=DEFAULT_LATENT_STUDENT_CODE_DIR,
    )
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
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    split_ratios = tuple(float(value) for value in args.split_ratios)
    if any(value < 0.0 for value in split_ratios) or sum(split_ratios) <= 0.0:
        raise ValueError("--split-ratios must be non-negative with a positive sum")
    for name in (
        "max_samples_per_split",
        "max_train_samples",
        "max_validation_samples",
        "max_test_samples",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_samples_per_split is not None:
        if (
            args.require_exact_samples_per_episode
            and args.max_samples_per_split % args.max_samples_per_episode != 0
        ):
            raise ValueError(
                "--max-samples-per-split must be divisible by "
                "--max-samples-per-episode when exact episode sizes are required"
            )


def split_sample_limit(args: argparse.Namespace, split_name: str) -> int | None:
    specific = getattr(args, f"max_{split_name}_samples")
    return specific if specific is not None else args.max_samples_per_split


def episode_identity(episode_samples: list[dict[str, Any]]) -> tuple[str, str]:
    if not episode_samples:
        raise ValueError("Cannot identify an empty episode")
    first = episode_samples[0]
    identity = (str(first["dataset_id"]), str(first["episode_id"]))
    for sample in episode_samples[1:]:
        current = (str(sample["dataset_id"]), str(sample["episode_id"]))
        if current != identity:
            raise ValueError(
                f"Episode group mixes identities {identity} and {current}"
            )
    return identity


def iter_scheduled_episode_groups(
    dataset: Any,
    *,
    schedule: str,
):
    """Yield physical episodes sequentially or round-robin across datasets."""
    members = list(getattr(dataset, "members", []))
    if schedule == "sequential" or not members:
        yield from iter_episode_sample_groups(dataset)
        return

    pending = deque(
        (member.dataset_id, iter(iter_episode_sample_groups(member.dataset)))
        for member in members
    )
    while pending:
        dataset_id, iterator = pending.popleft()
        try:
            episode_samples = next(iterator)
        except StopIteration:
            continue
        actual_dataset_id, _ = episode_identity(episode_samples)
        if actual_dataset_id != dataset_id:
            raise RuntimeError(
                "Combined dataset member label does not match emitted samples: "
                f"{dataset_id!r} != {actual_dataset_id!r}"
            )
        yield episode_samples
        pending.append((dataset_id, iterator))


def extract_latent_student_spatial_kv_chunked(
    batch: dict[str, Any],
    *,
    student: Any,
    processor: Any,
    device: torch.device,
    layer_index: int,
    expected_dim: int,
    spatial_token_count: int,
    prompt_template: str,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Honor --batch-size while retaining one output pack per episode."""
    total = len(batch["metadata"])
    kv_chunks: list[torch.Tensor] = []
    hidden_state_chunks: list[torch.Tensor] = []
    waypoint_chunks: list[torch.Tensor] = []
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        chunk = {
            "instructions": batch["instructions"][start:stop],
            "qwen_images": batch["qwen_images"][start:stop],
        }
        spatial_kv, spatial_hidden_states, latent_waypoints = extract_latent_student_spatial_kv(
            chunk,
            student=student,
            processor=processor,
            device=device,
            layer_index=layer_index,
            expected_dim=expected_dim,
            spatial_token_count=spatial_token_count,
            prompt_template=prompt_template,
        )
        kv_chunks.append(spatial_kv.cpu())
        hidden_state_chunks.append(spatial_hidden_states.cpu())
        waypoint_chunks.append(latent_waypoints.cpu())
    return (
        torch.cat(kv_chunks, dim=0),
        torch.cat(hidden_state_chunks, dim=0),
        torch.cat(waypoint_chunks, dim=0),
    )


def validate_dataset_configs(
    configs: list[Any],
    root: Path,
    *,
    normalize_actions: bool,
) -> None:
    resolved_root = root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {resolved_root}")

    missing: list[str] = []
    print("Selected extraction datasets:")
    for config in configs:
        data_dir = Path(config.data_dir).expanduser().resolve()
        status = "available" if data_dir.is_dir() else "MISSING"
        stats_status = "not required"
        if normalize_actions:
            adapter_kwargs = dict(getattr(config, "adapter_kwargs", {}))
            inline_stats = adapter_kwargs.get("action_stats")
            configured_stats = adapter_kwargs.get("action_stats_path")
            discovered_stats = (
                Path(configured_stats).expanduser().resolve()
                if configured_stats is not None
                else find_audit_json(data_dir)
            )
            if inline_stats is not None:
                stats_status = "inline"
            elif discovered_stats is not None and discovered_stats.is_file():
                stats_status = str(discovered_stats)
            else:
                stats_status = "MISSING"
                missing.append(f"{config.dataset_id} action stats/audit.json")
        print(
            f"  - {config.dataset_id}: {data_dir} [{status}], "
            f"action_stats=[{stats_status}]"
        )
        if not data_dir.is_dir():
            missing.append(f"{config.dataset_id}={data_dir}")
    if missing:
        raise FileNotFoundError(
            "Dataset preflight failed. Pass --dataset only for datasets present "
            "in this downloaded part and ensure each normalized dataset has an "
            "audit.json/action stats file:\n  - "
            + "\n  - ".join(missing)
        )


def _student_text_config(student: Any) -> Any:
    language_config = getattr(student._language_model, "config", None)
    if language_config is None:
        raise ValueError("LatentStudent language model has no config")
    return getattr(language_config, "text_config", language_config)


def validate_student_runtime_contract(
    student: Any,
    processor: Any,
    *,
    args: argparse.Namespace,
    cfg: Any,
) -> dict[str, Any]:
    """Fail before extraction if B2 would save the wrong layer or KV width."""
    text_config = _student_text_config(student)
    hidden_size = int(getattr(text_config, "hidden_size", -1))
    num_layers = int(getattr(text_config, "num_hidden_layers", -1))
    num_kv_heads = int(getattr(text_config, "num_key_value_heads", -1))
    head_dim = int(getattr(text_config, "head_dim", -1))
    layer_types = getattr(text_config, "layer_types", None)

    if hidden_size != int(cfg.model.qwen_hidden_size):
        raise ValueError(
            f"LatentStudent hidden_size={hidden_size} does not match "
            f"config qwen_hidden_size={cfg.model.qwen_hidden_size}"
        )
    if not 0 <= int(args.layer_index) < num_layers:
        raise ValueError(
            f"--layer-index {args.layer_index} is outside [0, {num_layers - 1}]"
        )
    layer_type = None
    if layer_types is not None:
        if len(layer_types) != num_layers:
            raise ValueError(
                f"LatentStudent has {len(layer_types)} layer_types for {num_layers} layers"
            )
        layer_type = str(layer_types[args.layer_index])
        if layer_type != "full_attention":
            raise ValueError(
                f"B2 KV extraction requires a full-attention layer, but layer "
                f"{args.layer_index} is {layer_type!r}"
            )

    computed_kv_dim = 2 * num_kv_heads * head_dim
    if computed_kv_dim != int(cfg.model.qwen_kv_dim):
        raise ValueError(
            "LatentStudent K/V layout is incompatible with the training config: "
            f"2 * {num_kv_heads} KV heads * {head_dim} head_dim = "
            f"{computed_kv_dim}, expected {cfg.model.qwen_kv_dim}"
        )
    if int(getattr(student, "M", -1)) != int(args.latent_count):
        raise ValueError(
            f"Loaded LatentStudent M={getattr(student, 'M', None)} does not match "
            f"--latent-count {args.latent_count}"
        )
    if int(getattr(student, "K", -1)) != int(args.spatial_token_count):
        raise ValueError(
            f"Loaded LatentStudent K={getattr(student, 'K', None)} does not match "
            f"--spatial-token-count {args.spatial_token_count}"
        )

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Processor has no tokenizer")
    expected_end_think_id = tokenizer_end_think_id(tokenizer)
    actual_end_think_id = int(getattr(student, "end_think_token_id", -1))
    if actual_end_think_id != expected_end_think_id:
        raise ValueError(
            f"LatentStudent </think> id {actual_end_think_id} does not match "
            f"processor tokenizer id {expected_end_think_id}"
        )

    tensors = [("spatial_tokens", student.spatial_tokens)]
    tensors.extend(
        (f"spatial_mlp.{name}", parameter)
        for name, parameter in student.spatial_mlp.named_parameters()
    )
    for name, tensor in tensors:
        if not bool(torch.isfinite(tensor.detach()).all()):
            raise ValueError(f"LatentStudent {name} contains NaN or Inf")

    contract = {
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "layer_index": int(args.layer_index),
        "layer_type": layer_type or "full_attention_assumed",
        "flattened_kv_dim": computed_kv_dim,
        "final_hidden_state_dim": hidden_size,
        "latent_count": int(args.latent_count),
        "spatial_token_count": int(args.spatial_token_count),
        "end_think_token_id": actual_end_think_id,
    }
    print("Validated B2 LatentStudent contract: " + json.dumps(contract, sort_keys=True))
    return contract


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
    spatial_hidden_states: torch.Tensor,
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
    if spatial_hidden_states.shape[:2] != (batch_size, args.spatial_token_count):
        raise ValueError(
            "Expected raw spatial hidden states [samples, spatial_tokens, dim], got "
            f"{tuple(spatial_hidden_states.shape)}"
        )
    if spatial_kv.ndim != 3 or int(spatial_kv.shape[2]) <= 0:
        raise ValueError(f"Invalid spatial KV shape {tuple(spatial_kv.shape)}")
    if not bool(torch.isfinite(spatial_kv.float()).all()):
        raise ValueError("Refusing to save spatial KV containing NaN or Inf")
    if spatial_hidden_states.ndim != 3 or int(spatial_hidden_states.shape[2]) <= 0:
        raise ValueError(
            f"Invalid raw spatial hidden-state shape {tuple(spatial_hidden_states.shape)}"
        )
    if not bool(torch.isfinite(spatial_hidden_states.float()).all()):
        raise ValueError("Refusing to save raw spatial hidden states containing NaN or Inf")
    if not bool(torch.isfinite(latent_waypoints.float()).all()):
        raise ValueError("Refusing to save latent waypoints containing NaN or Inf")
    if not bool(torch.isfinite(lang_tokens.float()).all()):
        raise ValueError("Refusing to save T5 tokens containing NaN or Inf")

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
    if path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to replace existing episode pack {path}. "
            "Use --resume to continue or a new --output-dir for a fresh run."
        )

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
        "qwen_anchor_hidden_states": spatial_hidden_states.cpu(),
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
            "qwen_hidden_state_dim": int(spatial_hidden_states.shape[2]),
            "image_pool_count": len(image_pool),
            "image_slot_count": int(batch["siglip_slot_mask"].shape[1]),
            "has_img_tokens": False,
            "has_image_slots": True,
            "has_latent_waypoints": True,
            "has_qwen_hidden_states": True,
            "qwen_cache_scope": "per_sample",
            "actions_normalized": bool(actions_normalized),
            "instruction": unique_instructions[0] if len(unique_instructions) == 1 else None,
        }
    ) + "\n"
    return batch_size, manifest_line


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            referenced = Path(str(row["path"]))
            if not referenced.is_absolute():
                referenced = (path.parent / referenced).resolve()
            if not referenced.is_file():
                raise FileNotFoundError(
                    f"{path}:{line_number} points to missing pack {referenced}"
                )
            rows.append(row)
    return rows


def prepare_episode_split_output(
    split_dir: Path,
    *,
    resume: bool,
    overwrite: bool,
) -> tuple[Path, list[dict[str, Any]], int]:
    """Prepare a durable manifest and return rows plus the next file index."""
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "manifest.jsonl"
    legacy_temporary = split_dir / "manifest.jsonl.tmp"
    existing_packs = sorted(split_dir.glob("episodes_*/*.pt"))
    existing_packs.extend(sorted(split_dir.glob("episode_*.pt")))

    if overwrite:
        manifest_path.unlink(missing_ok=True)
        legacy_temporary.unlink(missing_ok=True)
        for episode_directory in split_dir.glob("episodes_*"):
            if episode_directory.is_dir():
                shutil.rmtree(episode_directory)
        for episode_path in split_dir.glob("episode_*.pt"):
            episode_path.unlink()
        return manifest_path, [], 0

    if resume:
        if not manifest_path.exists() and legacy_temporary.exists():
            # Older versions wrote a complete incremental manifest to .tmp and
            # renamed it only after finishing the split. Preserve that work.
            os.replace(legacy_temporary, manifest_path)
        if not manifest_path.exists():
            if existing_packs:
                raise FileNotFoundError(
                    f"{split_dir} contains episode packs but no resumable manifest. "
                    "Recover it with scripts/build_b2_interrupted_manifests.py first."
                )
            return manifest_path, [], 0
        rows = load_manifest_rows(manifest_path)
        if not rows and existing_packs:
            raise ValueError(
                f"{manifest_path} is empty but {len(existing_packs)} packs exist"
            )
        referenced_packs = {
            (split_dir / str(row["path"])).resolve() for row in rows
        }
        orphan_packs = [
            path for path in existing_packs if path.resolve() not in referenced_packs
        ]
        if orphan_packs:
            preview = ", ".join(str(path) for path in orphan_packs[:3])
            raise ValueError(
                f"{split_dir} contains {len(orphan_packs)} pack(s) not recorded in "
                f"its manifest (for example: {preview}). Recover the manifest with "
                "scripts/build_b2_interrupted_manifests.py before resuming."
            )
        episode_numbers = [
            int(path.stem.removeprefix("episode_")) for path in existing_packs
        ]
        next_episode_index = max(episode_numbers, default=0)
        return manifest_path, rows, next_episode_index

    existing = [path for path in (manifest_path, legacy_temporary) if path.exists()]
    if existing or existing_packs:
        raise FileExistsError(
            f"Output split {split_dir} is not empty. Pass --resume to continue "
            "a compatible run, or choose a new --output-dir."
        )
    return manifest_path, [], 0


def write_durable_manifest_line(handle: Any, line: str) -> None:
    handle.write(line)
    handle.flush()
    os.fsync(handle.fileno())


RESUME_METADATA_KEYS = (
    "feature_type",
    "root",
    "datasets",
    "seed",
    "stage",
    "stage_count",
    "droid_stage_count",
    "split_ratios",
    "dataset_schedule",
    "normalize_actions",
    "action_target_mode",
    "pred_horizon",
    "cache_state_dim",
    "cache_action_dim",
    "max_samples_per_episode",
    "require_exact_samples_per_episode",
    "gripper_change_scope",
    "open_to_close_before",
    "open_to_close_after",
    "close_to_open_before",
    "close_to_open_after",
    "student_model_id",
    "processor_id",
    "spatial_parameters_path",
    "latent_student_code_dir",
    "latent_count",
    "spatial_token_count",
    "layer_index",
    "prompt_template",
    "qwen_kv_dim",
    "t5_model_id",
    "t5_precision",
    "save_padded_features",
    "image_history_size",
    "max_images_per_sample",
    "image_storage_codec",
    "image_jpeg_quality",
)


def write_or_validate_metadata(
    output_dir: Path,
    metadata: dict[str, Any],
    *,
    resume: bool,
    overwrite: bool,
) -> None:
    path = output_dir / "precompute_metadata.json"
    if resume and path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        mismatches = [
            key
            for key in RESUME_METADATA_KEYS
            if existing.get(key) != metadata.get(key)
        ]
        if mismatches:
            details = "; ".join(
                f"{key}: existing={existing.get(key)!r}, requested={metadata.get(key)!r}"
                for key in mismatches
            )
            raise ValueError(
                "Refusing to resume with incompatible extraction settings: " + details
            )
        return
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Pass --resume for a compatible interrupted "
            "run, --overwrite to replace selected splits, or use a new output dir."
        )

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


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
    manifest_path, completed_rows, next_episode_index = prepare_episode_split_output(
        split_dir,
        resume=args.resume,
        overwrite=args.overwrite,
    )

    sample_count = sum(int(row["num_samples"]) for row in completed_rows)
    episode_count = len(completed_rows)
    completed_occurrences = Counter(
        (str(row["dataset_id"]), str(row["episode_id"]))
        for row in completed_rows
    )
    samples_by_dataset = Counter()
    episodes_by_dataset = Counter()
    for row in completed_rows:
        dataset_id = str(row["dataset_id"])
        samples_by_dataset[dataset_id] += int(row["num_samples"])
        episodes_by_dataset[dataset_id] += 1
    skipped_wrong_size = 0
    skipped_no_image = 0
    skipped_completed = 0
    started = time.perf_counter()
    sample_limit = split_sample_limit(args, split_name)
    announced_dataset_ids: set[str] = set()

    if completed_rows:
        print(
            f"[{split_name}] resuming from {episode_count} episodes / "
            f"{sample_count} samples; next episode file index={next_episode_index}"
        )
    if sample_limit is not None and sample_count >= sample_limit:
        print(
            f"[{split_name}] sample limit {sample_limit} was already reached; "
            "nothing to resume."
        )
        return

    manifest_mode = "a" if completed_rows else "w"
    with manifest_path.open(manifest_mode, encoding="utf-8", buffering=1) as manifest:
        progress = tqdm(
            iter_scheduled_episode_groups(
                dataset,
                schedule=args.dataset_schedule,
            ),
            desc=f"B2 {split_name} [starting]",
            unit="episode",
            initial=episode_count,
        )
        for episode_samples in progress:
            dataset_id, episode_id = episode_identity(episode_samples)
            identity = (dataset_id, episode_id)
            if completed_occurrences[identity] > 0:
                completed_occurrences[identity] -= 1
                skipped_completed += 1
                continue

            if dataset_id not in announced_dataset_ids:
                tqdm.write(
                    f"[{split_name}] started dataset={dataset_id} "
                    f"(total episodes={episode_count}, samples={sample_count})"
                )
                announced_dataset_ids.add(dataset_id)
            progress.set_description(f"B2 {split_name} [{dataset_id}]")

            if sample_limit is not None:
                remaining = sample_limit - sample_count
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
                # Preserve the one-physical-episode-per-pack invariant. A later
                # short episode may still fit the remaining sample budget.
                continue

            unique_instructions = list(dict.fromkeys(map(str, batch["instructions"])))
            if len(unique_instructions) != 1:
                raise ValueError(
                    f"Episode {dataset_id}/{episode_id} contains "
                    f"{len(unique_instructions)} instructions, but the episode-pack "
                    "layout stores one shared T5 embedding."
                )

            spatial_kv, spatial_hidden_states, latent_waypoints = (
                extract_latent_student_spatial_kv_chunked(
                    batch,
                    student=student,
                    processor=processor,
                    device=device,
                    layer_index=args.layer_index,
                    expected_dim=cfg.model.qwen_kv_dim,
                    spatial_token_count=args.spatial_token_count,
                    prompt_template=args.prompt_template,
                    batch_size=args.batch_size,
                )
            )

            # Match the B0 episode-pack layout: language is shared at episode level.
            lang_tokens, lang_mask = precompute_t5_features_chunked(
                unique_instructions,
                tokenizer=t5_tokenizer,
                encoder=t5_encoder,
                max_lang_tokens=cfg.model.max_lang_tokens,
                expected_dim=cfg.model.lang_token_dim,
                device=device,
                batch_size=args.t5_batch_size,
            )
            saved, manifest_line = save_episode_pack(
                split_dir=split_dir,
                episode_index=next_episode_index,
                sample_start_index=sample_count,
                batch=batch,
                spatial_kv=spatial_kv,
                spatial_hidden_states=spatial_hidden_states,
                latent_waypoints=latent_waypoints,
                lang_tokens=lang_tokens,
                lang_mask=lang_mask,
                args=args,
                actions_normalized=actions_normalized,
            )
            write_durable_manifest_line(manifest, manifest_line)
            sample_count += saved
            episode_count += 1
            next_episode_index += 1
            samples_by_dataset[dataset_id] += saved
            episodes_by_dataset[dataset_id] += 1
            progress.set_postfix(
                current_dataset=dataset_id,
                dataset_episodes=episodes_by_dataset[dataset_id],
                dataset_samples=samples_by_dataset[dataset_id],
                total_episodes=episode_count,
                total_samples=sample_count,
            )

            if args.empty_cache_every > 0 and episode_count % args.empty_cache_every == 0:
                torch.cuda.empty_cache()

    elapsed = max(time.perf_counter() - started, 1e-9)
    print(
        f"[{split_name}] wrote {sample_count} samples in {episode_count} episode packs "
        f"to {split_dir} ({sample_count / elapsed:.1f} samples/s); "
        f"skipped_completed={skipped_completed}, "
        f"skipped_wrong_size={skipped_wrong_size}, skipped_no_image={skipped_no_image}"
    )
    print(
        f"[{split_name}] per-dataset samples: "
        + json.dumps(dict(sorted(samples_by_dataset.items())))
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    normalize_actions = not args.no_normalize_actions
    if args.action_target_mode == "absolute_state":
        normalize_actions = False

    dataset_root = args.root.expanduser().resolve()
    configs = build_lazy_configs(
        root=dataset_root,
        dataset_ids=args.dataset or list(OXE_DATASETS),
        max_episodes=args.max_episodes,
    )
    validate_dataset_configs(
        configs,
        dataset_root,
        normalize_actions=normalize_actions,
    )
    if args.num_workers != 0 or args.pin_memory:
        print(
            "Note: --num-workers and --pin-memory do not apply to this "
            "episode-at-a-time extractor; Qwen batching is controlled by "
            "--batch-size."
        )
    splits = build_combined_standardized_splits(
        configs=configs,
        split_ratios=args.split_ratios,
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
    student_contract = validate_student_runtime_contract(
        student,
        processor,
        args=args,
        cfg=cfg,
    )
    print("Loading T5 encoder...")
    t5_tokenizer, t5_encoder = load_t5(args, cfg)

    output_dir = args.output_dir
    if args.stage is not None and not args.no_stage_subdir:
        output_dir = output_dir / f"stage_{args.stage}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Validation first gives an interrupted long extraction a usable holdout.
    split_names = args.split or ["validation", "train", "test"]

    def resolved_id(value: str | Path | None) -> str | None:
        if value is None:
            return None
        candidate = Path(value).expanduser()
        return str(candidate.resolve()) if candidate.exists() else str(value)

    metadata = {
        "feature_type": FEATURE_TYPE,
        "config": args.config,
        "root": str(dataset_root),
        "splits": split_names,
        "datasets": [config.dataset_id for config in configs],
        "dataset_schedule": args.dataset_schedule,
        "seed": seed,
        "stage": args.stage,
        "stage_count": args.stage_count,
        "droid_stage_count": args.droid_stage_count,
        "split_ratios": [float(value) for value in args.split_ratios],
        "max_samples_per_split": args.max_samples_per_split,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "max_test_samples": args.max_test_samples,
        "normalize_actions": normalize_actions,
        "action_target_mode": args.action_target_mode,
        "pred_horizon": cfg.model.pred_horizon,
        "state_dim": cfg.model.resolved_cache_state_dim,
        "action_dim": cfg.model.resolved_cache_action_dim,
        "cache_state_dim": cfg.model.resolved_cache_state_dim,
        "cache_action_dim": cfg.model.resolved_cache_action_dim,
        "rdt_state_dim": cfg.model.state_dim,
        "rdt_action_dim": cfg.model.action_dim,
        "state_encoder_layout": cfg.model.state_encoder_layout,
        "action_encoder_layout": cfg.model.action_encoder_layout,
        "max_samples_per_episode": args.max_samples_per_episode,
        "require_exact_samples_per_episode": args.require_exact_samples_per_episode,
        "gripper_change_scope": args.gripper_change_scope,
        "open_to_close_before": args.open_to_close_before,
        "open_to_close_after": args.open_to_close_after,
        "close_to_open_before": args.close_to_open_before,
        "close_to_open_after": args.close_to_open_after,
        "student_model_id": resolved_id(args.student_model_id),
        "processor_id": resolved_id(args.processor_id or args.student_model_id),
        "spatial_parameters_path": (
            resolved_id(args.spatial_parameters_path)
            if args.spatial_parameters_path is not None
            else None
        ),
        "latent_student_code_dir": resolved_id(args.latent_student_code_dir),
        "latent_count": args.latent_count,
        "spatial_token_count": args.spatial_token_count,
        "layer_index": args.layer_index,
        "latent_student_contract": student_contract,
        "prompt_template": args.prompt_template,
        "qwen_kv_dim": cfg.model.qwen_kv_dim,
        "qwen_hidden_state_dim": cfg.model.qwen_hidden_size,
        "qwen_hidden_state_granularity": "five_final_layer_spatial_tokens_per_sample_step",
        "qwen_batch_size": args.batch_size,
        "qwen_cache_scope": "per_sample",
        "qwen_kv_granularity": "five_spatial_tokens_per_sample_step",
        "include_t5": True,
        "t5_model_id": resolve_model_id(
            args.t5_model_id,
            args.t5_fallback_model_id,
        ),
        "t5_precision": args.t5_precision,
        "save_padded_features": args.save_padded_features,
        "image_history_size": args.image_history_size,
        "max_images_per_sample": args.max_images_per_sample,
        "image_storage_codec": args.image_codec,
        "image_storage_lossless": args.image_codec == "png",
        "image_jpeg_quality": (
            args.image_jpeg_quality if args.image_codec == "jpeg" else None
        ),
        "cache_layout": "episode_pack",
        "episode_shards_per_directory": args.episode_shards_per_directory,
        "manifest_write_mode": "durable_per_episode",
    }
    write_or_validate_metadata(
        output_dir,
        metadata,
        resume=args.resume,
        overwrite=args.overwrite,
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
