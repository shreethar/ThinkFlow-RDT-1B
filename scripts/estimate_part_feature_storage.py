#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from precompute_all_features import (  # noqa: E402
    extract_qwen_kv,
    extract_siglip_features,
    extract_t5_features,
    load_models,
    standardized_collate_fn,
)
from thinkflow_rdt.adapters.combined_lazy import (  # noqa: E402
    SPLIT_NAMES,
    build_combined_standardized_splits,
    default_lazy_standardized_dataset_configs,
)
from thinkflow_rdt.adapters.sample_filtering import (  # noqa: E402
    DEFAULT_GRIPPER_WINDOW_AFTER,
    DEFAULT_GRIPPER_WINDOW_BEFORE,
    DEFAULT_MAX_SAMPLES_PER_EPISODE,
)
from thinkflow_rdt.config import load_config  # noqa: E402


DEFAULT_REPO_ID = "shreethar/FYP-Stage-3-part-1"
DEFAULT_LOCAL_DIR = REPO_ROOT / "dataset" / "hf_parts" / "part_1"
DEFAULT_MODEL_ROOT = Path("/home/ubuntu/models")
DATASET_IDS = ("bc_z", "bridge", "droid", "fractal", "kuka")


@dataclass
class SizeStats:
    count: int = 0
    total: int = 0
    max_value: int = 0

    def add(self, value: int) -> None:
        self.count += 1
        self.total += int(value)
        self.max_value = max(self.max_value, int(value))

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else math.nan


class EpisodeLimitedIterable(IterableDataset):
    def __init__(
        self,
        datasets: Iterable[tuple[str, Iterable[dict[str, Any]]]],
        *,
        max_episodes: int,
    ) -> None:
        self.datasets = list(datasets)
        self.max_episodes = int(max_episodes)

    def __iter__(self):
        seen: set[tuple[str, str]] = set()
        for _, dataset in self.datasets:
            for sample in dataset:
                key = (str(sample["dataset_id"]), str(sample["episode_id"]))
                if key not in seen:
                    if len(seen) >= self.max_episodes:
                        return
                    seen.add(key)
                yield sample


def parse_split_values(raw_splits: list[str] | None) -> list[str]:
    if not raw_splits or "all" in raw_splits:
        return list(SPLIT_NAMES)
    return list(raw_splits)


def parse_dataset_values(raw_datasets: list[str] | None) -> list[str] | None:
    if not raw_datasets or "all" in raw_datasets:
        return None
    return list(raw_datasets)


def snapshot_download_if_needed(args: argparse.Namespace) -> None:
    if not args.download:
        return
    if args.local_dir.exists() and any(args.local_dir.iterdir()) and not args.force_download:
        print(f"Dataset already exists at {args.local_dir}; skipping download.")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Downloading requires huggingface_hub. Install it or pass --no-download "
            "after downloading the repo manually."
        ) from exc

    args.local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.local_dir),
        local_dir_use_symlinks=False,
        force_download=args.force_download,
    )


def make_splits(args: argparse.Namespace, cfg: Any):
    dataset_ids = parse_dataset_values(args.dataset)
    configs = default_lazy_standardized_dataset_configs(
        dataset_ids=dataset_ids,
        root=args.local_dir.expanduser().resolve(),
    )
    return build_combined_standardized_splits(
        configs=configs,
        seed=args.seed,
        horizon=cfg.model.pred_horizon,
        normalize_actions=True,
        filter_empty_language=not args.include_empty_language,
        max_samples_per_episode=(
            None if args.max_samples_per_episode < 0 else args.max_samples_per_episode
        ),
        gripper_window_before=args.gripper_window_before,
        gripper_window_after=args.gripper_window_after,
    )


def iter_selected_split_members(splits: dict[str, Any], split_names: list[str]):
    for split_name in split_names:
        dataset = splits[split_name]
        for member in dataset.members:
            yield f"{split_name}/{member.dataset_id}", member.dataset


def tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def serialized_size(obj: Any) -> int:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.tell()


def mib(value: float) -> float:
    return float(value) / (1024.0 * 1024.0)


def gib(value: float) -> float:
    return float(value) / (1024.0 * 1024.0 * 1024.0)


def sample_record_and_component_sizes(
    *,
    batch: dict[str, Any],
    batch_index: int,
    qwen_kv: torch.Tensor,
    lang_tokens: torch.Tensor,
    lang_mask: torch.Tensor,
    img_tokens: torch.Tensor,
    img_mask: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, int]]:
    metadata = batch["metadata"][batch_index]
    record = {
        "qwen_kv": qwen_kv[batch_index].cpu(),
        "lang_tokens": lang_tokens[batch_index].cpu(),
        "lang_mask": lang_mask[batch_index].cpu(),
        "img_tokens": img_tokens[batch_index].cpu(),
        "img_mask": img_mask[batch_index].cpu(),
        "state": batch["state"][batch_index].cpu(),
        "actions": batch["actions"][batch_index].cpu(),
        "action_time_mask": batch["action_time_mask"][batch_index].cpu(),
        "action_dim_mask": batch["action_dim_mask"][batch_index].cpu(),
        "ctrl_freq": float(batch["ctrl_freq"][batch_index].item()),
        **metadata,
    }
    component_sizes = {
        "record_pt": serialized_size(record),
        "qwen_kv_raw": tensor_nbytes(record["qwen_kv"]),
        "t5_raw": tensor_nbytes(record["lang_tokens"]) + tensor_nbytes(record["lang_mask"]),
        "siglip_raw": tensor_nbytes(record["img_tokens"]) + tensor_nbytes(record["img_mask"]),
        "state_action_raw": (
            tensor_nbytes(record["state"])
            + tensor_nbytes(record["actions"])
            + tensor_nbytes(record["action_time_mask"])
            + tensor_nbytes(record["action_dim_mask"])
        ),
    }
    return record, component_sizes


def count_total_steps(
    *,
    splits: dict[str, Any],
    split_names: list[str],
) -> dict[str, Any]:
    total = 0
    per_split: dict[str, int] = {}
    per_dataset: dict[str, int] = defaultdict(int)
    per_dataset_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for split_name in split_names:
        split_total = 0
        for sample in tqdm(splits[split_name], desc=f"count {split_name}", unit="sample"):
            dataset_id = str(sample["dataset_id"])
            total += 1
            split_total += 1
            per_dataset[dataset_id] += 1
            per_dataset_split[dataset_id][split_name] += 1
        per_split[split_name] = split_total

    return {
        "total_steps": total,
        "per_split": per_split,
        "per_dataset": dict(per_dataset),
        "per_dataset_split": {
            dataset_id: dict(split_counts)
            for dataset_id, split_counts in per_dataset_split.items()
        },
    }


def estimate_features(args: argparse.Namespace, cfg: Any, splits: dict[str, Any]) -> dict[str, Any]:
    split_names = parse_split_values(args.split)
    datasets = list(iter_selected_split_members(splits, split_names))
    probe_dataset = EpisodeLimitedIterable(datasets, max_episodes=args.probe_episodes)
    dataloader = DataLoader(
        probe_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=lambda samples: standardized_collate_fn(
            samples,
            max_images_per_sample=args.max_images_per_sample,
            image_history_size=args.image_history_size,
            image_jpeg_quality=args.image_jpeg_quality,
            skip_no_image=not args.keep_no_image,
        ),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device_map == "cuda" and device.type != "cuda":
        args.device_map = "cpu"
    models = load_models(args, cfg, device)

    step_stats = {
        "record_pt": SizeStats(),
        "qwen_kv_raw": SizeStats(),
        "t5_raw": SizeStats(),
        "siglip_raw": SizeStats(),
        "state_action_raw": SizeStats(),
    }
    episode_bytes: dict[tuple[str, str], int] = defaultdict(int)
    episode_steps: dict[tuple[str, str], int] = defaultdict(int)
    dataset_step_counts: dict[str, int] = defaultdict(int)
    image_slot_counts: dict[int, int] = defaultdict(int)
    image_valid_counts: dict[int, int] = defaultdict(int)

    sample_count = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(dataloader, desc="extract probe", unit="batch")):
            if batch is None:
                continue
            qwen_kv = extract_qwen_kv(
                batch,
                models["qwen_processor"],
                models["qwen_vlm"],
                device=device,
                layer_index=args.qwen_layer_index,
                max_new_tokens=args.qwen_max_new_tokens,
                expected_dim=cfg.model.qwen_kv_dim,
            )
            lang_tokens, lang_mask = extract_t5_features(
                batch,
                models["t5_tokenizer"],
                models["t5_encoder"],
                max_lang_tokens=cfg.model.max_lang_tokens,
                expected_dim=cfg.model.lang_token_dim,
                device=device,
            )
            img_tokens, img_mask = extract_siglip_features(
                batch,
                models["siglip_processor"],
                models["siglip_encoder"],
                max_img_tokens=cfg.model.image_tokens,
                expected_dim=cfg.model.img_token_dim,
                device=device,
            )

            for local_index in range(qwen_kv.shape[0]):
                record, sizes = sample_record_and_component_sizes(
                    batch=batch,
                    batch_index=local_index,
                    qwen_kv=qwen_kv,
                    lang_tokens=lang_tokens,
                    lang_mask=lang_mask,
                    img_tokens=img_tokens,
                    img_mask=img_mask,
                )
                for key, value in sizes.items():
                    step_stats[key].add(value)
                episode_key = (str(record["dataset_id"]), str(record["episode_id"]))
                episode_bytes[episode_key] += sizes["record_pt"]
                episode_steps[episode_key] += 1
                dataset_step_counts[str(record["dataset_id"])] += 1
                image_slot_counts[int(record.get("image_slot_count", 0))] += 1
                image_valid_counts[int(record.get("image_count", 0))] += 1
                sample_count += 1

            if args.max_batches is not None and batch_index + 1 >= args.max_batches:
                break

    episode_size_values = list(episode_bytes.values())
    episode_step_values = list(episode_steps.values())
    return {
        "probe_episodes": len(episode_bytes),
        "probe_steps": sample_count,
        "probe_dataset_step_counts": dict(dataset_step_counts),
        "image_slot_count_histogram": dict(sorted(image_slot_counts.items())),
        "valid_image_count_histogram": dict(sorted(image_valid_counts.items())),
        "per_step": {
            key: {
                "mean_bytes": stats.mean,
                "mean_mib": mib(stats.mean),
                "max_bytes": stats.max_value,
                "max_mib": mib(stats.max_value),
            }
            for key, stats in step_stats.items()
        },
        "per_episode": {
            "mean_steps": float(np_mean(episode_step_values)),
            "max_steps": max(episode_step_values) if episode_step_values else 0,
            "mean_bytes": float(np_mean(episode_size_values)),
            "mean_mib": mib(np_mean(episode_size_values)),
            "max_bytes": max(episode_size_values) if episode_size_values else 0,
            "max_mib": mib(max(episode_size_values) if episode_size_values else 0),
        },
    }


def np_mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else math.nan


def print_summary(report: dict[str, Any]) -> None:
    print("\nFeature Storage Probe")
    print(json.dumps(report["probe"], indent=2))
    total = report.get("total_steps")
    if total is not None:
        print("\nTotal standardized steps in selected part")
        print(json.dumps(total, indent=2))

    total_steps = total["total_steps"] if total is not None else None
    mean_record_bytes = report["probe"]["per_step"]["record_pt"]["mean_bytes"]
    if total_steps is not None and mean_record_bytes == mean_record_bytes:
        estimated = mean_record_bytes * total_steps
        print("\nEstimated full feature cache from probe mean")
        print(f"  total steps: {total_steps}")
        print(f"  mean per step: {mib(mean_record_bytes):.3f} MiB")
        print(f"  estimated total: {gib(estimated):.2f} GiB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a raw staged dataset part, run full Qwen/T5/SigLIP feature "
            "extraction on a limited episode probe, and estimate feature-cache storage."
        )
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--delete-dataset-after", action="store_true")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "b0_rdt1b_lora.yaml"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--split",
        action="append",
        choices=["all", *SPLIT_NAMES],
        help="Split to include. Repeat for multiple. Defaults to all.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=["all", *DATASET_IDS],
        help="Dataset to include. Repeat for multiple. Defaults to all.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-episodes", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--max-samples-per-episode",
        type=int,
        default=-1,
        help=(
            "Sampling cap for counting/probe. -1 means all valid language steps; "
            "64 matches the standard loader cap."
        ),
    )
    parser.add_argument("--include-empty-language", action="store_true")
    parser.add_argument("--gripper-window-before", type=int, default=DEFAULT_GRIPPER_WINDOW_BEFORE)
    parser.add_argument("--gripper-window-after", type=int, default=DEFAULT_GRIPPER_WINDOW_AFTER)
    parser.add_argument("--skip-total-count", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--image-history-size", type=int, default=2)
    parser.add_argument("--max-images-per-sample", type=int, default=6)
    parser.add_argument("--image-jpeg-quality", type=int, default=90)
    parser.add_argument("--keep-no-image", action="store_true")
    parser.add_argument("--qwen-model-id", default="shreethar/stage1_unsloth")
    parser.add_argument("--qwen-layer-index", type=int, default=7)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--t5-model-id", default=str(DEFAULT_MODEL_ROOT / "t5-v1_1-xxl"))
    parser.add_argument("--t5-fallback-model-id", default="google/t5-v1_1-xxl")
    parser.add_argument(
        "--siglip-model-id",
        default=str(DEFAULT_MODEL_ROOT / "siglip-so400m-patch14-384"),
    )
    parser.add_argument(
        "--siglip-fallback-model-id",
        default="google/siglip-so400m-patch14-384",
    )
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.feature_set = "all"
    if args.probe_episodes <= 0:
        raise ValueError("--probe-episodes must be positive")
    snapshot_download_if_needed(args)

    cfg = load_config(args.config)
    splits = make_splits(args, cfg)
    split_names = parse_split_values(args.split)

    total_report = None
    if not args.skip_total_count:
        total_report = count_total_steps(splits=splits, split_names=split_names)
        splits = make_splits(args, cfg)

    probe_report = estimate_features(args, cfg, splits)
    report = {
        "repo_id": args.repo_id,
        "local_dir": str(args.local_dir),
        "config": args.config,
        "splits": split_names,
        "datasets": parse_dataset_values(args.dataset) or list(DATASET_IDS),
        "count_policy": {
            "max_samples_per_episode": None
            if args.max_samples_per_episode < 0
            else args.max_samples_per_episode,
            "filter_empty_language": not args.include_empty_language,
            "image_history_size": args.image_history_size,
            "max_images_per_sample": args.max_images_per_sample,
            "image_tokens": cfg.model.image_tokens,
            "pred_horizon": cfg.model.pred_horizon,
        },
        "total_steps": total_report,
        "probe": probe_report,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_summary(report)

    if args.delete_dataset_after:
        shutil.rmtree(args.local_dir, ignore_errors=True)
        print(f"Deleted downloaded dataset directory: {args.local_dir}")


if __name__ == "__main__":
    main()
