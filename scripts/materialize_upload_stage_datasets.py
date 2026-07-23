#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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


DEFAULT_REPO_TEMPLATE = "shreethar/FYP-Stage-3-part-{stage}"
DEFAULT_TOKEN_KEYS = ("hf_token", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
DEFAULT_DATASET_IDS = ("bc_z", "bridge", "droid", "fractal", "kuka")
IMAGE_KEYS = ("primary", "wrist", "secondary")
CTRL_FREQ_BY_DATASET = {
    "bc_z": 10.0,
    "bridge": 10.0,
    "droid": 15.0,
    "fractal": 3.0,
    "kuka": 3.0,
}


@dataclass
class ShardWriter:
    path: Path
    split_name: str
    stage: int
    index: int
    handle: tarfile.TarFile = field(init=False)
    sample_count: int = 0
    dataset_counts: dict[str, int] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = tarfile.open(self.path, mode="w")

    def close(self) -> None:
        self.handle.close()


def load_env_values(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    with env_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_hf_token(env_path: Path, token_key: str | None) -> str:
    values = load_env_values(env_path)
    keys = (token_key,) if token_key is not None else DEFAULT_TOKEN_KEYS
    for key in keys:
        token = values.get(key)
        if token:
            return token
    raise ValueError(f"No Hugging Face token found in {env_path} using keys {keys}")


def import_huggingface_hub() -> tuple[Any, Any]:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:
        raise ImportError(
            "Uploading requires huggingface_hub. Run through the project env, e.g. "
            "uv run python scripts/materialize_upload_stage_datasets.py"
        ) from exc
    return HfApi, create_repo


def select_dataset_ids(raw_dataset_args: list[str] | None, stage: int, droid_stage_count: int) -> list[str]:
    dataset_ids = (
        list(DEFAULT_DATASET_IDS)
        if not raw_dataset_args or "all" in raw_dataset_args
        else list(raw_dataset_args)
    )
    if stage > droid_stage_count:
        dataset_ids = [dataset_id for dataset_id in dataset_ids if dataset_id != "droid"]
    return dataset_ids


def parse_stage_values(raw_stage: str) -> list[int]:
    if raw_stage == "all":
        return [1, 2, 3]
    return [int(raw_stage)]


def parse_split_values(raw_splits: list[str] | None) -> list[str]:
    if not raw_splits or "all" in raw_splits:
        return list(SPLIT_NAMES)
    return list(raw_splits)


def as_rgb_pil(image: Any) -> Image.Image | None:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.size == 0 or not np.any(array):
        return None
    return Image.fromarray(array.astype(np.uint8)).convert("RGB")


def image_to_bytes(image: Image.Image, image_format: str, jpeg_quality: int) -> bytes:
    buffer = io.BytesIO()
    if image_format == "jpeg":
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    elif image_format == "png":
        image.save(buffer, format="PNG", optimize=True)
    else:
        raise ValueError(f"Unsupported image format: {image_format}")
    return buffer.getvalue()


def arrays_to_npz_bytes(sample: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        state=np.asarray(sample["state"], dtype=np.float32),
        state_mask=np.asarray(sample.get("state_mask", np.ones((7,), dtype=np.float32)), dtype=np.float32),
        actions=np.asarray(sample["actions"], dtype=np.float32),
        actions_mask=np.asarray(sample["actions_mask"], dtype=np.float32),
        action_dim_mask=np.asarray(sample.get("action_dim_mask", np.ones((7,), dtype=np.float32)), dtype=np.float32),
    )
    return buffer.getvalue()


def add_bytes_to_tar(tar: tarfile.TarFile, member_name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(member_name)
    info.size = len(payload)
    info.mtime = 0
    tar.addfile(info, io.BytesIO(payload))


def safe_sample_prefix(global_index: int, sample: dict[str, Any]) -> str:
    dataset_id = str(sample["dataset_id"])
    digest_input = (
        f"{dataset_id}:{sample['episode_id']}:{sample['step_idx']}:{global_index}"
    ).encode("utf-8")
    import hashlib

    digest = hashlib.blake2b(digest_input, digest_size=6).hexdigest()
    return f"{global_index:09d}_{dataset_id}_{digest}"


def add_sample_to_shard(
    writer: ShardWriter,
    *,
    sample: dict[str, Any],
    global_index: int,
    image_format: str,
    jpeg_quality: int,
) -> dict[str, Any]:
    prefix = safe_sample_prefix(global_index, sample)
    sample_dir = f"samples/{prefix}"
    arrays_member = f"{sample_dir}/arrays.npz"
    add_bytes_to_tar(writer.handle, arrays_member, arrays_to_npz_bytes(sample))

    image_members: dict[str, str] = {}
    image_mask: dict[str, int] = {}
    images = sample.get("images", {})
    image_suffix = "jpg" if image_format == "jpeg" else "png"
    for key in IMAGE_KEYS:
        pil_image = as_rgb_pil(images.get(key))
        if pil_image is None:
            image_mask[key] = 0
            continue
        member_name = f"{sample_dir}/{key}.{image_suffix}"
        add_bytes_to_tar(
            writer.handle,
            member_name,
            image_to_bytes(pil_image, image_format, jpeg_quality),
        )
        image_members[key] = member_name
        image_mask[key] = 1

    dataset_id = str(sample["dataset_id"])
    metadata = {
        "format_version": 1,
        "sample_id": prefix,
        "dataset_id": dataset_id,
        "episode_id": str(sample["episode_id"]),
        "step_idx": str(sample["step_idx"]),
        "split": writer.split_name,
        "stage": writer.stage,
        "instruction": str(sample["instruction"]),
        "arrays": arrays_member,
        "images": image_members,
        "image_mask": image_mask,
        "ctrl_freq": float(sample.get("ctrl_freq", CTRL_FREQ_BY_DATASET.get(dataset_id, 10.0))),
        "actions_are_normalized": True,
        "gripper_convention_before_normalization": "0=open, 1=closed",
    }
    metadata_member = f"{sample_dir}/metadata.json"
    add_bytes_to_tar(
        writer.handle,
        metadata_member,
        (json.dumps(metadata, ensure_ascii=False) + "\n").encode("utf-8"),
    )

    writer.sample_count += 1
    writer.dataset_counts[dataset_id] = writer.dataset_counts.get(dataset_id, 0) + 1
    manifest_record = {
        "stage": writer.stage,
        "split": writer.split_name,
        "shard": writer.path.name,
        "sample_id": prefix,
        "metadata": metadata_member,
        "arrays": arrays_member,
        "dataset_id": dataset_id,
        "episode_id": metadata["episode_id"],
        "step_idx": metadata["step_idx"],
        "image_mask": image_mask,
    }
    writer.records.append(manifest_record)
    return manifest_record


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def repo_id_for_stage(template: str, stage: int) -> str:
    return template.format(stage=stage)


def upload_file(
    api: Any,
    *,
    local_path: Path,
    path_in_repo: str,
    repo_id: str,
    token: str | None,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"DRY RUN: would upload {local_path} -> {repo_id}/{path_in_repo}")
        return
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Upload {path_in_repo}",
    )


def discover_audit_paths(root: Path, dataset_ids: Iterable[str]) -> dict[str, Path]:
    nested_mock_root = root / "dataset" / "mock_dataset"
    mock_root = root if (root / "format.json").exists() or (root / "bc_z_dataset").exists() else nested_mock_root
    mapping = {
        "bc_z": mock_root / "bc_z_dataset" / "audit.json",
        "bridge": mock_root / "bridge_dataset" / "audit.json",
        "droid": mock_root / "droid_dataset" / "audit.json",
        "fractal": mock_root / "fractal_dataset" / "audit.json",
        "kuka": mock_root / "kuka_dataset" / "audit.json",
    }
    return {
        dataset_id: path
        for dataset_id, path in mapping.items()
        if dataset_id in set(dataset_ids) and path.exists()
    }


def build_readme(stage: int, repo_id: str, manifest: dict[str, Any]) -> str:
    datasets = ", ".join(manifest["datasets"])
    lines = [
        "---",
        "license: other",
        "task_categories:",
        "- robotics",
        f"pretty_name: FYP Stage 3 Part {stage}",
        "---",
        "",
        f"# FYP Stage 3 Part {stage}",
        "",
        "This repository contains materialized ThinkFlow RDT standardized samples.",
        "",
        f"- Repo: `{repo_id}`",
        f"- Stage: `{stage}`",
        f"- Datasets: `{datasets}`",
        "- Samples are already filtered, split, sampled, and action-normalized.",
        "- Canonical raw gripper convention before normalization: `0=open`, `1=closed`.",
        "",
        "Each `.tar` shard contains per-sample:",
        "",
        "- `metadata.json` with instruction, ids, split/stage, image paths, and masks",
        "- `arrays.npz` with `state`, `state_mask`, `actions`, `actions_mask`, `action_dim_mask`",
        "- compressed RGB image files for available views",
        "",
        "Use `materialized_manifest.jsonl` to enumerate samples.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def create_repo_if_needed(
    *,
    api: Any,
    create_repo: Any,
    repo_id: str,
    token: str | None,
    private: bool,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"DRY RUN: would create/ensure dataset repo {repo_id}")
        return
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )


def materialize_split(
    *,
    dataset: Any,
    split_name: str,
    stage: int,
    stage_dir: Path,
    repo_id: str,
    api: Any,
    token: str | None,
    dry_run: bool,
    upload: bool,
    delete_after_upload: bool,
    samples_per_shard: int,
    max_samples_per_split: int | None,
    image_format: str,
    jpeg_quality: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_dir = stage_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    writer: ShardWriter | None = None
    sample_count = 0
    shard_index = 0

    def finalize_writer(current: ShardWriter | None) -> None:
        if current is None:
            return
        current.close()
        shard_summary = {
            "stage": stage,
            "split": split_name,
            "path": f"{split_name}/{current.path.name}",
            "sample_count": current.sample_count,
            "dataset_counts": current.dataset_counts,
            "size_bytes": current.path.stat().st_size,
        }
        shard_summaries.append(shard_summary)
        if upload:
            upload_file(
                api,
                local_path=current.path,
                path_in_repo=f"{split_name}/{current.path.name}",
                repo_id=repo_id,
                token=token,
                dry_run=dry_run,
            )
            if delete_after_upload and not dry_run:
                current.path.unlink(missing_ok=True)

    progress = tqdm(dataset, desc=f"stage {stage} {split_name}", unit="sample")
    for sample in progress:
        if max_samples_per_split is not None and sample_count >= max_samples_per_split:
            break
        if writer is None or writer.sample_count >= samples_per_shard:
            finalize_writer(writer)
            shard_name = f"{split_name}-stage{stage}-{shard_index:06d}.tar"
            writer = ShardWriter(
                path=split_dir / shard_name,
                split_name=split_name,
                stage=stage,
                index=shard_index,
            )
            shard_index += 1

        record = add_sample_to_shard(
            writer,
            sample=sample,
            global_index=sample_count,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
        )
        all_records.append(record)
        sample_count += 1
        progress.set_postfix(samples=sample_count)

    finalize_writer(writer)
    return all_records, {
        "sample_count": sample_count,
        "shards": shard_summaries,
    }


def materialize_stage(
    *,
    args: argparse.Namespace,
    stage: int,
    api: Any,
    create_repo: Any,
    token: str | None,
) -> None:
    dataset_ids = select_dataset_ids(args.dataset, stage, args.droid_stage_count)
    split_names = parse_split_values(args.split)
    repo_id = repo_id_for_stage(args.repo_id_template, stage)
    stage_dir = args.output_dir.expanduser().resolve() / f"part_{stage}"
    if not dataset_ids:
        print(f"[stage {stage}] no datasets selected; skipping")
        return
    if args.dry_run:
        print(f"DRY RUN: stage={stage}")
        print(f"DRY RUN: repo_id={repo_id}")
        print(f"DRY RUN: output_dir={stage_dir}")
        print(f"DRY RUN: datasets={dataset_ids}")
        print(f"DRY RUN: splits={split_names}")
        print(f"DRY RUN: would materialize tar shards with {args.samples_per_shard} samples/shard")
        return
    if args.overwrite and stage_dir.exists() and not args.delete_after_upload:
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    create_repo_if_needed(
        api=api,
        create_repo=create_repo,
        repo_id=repo_id,
        token=token,
        private=args.private,
        dry_run=args.dry_run or not args.upload,
    )

    configs = default_lazy_standardized_dataset_configs(
        dataset_ids=dataset_ids,
        root=args.root.expanduser().resolve(),
    )
    splits = build_combined_standardized_splits(
        configs=configs,
        seed=args.seed,
        stage=stage,
        stage_count=args.stage_count,
        droid_stage_count=args.droid_stage_count,
        horizon=args.horizon,
        normalize_actions=True,
        filter_empty_language=not args.include_empty_language,
        max_samples_per_episode=(
            None if args.max_samples_per_episode < 0 else args.max_samples_per_episode
        ),
        gripper_window_before=args.gripper_window_before,
        gripper_window_after=args.gripper_window_after,
    )

    manifest_records: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    for split_name in split_names:
        records, summary = materialize_split(
            dataset=splits[split_name],
            split_name=split_name,
            stage=stage,
            stage_dir=stage_dir,
            repo_id=repo_id,
            api=api,
            token=token,
            dry_run=args.dry_run,
            upload=args.upload,
            delete_after_upload=args.delete_after_upload,
            samples_per_shard=args.samples_per_shard,
            max_samples_per_split=args.max_samples_per_split,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
        )
        manifest_records.extend(records)
        split_summaries[split_name] = summary

    manifest_jsonl = stage_dir / "materialized_manifest.jsonl"
    with manifest_jsonl.open("w", encoding="utf-8") as handle:
        for record in manifest_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    audit_paths = discover_audit_paths(args.root.expanduser().resolve(), dataset_ids)
    manifest = {
        "repo_id": repo_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "format": "materialized_standardized_tar_v1",
        "stage": stage,
        "stage_count": args.stage_count,
        "droid_stage_count": args.droid_stage_count,
        "datasets": dataset_ids,
        "splits": split_names,
        "seed": args.seed,
        "horizon": args.horizon,
        "filter_empty_language": not args.include_empty_language,
        "max_samples_per_episode": None
        if args.max_samples_per_episode < 0
        else args.max_samples_per_episode,
        "gripper_window_before": args.gripper_window_before,
        "gripper_window_after": args.gripper_window_after,
        "actions_are_normalized": True,
        "split_summaries": split_summaries,
    }
    manifest_json = stage_dir / "materialized_dataset_manifest.json"
    readme_path = stage_dir / "README.md"
    write_json(manifest_json, manifest)
    readme_path.write_text(build_readme(stage, repo_id, manifest), encoding="utf-8")

    if args.upload:
        upload_file(
            api,
            local_path=manifest_jsonl,
            path_in_repo="materialized_manifest.jsonl",
            repo_id=repo_id,
            token=token,
            dry_run=args.dry_run,
        )
        upload_file(
            api,
            local_path=manifest_json,
            path_in_repo="materialized_dataset_manifest.json",
            repo_id=repo_id,
            token=token,
            dry_run=args.dry_run,
        )
        upload_file(
            api,
            local_path=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
            dry_run=args.dry_run,
        )
        for dataset_id, audit_path in audit_paths.items():
            upload_file(
                api,
                local_path=audit_path,
                path_in_repo=f"audits/{dataset_id}_audit.json",
                repo_id=repo_id,
                token=token,
                dry_run=args.dry_run,
            )

    print(f"[stage {stage}] wrote {len(manifest_records)} materialized samples to {stage_dir}")
    print(f"[stage {stage}] repo: {repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize filtered/normalized standardized samples into tar shards "
            "and upload stage-specific Hugging Face dataset repos."
        )
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "materialized_stage_datasets")
    parser.add_argument("--repo-id-template", default=DEFAULT_REPO_TEMPLATE)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--token-key", default=None)
    parser.add_argument("--stage", choices=["all", "1", "2", "3"], default="all")
    parser.add_argument("--stage-count", type=int, default=3)
    parser.add_argument("--droid-stage-count", type=int, default=2)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=["all", *DEFAULT_DATASET_IDS],
        help="Dataset to include. Repeat for multiple. Defaults to all.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=["all", *SPLIT_NAMES],
        help="Split to include. Repeat for multiple. Defaults to all.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--max-samples-per-episode", type=int, default=DEFAULT_MAX_SAMPLES_PER_EPISODE)
    parser.add_argument("--gripper-window-before", type=int, default=DEFAULT_GRIPPER_WINDOW_BEFORE)
    parser.add_argument("--gripper-window-after", type=int, default=DEFAULT_GRIPPER_WINDOW_AFTER)
    parser.add_argument("--include-empty-language", action="store_true")
    parser.add_argument("--samples-per-shard", type=int, default=1024)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--image-format", choices=["jpeg", "png"], default="jpeg")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--upload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--delete-after-upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_shard <= 0:
        raise ValueError("--samples-per-shard must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")

    if args.upload and not args.dry_run:
        token = load_hf_token(args.env_file.expanduser().resolve(), args.token_key)
        HfApi, create_repo = import_huggingface_hub()
        api = HfApi(token=token)
    else:
        token = None

        def create_repo(**_: Any) -> None:
            return None

        api = None

    for stage in parse_stage_values(args.stage):
        materialize_stage(
            args=args,
            stage=stage,
            api=api,
            create_repo=create_repo,
            token=token,
        )

    print("Done.")


if __name__ == "__main__":
    main()
