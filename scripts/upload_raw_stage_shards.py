#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_TEMPLATE = "shreethar/FYP-Stage-3-part-{part}"
DEFAULT_TOKEN_KEYS = ("hf_token", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
PARTS = (1, 2, 3)
DEFAULT_DATASET_IDS = ("bc_z", "bridge", "droid", "fractal", "kuka")


@dataclass(frozen=True)
class RawShardDatasetConfig:
    dataset_id: str
    data_dir: Path
    audit_json: Path
    shard_template: str
    part_targets: dict[int, int | None]

    @property
    def dataset_info_json(self) -> Path:
        return self.data_dir / "dataset_info.json"

    def shard_filename(self, shard_index: int) -> str:
        return self.shard_template.format(shard=shard_index)

    def shard_path(self, shard_index: int) -> Path:
        return self.data_dir / self.shard_filename(shard_index)


@dataclass(frozen=True)
class PartSelection:
    dataset_id: str
    part: int
    target_episodes: int | None
    episode_interval: tuple[int, int] | None
    shard_indices: list[int]
    raw_episode_count: int
    missing_files: list[Path]
    allow_patterns: list[str]


def mock_root_from_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    nested = root / "dataset" / "mock_dataset"
    if nested.exists():
        return nested
    return root


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def default_dataset_configs(root: Path) -> dict[str, RawShardDatasetConfig]:
    mock_root = mock_root_from_root(root)
    return {
        "fractal": RawShardDatasetConfig(
            dataset_id="fractal",
            data_dir=mock_root / "fractal_dataset" / "data",
            audit_json=mock_root / "fractal_dataset" / "audit.json",
            shard_template="fractal20220817_data-train.tfrecord-{shard:05d}-of-01024",
            part_targets={1: 8436, 2: 8436, 3: 8437},
        ),
        "kuka": RawShardDatasetConfig(
            dataset_id="kuka",
            data_dir=mock_root / "kuka_dataset" / "data",
            audit_json=mock_root / "kuka_dataset" / "audit.json",
            shard_template="kuka-train.tfrecord-{shard:05d}-of-01024",
            part_targets={1: 16739, 2: 16739, 3: 16739},
        ),
        "bridge": RawShardDatasetConfig(
            dataset_id="bridge",
            data_dir=first_existing(
                mock_root / "bridge_dataset" / "data",
                mock_root / "bridge_dataset" / "bridge_subset",
            ),
            audit_json=mock_root / "bridge_dataset" / "audit.json",
            shard_template="bridge_data_v2-train.tfrecord-{shard:05d}-of-01024",
            part_targets={1: 11675, 2: 11676, 3: 11676},
        ),
        "droid": RawShardDatasetConfig(
            dataset_id="droid",
            data_dir=first_existing(
                mock_root / "droid_dataset" / "data",
                mock_root / "droid_dataset" / "droid_100" / "1.0.0",
            ),
            audit_json=mock_root / "droid_dataset" / "audit.json",
            shard_template="droid_101-train.tfrecord-{shard:05d}-of-02048",
            part_targets={1: 15015, 2: 15014, 3: None},
        ),
        "bc_z": RawShardDatasetConfig(
            dataset_id="bc_z",
            data_dir=mock_root / "bc_z_dataset" / "data",
            audit_json=mock_root / "bc_z_dataset" / "audit.json",
            shard_template="bc_z-train.array_record-{shard:05d}-of-01024",
            part_targets={1: 5008, 2: 5009, 3: 5009},
        ),
    }


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
            "uv run python scripts/upload_raw_stage_shards.py"
        ) from exc
    return HfApi, create_repo


def select_dataset_ids(raw_dataset_args: list[str] | None) -> list[str]:
    if not raw_dataset_args or "all" in raw_dataset_args:
        return list(DEFAULT_DATASET_IDS)
    return list(raw_dataset_args)


def select_parts(raw_part_args: list[int] | None) -> list[int]:
    if not raw_part_args:
        return list(PARTS)
    return sorted(set(raw_part_args))


def read_train_shard_lengths(dataset_info_json: Path) -> list[int]:
    if not dataset_info_json.exists():
        raise FileNotFoundError(dataset_info_json)
    with dataset_info_json.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    train_split = next(
        (split for split in info.get("splits", []) if split.get("name") == "train"),
        None,
    )
    if train_split is None:
        raise ValueError(f"No train split found in {dataset_info_json}")
    return [int(value) for value in train_split["shardLengths"]]


def cumulative_intervals(lengths: list[int]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = 0
    for length in lengths:
        stop = start + int(length)
        intervals.append((start, stop))
        start = stop
    return intervals


def target_intervals(part_targets: dict[int, int | None]) -> dict[int, tuple[int, int] | None]:
    intervals: dict[int, tuple[int, int] | None] = {}
    start = 0
    for part in PARTS:
        target = part_targets.get(part)
        if target is None:
            intervals[part] = None
            continue
        stop = start + int(target)
        intervals[part] = (start, stop)
        start = stop
    return intervals


def shard_intersects_interval(shard_interval: tuple[int, int], interval: tuple[int, int]) -> bool:
    shard_start, shard_stop = shard_interval
    start, stop = interval
    return shard_start < stop and shard_stop > start


def shard_starts_in_interval(shard_interval: tuple[int, int], interval: tuple[int, int]) -> bool:
    shard_start, _ = shard_interval
    start, stop = interval
    return start <= shard_start < stop


def metadata_allow_patterns(data_dir: Path) -> list[str]:
    patterns: list[str] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if ".tfrecord-" in path.name or ".array_record-" in path.name:
            continue
        patterns.append(path.relative_to(data_dir).as_posix())
    return patterns


def select_shards_for_dataset(
    config: RawShardDatasetConfig,
    *,
    part: int,
    boundary_policy: str,
    skip_missing: bool,
) -> PartSelection:
    lengths = read_train_shard_lengths(config.dataset_info_json)
    shard_intervals = cumulative_intervals(lengths)
    intervals = target_intervals(config.part_targets)
    interval = intervals[part]
    target = config.part_targets.get(part)
    if interval is None or target is None:
        return PartSelection(
            dataset_id=config.dataset_id,
            part=part,
            target_episodes=None,
            episode_interval=None,
            shard_indices=[],
            raw_episode_count=0,
            missing_files=[],
            allow_patterns=metadata_allow_patterns(config.data_dir),
        )

    if boundary_policy == "unique":
        selected = [
            index
            for index, shard_interval in enumerate(shard_intervals)
            if shard_starts_in_interval(shard_interval, interval)
        ]
    elif boundary_policy == "overlap":
        selected = [
            index
            for index, shard_interval in enumerate(shard_intervals)
            if shard_intersects_interval(shard_interval, interval)
        ]
    else:
        raise ValueError(f"Unknown boundary policy: {boundary_policy}")

    missing = [config.shard_path(index) for index in selected if not config.shard_path(index).exists()]
    if missing and not skip_missing:
        preview = "\n".join(str(path) for path in missing[:10])
        suffix = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        raise FileNotFoundError(
            f"{config.dataset_id} part {part} is missing {len(missing)} selected shard files:\n"
            f"{preview}{suffix}"
        )
    if skip_missing:
        selected = [index for index in selected if config.shard_path(index).exists()]

    allow_patterns = metadata_allow_patterns(config.data_dir)
    allow_patterns.extend(config.shard_filename(index) for index in selected)
    raw_episode_count = sum(lengths[index] for index in selected)
    return PartSelection(
        dataset_id=config.dataset_id,
        part=part,
        target_episodes=target,
        episode_interval=interval,
        shard_indices=selected,
        raw_episode_count=raw_episode_count,
        missing_files=missing,
        allow_patterns=allow_patterns,
    )


def repo_id_for_part(template: str, part: int) -> str:
    return template.format(part=part, stage=part)


def upload_text_file(
    api: Any,
    *,
    text: str,
    path_in_repo: str,
    repo_id: str,
    token: str | None,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"DRY RUN: would upload generated {path_in_repo} -> {repo_id}")
        return
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    try:
        api.upload_file(
            path_or_fileobj=str(temp_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Upload {path_in_repo}",
        )
    finally:
        temp_path.unlink(missing_ok=True)


def upload_dataset_selection(
    api: Any,
    *,
    config: RawShardDatasetConfig,
    selection: PartSelection,
    repo_id: str,
    token: str | None,
    dry_run: bool,
) -> None:
    data_path_in_repo = f"{config.dataset_id}/data"
    audit_path_in_repo = f"{config.dataset_id}/audit.json"

    if selection.shard_indices:
        first = selection.shard_indices[0]
        last = selection.shard_indices[-1]
        shard_summary = f"shards {first:05d}-{last:05d} ({len(selection.shard_indices)} files)"
    else:
        shard_summary = "no shards"
    print(
        f"[{repo_id}] {config.dataset_id}: {shard_summary}, "
        f"target={selection.target_episodes}, raw={selection.raw_episode_count}"
    )
    if selection.missing_files:
        preview = ", ".join(path.name for path in selection.missing_files[:5])
        suffix = "" if len(selection.missing_files) <= 5 else f", ... +{len(selection.missing_files) - 5}"
        print(
            f"[{repo_id}] {config.dataset_id}: missing {len(selection.missing_files)} "
            f"selected local shard files: {preview}{suffix}"
        )

    if dry_run:
        print(f"DRY RUN: would upload folder {config.data_dir} -> {data_path_in_repo}")
        print(f"DRY RUN: allow_patterns={len(selection.allow_patterns)} files")
        print(f"DRY RUN: would upload {config.audit_json} -> {audit_path_in_repo}")
        return

    if not selection.shard_indices:
        return
    api.upload_folder(
        folder_path=str(config.data_dir),
        path_in_repo=data_path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        allow_patterns=selection.allow_patterns,
        commit_message=f"Upload {config.dataset_id} raw shards for part {selection.part}",
    )
    api.upload_file(
        path_or_fileobj=str(config.audit_json),
        path_in_repo=audit_path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Upload {config.dataset_id} audit for part {selection.part}",
    )


def build_manifest(
    *,
    repo_id: str,
    part: int,
    selections: list[PartSelection],
    boundary_policy: str,
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for selection in selections:
        if not selection.shard_indices:
            continue
        datasets[selection.dataset_id] = {
            "data_path": f"{selection.dataset_id}/data",
            "audit_path": f"{selection.dataset_id}/audit.json",
            "target_episodes": selection.target_episodes,
            "episode_interval": selection.episode_interval,
            "raw_episode_count_from_selected_shards": selection.raw_episode_count,
            "shard_count": len(selection.shard_indices),
            "first_shard": selection.shard_indices[0],
            "last_shard": selection.shard_indices[-1],
            "shards": selection.shard_indices,
            "missing_file_count": len(selection.missing_files),
        }
    return {
        "repo_id": repo_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "format": "raw_tfds_shards_by_stage_part",
        "part": part,
        "boundary_policy": boundary_policy,
        "note": (
            "Raw shards are uploaded directly. Empty-language samples, Kuka failures, "
            "unused timesteps, train/validation/test splitting, and action normalization "
            "are still handled by the standardized loaders using the included audit.json files."
        ),
        "datasets": datasets,
    }


def build_readme(manifest: dict[str, Any]) -> str:
    lines = [
        "---",
        "license: other",
        "task_categories:",
        "- robotics",
        f"pretty_name: FYP Stage 3 Part {manifest['part']} Raw Shards",
        "---",
        "",
        f"# FYP Stage 3 Part {manifest['part']}",
        "",
        "This repository contains raw TFDS/ArrayRecord shards for one stage part of the ThinkFlow RDT dataset subset.",
        "",
        "The files are not materialized samples. The standardized loader still applies filtering, sampling, splitting, and normalization.",
        "",
        f"Boundary policy: `{manifest['boundary_policy']}`",
        "",
        "## Datasets",
        "",
    ]
    for dataset_id, dataset in manifest["datasets"].items():
        lines.extend(
            [
                f"### `{dataset_id}`",
                "",
                f"- Data path: `{dataset['data_path']}`",
                f"- Audit path: `{dataset['audit_path']}`",
                f"- Target episodes: `{dataset['target_episodes']}`",
                f"- Raw episodes from selected shards: `{dataset['raw_episode_count_from_selected_shards']}`",
                f"- Shards: `{dataset['first_shard']:05d}` to `{dataset['last_shard']:05d}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def create_repo_if_needed(
    *,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload raw local shards into stage/part-specific Hugging Face dataset repos."
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--repo-id-template", default=DEFAULT_REPO_TEMPLATE)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--token-key", default=None)
    parser.add_argument(
        "--part",
        action="append",
        type=int,
        choices=PARTS,
        help="Part/stage repo to upload. Repeat for multiple. Defaults to all.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=["all", *DEFAULT_DATASET_IDS],
        help="Dataset to upload. Repeat for multiple. Defaults to all.",
    )
    parser.add_argument(
        "--boundary-policy",
        choices=["unique", "overlap"],
        default="unique",
        help=(
            "unique assigns each raw shard to only one part based on shard start episode; "
            "overlap uploads boundary shards to both neighboring parts if needed."
        ),
    )
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    dataset_ids = select_dataset_ids(args.dataset)
    parts = select_parts(args.part)
    configs = default_dataset_configs(root)

    if args.dry_run:
        api = None
        token = None

        def create_repo(**_: Any) -> None:
            return None

    else:
        token = load_hf_token(args.env_file.expanduser().resolve(), args.token_key)
        HfApi, create_repo = import_huggingface_hub()
        api = HfApi(token=token)

    for part in parts:
        repo_id = repo_id_for_part(args.repo_id_template, part)
        selected_configs = [
            configs[dataset_id]
            for dataset_id in dataset_ids
            if not (dataset_id == "droid" and part == 3)
        ]
        create_repo_if_needed(
            create_repo=create_repo,
            repo_id=repo_id,
            token=token,
            private=args.private,
            dry_run=args.dry_run,
        )

        selections: list[PartSelection] = []
        for config in selected_configs:
            selection = select_shards_for_dataset(
                config,
                part=part,
                boundary_policy=args.boundary_policy,
                skip_missing=args.skip_missing or args.dry_run,
            )
            selections.append(selection)
            upload_dataset_selection(
                api,
                config=config,
                selection=selection,
                repo_id=repo_id,
                token=token,
                dry_run=args.dry_run,
            )

        manifest = build_manifest(
            repo_id=repo_id,
            part=part,
            selections=selections,
            boundary_policy=args.boundary_policy,
        )
        upload_text_file(
            api,
            text=json.dumps(manifest, indent=2) + "\n",
            path_in_repo="raw_stage_manifest.json",
            repo_id=repo_id,
            token=token,
            dry_run=args.dry_run,
        )
        upload_text_file(
            api,
            text=build_readme(manifest),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
            dry_run=args.dry_run,
        )

    print("Done.")


if __name__ == "__main__":
    main()
