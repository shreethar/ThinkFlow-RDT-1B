from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "shreethar/FYP-Stage-3"
DEFAULT_TOKEN_KEYS = ("hf_token", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
DEFAULT_DATASET_IDS = ("bc_z", "bridge", "droid", "fractal", "kuka")
DEFAULT_IGNORE_PATTERNS = (
    "*.ipynb",
    "*.gif",
    "*.stackdump",
    ".env",
    "__pycache__/*",
    ".ipynb_checkpoints/*",
)


@dataclass(frozen=True)
class DatasetUploadConfig:
    dataset_id: str
    data_dir: Path
    audit_json: Path
    path_in_repo: str


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def default_upload_configs(root: Path) -> dict[str, DatasetUploadConfig]:
    mock_root = root / "dataset" / "mock_dataset"
    return {
        "bc_z": DatasetUploadConfig(
            dataset_id="bc_z",
            data_dir=mock_root / "bc_z_dataset" / "data",
            audit_json=mock_root / "bc_z_dataset" / "audit.json",
            path_in_repo="bc_z/data",
        ),
        "bridge": DatasetUploadConfig(
            dataset_id="bridge",
            data_dir=first_existing(
                mock_root / "bridge_dataset" / "data",
                mock_root / "bridge_dataset" / "bridge_subset",
            ),
            audit_json=mock_root / "bridge_dataset" / "audit.json",
            path_in_repo="bridge/data",
        ),
        "droid": DatasetUploadConfig(
            dataset_id="droid",
            data_dir=first_existing(
                mock_root / "droid_dataset" / "data",
                mock_root / "droid_dataset" / "droid_100" / "1.0.0",
            ),
            audit_json=mock_root / "droid_dataset" / "audit.json",
            path_in_repo="droid/data",
        ),
        "fractal": DatasetUploadConfig(
            dataset_id="fractal",
            data_dir=mock_root / "fractal_dataset" / "data",
            audit_json=mock_root / "fractal_dataset" / "audit.json",
            path_in_repo="fractal/data",
        ),
        "kuka": DatasetUploadConfig(
            dataset_id="kuka",
            data_dir=mock_root / "kuka_dataset" / "data",
            audit_json=mock_root / "kuka_dataset" / "audit.json",
            path_in_repo="kuka/data",
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
            value = value.strip().strip('"').strip("'")
            values[key.strip()] = value
    return values


def load_hf_token(env_path: Path, token_key: str | None) -> str:
    env_values = load_env_values(env_path)
    keys = (token_key,) if token_key is not None else DEFAULT_TOKEN_KEYS
    for key in keys:
        token = env_values.get(key)
        if token:
            return token
    key_list = ", ".join(keys)
    raise ValueError(f"No Hugging Face token found in {env_path} using key(s): {key_list}")


def import_huggingface_hub() -> tuple[Any, Any]:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:
        raise ImportError(
            "Uploading requires huggingface_hub. Run this through the project "
            "environment, for example: uv run python scripts/upload_fyp_stage3_dataset.py"
        ) from exc
    return HfApi, create_repo


def load_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_manifest(configs: list[DatasetUploadConfig], repo_id: str) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for config in configs:
        audit = load_json_or_empty(config.audit_json)
        normalization = audit.get("action_normalization", {})
        source = normalization.get("source", {})
        datasets[config.dataset_id] = {
            "data_path": config.path_in_repo,
            "audit_path": f"{config.dataset_id}/audit.json",
            "local_data_dir": str(config.data_dir),
            "dataset_name": audit.get("dataset_name"),
            "control_frequency": audit.get("control_frequency"),
            "standardized_mapping": audit.get("standardized_mapping"),
            "action_normalization": {
                "method": normalization.get("method"),
                "dim_names": normalization.get("dim_names"),
                "q01": normalization.get("q01"),
                "q99": normalization.get("q99"),
                "source": {
                    "stats_scope": source.get("stats_scope"),
                    "filter_empty_language": source.get("filter_empty_language"),
                    "max_samples_per_episode": source.get("max_samples_per_episode"),
                    "gripper_window_before": source.get("gripper_window_before"),
                    "gripper_window_after": source.get("gripper_window_after"),
                    "kuka_only_success": source.get("kuka_only_success"),
                    "num_episodes_used": source.get("num_episodes_used"),
                    "num_sampled_start_steps": source.get("num_sampled_start_steps"),
                    "num_action_rows_used_for_stats": source.get(
                        "num_action_rows_used_for_stats"
                    ),
                },
            },
        }

    return {
        "repo_id": repo_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "format": "Raw source shards plus per-dataset audit metadata for ThinkFlow RDT standardized adapters.",
        "split_policy": "Use thinkflow_rdt.adapters.build_combined_standardized_splits for deterministic episode-level 80/10/10 train/validation/test splits.",
        "datasets": datasets,
    }


def build_dataset_card(manifest: dict[str, Any]) -> str:
    lines = [
        "---",
        "license: other",
        "task_categories:",
        "- robotics",
        "pretty_name: FYP Stage 3 Robot Manipulation Dataset",
        "---",
        "",
        "# FYP Stage 3",
        "",
        "This repository contains local source shards and audit metadata for the ThinkFlow RDT standardized robotics dataset adapters.",
        "",
        "The standardized loader emits samples with:",
        "",
        "```python",
        "{",
        '    "dataset_id": str,',
        '    "episode_id": str,',
        '    "step_idx": str,',
        '    "instruction": str,',
        '    "images": {"primary": PIL.Image | None, "wrist": PIL.Image | None, "secondary": PIL.Image | None},',
        '    "image_mask": {"primary": int, "wrist": int, "secondary": int},',
        '    "state": np.ndarray,        # [7]',
        '    "state_mask": np.ndarray,   # [7]',
        '    "actions": np.ndarray,      # [32, 7]',
        '    "actions_mask": np.ndarray, # [32]',
        "}",
        "```",
        "",
        "Canonical gripper convention is `0=open`, `1=closed` before action normalization.",
        "",
        "Use `thinkflow_rdt.adapters.build_combined_standardized_splits` to recreate deterministic episode-level splits:",
        "",
        "```python",
        "from thinkflow_rdt.adapters import build_combined_standardized_splits",
        "",
        "splits = build_combined_standardized_splits(",
        "    root='.',",
        "    seed=42,",
        "    normalize_actions=True,",
        ")",
        "train_ds = splits['train']",
        "val_ds = splits['validation']",
        "test_ds = splits['test']",
        "```",
        "",
        "Split policy: 80% train, 10% validation, 10% test, split at episode level per dataset.",
        "",
        "## Included Datasets",
        "",
    ]

    for dataset_id, entry in manifest["datasets"].items():
        norm = entry.get("action_normalization", {})
        source = norm.get("source", {})
        lines.extend(
            [
                f"### `{dataset_id}`",
                "",
                f"- Data path: `{entry['data_path']}`",
                f"- Audit path: `{entry['audit_path']}`",
                f"- Control frequency: `{entry.get('control_frequency')}`",
                f"- Normalization stats scope: `{source.get('stats_scope')}`",
                f"- Episodes used for stats: `{source.get('num_episodes_used')}`",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def select_dataset_ids(raw_dataset_args: list[str] | None) -> list[str]:
    if not raw_dataset_args or "all" in raw_dataset_args:
        return list(DEFAULT_DATASET_IDS)
    return raw_dataset_args


def upload_metadata_files(
    api: Any,
    *,
    repo_id: str,
    token: str | None,
    manifest: dict[str, Any],
    dry_run: bool,
) -> None:
    if dry_run:
        print("DRY RUN: would upload README.md and dataset_manifest.json")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        manifest_path = tmp_path / "dataset_manifest.json"
        readme_path = tmp_path / "README.md"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        readme_path.write_text(build_dataset_card(manifest), encoding="utf-8")

        api.upload_file(
            path_or_fileobj=str(manifest_path),
            path_in_repo="dataset_manifest.json",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Upload standardized dataset manifest",
        )
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Upload dataset card",
        )


def upload_extra_files(
    api: Any,
    *,
    root: Path,
    repo_id: str,
    token: str | None,
    dry_run: bool,
) -> None:
    mock_root = root / "dataset" / "mock_dataset"
    for local_name in ("format.json", "sample.json"):
        path = mock_root / local_name
        if not path.exists():
            continue
        if dry_run:
            print(f"DRY RUN: would upload {path} -> {local_name}")
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=local_name,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Upload {local_name}",
        )


def upload_dataset_config(
    api: Any,
    *,
    config: DatasetUploadConfig,
    repo_id: str,
    token: str | None,
    ignore_patterns: list[str],
    dry_run: bool,
) -> None:
    if not config.data_dir.exists():
        raise FileNotFoundError(config.data_dir)
    if not config.audit_json.exists():
        raise FileNotFoundError(config.audit_json)

    audit_path_in_repo = f"{config.dataset_id}/audit.json"
    if dry_run:
        print(f"DRY RUN: would upload {config.audit_json} -> {audit_path_in_repo}")
        print(f"DRY RUN: would upload {config.data_dir} -> {config.path_in_repo}")
        return

    api.upload_file(
        path_or_fileobj=str(config.audit_json),
        path_in_repo=audit_path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Upload {config.dataset_id} audit",
    )
    api.upload_folder(
        folder_path=str(config.data_dir),
        path_in_repo=config.path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        ignore_patterns=ignore_patterns,
        commit_message=f"Upload {config.dataset_id} data",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the FYP Stage 3 dataset source bundle to Hugging Face."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument(
        "--token-key",
        default=None,
        help="Token key inside .env. Defaults to hf_token, HF_TOKEN, then HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=["all", *DEFAULT_DATASET_IDS],
        help="Dataset to upload. Repeat for multiple. Defaults to all.",
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--private", action="store_true", help="Create repo as private.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-notebooks",
        action="store_true",
        help="Do not ignore notebooks/gifs while uploading data folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    selected_ids = select_dataset_ids(args.dataset)
    all_configs = default_upload_configs(root)
    configs = [all_configs[dataset_id] for dataset_id in selected_ids]
    ignore_patterns = [] if args.include_notebooks else list(DEFAULT_IGNORE_PATTERNS)

    manifest = build_manifest(configs, args.repo_id)

    if args.dry_run:
        print(f"DRY RUN: repo_id={args.repo_id}")
        print(f"DRY RUN: selected datasets={selected_ids}")
        token = None
        api = None
    else:
        token = load_hf_token(args.env_file.expanduser().resolve(), args.token_key)
        HfApi, create_repo = import_huggingface_hub()
        create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
            token=token,
        )
        api = HfApi(token=token)

    upload_metadata_files(
        api,
        repo_id=args.repo_id,
        token=token,
        manifest=manifest,
        dry_run=args.dry_run,
    )
    upload_extra_files(
        api,
        root=root,
        repo_id=args.repo_id,
        token=token,
        dry_run=args.dry_run,
    )

    for config in configs:
        print(f"[{config.dataset_id}] {config.data_dir} -> {config.path_in_repo}")
        upload_dataset_config(
            api,
            config=config,
            repo_id=args.repo_id,
            token=token,
            ignore_patterns=ignore_patterns,
            dry_run=args.dry_run,
        )

    print("Done.")


if __name__ == "__main__":
    main()
