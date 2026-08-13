#!/usr/bin/env python
"""Compare two SmolVLA Hub checkpoints without constructing an environment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors import safe_open


ARCHITECTURE_FIELDS = (
    "type",
    "vlm_model_name",
    "attention_mode",
    "num_vlm_layers",
    "num_expert_layers",
    "self_attn_every_n_layers",
    "expert_width_multiplier",
    "chunk_size",
    "max_state_dim",
    "max_action_dim",
    "pad_language_to",
    "add_image_special_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="lerobot/smolvla_base")
    parser.add_argument("--candidate", default="lerobot/smolvla_libero")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def download(repo_id: str, filename: str, args: argparse.Namespace) -> Path:
    return Path(
        hf_hub_download(
            repo_id,
            filename,
            revision=args.revision,
            local_files_only=args.local_files_only,
        )
    )


def tensor_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    with safe_open(path, framework="pt", device="cpu") as tensors:
        return {name: tuple(tensors.get_slice(name).get_shape()) for name in tensors.keys()}


def feature_summary(config: dict) -> dict:
    return {
        "inputs": config.get("input_features", {}),
        "outputs": config.get("output_features", {}),
        "empty_cameras": config.get("empty_cameras"),
        "n_action_steps": config.get("n_action_steps"),
        "num_steps": config.get("num_steps"),
    }


def main() -> None:
    args = parse_args()
    base_config = json.loads(download(args.base, "config.json", args).read_text())
    candidate_config = json.loads(download(args.candidate, "config.json", args).read_text())
    base_shapes = tensor_shapes(download(args.base, "model.safetensors", args))
    candidate_shapes = tensor_shapes(download(args.candidate, "model.safetensors", args))

    architecture = {
        field: {"base": base_config.get(field), "candidate": candidate_config.get(field)}
        for field in ARCHITECTURE_FIELDS
        if base_config.get(field) != candidate_config.get(field)
    }
    base_keys = set(base_shapes)
    candidate_keys = set(candidate_shapes)
    common = base_keys & candidate_keys
    shape_mismatches = {
        key: {"base": base_shapes[key], "candidate": candidate_shapes[key]}
        for key in sorted(common)
        if base_shapes[key] != candidate_shapes[key]
    }
    report = {
        "base": args.base,
        "candidate": args.candidate,
        "feature_contract": {
            "base": feature_summary(base_config),
            "candidate": feature_summary(candidate_config),
        },
        "architecture_field_differences": architecture,
        "weights": {
            "base_tensor_count": len(base_shapes),
            "candidate_tensor_count": len(candidate_shapes),
            "common_tensor_count": len(common),
            "base_only_count": len(base_keys - candidate_keys),
            "candidate_only_count": len(candidate_keys - base_keys),
            "common_shape_mismatch_count": len(shape_mismatches),
            "base_only_prefixes": Counter(key.split(".")[0] for key in base_keys - candidate_keys),
            "candidate_only_prefixes": Counter(
                key.split(".")[0] for key in candidate_keys - base_keys
            ),
            "shape_mismatch_examples": dict(list(shape_mismatches.items())[:30]),
        },
        "same_serialized_architecture": not architecture and not shape_mismatches,
        "direct_weight_compatible": (
            base_keys == candidate_keys and not shape_mismatches
        ),
    }
    print(json.dumps(report, indent=2, default=dict))


if __name__ == "__main__":
    main()
