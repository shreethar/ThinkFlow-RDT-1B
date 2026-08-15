"""Create a clean Qwen-KV SmolVLA bootstrap from ``lerobot/smolvla_base``.

The base checkpoint supplies all native SmolVLA weights.  Only the additional
per-cross-attention Qwen K/V projections and logit biases are newly initialized.
The result is a normal custom-policy checkpoint that the literal
``lerobot-train --policy.path=...`` command can load.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

if "--local-files-only" in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

from .configuration import make_libero_kv_config
from .cached_libero import list_shards
from .modeling import KVSmolVLAPolicy
from .stats import load_or_compute_cache_stats


LIBERO_SUITES = ("libero_10", "libero_spatial", "libero_goal", "libero_object")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="lerobot/smolvla_base")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/smolvla_base_qwen_kv_init"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Compute --stats from these cached suites when the stats file does not exist.",
    )
    parser.add_argument("--suites", nargs="+", choices=LIBERO_SUITES, default=list(LIBERO_SUITES))
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("outputs/smolvla_base_qwen_kv_all_suites/cache_stats.pt"),
        help="Native 8D-state/7D-action statistics saved by cache training.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help=(
            "Actions executed per re-plan. Defaults to 4 for cache training, or "
            "preserves the source checkpoint when --preserve-base-processors is set."
        ),
    )
    parser.add_argument("--external-logit-bias-init", type=float, default=-1.0)
    parser.add_argument("--external-kv-ranking-weight", type=float, default=0.1)
    parser.add_argument("--external-kv-ranking-margin", type=float, default=0.01)
    parser.add_argument("--external-kv-adapter-warmup-steps", type=int, default=1000)
    parser.add_argument("--external-kv-adapter-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--action-expert-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--external-kv-token-count", type=int, choices=[1, 5], default=1)
    parser.add_argument(
        "--external-kv-optional",
        action="store_true",
        help="Save a checkpoint that bypasses external K/V when qwen_kv is absent.",
    )
    parser.add_argument(
        "--preserve-base-processors",
        action="store_true",
        help=(
            "Load and save the base checkpoint's own pre/post-processors. Use this "
            "for a behavior-preserving conversion control such as smolvla_libero."
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    stats_path = args.stats.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty bootstrap directory: {output_dir}"
        )
    if not args.preserve_base_processors and not stats_path.exists():
        if args.cache_root is None:
            raise FileNotFoundError(
                f"Missing cache statistics {stats_path}. Pass --cache-root to compute "
                "them from native state8/action7 shards."
            )
        cache_root = args.cache_root.expanduser().resolve()
        shard_paths = [
            path
            for suite in args.suites
            for path in list_shards(cache_root, suite, split="train")
        ]
        print(f"Computing native LIBERO statistics from {len(shard_paths)} shards")
        load_or_compute_cache_stats(
            stats_path,
            shard_paths,
            chunk_size=args.chunk_size,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    payload = None
    stats = None
    if not args.preserve_base_processors:
        payload = torch.load(stats_path, map_location="cpu", weights_only=True)
        if payload.get("schema") != "libero_native_state8_action7_v1":
            raise ValueError(f"Incompatible statistics schema in {stats_path}")
        stats = payload["stats"]

    source_config = PreTrainedConfig.from_pretrained(
        args.base,
        local_files_only=args.local_files_only,
    )
    n_action_steps = (
        args.n_action_steps
        if args.n_action_steps is not None
        else (
            int(source_config.n_action_steps)
            if args.preserve_base_processors
            else 4
        )
    )
    config = make_libero_kv_config(
        args.base,
        device=args.device,
        chunk_size=args.chunk_size,
        n_action_steps=n_action_steps,
        train_expert_only=(
            bool(source_config.train_expert_only)
            if args.preserve_base_processors
            else True
        ),
        freeze_vision_encoder=(
            bool(source_config.freeze_vision_encoder)
            if args.preserve_base_processors
            else True
        ),
        external_kv_logit_bias_init=args.external_logit_bias_init,
        external_kv_token_count=args.external_kv_token_count,
        external_kv_required=not args.external_kv_optional,
        external_kv_ranking_weight=args.external_kv_ranking_weight,
        external_kv_ranking_margin=args.external_kv_ranking_margin,
        external_kv_adapter_warmup_steps=args.external_kv_adapter_warmup_steps,
        external_kv_adapter_learning_rate=args.external_kv_adapter_learning_rate,
        action_expert_learning_rate=args.action_expert_learning_rate,
        # The official smolvla_libero processor maps observations to camera1/camera2.
        # A behavior-preserving conversion must keep those exact feature names;
        # replacing them with cache-training names image/image2 breaks inference.
        preserve_pretrained_features=args.preserve_base_processors,
        local_files_only=args.local_files_only,
    )
    print(f"Loading native SmolVLA weights from {args.base}")
    print("Expected: missing keys only for newly initialized external Qwen K/V adapters")
    policy = KVSmolVLAPolicy.from_pretrained(
        args.base,
        config=config,
        local_files_only=args.local_files_only,
        strict=False,
    )
    if args.preserve_base_processors:
        # A conversion control must retain the source checkpoint's normalization.
        # Recomputing cache statistics here would change policy behavior even when
        # external Qwen K/V fusion is disabled.
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            args.base,
        )
    else:
        preprocessor, postprocessor = make_smolvla_pre_post_processors(
            config,
            dataset_stats=stats,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    bootstrap = {
        "base": args.base,
        "seed": args.seed,
        "stats": str(stats_path) if payload is not None else None,
        "processor_source": args.base if args.preserve_base_processors else "cache_stats",
        "samples_in_stats": int(payload.get("num_samples", 0)) if payload else 0,
        "policy_type": config.type,
        "input_features": {
            key: {"type": value.type.value, "shape": list(value.shape)}
            for key, value in config.input_features.items()
        },
        "output_features": {
            key: {"type": value.type.value, "shape": list(value.shape)}
            for key, value in config.output_features.items()
        },
        "chunk_size": config.chunk_size,
        "n_action_steps": config.n_action_steps,
        "external_kv_width": config.external_kv_width,
        "external_kv_token_count": config.external_kv_token_count,
        "external_kv_required": config.external_kv_required,
        "external_kv_logit_bias_init": config.external_kv_logit_bias_init,
        "external_kv_ranking_weight": config.external_kv_ranking_weight,
        "external_kv_ranking_margin": config.external_kv_ranking_margin,
        "external_kv_adapter_warmup_steps": config.external_kv_adapter_warmup_steps,
        "external_kv_adapter_learning_rate": config.external_kv_adapter_learning_rate,
        "action_expert_learning_rate": config.action_expert_learning_rate,
        "initialization": {
            "native_smolvla": "pretrained base weights",
            "external_qwen_kv_adapters": "new seeded initialization",
        },
    }
    (output_dir / "bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n")
    print(json.dumps(bootstrap, indent=2))
    print(f"Saved clean Qwen-KV bootstrap: {output_dir}")


if __name__ == "__main__":
    main()
