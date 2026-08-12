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
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

from .configuration import make_libero_kv_config
from .modeling import KVSmolVLAPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="lerobot/smolvla_base")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/smolvla_base_qwen_kv_init"),
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("outputs/smolvla_base_qwen_kv_all_suites/cache_stats.pt"),
        help="Native 8D-state/7D-action statistics saved by cache training.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=4)
    parser.add_argument("--external-logit-bias-init", type=float, default=-4.0)
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
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing cache statistics {stats_path}. Run cache statistics/training once, "
            "or pass --stats to a compatible state8/action7 cache_stats.pt."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    payload = torch.load(stats_path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "libero_native_state8_action7_v1":
        raise ValueError(f"Incompatible statistics schema in {stats_path}")
    stats = payload["stats"]

    config = make_libero_kv_config(
        args.base,
        device=args.device,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        train_expert_only=True,
        freeze_vision_encoder=True,
        external_kv_logit_bias_init=args.external_logit_bias_init,
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
        "stats": str(stats_path),
        "samples_in_stats": int(payload.get("num_samples", 0)),
        "policy_type": config.type,
        "state_dim": 8,
        "action_dim": 7,
        "external_kv_width": config.external_kv_width,
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
