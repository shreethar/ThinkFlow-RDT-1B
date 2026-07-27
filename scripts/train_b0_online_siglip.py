#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace

from thinkflow_rdt.config import load_config
from thinkflow_rdt.train import train


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train B0 with cached Qwen/T5 features and frozen SigLIP computed "
            "online from cached image slots."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--siglip-model-id",
        default="/home/ubuntu/models/siglip-so400m-patch14-384",
    )
    parser.add_argument(
        "--siglip-fallback-model-id",
        default="google/siglip-so400m-patch14-384",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Randomly initialize the RDT core; useful only for smoke tests.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override training.max_steps from the YAML (optimizer updates).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output_dir so trial checkpoints do not replace a full run.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Override training.wandb_run_name from the YAML.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Override training.log_every (measured in optimizer updates).",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.log_every is not None and args.log_every <= 0:
        parser.error("--log-every must be positive")
    training = cfg.training
    if args.max_steps is not None:
        training = replace(training, max_steps=args.max_steps)
    if args.wandb_run_name is not None:
        training = replace(training, wandb_run_name=args.wandb_run_name)
    if args.log_every is not None:
        training = replace(training, log_every=args.log_every)
    cfg = replace(
        cfg,
        training=training,
        output_dir=args.output_dir or cfg.output_dir,
    )
    cfg.validate()
    train(
        cfg,
        load_pretrained=not args.no_pretrained,
        online_siglip_model_id=args.siglip_model_id,
        online_siglip_fallback_model_id=args.siglip_fallback_model_id,
    )


if __name__ == "__main__":
    main()
