#!/usr/bin/env python
from __future__ import annotations

import argparse

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
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(
        cfg,
        load_pretrained=not args.no_pretrained,
        online_siglip_model_id=args.siglip_model_id,
        online_siglip_fallback_model_id=args.siglip_fallback_model_id,
    )


if __name__ == "__main__":
    main()
