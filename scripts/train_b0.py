#!/usr/bin/env python
from __future__ import annotations

import argparse

from thinkflow_rdt.config import load_config
from thinkflow_rdt.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Randomly initialize the RDT core; useful only for smoke tests.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, load_pretrained=not args.no_pretrained)


if __name__ == "__main__":
    main()
