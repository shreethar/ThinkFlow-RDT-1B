#!/usr/bin/env python
from __future__ import annotations

import argparse

from thinkflow_rdt.config import load_config
from thinkflow_rdt.model import SFTConditionedRDT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--load-pretrained", action="store_true")
    args = parser.parse_args()
    model = SFTConditionedRDT(
        load_config(args.config), load_pretrained=args.load_pretrained
    )
    for name in model.lora_targets:
        print(name)
    print(model.trainable_parameter_report())


if __name__ == "__main__":
    main()
