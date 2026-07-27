#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thinkflow_rdt.checkpoint import FULL_RDT_FILE, INTERFACE_FILE, METADATA_FILE, load_trainable_artifact
from thinkflow_rdt.config import load_config
from thinkflow_rdt.model import SFTConditionedRDT


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge an RDT LoRA artifact into its base transformer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint directory containing rdt_lora and interfaces.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = SFTConditionedRDT(cfg, load_pretrained=True)
    load_trainable_artifact(model, args.checkpoint, trainable=False)
    if not hasattr(model.runner.model, "merge_and_unload"):
        raise TypeError("Checkpoint model is not PEFT/LoRA")
    model.runner.model = model.runner.model.merge_and_unload(safe_merge=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.runner.model.state_dict(), args.output_dir / FULL_RDT_FILE)
    interfaces = torch.load(args.checkpoint / INTERFACE_FILE, map_location="cpu", weights_only=True)
    interfaces["_rdt_artifact_format"] = "full"
    torch.save(interfaces, args.output_dir / INTERFACE_FILE)
    metadata = json.loads((args.checkpoint / METADATA_FILE).read_text(encoding="utf-8"))
    metadata["artifact_format"] = "full"
    metadata["merged_from"] = str(args.checkpoint.resolve())
    (args.output_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Merged artifact written to {args.output_dir}")


if __name__ == "__main__":
    main()
