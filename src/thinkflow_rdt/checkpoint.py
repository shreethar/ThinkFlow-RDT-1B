from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict


INTERFACE_FILE = "interfaces.pt"
METADATA_FILE = "metadata.json"
ADAPTER_DIR = "rdt_lora"


def save_trainable_artifact(model, output_dir: str | Path, metadata: dict[str, Any]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.runner.model.save_pretrained(output / ADAPTER_DIR)
    torch.save(
        {
            "lang_adaptor": model.runner.lang_adaptor.state_dict(),
            "img_adaptor": model.runner.img_adaptor.state_dict(),
            "state_adaptor": model.runner.state_adaptor.state_dict(),
        },
        output / INTERFACE_FILE,
    )
    with (output / METADATA_FILE).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def load_trainable_artifact(model, artifact_dir: str | Path, trainable: bool) -> None:
    artifact = Path(artifact_dir)
    adapter_dir = artifact / ADAPTER_DIR
    if not adapter_dir.exists():
        raise FileNotFoundError(adapter_dir)
    # The model constructor has already created the same default LoRA layout.
    # Load the saved adapter into that layout instead of double-wrapping it.
    adapter_state = load_peft_weights(str(adapter_dir), device="cpu")
    set_peft_model_state_dict(
        model.runner.model,
        adapter_state,
        adapter_name="default",
    )
    for name, parameter in model.runner.model.named_parameters():
        if "lora_" in name or "modules_to_save" in name:
            parameter.requires_grad = trainable
    interfaces = torch.load(
        artifact / INTERFACE_FILE,
        map_location="cpu",
        weights_only=False,
    )
    model.runner.lang_adaptor.load_state_dict(interfaces["lang_adaptor"])
    model.runner.img_adaptor.load_state_dict(interfaces["img_adaptor"])
    model.runner.state_adaptor.load_state_dict(interfaces["state_adaptor"])
