from __future__ import annotations

from typing import Iterable

import torch.nn as nn
from peft import LoraConfig, get_peft_model

from .config import LoraConfigData


SELF_ATTN_SUFFIXES = ("attn.qkv", "attn.proj")
CROSS_ATTN_SUFFIXES = (
    "cross_attn.q",
    "cross_attn.kv",
    "cross_attn.proj",
)
FFN_SUFFIXES = ("ffn.fc1", "ffn.fc2")


def find_lora_targets(rdt_core: nn.Module, cfg: LoraConfigData) -> list[str]:
    suffixes: list[str] = []
    if cfg.target_self_attention:
        suffixes.extend(SELF_ATTN_SUFFIXES)
    if cfg.target_cross_attention:
        suffixes.extend(CROSS_ATTN_SUFFIXES)
    if cfg.target_ffn:
        suffixes.extend(FFN_SUFFIXES)
    if not suffixes:
        raise ValueError("At least one LoRA target family must be enabled")

    targets: list[str] = []
    for name, module in rdt_core.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith("blocks."):
            continue
        if any(name.endswith(suffix) for suffix in suffixes):
            targets.append(name)
    if not targets:
        available = [
            name for name, module in rdt_core.named_modules()
            if isinstance(module, nn.Linear)
        ]
        raise RuntimeError(
            "No RDT LoRA targets matched. First linear modules: "
            + ", ".join(available[:30])
        )
    return targets


def apply_lora(rdt_core: nn.Module, cfg: LoraConfigData) -> tuple[nn.Module, list[str]]:
    targets = find_lora_targets(rdt_core, cfg)
    modules_to_save = ["final_layer"] if cfg.train_final_layer else None
    peft_cfg = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=targets,
        modules_to_save=modules_to_save,
        bias="none",
    )
    wrapped = get_peft_model(rdt_core, peft_cfg)
    return wrapped, targets


def count_parameters(module: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return trainable, total
