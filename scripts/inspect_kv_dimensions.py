#!/usr/bin/env python3
"""Measure Qwen3.5 and RDT attention K/V widths with real forward projections."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText


def layer_key_values(cache, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "layers"):
        return cache.layers[layer_index].keys, cache.layers[layer_index].values
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_index], cache.value_cache[layer_index]
    return cache[layer_index][:2]


def flattened_width(tensor: torch.Tensor) -> int:
    # Cache layout is [batch, heads, sequence, head_dim].
    return int(tensor.shape[1] * tensor.shape[-1])


def inspect_qwen(args: argparse.Namespace) -> tuple[int, int]:
    dtype = torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(
        args.qwen_model,
        local_files_only=not args.allow_download,
        dtype=dtype,
        device_map=args.device,
        attn_implementation="sdpa",
    )
    model.eval()
    text_config = model.config.text_config
    token_id = text_config.eos_token_id
    if token_id is None:
        token_id = 0
    input_ids = torch.tensor([[token_id]], device=args.device)
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
            return_dict=True,
        )
    keys, values = layer_key_values(output.past_key_values, args.qwen_layer)
    key_width = flattened_width(keys)
    value_width = flattened_width(values)
    print(f"Qwen model: {model.__class__.__name__}")
    print(
        f"Qwen layer {args.qwen_layer}: "
        f"{text_config.layer_types[args.qwen_layer]}"
    )
    print(f"Qwen K: shape={tuple(keys.shape)}, flattened width={key_width}")
    print(f"Qwen V: shape={tuple(values.shape)}, flattened width={value_width}")
    print(f"Qwen concatenated K+V width: {key_width + value_width}")
    del output, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return key_width, value_width


def inspect_rdt(args: argparse.Namespace) -> tuple[int, int]:
    repo = Path(args.rdt_repo).expanduser().resolve()
    sys.path.insert(0, str(repo))
    from models.rdt.model import RDT  # type: ignore

    model = RDT(
        output_dim=7,
        horizon=64,
        hidden_size=args.rdt_hidden_size,
        depth=1,
        num_heads=args.rdt_heads,
        max_lang_cond_len=128,
        img_cond_len=4374,
        dtype=torch.bfloat16,
    ).to(args.device)
    attention = model.blocks[0].attn
    tokens = torch.zeros(
        1,
        2,
        args.rdt_hidden_size,
        dtype=torch.bfloat16,
        device=args.device,
    )
    with torch.no_grad():
        qkv = attention.qkv(tokens).reshape(
            1,
            2,
            3,
            attention.num_heads,
            attention.head_dim,
        ).permute(2, 0, 3, 1, 4)
    _, keys, values = qkv.unbind(0)
    key_width = flattened_width(keys)
    value_width = flattened_width(values)
    print(f"RDT attention heads: {attention.num_heads}")
    print(f"RDT head dimension: {attention.head_dim}")
    print(f"RDT K: shape={tuple(keys.shape)}, flattened width={key_width}")
    print(f"RDT V: shape={tuple(values.shape)}, flattened width={value_width}")
    print(f"RDT concatenated K+V width: {key_width + value_width}")
    return key_width, value_width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-model", default="shreethar/stage1_unsloth")
    parser.add_argument("--qwen-layer", type=int, default=7)
    parser.add_argument(
        "--rdt-repo",
        default="/home/ubuntu/RoboticsDiffusionTransformer",
    )
    parser.add_argument("--rdt-hidden-size", type=int, default=2048)
    parser.add_argument("--rdt-heads", type=int, default=32)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads instead of requiring a cached model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qwen_key_width, qwen_value_width = inspect_qwen(args)
    rdt_key_width, rdt_value_width = inspect_rdt(args)
    qwen_width = qwen_key_width + qwen_value_width
    rdt_width = rdt_key_width + rdt_value_width
    print(f"Required concatenated-KV projector: Linear({qwen_width}, {rdt_width})")


if __name__ == "__main__":
    main()
