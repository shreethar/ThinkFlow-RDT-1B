#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import math
from typing import Iterable

import torch

from thinkflow_rdt.config import load_config
from thinkflow_rdt.model import SFTConditionedRDT
from thinkflow_rdt.train import create_optimizer, seed_everything


def make_smooth_actions(batch_size: int, horizon: int, action_dim: int, device: torch.device) -> torch.Tensor:
    """Create a deterministic, bounded target chunk that is easier to memorize than white noise."""
    t = torch.linspace(0.0, 1.0, horizon, device=device, dtype=torch.float32)
    channels = []
    for dim in range(action_dim):
        frequency = dim + 1
        if dim == action_dim - 1:
            # Gripper-like channel: closed for first half, open for second half.
            value = torch.where(t < 0.5, -torch.ones_like(t), torch.ones_like(t))
        elif dim % 2 == 0:
            value = 0.25 * torch.sin(2.0 * math.pi * frequency * t)
        else:
            value = 0.25 * torch.cos(2.0 * math.pi * frequency * t)
        channels.append(value)
    action = torch.stack(channels, dim=-1)
    return action.unsqueeze(0).repeat(batch_size, 1, 1)


def make_synthetic_batch(cfg, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 123)

    # These tensors stand in for frozen Qwen hidden states. They are deliberately
    # created in float32; SFTConditionedRDT casts them to its configured dtype.
    lang_tokens = torch.randn(
        batch_size,
        min(16, cfg.model.max_lang_tokens),
        cfg.model.qwen_hidden_size,
        generator=generator,
    )
    img_tokens = torch.randn(
        batch_size,
        cfg.model.image_tokens,
        cfg.model.qwen_hidden_size,
        generator=generator,
    )
    state = torch.randn(
        batch_size,
        cfg.model.state_dim,
        generator=generator,
    ).clamp(-1.0, 1.0)
    actions = make_smooth_actions(
        batch_size,
        cfg.model.pred_horizon,
        cfg.model.action_dim,
        device=torch.device("cpu"),
    )

    return {
        "lang_tokens": lang_tokens.to(device),
        "img_tokens": img_tokens.to(device),
        "state": state.to(device),
        "actions": actions.to(device),
        "lang_mask": torch.ones(batch_size, lang_tokens.shape[1], dtype=torch.bool, device=device),
        "img_mask": torch.ones(batch_size, img_tokens.shape[1], dtype=torch.bool, device=device),
        "action_time_mask": torch.ones(batch_size, cfg.model.pred_horizon, dtype=torch.bool, device=device),
        "action_dim_mask": torch.ones(batch_size, cfg.model.action_dim, dtype=torch.float32, device=device),
        "ctrl_freq": torch.full((batch_size,), 10.0, dtype=torch.float32, device=device),
    }


def first_named_parameter(model: torch.nn.Module, predicates: Iterable[str], trainable: bool | None = None):
    for name, parameter in model.named_parameters():
        if trainable is not None and parameter.requires_grad != trainable:
            continue
        if any(token in name for token in predicates):
            return name, parameter
    raise RuntimeError(f"Could not find a parameter matching: {tuple(predicates)}")


def clone_parameter(parameter: torch.Tensor) -> torch.Tensor:
    return parameter.detach().float().cpu().clone()


def changed(before: torch.Tensor, after_parameter: torch.Tensor) -> float:
    after = after_parameter.detach().float().cpu()
    return float((after - before).abs().max())


def grad_norm(parameter: torch.Tensor) -> float | None:
    if parameter.grad is None:
        return None
    return float(parameter.grad.detach().float().norm().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-only forward/backward and tiny overfit test.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--feature-file",
        default=None,
        help="Optional .pt file containing one real frozen-Qwen feature sample. Keys: lang_tokens [L,2560], img_tokens [I,2560], and optionally state/actions/masks/ctrl_freq.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Randomly initialize the RDT core. Use with tiny_smoke.yaml, not the final 1B test.",
    )
    parser.add_argument(
        "--fixed-diffusion-rng",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset RNG before every forward so noise, timestep, and LoRA dropout are fixed. This makes the memorization test unambiguous.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)

    if not torch.cuda.is_available():
        if cfg.model.dtype.lower() in {"bf16", "bfloat16", "fp16", "float16"}:
            raise RuntimeError("The 1B bf16/fp16 smoke test requires a CUDA GPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    print("\n[1/6] Constructing model...")
    model = SFTConditionedRDT(cfg, load_pretrained=not args.no_pretrained)
    model.to(device)
    model.train()
    print(model.trainable_parameter_report())
    print(f"Device: {device}; compute dtype: {model.model_dtype}")

    print("\n[2/6] Creating one fixed batch...")
    if args.feature_file is None:
        batch = make_synthetic_batch(cfg, args.batch_size, device)
        print("  Source: synthetic frozen-Qwen features")
    else:
        sample = torch.load(args.feature_file, map_location="cpu", weights_only=False)
        required = {"lang_tokens", "img_tokens"}
        missing = required.difference(sample)
        if missing:
            raise KeyError(f"Feature file is missing keys: {sorted(missing)}")

        lang = sample["lang_tokens"].float()
        img = sample["img_tokens"].float()
        if lang.ndim == 3 and lang.shape[0] == 1:
            lang = lang[0]
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]
        if lang.ndim != 2 or lang.shape[-1] != cfg.model.qwen_hidden_size:
            raise ValueError(f"lang_tokens must be [L,{cfg.model.qwen_hidden_size}], got {tuple(lang.shape)}")
        if img.shape != (cfg.model.image_tokens, cfg.model.qwen_hidden_size):
            raise ValueError(
                f"img_tokens must be [{cfg.model.image_tokens},{cfg.model.qwen_hidden_size}], got {tuple(img.shape)}. "
                "Pool/pad the visual features before this smoke test."
            )

        state = sample.get("state", torch.zeros(cfg.model.state_dim)).float().reshape(-1)
        if state.numel() != cfg.model.state_dim:
            raise ValueError(f"state must have {cfg.model.state_dim} values, got {state.numel()}")
        actions = sample.get("actions")
        if actions is None:
            actions = make_smooth_actions(1, cfg.model.pred_horizon, cfg.model.action_dim, torch.device("cpu"))[0]
        actions = actions.float()
        if actions.shape != (cfg.model.pred_horizon, cfg.model.action_dim):
            raise ValueError(
                f"actions must be [{cfg.model.pred_horizon},{cfg.model.action_dim}], got {tuple(actions.shape)}"
            )

        lang_mask = sample.get("lang_mask", torch.ones(lang.shape[0], dtype=torch.bool)).bool()
        img_mask = sample.get("img_mask", torch.ones(img.shape[0], dtype=torch.bool)).bool()
        action_time_mask = sample.get(
            "action_time_mask", torch.ones(cfg.model.pred_horizon, dtype=torch.bool)
        ).bool()
        action_dim_mask = sample.get(
            "action_dim_mask", torch.ones(cfg.model.action_dim, dtype=torch.float32)
        ).float()
        ctrl_freq_value = float(sample.get("ctrl_freq", 10.0))

        batch = {
            "lang_tokens": lang.unsqueeze(0).repeat(args.batch_size, 1, 1).to(device),
            "img_tokens": img.unsqueeze(0).repeat(args.batch_size, 1, 1).to(device),
            "state": state.unsqueeze(0).repeat(args.batch_size, 1).to(device),
            "actions": actions.unsqueeze(0).repeat(args.batch_size, 1, 1).to(device),
            "lang_mask": lang_mask.unsqueeze(0).repeat(args.batch_size, 1).to(device),
            "img_mask": img_mask.unsqueeze(0).repeat(args.batch_size, 1).to(device),
            "action_time_mask": action_time_mask.unsqueeze(0).repeat(args.batch_size, 1).to(device),
            "action_dim_mask": action_dim_mask.unsqueeze(0).repeat(args.batch_size, 1).to(device),
            "ctrl_freq": torch.full((args.batch_size,), ctrl_freq_value, dtype=torch.float32, device=device),
        }
        print(f"  Source: real feature file {args.feature_file}")
    for key, value in batch.items():
        print(f"  {key:22s} {tuple(value.shape)} {value.dtype}")

    print("\n[3/6] Selecting representative parameters...")
    lora_name, lora_param = first_named_parameter(model, ("lora_A",), trainable=True)
    interface_name, interface_param = first_named_parameter(
        model, ("runner.lang_adaptor", "runner.img_adaptor", "runner.state_adaptor"), trainable=True
    )
    final_name, final_param = first_named_parameter(model, ("final_layer",), trainable=True)
    frozen_name, frozen_param = first_named_parameter(
        model, ("base_layer.weight", "attn.qkv.weight"), trainable=False
    )
    print("  LoRA:      ", lora_name)
    print("  Interface: ", interface_name)
    print("  Final head:", final_name)
    print("  Frozen RDT:", frozen_name)

    snapshots = {
        "lora": clone_parameter(lora_param),
        "interface": clone_parameter(interface_param),
        "final": clone_parameter(final_param),
        "frozen": clone_parameter(frozen_param),
    }

    optimizer = create_optimizer(model, cfg)
    initial_loss = None
    losses: list[float] = []

    print("\n[4/6] Forward/backward and tiny memorization test...")
    for step in range(args.steps):
        if args.fixed_diffusion_rng:
            # Makes sampled noise, timestep and dropout identical each iteration.
            torch.manual_seed(cfg.seed + 999)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed + 999)

        optimizer.zero_grad(set_to_none=True)
        metrics = model(batch)
        loss = metrics["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()

        if step == 0:
            print("  First backward gradient norms:")
            print(f"    LoRA       : {grad_norm(lora_param)}")
            print(f"    Interface  : {grad_norm(interface_param)}")
            print(f"    Final head : {grad_norm(final_param)}")
            print(f"    Frozen RDT : {grad_norm(frozen_param)} (must be None)")
            if lora_param.grad is None:
                raise RuntimeError("LoRA parameter received no gradient")
            if interface_param.grad is None:
                raise RuntimeError("Interface parameter received no gradient")
            if final_param.grad is None:
                raise RuntimeError("Final action head received no gradient")
            if frozen_param.grad is not None:
                raise RuntimeError("A frozen base-RDT parameter received a gradient")

        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            cfg.training.max_grad_norm,
        )
        optimizer.step()

        value = float(loss.detach().float().cpu())
        losses.append(value)
        initial_loss = value if initial_loss is None else initial_loss
        print(f"  step={step:03d} loss={value:.6f} mae={float(metrics['train_target_mae']):.6f}")

    print("\n[5/6] Checking which parameters changed...")
    deltas = {
        "LoRA": changed(snapshots["lora"], lora_param),
        "Interface": changed(snapshots["interface"], interface_param),
        "Final head": changed(snapshots["final"], final_param),
        "Frozen RDT": changed(snapshots["frozen"], frozen_param),
    }
    for name, delta in deltas.items():
        print(f"  {name:12s} max |Δ| = {delta:.8e}")

    if deltas["LoRA"] == 0.0:
        raise RuntimeError("LoRA weights did not update")
    if deltas["Interface"] == 0.0:
        raise RuntimeError("Condition/state adaptor did not update")
    if deltas["Final head"] == 0.0:
        raise RuntimeError("Final action head did not update")
    if deltas["Frozen RDT"] != 0.0:
        raise RuntimeError("Frozen base RDT weights changed")

    print("\n[6/6] Learning verdict...")
    final_loss = losses[-1]
    ratio = final_loss / max(float(initial_loss), 1e-12)
    print(f"  initial loss: {initial_loss:.6f}")
    print(f"  final loss:   {final_loss:.6f}")
    print(f"  final/initial:{ratio:.4f}")
    if final_loss >= float(initial_loss):
        raise RuntimeError(
            "The fixed-batch loss did not decrease. Check learning rates, gradients, dtype, and pretrained loading."
        )
    print("\nPASS: model construction, forward pass, backward pass, parameter freezing, and learning all worked.")


if __name__ == "__main__":
    main()
