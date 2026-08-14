"""Validate whether a trained SmolVLA policy uses its cached Qwen K/V input.

The paired ablations reuse exactly the same cached observations, actions,
diffusion noise, and diffusion times. Only Qwen K/V changes, so the reported
loss deltas and prediction deltas isolate the effect of the fusion path. The
same script supports B0 (one token) and B2 (five tokens) by reading the token
count from the checkpoint configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate

REPO_ROOT = Path(__file__).resolve().parents[2]
for extra_path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed

from .cached_libero import list_shards, sample_from_pack
from .configuration import KVSmolVLAConfig
from .modeling import KVSmolVLAPolicy


LIBERO_SUITES = ("libero_10", "libero_spatial", "libero_goal", "libero_object")
VARIANTS = ("matched", "zero", "shuffle", "cross_task")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=None,
        help="Optional pre-training bootstrap checkpoint used to measure adapter drift.",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--suites", nargs="+", choices=LIBERO_SUITES, default=list(LIBERO_SUITES))
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--prediction-samples",
        type=int,
        default=32,
        help="Number of examples used for full iterative action-sampling sensitivity.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-gradient-diagnostic", action="store_true")
    return parser.parse_args()


def resolve_policy_checkpoint(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        if resolved.name != "train_config.json":
            raise ValueError(f"Expected train_config.json, got {resolved.name!r}")
        resolved = resolved.parent
    if (resolved / "pretrained_model").is_dir():
        resolved = resolved / "pretrained_model"
    required = ("config.json", "model.safetensors", "policy_preprocessor.json")
    missing = [name for name in required if not (resolved / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint {resolved} is missing {missing}")
    return resolved


def _num_pack_samples(pack: dict[str, Any]) -> int:
    qwen = torch.as_tensor(pack["qwen_kv"])
    if qwen.ndim not in (2, 3):
        raise ValueError(f"Unexpected qwen_kv shape {tuple(qwen.shape)}")
    return int(qwen.shape[0])


def load_balanced_examples(
    cache_root: Path,
    suites: list[str],
    split: str,
    num_samples: int,
    chunk_size: int,
    token_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Load a deterministic, approximately suite-balanced validation subset."""

    rng = random.Random(seed)
    target_per_suite = math.ceil(num_samples / len(suites))
    by_suite: dict[str, list[dict[str, Any]]] = {}
    for suite_offset, suite in enumerate(suites):
        suite_rng = random.Random(seed + 1009 * (suite_offset + 1))
        paths = list_shards(cache_root, suite, split=split)
        suite_rng.shuffle(paths)
        examples: list[dict[str, Any]] = []
        for path in paths:
            pack = torch.load(path, map_location="cpu", weights_only=False)
            indices = list(range(_num_pack_samples(pack)))
            suite_rng.shuffle(indices)
            for sample_index in indices:
                try:
                    sample = sample_from_pack(
                        pack,
                        sample_index,
                        chunk_size=chunk_size,
                        expected_qwen_tokens=token_count,
                    )
                except ValueError as exc:
                    if "camera slot" in str(exc):
                        continue
                    raise
                sample["_suite"] = suite
                sample["_source"] = f"{path.name}:{sample_index}"
                examples.append(sample)
                if len(examples) >= target_per_suite:
                    break
            if len(examples) >= target_per_suite:
                break
        if not examples:
            raise RuntimeError(f"No usable samples found for {suite}/{split}")
        by_suite[suite] = examples
        print(f"Loaded {len(examples)} examples from {suite}/{split}")

    # Round-robin keeps every prefix balanced, including when num_samples is small.
    result: list[dict[str, Any]] = []
    for index in range(target_per_suite):
        order = suites.copy()
        rng.shuffle(order)
        for suite in order:
            if index < len(by_suite[suite]):
                result.append(by_suite[suite][index])
                if len(result) == num_samples:
                    return result
    return result


def make_derangement(size: int, seed: int) -> list[int]:
    if size < 2:
        raise ValueError("At least two samples are required for shuffled-Qwen validation")
    rng = random.Random(seed)
    indices = list(range(size))
    for _ in range(1000):
        rng.shuffle(indices)
        if all(index != donor for index, donor in enumerate(indices)):
            return indices.copy()
    return list(range(1, size)) + [0]


def make_cross_task_donors(examples: list[dict[str, Any]], seed: int) -> list[int]:
    """Prefer another suite; otherwise use another instruction in the same suite."""

    rng = random.Random(seed)
    donors: list[int] = []
    for index, example in enumerate(examples):
        candidates = [
            other
            for other, donor in enumerate(examples)
            if donor["_suite"] != example["_suite"]
        ]
        if not candidates:
            candidates = [
                other
                for other, donor in enumerate(examples)
                if donor["task"] != example["task"]
            ]
        if not candidates:
            candidates = [other for other in range(len(examples)) if other != index]
        if not candidates:
            raise ValueError("Cross-task validation requires at least two distinct samples")
        donors.append(rng.choice(candidates))
    return donors


def paired_summary(baseline: np.ndarray, values: np.ndarray) -> dict[str, float]:
    delta = values - baseline
    stderr = float(delta.std(ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else 0.0
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "delta_from_matched": float(delta.mean()),
        "delta_percent": float(100.0 * delta.mean() / max(float(baseline.mean()), 1e-12)),
        "paired_delta_stderr": stderr,
        "paired_delta_ci95_low": float(delta.mean() - 1.96 * stderr),
        "paired_delta_ci95_high": float(delta.mean() + 1.96 * stderr),
        "matched_lower_fraction": float(np.mean(baseline < values)),
    }


def adapter_parameter_diagnostics(policy: KVSmolVLAPolicy) -> dict[str, Any]:
    selected = {
        name: parameter.detach().float().cpu()
        for name, parameter in policy.named_parameters()
        if "external_key_projections" in name
        or "external_value_projections" in name
        or "external_logit_biases" in name
    }
    biases = {
        name: float(value.item())
        for name, value in selected.items()
        if "external_logit_biases" in name
    }
    projection_norms = {
        name: float(value.norm().item())
        for name, value in selected.items()
        if "external_logit_biases" not in name
    }
    return {
        "parameter_count": int(sum(value.numel() for value in selected.values())),
        "logit_biases": biases,
        "projection_weight_norms": projection_norms,
    }


def reference_drift(
    policy: KVSmolVLAPolicy, reference_checkpoint: Path | None
) -> dict[str, Any] | None:
    if reference_checkpoint is None:
        return None
    from safetensors.torch import load_file

    reference = resolve_policy_checkpoint(reference_checkpoint)
    reference_state = load_file(str(reference / "model.safetensors"), device="cpu")
    per_parameter: dict[str, dict[str, float]] = {}
    missing: list[str] = []
    for name, parameter in policy.named_parameters():
        if not any(
            marker in name
            for marker in (
                "external_key_projections",
                "external_value_projections",
                "external_logit_biases",
            )
        ):
            continue
        if name not in reference_state:
            missing.append(name)
            continue
        current = parameter.detach().float().cpu()
        initial = reference_state[name].float()
        difference = current - initial
        per_parameter[name] = {
            "initial_norm": float(initial.norm().item()),
            "current_norm": float(current.norm().item()),
            "delta_norm": float(difference.norm().item()),
            "relative_delta": float(
                difference.norm().item() / max(initial.norm().item(), 1e-12)
            ),
        }
    return {
        "reference_checkpoint": str(reference),
        "matched_parameters": len(per_parameter),
        "missing_parameters": missing,
        "parameters": per_parameter,
    }


def qwen_for_variant(
    variant: str,
    batch_indices: list[int],
    examples: list[dict[str, Any]],
    shuffle_donors: list[int],
    cross_donors: list[int],
) -> torch.Tensor:
    matched = torch.stack([examples[index]["qwen_kv"] for index in batch_indices])
    if variant == "matched":
        return matched
    if variant == "zero":
        return torch.zeros_like(matched)
    donors = shuffle_donors if variant == "shuffle" else cross_donors
    return torch.stack([examples[donors[index]]["qwen_kv"] for index in batch_indices])


def prepare_batch(
    examples: list[dict[str, Any]],
    batch_indices: list[int],
    preprocessor: Any,
    device: torch.device,
) -> dict[str, Any]:
    raw_examples = [
        {key: value for key, value in examples[index].items() if not key.startswith("_") and key != "qwen_kv"}
        for index in batch_indices
    ]
    processed = preprocessor(default_collate(raw_examples))
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in processed.items()
    }


def gradient_diagnostic(
    policy: KVSmolVLAPolicy,
    batch: dict[str, Any],
    qwen: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
) -> dict[str, Any]:
    parameters = [
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
        and any(
            marker in name
            for marker in (
                "external_key_projections",
                "external_value_projections",
                "external_logit_biases",
            )
        )
    ]
    if not parameters:
        return {"error": "No trainable external Qwen adapter parameters found"}
    grad_batch = dict(batch)
    grad_batch[policy.config.external_kv_key] = qwen
    policy.zero_grad(set_to_none=True)
    loss, _ = policy(grad_batch, noise=noise, time=time, reduction="mean")
    gradients = torch.autograd.grad(
        loss,
        [parameter for _, parameter in parameters],
        allow_unused=True,
    )
    per_parameter = {}
    squared_total = 0.0
    for (name, _), gradient in zip(parameters, gradients, strict=True):
        norm = 0.0 if gradient is None else float(gradient.detach().float().norm().item())
        per_parameter[name] = norm
        squared_total += norm * norm
    return {
        "loss": float(loss.detach().item()),
        "total_grad_norm": math.sqrt(squared_total),
        "nonzero_parameter_fraction": float(
            np.mean([value > 0.0 for value in per_parameter.values()])
        ),
        "per_parameter_grad_norm": per_parameter,
    }


def main() -> None:
    args = parse_args()
    if args.num_samples < 2 or args.batch_size < 1:
        raise ValueError("--num-samples must be >= 2 and --batch-size must be >= 1")
    checkpoint = resolve_policy_checkpoint(args.checkpoint)
    cache_root = args.cache_root.expanduser().resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    set_seed(args.seed)
    register_third_party_plugins()
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if not isinstance(config, KVSmolVLAConfig):
        raise TypeError(f"Expected smolvla_qwen_kv, got {type(config).__name__}")
    config.device = str(device)
    config.load_vlm_weights = False
    policy = KVSmolVLAPolicy.from_pretrained(
        checkpoint,
        config=config,
        local_files_only=True,
        strict=True,
    ).eval()
    preprocessor, _postprocessor = make_pre_post_processors(
        config,
        str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    examples = load_balanced_examples(
        cache_root,
        args.suites,
        args.split,
        args.num_samples,
        config.chunk_size,
        config.external_kv_token_count,
        args.seed,
    )
    shuffle_donors = make_derangement(len(examples), args.seed + 17)
    cross_donors = make_cross_task_donors(examples, args.seed + 31)
    losses: dict[str, list[float]] = {variant: [] for variant in VARIANTS}
    prediction_deltas: dict[str, list[float]] = {
        variant: [] for variant in VARIANTS if variant != "matched"
    }
    first_gradient_inputs = None
    sampled_predictions = 0
    generator = torch.Generator(device=device).manual_seed(args.seed + 101)
    def amp_context():
        return torch.autocast(device_type=device.type) if config.use_amp else nullcontext()

    for start in range(0, len(examples), args.batch_size):
        batch_indices = list(range(start, min(start + args.batch_size, len(examples))))
        batch = prepare_batch(examples, batch_indices, preprocessor, device)
        batch_size = len(batch_indices)
        noise = torch.randn(
            batch_size,
            config.chunk_size,
            config.max_action_dim,
            generator=generator,
            device=device,
        )
        # A fixed stratified set covers the full training diffusion-time interval.
        time = (
            torch.arange(batch_size, device=device, dtype=torch.float32) + 0.5
        ) / batch_size
        time = torch.remainder(time + (start / max(len(examples), 1)), 1.0)
        time = time.clamp_(0.001, 0.999)
        variant_qwen = {
            variant: qwen_for_variant(
                variant, batch_indices, examples, shuffle_donors, cross_donors
            ).to(device)
            for variant in VARIANTS
        }
        if first_gradient_inputs is None:
            first_gradient_inputs = (batch, variant_qwen["matched"], noise, time)

        with torch.no_grad(), amp_context():
            for variant in VARIANTS:
                variant_batch = dict(batch)
                variant_batch[config.external_kv_key] = variant_qwen[variant]
                per_sample, _ = policy(
                    variant_batch,
                    noise=noise,
                    time=time,
                    reduction="none",
                )
                values = per_sample.detach().float().reshape(batch_size, -1).mean(dim=1)
                losses[variant].extend(values.cpu().tolist())

            remaining = max(args.prediction_samples - sampled_predictions, 0)
            if remaining:
                take = min(remaining, batch_size)
                prediction_noise = noise[:take]
                outputs = {}
                for variant in VARIANTS:
                    prediction_batch = {
                        key: value[:take] if isinstance(value, torch.Tensor) else value[:take]
                        if isinstance(value, list)
                        else value
                        for key, value in batch.items()
                    }
                    prediction_batch[config.external_kv_key] = variant_qwen[variant][:take]
                    normalized = policy.predict_action_chunk(
                        prediction_batch, noise=prediction_noise
                    )
                    outputs[variant] = normalized.detach().float()
                for variant in prediction_deltas:
                    difference = outputs[variant] - outputs["matched"]
                    per_sample_mse = difference.square().flatten(1).mean(dim=1)
                    prediction_deltas[variant].extend(per_sample_mse.cpu().tolist())
                sampled_predictions += take
        print(f"Validated {min(start + batch_size, len(examples))}/{len(examples)} samples")

    matched = np.asarray(losses["matched"], dtype=np.float64)
    loss_report = {
        variant: paired_summary(matched, np.asarray(values, dtype=np.float64))
        for variant, values in losses.items()
    }
    prediction_report = {
        variant: {
            "mse_from_matched_mean": float(np.mean(values)),
            "mse_from_matched_std": float(np.std(values)),
            "examples": len(values),
        }
        for variant, values in prediction_deltas.items()
    }

    gradient_report = None
    if not args.skip_gradient_diagnostic and first_gradient_inputs is not None:
        batch, qwen, noise, time = first_gradient_inputs
        with amp_context():
            gradient_report = gradient_diagnostic(policy, batch, qwen, noise, time)

    report = {
        "checkpoint": str(checkpoint),
        "cache_root": str(cache_root),
        "split": args.split,
        "suites": args.suites,
        "samples": len(examples),
        "qwen_token_count": config.external_kv_token_count,
        "qwen_width": config.external_kv_width,
        "seed": args.seed,
        "paired_imitation_loss": loss_report,
        "sampled_action_sensitivity": prediction_report,
        "adapter_parameters": adapter_parameter_diagnostics(policy),
        "adapter_drift": reference_drift(policy, args.reference_checkpoint),
        "gradient_diagnostic": gradient_report,
        "interpretation": {
            "content_use": "Shuffle/cross-task loss deltas above zero indicate content-specific Qwen use.",
            "presence_only": "A zero-Qwen penalty without shuffle/cross-task penalties suggests reliance on a constant Qwen-side offset rather than sample-specific content.",
            "ignored": "Near-zero paired loss and sampled-action deltas indicate that the policy is effectively ignoring Qwen K/V.",
        },
    }
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"Qwen tokens: {config.external_kv_token_count}")
    for variant in VARIANTS:
        metrics = loss_report[variant]
        print(
            f"{variant:>10}: loss={metrics['mean']:.6f} "
            f"delta={metrics['delta_from_matched']:+.6f} "
            f"({metrics['delta_percent']:+.2f}%)"
        )
    for variant, metrics in prediction_report.items():
        print(
            f"action {variant:>10}: MSE from matched="
            f"{metrics['mse_from_matched_mean']:.6e}"
        )
    if gradient_report is not None:
        print(
            "Qwen adapter gradient norm: "
            f"{gradient_report.get('total_grad_norm', float('nan')):.6e}"
        )


if __name__ == "__main__":
    main()
