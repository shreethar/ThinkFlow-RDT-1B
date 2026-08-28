#!/usr/bin/env python
"""Generate simulator-labelled LIBERO recovery examples for B0 continuation.

This is deliberately an offline recovery-data pass, not a claim of full DAgger.
Each sample starts from a recorded demonstration state shortly before the first
gripper transition, translates the end effector away from the demonstrated
path, and asks a feedback oracle to track the remaining recorded EEF positions.
The oracle's commands (not the original open-loop commands) become the target.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/thinkflow-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/thinkflow-matplotlib")

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from inspect_libero_outputs import install_robosuite_mujoco_compatibility  # noqa: E402
from precompute_all_features import (  # noqa: E402
    QWEN_TRAJECTORY_PROMPT_TEMPLATE,
    extract_qwen_kv,
    extract_t5_features,
    load_models,
    save_sample_shard,
    standardized_collate_fn,
    unique_instruction_indices,
)
from replay_libero_demo_actions import find_demo_file  # noqa: E402
from thinkflow_rdt.adapters.libero import (  # noqa: E402
    LIBERO_ACTION_DIM,
    LIBERO_STATE_DIM,
    libero_action_to_rdt,
    libero_observation_to_rdt,
)
from thinkflow_rdt.config import load_config  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402


DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@dataclass(frozen=True)
class Candidate:
    suite: str
    task_id: int
    task_name: str
    instruction: str
    demo_path: Path
    demo_name: str
    demo_index: int
    anchor: int
    anchor_kind: str
    variant: int
    split: str


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def numeric_demo_index(name: str, fallback: int) -> int:
    suffix = name.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else fallback


def first_gripper_transition(actions: np.ndarray, *, threshold: float = 0.0) -> int | None:
    """Return the first persistent sign transition in raw gripper commands."""
    values = np.asarray(actions, dtype=np.float32)[:, 6]
    labels = values >= threshold
    for index in range(1, len(labels)):
        if labels[index] == labels[index - 1]:
            continue
        stop = min(len(labels), index + 3)
        if bool(np.all(labels[index:stop] == labels[index])):
            return index
    return None


def perturb_offset(rng: np.random.Generator, *, variant: int, minimum: float, maximum: float) -> np.ndarray:
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    radius = float(rng.uniform(minimum, maximum))
    z = float(rng.uniform(0.003, min(0.015, 0.5 * maximum))) if variant % 2 else 0.0
    return np.asarray([radius * np.cos(angle), radius * np.sin(angle), z], dtype=np.float32)


def position_feedback_action(
    target_xyz: np.ndarray,
    live_xyz: np.ndarray,
    reference_action: np.ndarray,
    *,
    gain: float,
    command_limit: float,
) -> np.ndarray:
    """Track a world-space EEF target while preserving demonstrated rotation/gripper."""
    action = np.asarray(reference_action, dtype=np.float32)[:7].copy()
    correction = float(gain) * (
        np.asarray(target_xyz, dtype=np.float32) - np.asarray(live_xyz, dtype=np.float32)
    )
    action[:3] = np.clip(correction, -float(command_limit), float(command_limit))
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def demo_names(handle: h5py.File) -> list[str]:
    root = handle["data"] if "data" in handle else handle
    names = [name for name in root if hasattr(root[name], "keys")]
    return sorted(names, key=lambda name: numeric_demo_index(name, 10**9))


def demo_arrays(group: h5py.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "states" not in group or "actions" not in group or "obs" not in group:
        raise KeyError(f"{group.name} must contain states, actions, and obs")
    states = np.asarray(group["states"], dtype=np.float64)
    actions = np.asarray(group["actions"], dtype=np.float32)[:, :7]
    obs = group["obs"]
    for key in ("ee_pos", "robot0_eef_pos"):
        if key in obs:
            eef_xyz = np.asarray(obs[key], dtype=np.float32)[:, :3]
            break
    else:
        raise KeyError(f"{group.name}/obs has no EEF position array")
    usable = min(len(states), len(actions), len(eef_xyz))
    return states[:usable], actions[:usable], eef_xyz[:usable]


def build_candidates(args: argparse.Namespace, benchmarks: dict[str, Any]) -> list[Candidate]:
    candidates: list[Candidate] = []
    before_offsets = sorted(set(int(value) for value in args.anchor_before), reverse=True)
    for suite in args.suites:
        benchmark = benchmarks[suite]
        dataset_dir = args.dataset_root / suite
        for task_id in range(args.tasks_per_suite):
            task = benchmark.get_task(task_id)
            demo_path = find_demo_file(dataset_dir, task_name=task.name)
            with h5py.File(demo_path, "r") as handle:
                root = handle["data"] if "data" in handle else handle
                for fallback_index, name in enumerate(demo_names(handle)):
                    demo_index = numeric_demo_index(name, fallback_index)
                    actions = np.asarray(root[name]["actions"], dtype=np.float32)[:, :7]
                    transition = first_gripper_transition(actions)
                    if transition is None:
                        if args.require_gripper_transition:
                            continue
                        transition = max(args.minimum_anchor + max(before_offsets), len(actions) // 2)
                    split = (
                        "validation"
                        if demo_index % args.validation_mod == args.validation_remainder
                        else "train"
                    )
                    for variant, before in enumerate(before_offsets):
                        anchor = int(transition - before)
                        if anchor < args.minimum_anchor or anchor + args.horizon > len(actions):
                            continue
                        candidates.append(
                            Candidate(
                                suite=suite,
                                task_id=task_id,
                                task_name=task.name,
                                instruction=task.language,
                                demo_path=demo_path,
                                demo_name=name,
                                demo_index=demo_index,
                                anchor=anchor,
                                anchor_kind=f"first_gripper_transition_minus_{before}",
                                variant=variant,
                                split=split,
                            )
                        )
    return candidates


def balanced_candidates(
    candidates: Iterable[Candidate], *, split: str, limit: int, seed: int
) -> list[Candidate]:
    groups: dict[tuple[str, int], list[Candidate]] = {}
    for candidate in candidates:
        if candidate.split == split:
            groups.setdefault((candidate.suite, candidate.task_id), []).append(candidate)
    rng = np.random.default_rng(seed + (0 if split == "train" else 1))
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    rng.shuffle(keys)
    output: list[Candidate] = []
    while keys and len(output) < limit:
        next_keys: list[tuple[str, int]] = []
        for key in keys:
            values = groups[key]
            if values and len(output) < limit:
                output.append(values.pop())
            if values:
                next_keys.append(key)
        keys = next_keys
    return output


def apply_translation_perturbation(
    env: Any,
    observation: dict[str, Any],
    *,
    target_xyz: np.ndarray,
    gripper_command: float,
    gain: float,
    command_limit: float,
    steps: int,
    tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any] | None, float]:
    previous = None
    for _ in range(steps):
        live_xyz = libero_observation_to_rdt(observation)["state"][:3]
        error = float(np.linalg.norm(target_xyz - live_xyz))
        if error <= tolerance:
            break
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(gain * (target_xyz - live_xyz), -command_limit, command_limit)
        action[6] = float(gripper_command)
        previous = observation
        observation, _, _, _ = env.step(action)
    final_xyz = libero_observation_to_rdt(observation)["state"][:3]
    return observation, previous, float(np.linalg.norm(target_xyz - final_xyz))


def collect_recovery_sample(
    env: Any,
    candidate: Candidate,
    *,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    with h5py.File(candidate.demo_path, "r") as handle:
        root = handle["data"] if "data" in handle else handle
        states, demo_actions, eef_xyz = demo_arrays(root[candidate.demo_name])

    observation = env.reset()
    observation = env.set_init_state(states[candidate.anchor])
    anchor_live_xyz = libero_observation_to_rdt(observation)["state"][:3]
    offset = perturb_offset(
        rng,
        variant=candidate.variant,
        minimum=args.perturb_min,
        maximum=args.perturb_max,
    )
    desired_xyz = anchor_live_xyz + offset
    observation, previous_observation, perturb_residual = apply_translation_perturbation(
        env,
        observation,
        target_xyz=desired_xyz,
        gripper_command=float(demo_actions[candidate.anchor, 6]),
        gain=args.position_gain,
        command_limit=args.command_limit,
        steps=args.perturb_steps,
        tolerance=args.perturb_tolerance,
    )
    perturbed_xyz = libero_observation_to_rdt(observation)["state"][:3]
    achieved_perturbation = float(np.linalg.norm(perturbed_xyz - eef_xyz[candidate.anchor]))
    initial_tracking_error = achieved_perturbation

    raw_targets: list[np.ndarray] = []
    tracking_errors: list[float] = []
    success = False
    for horizon_index in range(args.horizon):
        reference_index = min(candidate.anchor + horizon_index, len(demo_actions) - 1)
        target_index = min(candidate.anchor + horizon_index + 1, len(eef_xyz) - 1)
        live_xyz = libero_observation_to_rdt(observation)["state"][:3]
        action = position_feedback_action(
            eef_xyz[target_index],
            live_xyz,
            demo_actions[reference_index],
            gain=args.position_gain,
            command_limit=args.command_limit,
        )
        raw_targets.append(action)
        observation, _, done, _ = env.step(action)
        live_after = libero_observation_to_rdt(observation)["state"][:3]
        tracking_errors.append(float(np.linalg.norm(live_after - eef_xyz[target_index])))
        success = bool(done) or bool(env.check_success())

    first_ten_stop = min(10, len(tracking_errors))
    first_ten_tail = tracking_errors[max(0, first_ten_stop - 3):first_ten_stop]
    recovery_error_h10 = float(np.mean(first_ten_tail))
    accepted = (
        achieved_perturbation >= args.minimum_achieved_perturbation
        and recovery_error_h10 <= args.max_recovery_error_h10
        and recovery_error_h10 <= initial_tracking_error * args.max_error_ratio
    )
    diagnostic = {
        "suite": candidate.suite,
        "task_id": candidate.task_id,
        "task_name": candidate.task_name,
        "demo_name": candidate.demo_name,
        "demo_index": candidate.demo_index,
        "anchor": candidate.anchor,
        "anchor_kind": candidate.anchor_kind,
        "variant": candidate.variant,
        "split": candidate.split,
        "requested_offset_xyz": offset.astype(float).tolist(),
        "achieved_perturbation_m": achieved_perturbation,
        "perturb_target_residual_m": perturb_residual,
        "initial_tracking_error_m": initial_tracking_error,
        "recovery_error_h10_m": recovery_error_h10,
        "final_tracking_error_m": tracking_errors[-1],
        "simulator_success": success,
        "accepted": accepted,
    }
    if not accepted:
        return None, diagnostic

    # The cached observation must be the perturbed state before the recovery
    # horizon, not the state after executing the oracle targets.
    observation = env.reset()
    observation = env.set_init_state(states[candidate.anchor])
    observation, previous_observation, _ = apply_translation_perturbation(
        env,
        observation,
        target_xyz=desired_xyz,
        gripper_command=float(demo_actions[candidate.anchor, 6]),
        gain=args.position_gain,
        command_limit=args.command_limit,
        steps=args.perturb_steps,
        tolerance=args.perturb_tolerance,
    )
    converted = libero_observation_to_rdt(observation)
    previous_converted = (
        None if previous_observation is None else libero_observation_to_rdt(previous_observation)
    )
    from PIL import Image

    current_images = {
        "primary": Image.fromarray(converted["primary"]).convert("RGB"),
        "wrist": None if converted["wrist"] is None else Image.fromarray(converted["wrist"]).convert("RGB"),
        "secondary": None,
    }
    if previous_converted is None:
        previous_images = current_images
        previous_mask = {"primary": 0, "wrist": 0, "secondary": 0}
    else:
        previous_images = {
            "primary": Image.fromarray(previous_converted["primary"]).convert("RGB"),
            "wrist": None if previous_converted["wrist"] is None else Image.fromarray(previous_converted["wrist"]).convert("RGB"),
            "secondary": None,
        }
        previous_mask = {
            "primary": 1,
            "wrist": int(previous_images["wrist"] is not None),
            "secondary": 0,
        }
    current_mask = {
        "primary": 1,
        "wrist": int(current_images["wrist"] is not None),
        "secondary": 0,
    }
    episode_id = (
        f"recovery:{candidate.suite}:task{candidate.task_id}:"
        f"{candidate.demo_name}:a{candidate.anchor}:v{candidate.variant}"
    )
    sample = {
        # A distinct label keeps recovery validation visible as separate W&B
        # strata while retaining the same 11D/10D LIBERO tensor contract.
        "dataset_id": f"{candidate.suite}_recovery",
        "episode_id": episode_id,
        "step_idx": str(candidate.anchor),
        "instruction": candidate.instruction,
        "images": current_images,
        "image_mask": current_mask,
        "image_history": [previous_images, current_images],
        "image_history_mask": [previous_mask, current_mask],
        "state": converted["state"].astype(np.float32),
        "state_mask": np.ones(LIBERO_STATE_DIM, dtype=np.float32),
        "actions": libero_action_to_rdt(np.asarray(raw_targets, dtype=np.float32)),
        "actions_mask": np.ones(args.horizon, dtype=np.float32),
        "action_dim_mask": np.ones(LIBERO_ACTION_DIM, dtype=np.float32),
        "ctrl_freq": 20.0,
        "recovery_diagnostic": diagnostic,
    }
    return sample, diagnostic


def repeat_manifest(unique_path: Path, output_path: Path, repeat: int) -> None:
    lines = [line for line in unique_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for _ in range(repeat):
            for line in lines:
                handle.write(line + "\n")
    os.replace(temporary, output_path)


def load_qwen_only(args: argparse.Namespace) -> dict[str, Any]:
    """Load Qwen without forcing a redundant 44.5 GB T5-XXL download."""
    processor_id = args.qwen_processor_id or args.qwen_model_id
    processor = AutoProcessor.from_pretrained(processor_id)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.qwen_model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    model.eval().requires_grad_(False)
    return {"qwen_processor": processor, "qwen_vlm": model}


def cached_t5_instruction_features(
    cache_root: Path,
    suites: Iterable[str],
    required_instructions: set[str],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Reuse deterministic frozen-T5 outputs from the existing LIBERO cache."""
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for suite in suites:
        for split in ("train", "validation"):
            manifest_path = cache_root / suite / split / "manifest.jsonl"
            if not manifest_path.is_file():
                continue
            seen_tasks: set[str] = set()
            with manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    episode_id = str(item.get("first_episode_id", ""))
                    task_key = episode_id.split(":demo", 1)[0]
                    if task_key in seen_tasks:
                        continue
                    seen_tasks.add(task_key)
                    path = Path(item["path"])
                    path = path if path.is_absolute() else manifest_path.parent / path
                    record = torch.load(path, map_location="cpu", weights_only=True)
                    instructions = [str(value) for value in record["instructions"]]
                    sample_lang_index = torch.as_tensor(record["sample_lang_index"], dtype=torch.long)
                    for sample_index, instruction in enumerate(instructions):
                        if instruction not in required_instructions or instruction in result:
                            continue
                        pool_index = int(sample_lang_index[sample_index].item())
                        tokens = torch.as_tensor(record["lang_tokens"][pool_index]).cpu()
                        mask = torch.as_tensor(record["lang_mask"][pool_index], dtype=torch.bool).cpu()
                        result[instruction] = (tokens, mask)
                    if required_instructions.issubset(result):
                        return result
    missing = sorted(required_instructions.difference(result))
    if missing:
        raise KeyError(
            f"Existing cache {cache_root} has no T5 features for {len(missing)} instructions; "
            f"examples: {missing[:5]}"
        )
    return result


def pad_cached_t5_features(
    instructions: list[str],
    feature_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    max_lang_tokens: int,
    expected_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.zeros(
        len(instructions), max_lang_tokens, expected_dim, dtype=torch.bfloat16
    )
    output_mask = torch.zeros(len(instructions), max_lang_tokens, dtype=torch.bool)
    for index, instruction in enumerate(instructions):
        tokens, mask = feature_cache[instruction]
        tokens = torch.as_tensor(tokens)
        mask = torch.as_tensor(mask, dtype=torch.bool)
        if tokens.ndim != 2 or tokens.shape[-1] != expected_dim:
            raise ValueError(
                f"Cached T5 feature for {instruction!r} has shape {tuple(tokens.shape)}"
            )
        valid_tokens = tokens[mask][:max_lang_tokens]
        output[index, : len(valid_tokens)] = valid_tokens.to(torch.bfloat16)
        output_mask[index, : len(valid_tokens)] = True
    return output, output_mask


def save_feature_batch(
    samples: list[dict[str, Any]],
    *,
    split_dir: Path,
    manifest_handle: Any,
    shard_index: int,
    sample_start_index: int,
    models: dict[str, Any],
    t5_feature_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    cfg: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> int:
    batch = standardized_collate_fn(
        samples,
        max_images_per_sample=6,
        image_history_size=2,
        image_jpeg_quality=args.image_jpeg_quality,
        skip_no_image=True,
        encode_image_slots=False,
    )
    if batch is None or len(batch["metadata"]) != len(samples):
        raise RuntimeError("Recovery batch unexpectedly lost samples during collation")
    for metadata, sample in zip(batch["metadata"], samples):
        metadata["recovery"] = sample["recovery_diagnostic"]

    qwen_kv = extract_qwen_kv(
        batch,
        models["qwen_processor"],
        models["qwen_vlm"],
        device=device,
        layer_index=args.qwen_layer_index,
        max_new_tokens=args.qwen_max_new_tokens,
        expected_dim=cfg.model.qwen_kv_dim,
        stop_at_think_end=True,
        prompt_template=QWEN_TRAJECTORY_PROMPT_TEMPLATE,
        enable_thinking=False,
    )
    unique_instructions, sample_lang_index = unique_instruction_indices(batch["instructions"])
    if t5_feature_cache is not None:
        lang_tokens, lang_mask = pad_cached_t5_features(
            unique_instructions,
            t5_feature_cache,
            max_lang_tokens=cfg.model.max_lang_tokens,
            expected_dim=cfg.model.lang_token_dim,
        )
    else:
        lang_tokens, lang_mask = extract_t5_features(
            {"instructions": unique_instructions},
            models["t5_tokenizer"],
            models["t5_encoder"],
            max_lang_tokens=cfg.model.max_lang_tokens,
            expected_dim=cfg.model.lang_token_dim,
            device=device,
        )
    new_count, _ = save_sample_shard(
        split_dir=split_dir,
        manifest_handle=manifest_handle,
        shard_index=shard_index,
        sample_start_index=sample_start_index,
        batch=batch,
        qwen_kv=qwen_kv,
        lang_tokens=lang_tokens,
        lang_mask=lang_mask,
        sample_lang_index=sample_lang_index,
        image_history_size=2,
        image_jpeg_quality=args.image_jpeg_quality,
        save_padded_features=False,
        image_codec=args.image_codec,
    )
    return new_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_b0_native128_recovery_continue.yaml")
    parser.add_argument("--libero-root", type=Path, default=Path("/home/ubuntu/LIBERO"))
    parser.add_argument("--dataset-root", type=Path, default=Path("libero-dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("cache_features_libero_b0_recovery"))
    parser.add_argument("--suites", nargs="+", choices=DEFAULT_SUITES, default=list(DEFAULT_SUITES))
    parser.add_argument("--tasks-per-suite", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--validation-samples", type=int, default=128)
    parser.add_argument("--train-repeat", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--anchor-before", type=int, nargs="+", default=[12, 6])
    parser.add_argument("--minimum-anchor", type=int, default=2)
    parser.add_argument("--validation-mod", type=int, default=10)
    parser.add_argument("--validation-remainder", type=int, default=0)
    parser.add_argument("--require-gripper-transition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--perturb-min", type=float, default=0.02)
    parser.add_argument("--perturb-max", type=float, default=0.04)
    parser.add_argument("--minimum-achieved-perturbation", type=float, default=0.012)
    parser.add_argument("--perturb-steps", type=int, default=8)
    parser.add_argument("--perturb-tolerance", type=float, default=0.004)
    parser.add_argument("--position-gain", type=float, default=20.0)
    parser.add_argument("--command-limit", type=float, default=0.8)
    parser.add_argument("--max-recovery-error-h10", type=float, default=0.035)
    parser.add_argument("--max-error-ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--image-codec", choices=["png", "jpeg"], default="png")
    parser.add_argument("--image-jpeg-quality", type=int, default=90)
    parser.add_argument("--simulator-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--qwen-model-id", default="model/model/stage1_unsloth")
    parser.add_argument("--qwen-processor-id", default=None)
    parser.add_argument("--qwen-layer-index", type=int, default=7)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--t5-model-id", default="/home/ubuntu/RoboticsDiffusionTransformer/google/t5-v1_1-xxl")
    parser.add_argument("--t5-fallback-model-id", default="google/t5-v1_1-xxl")
    parser.add_argument("--t5-precision", choices=["bf16", "8bit"], default="bf16")
    parser.add_argument(
        "--reuse-t5-cache-root",
        type=Path,
        default=Path("cache_features_libero_b0_raw_ortho6d"),
        help=(
            "Reuse frozen T5 instruction embeddings from this existing LIBERO cache. "
            "This avoids reloading T5-XXL."
        ),
    )
    parser.add_argument(
        "--load-t5-online",
        action="store_true",
        help="Load T5-XXL instead of reusing --reuse-t5-cache-root.",
    )
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.horizon != 64:
        raise ValueError("The current LIBERO B0 model/cache contract requires --horizon 64")
    if args.batch_size <= 0 or args.train_repeat <= 0:
        raise ValueError("--batch-size and --train-repeat must be positive")
    if not (0.0 < args.perturb_min <= args.perturb_max):
        raise ValueError("Require 0 < --perturb-min <= --perturb-max")
    if not 0 <= args.validation_remainder < args.validation_mod:
        raise ValueError("validation remainder must be inside [0, validation_mod)")


def main() -> None:
    args = parse_args()
    validate_args(args)
    cfg = load_config(args.config)
    if str(args.libero_root) not in sys.path:
        sys.path.insert(0, str(args.libero_root))
    install_robosuite_mujoco_compatibility()
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} is not empty; pass --overwrite to replace it"
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = {suite: get_benchmark(suite)(0) for suite in args.suites}
    all_candidates = build_candidates(args, benchmarks)
    requested_by_split = {
        "train": args.train_samples,
        "validation": args.validation_samples,
    }
    selected = {
        split: balanced_candidates(
            all_candidates,
            split=split,
            # Keep replacement candidates available when a perturbation fails
            # the oracle-quality gate.
            limit=max(requested * 2, requested + 128),
            seed=args.seed,
        )
        for split, requested in requested_by_split.items()
    }
    for split, requested in (("train", args.train_samples), ("validation", args.validation_samples)):
        if len(selected[split]) < requested:
            raise RuntimeError(
                f"Only {len(selected[split])} {split} candidates are available; requested {requested}. "
                "Reduce the sample count, add anchor offsets, or disable --require-gripper-transition."
            )

    models = None
    t5_feature_cache = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.simulator_only:
        # load_models inspects this field to decide whether SigLIP is cached.
        args.feature_set = "qwen_t5"
        required_instructions = {candidate.instruction for values in selected.values() for candidate in values}
        if not args.load_t5_online:
            t5_feature_cache = cached_t5_instruction_features(
                args.reuse_t5_cache_root,
                args.suites,
                required_instructions,
            )
            print(
                f"Reusing frozen T5 features for {len(t5_feature_cache)} instructions "
                f"from {args.reuse_t5_cache_root}",
                flush=True,
            )
            models = load_qwen_only(args)
        else:
            models = load_models(args, cfg, device)

    rng = np.random.default_rng(args.seed)
    diagnostics: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "generator": "offline_translation_perturbation_feedback_oracle_v1",
        "strict_dagger": False,
        "config": vars(args),
        "candidate_count": len(all_candidates),
        "started_at_unix": time.time(),
        "splits": {},
    }
    envs: dict[tuple[str, int], Any] = {}
    try:
        for split in ("train", "validation"):
            split_dir = args.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            unique_manifest = split_dir / "manifest_unique.jsonl"
            temporary_manifest = unique_manifest.with_suffix(".jsonl.tmp")
            accepted_samples: list[dict[str, Any]] = []
            accepted_count = 0
            attempted_count = 0
            sample_count = 0
            shard_count = 0
            rejected = 0
            with temporary_manifest.open("w", encoding="utf-8") as manifest_handle:
                for candidate in selected[split]:
                    if accepted_count >= requested_by_split[split]:
                        break
                    attempted_count += 1
                    key = (candidate.suite, candidate.task_id)
                    if key not in envs:
                        benchmark = benchmarks[candidate.suite]
                        envs[key] = OffScreenRenderEnv(
                            bddl_file_name=benchmark.get_task_bddl_file_path(candidate.task_id),
                            camera_heights=128,
                            camera_widths=128,
                            horizon=args.horizon + args.perturb_steps + 20,
                        )
                    sample, diagnostic = collect_recovery_sample(
                        envs[key], candidate, rng=rng, args=args
                    )
                    diagnostics.append(diagnostic)
                    if sample is None:
                        rejected += 1
                        continue
                    accepted_samples.append(sample)
                    accepted_count += 1
                    print(
                        f"[{split}] {accepted_count}/{requested_by_split[split]} accepted "
                        f"dataset={candidate.suite} task={candidate.task_id} demo={candidate.demo_name} "
                        f"anchor={candidate.anchor} perturb={diagnostic['achieved_perturbation_m']:.3f}m "
                        f"h10={diagnostic['recovery_error_h10_m']:.3f}m",
                        flush=True,
                    )
                    if args.simulator_only:
                        continue
                    if len(accepted_samples) >= args.batch_size:
                        assert models is not None
                        sample_count = save_feature_batch(
                            accepted_samples,
                            split_dir=split_dir,
                            manifest_handle=manifest_handle,
                            shard_index=shard_count,
                            sample_start_index=sample_count,
                            models=models,
                            t5_feature_cache=t5_feature_cache,
                            cfg=cfg,
                            device=device,
                            args=args,
                        )
                        shard_count += 1
                        accepted_samples.clear()
                if not args.simulator_only and accepted_samples:
                    assert models is not None
                    sample_count = save_feature_batch(
                        accepted_samples,
                        split_dir=split_dir,
                        manifest_handle=manifest_handle,
                        shard_index=shard_count,
                        sample_start_index=sample_count,
                        models=models,
                        t5_feature_cache=t5_feature_cache,
                        cfg=cfg,
                        device=device,
                        args=args,
                    )
                    shard_count += 1
                    accepted_samples.clear()
            if args.simulator_only:
                temporary_manifest.unlink(missing_ok=True)
            else:
                os.replace(temporary_manifest, unique_manifest)
                repeat = args.train_repeat if split == "train" else 1
                repeat_manifest(unique_manifest, split_dir / "manifest.jsonl", repeat)
            split_diagnostics = [item for item in diagnostics if item["split"] == split]
            if accepted_count < requested_by_split[split]:
                raise RuntimeError(
                    f"Only {accepted_count}/{requested_by_split[split]} requested {split} "
                    "samples passed the recovery quality gate. Inspect "
                    "recovery_diagnostics.jsonl or relax the quality thresholds."
                )
            summary["splits"][split] = {
                "candidate_pool": len(selected[split]),
                "attempted": attempted_count,
                "accepted": sum(bool(item["accepted"]) for item in split_diagnostics),
                "rejected": rejected,
                "saved_samples": sample_count,
                "shards": shard_count,
                "manifest_repeat": args.train_repeat if split == "train" else 1,
            }
    finally:
        for env in envs.values():
            env.close()

    summary["finished_at_unix"] = time.time()
    summary["elapsed_sec"] = summary["finished_at_unix"] - summary["started_at_unix"]
    (args.output_dir / "recovery_diagnostics.jsonl").write_text(
        "".join(json.dumps(item, default=json_default) + "\n" for item in diagnostics),
        encoding="utf-8",
    )
    (args.output_dir / "precompute_metadata.json").write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
