#!/usr/bin/env python
"""Interactive Gradio demo for B0/B2/B3 RDT checkpoints in LIBERO."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/thinkflow-rdt-demo-cache")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/thinkflow-rdt-demo-matplotlib")

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    SiglipImageProcessor,
    SiglipVisionModel,
)

REPO_ROOT = Path(__file__).resolve().parent
for extra_path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from precompute_all_features import (  # noqa: E402
    extract_qwen_kv,
    extract_siglip_features,
    extract_t5_features,
    standardized_collate_fn,
)
from precompute_latent_student_kv import (  # noqa: E402
    extract_latent_student_spatial_kv,
    load_student_and_processor,
)
from rollout_libero_rdt import (  # noqa: E402
    B2_TRAJECTORY_PROMPT,
    frame_for_video,
    install_robosuite_mujoco_compatibility,
    load_feature_metadata,
    load_t5_encoder,
    native_rdt_action_to_libero_7d,
    native_rdt_policy_inputs,
    resolve_model_id,
    rollout_sample,
    t5_device_from_encoder,
)
from run_precompute_32frame_episode_packs_latent_student_kv import (  # noqa: E402
    validate_student_runtime_contract,
)
from evaluate_hf_rdt_libero_goal import (  # noqa: E402
    build_policy as build_upstream_policy,
    load_checkpoint_config as load_upstream_checkpoint_config,
)
from thinkflow_rdt.checkpoint import load_trainable_artifact  # noqa: E402
from thinkflow_rdt.config import load_config  # noqa: E402
from thinkflow_rdt.model import SFTConditionedRDT  # noqa: E402


SUITE_LABELS = {
    "LIBERO Spatial": "libero_spatial",
    "LIBERO Long": "libero_10",
}
VARIANT_LABELS = {"B0": "b0", "B2": "b2", "B3": "b3"}
ARTIFACT_FILES = ("rdt_full.pt", "interfaces.pt", "metadata.json")


def overlay_visual_trajectory(
    frame: np.ndarray,
    waypoints: np.ndarray | None,
) -> np.ndarray:
    """Draw normalized LatentStudent ``(x, y)`` waypoints on an RGB frame.

    The waypoint head and ``libero_image_to_rgb`` both use conventional
    top-left image coordinates, so no additional vertical flip belongs here.
    Invalid points are ignored defensively; valid coordinates are clipped to
    the visible image instead of allowing a malformed overlay outside it.
    """
    rendered = np.asarray(frame, dtype=np.uint8).copy()
    if waypoints is None:
        return rendered

    values = np.asarray(waypoints, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(
            f"Expected visual waypoints [K,2], got {tuple(values.shape)}"
        )
    valid = np.isfinite(values).all(axis=1)
    values = np.clip(values[valid], 0.0, 1.0)
    if len(values) == 0:
        return rendered

    height, width = rendered.shape[:2]
    points = [
        (
            int(round(float(x) * max(width - 1, 0))),
            int(round(float(y) * max(height - 1, 0))),
        )
        for x, y in values
    ]
    image = Image.fromarray(rendered)
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(width, height) / 128))
    radius = max(4, round(min(width, height) / 80))
    if len(points) > 1:
        draw.line(points, fill=(255, 220, 0), width=line_width)
    for index, (x, y) in enumerate(points):
        fill = (
            (40, 220, 80)
            if index == 0
            else (240, 70, 70)
            if index == len(points) - 1
            else (30, 180, 255)
        )
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=fill,
            outline=(255, 255, 255),
            width=max(1, line_width // 2),
        )
        draw.text((x + radius + 2, y - radius), str(index + 1), fill=(255, 255, 255))
    return np.asarray(image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/libero_rdt_demo_models.json"),
    )
    parser.add_argument(
        "--success-map",
        type=Path,
        default=Path("configs/libero_rdt_demo_successes.json"),
    )
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=Path("output_3/checkpoints/RDT-1B-LIBERO-Base"),
    )
    parser.add_argument("--libero-root", type=Path, default=Path("/home/ubuntu/LIBERO"))
    parser.add_argument(
        "--rdt-repo",
        type=Path,
        default=Path("/home/ubuntu/RoboticsDiffusionTransformer"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output_2/gradio_rdt_libero"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qwen-layer-index", type=int, default=7)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    parser.add_argument("--latent-count", type=int, default=6)
    parser.add_argument("--spatial-token-count", type=int, default=5)
    parser.add_argument(
        "--latent-student-code-dir",
        type=Path,
        default=Path("/home/ubuntu/VLA-FYP/train/stage2"),
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="flash_attention_2",
    )
    parser.add_argument(
        "--student-precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--t5-model-id", default="google/t5-v1_1-xxl")
    parser.add_argument("--t5-fallback-model-id", default="google/t5-v1_1-xxl")
    parser.add_argument("--t5-precision", choices=("bf16", "8bit"), default="bf16")
    parser.add_argument("--siglip-model-id", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--siglip-fallback-model-id", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--video-resolution", type=int, default=512)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def is_artifact(path: Path) -> bool:
    return all((path / filename).is_file() for filename in ARTIFACT_FILES)


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def resolve_checkpoint(path: Path) -> Path:
    """Resolve an artifact, preferring final and then the latest checkpoint."""
    path = resolve_path(path)
    if is_artifact(path):
        return path
    final = path / "final"
    if is_artifact(final):
        return final
    candidates = sorted(
        (candidate for candidate in path.glob("checkpoint-*") if is_artifact(candidate)),
        key=checkpoint_step,
    )
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No complete RDT artifact found at {path}. Expected final/ or "
        "checkpoint-N/ containing rdt_full.pt, interfaces.pt, and metadata.json."
    )


def resolve_local_or_hub(model_id: str, *, local_files_only: bool) -> str:
    raw = Path(model_id).expanduser()
    local = resolve_path(raw)
    if local.exists():
        return str(local)
    if raw.is_absolute() or model_id.startswith(("./", "../")):
        raise FileNotFoundError(local)
    if "/" not in model_id:
        raise FileNotFoundError(local)
    # A local snapshot is required so the LatentStudent wrapper can also find
    # packaged spatial_parameters.pt (B3 slots + waypoint MLP).
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, local_files_only=local_files_only)


@dataclass(frozen=True)
class ModelSpec:
    suite: str
    variant: str
    checkpoint: Path
    config: Path
    cache_root: Path
    qwen_model_id: str | None = None
    student_model_id: str | None = None
    processor_id: str | None = None
    controller: str = "thinkflow_artifact"
    conditioning: str = "enabled"

    @property
    def key(self) -> tuple[str, str]:
        return self.suite, self.variant


def load_registry(path: Path) -> dict[tuple[str, str], ModelSpec]:
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    specs: dict[tuple[str, str], ModelSpec] = {}
    for row in payload.get("models", []):
        spec = ModelSpec(
            suite=str(row["suite"]),
            variant=str(row["variant"]).lower(),
            checkpoint=resolve_path(row["checkpoint"]),
            config=resolve_path(row["config"]),
            cache_root=resolve_path(row["cache_root"]),
            qwen_model_id=row.get("qwen_model_id"),
            student_model_id=row.get("student_model_id"),
            processor_id=row.get("processor_id"),
            controller=str(row.get("controller", "thinkflow_artifact")),
            conditioning=str(row.get("conditioning", "enabled")),
        )
        if spec.suite not in SUITE_LABELS.values() or spec.variant not in VARIANT_LABELS.values():
            raise ValueError(f"Unsupported registry entry: {row}")
        if spec.key in specs:
            raise ValueError(f"Duplicate registry entry for {spec.key}")
        specs[spec.key] = spec
    return specs


@dataclass
class LoadedBundle:
    spec: ModelSpec
    checkpoint: Path
    cfg: Any
    benchmark: Any
    model: Any
    t5_tokenizer: Any
    t5: Any
    siglip_processor: Any
    siglip: Any
    qwen_processor: Any
    qwen: Any = None
    latent_student: Any = None


class RDTDemoRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if str(args.libero_root) not in sys.path:
            sys.path.insert(0, str(args.libero_root))
        install_robosuite_mujoco_compatibility()
        from libero.libero.benchmark import get_benchmark
        from libero.libero.envs import OffScreenRenderEnv

        self.get_benchmark = get_benchmark
        self.OffScreenRenderEnv = OffScreenRenderEnv
        self.registry = load_registry(args.registry)
        self.success_map_path = resolve_path(args.success_map)
        self.success_contract: dict[str, Any] = {}
        self.success_results: dict[str, Any] = {}
        self.success_map_mtime_ns = -1
        self.success_map_lock = threading.RLock()
        self.reload_success_map(force=True)
        self.catalogs = {
            suite: get_benchmark(suite)(0) for suite in SUITE_LABELS.values()
        }
        self.bundle: LoadedBundle | None = None
        self.language_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.stop_requested = threading.Event()
        self.lock = threading.RLock()
        self.busy = False
        resolve_path(args.output_dir).mkdir(parents=True, exist_ok=True)

    def reload_success_map(self, *, force: bool = False) -> bool:
        """Atomically refresh exact-seed results when their JSON file changes."""
        try:
            mtime_ns = self.success_map_path.stat().st_mtime_ns
            if not force and mtime_ns == self.success_map_mtime_ns:
                return False
            payload = json.loads(self.success_map_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            # Keep the last valid snapshot if a filesystem update is in flight.
            return False
        with self.success_map_lock:
            self.success_contract = payload.get("evaluation_contract", {})
            self.success_results = payload.get("results", {})
            self.success_map_mtime_ns = mtime_ns
        return True

    def task_choices(self, suite_label: str) -> list[str]:
        benchmark = self.catalogs[SUITE_LABELS[suite_label]]
        return [
            f"{index}: {benchmark.get_task(index).language}" for index in range(10)
        ]

    def variant_choices(self, suite_label: str) -> list[str]:
        suite = SUITE_LABELS[suite_label]
        return [
            label
            for label, variant in VARIANT_LABELS.items()
            if (suite, variant) in self.registry
        ]

    def state_ui(
        self,
        suite_label: str,
        variant_label: str,
        task_label: str,
    ) -> dict[str, Any]:
        self.reload_success_map()
        suite = SUITE_LABELS[suite_label]
        variant = VARIANT_LABELS[variant_label]
        task_id = int(task_label.split(":", 1)[0])
        states = (
            self.success_results.get(suite, {})
            .get(variant, {})
            .get("tasks", {})
            .get(str(task_id), [])
        )
        states = list(states[:5]) + [None] * max(0, 5 - len(states))
        choices = []
        for index, result in enumerate(states[:5]):
            outcome = "1" if result is True else "0" if result is False else "9"
            choices.append((f"{index}-{outcome}", index))
        preferred = next(
            (index for index, result in enumerate(states[:5]) if result is True),
            0,
        )
        return {"choices": choices, "value": preferred}

    def task_instruction(self, suite_label: str, task_label: str) -> str:
        task_id = int(task_label.split(":", 1)[0])
        return self.catalogs[SUITE_LABELS[suite_label]].get_task(task_id).language

    def suite_changed(self, suite_label: str) -> tuple[Any, str, str]:
        choices = self.task_choices(suite_label)
        message = "Selection changed. Click **Load model** before running."
        return choices, choices[0], message

    def default_instruction(self, suite_label: str, task_label: str) -> str:
        return self.task_instruction(suite_label, task_label)

    def _free_bundle(self) -> None:
        self.bundle = None
        self.language_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass

    def unload(self) -> tuple[str, dict[str, Any]]:
        with self.lock:
            if self.busy:
                return "Cannot unload while a rollout is active; click Stop first.", {}
            previous = None if self.bundle is None else self.bundle.checkpoint
            self._free_bundle()
        return f"Unloaded {previous}" if previous else "No model was loaded.", {}

    def _load_latent_student(self, spec: ModelSpec, cfg: Any) -> tuple[Any, Any]:
        if not spec.student_model_id:
            raise ValueError(f"{spec.variant.upper()} requires student_model_id in the registry")
        student_id = resolve_local_or_hub(
            spec.student_model_id, local_files_only=self.args.local_files_only
        )
        processor_source = spec.processor_id or spec.student_model_id
        processor_id = resolve_local_or_hub(
            processor_source, local_files_only=self.args.local_files_only
        )
        loader_args = SimpleNamespace(
            student_model_id=student_id,
            processor_id=processor_id,
            latent_student_code_dir=self.args.latent_student_code_dir,
            attn_implementation=self.args.attn_implementation,
            student_precision=self.args.student_precision,
            spatial_parameters_path=None,
            latent_count=self.args.latent_count,
            spatial_token_count=self.args.spatial_token_count,
            layer_index=self.args.qwen_layer_index,
        )
        student, processor = load_student_and_processor(loader_args, self.device)
        validate_student_runtime_contract(student, processor, args=loader_args, cfg=cfg)
        return student, processor

    def load_model(self, suite_label: str, variant_label: str) -> tuple[str, dict[str, Any]]:
        suite = SUITE_LABELS[suite_label]
        variant = VARIANT_LABELS[variant_label]
        spec = self.registry.get((suite, variant))
        if spec is None:
            raise KeyError(f"No registry entry for {suite}/{variant}")
        with self.lock:
            if self.busy:
                return "Cannot switch models while a rollout is active.", {}
            self._free_bundle()
            upstream_ema = spec.controller == "upstream_ema"
            checkpoint = (
                resolve_path(spec.checkpoint)
                if upstream_ema
                else resolve_checkpoint(spec.checkpoint)
            )
            if not spec.config.is_file():
                raise FileNotFoundError(spec.config)
            artifact_metadata = (
                {"global_step": None, "config": {"model": {}}}
                if upstream_ema
                else json.loads(
                    (checkpoint / "metadata.json").read_text(encoding="utf-8")
                )
            )
            artifact_model = artifact_metadata.get("config", {}).get("model", {})
            artifact_variant = str(
                artifact_model.get("conditioning_variant", "")
            ).lower()
            if artifact_variant and artifact_variant != variant:
                raise ValueError(
                    f"Checkpoint declares {artifact_variant.upper()}, but "
                    f"{variant.upper()} was selected: {checkpoint}"
                )
            cfg = load_config(spec.config)
            if upstream_ema:
                cfg = replace(
                    cfg,
                    model=replace(
                        cfg.model,
                        qwen_fusion="none",
                        conditioning_variant=variant,
                        gradient_checkpointing=False,
                        allow_random_frozen_state_adaptor=True,
                    ),
                )
            if cfg.model.conditioning_variant != variant:
                # B2 and B3 share architecture/configuration; the cache/model
                # metadata distinguishes their extracted feature semantics.
                cfg = replace(cfg, model=replace(cfg.model, conditioning_variant=variant))
            expected_fusion = (
                "hidden_cross_attention"
                if variant == "b0"
                else "hidden_waypoint_cross_attention"
            )
            if not upstream_ema and cfg.model.qwen_fusion != expected_fusion:
                raise ValueError(
                    f"{variant.upper()} requires {expected_fusion}, got {cfg.model.qwen_fusion}"
                )
            if cfg.model.native_rdt_128_mapping != "libero_joint_eef_delta":
                raise ValueError("Demo supports only the native Libero_RDT joint/action mapping")

            metadata = load_feature_metadata(spec.cache_root)
            t5_id = metadata.get("t5_model_id") or self.args.t5_model_id
            t5_tokenizer, t5 = load_t5_encoder(
                model_id=t5_id,
                fallback_model_id=self.args.t5_fallback_model_id,
                precision=self.args.t5_precision,
                device_map=self.args.device_map,
                cfg=cfg,
            )
            siglip_id = resolve_model_id(
                metadata.get("siglip_model_id", self.args.siglip_model_id),
                self.args.siglip_fallback_model_id,
            )
            siglip_processor = SiglipImageProcessor.from_pretrained(siglip_id)
            siglip = SiglipVisionModel.from_pretrained(
                siglip_id,
                torch_dtype=torch.bfloat16,
                device_map=self.args.device_map,
            ).eval()
            siglip.requires_grad_(False)

            qwen = None
            latent_student = None
            if variant == "b0":
                qwen_id = spec.qwen_model_id or metadata.get(
                    "qwen_model_id", "shreethar/stage1_unsloth"
                )
                processor_id = spec.processor_id or metadata.get(
                    "qwen_processor_id", qwen_id
                )
                qwen_id = resolve_local_or_hub(
                    qwen_id, local_files_only=self.args.local_files_only
                )
                processor_id = resolve_local_or_hub(
                    processor_id, local_files_only=self.args.local_files_only
                )
                qwen_processor = AutoProcessor.from_pretrained(
                    processor_id, local_files_only=self.args.local_files_only
                )
                qwen_processor.tokenizer.padding_side = "left"
                qwen = AutoModelForImageTextToText.from_pretrained(
                    qwen_id,
                    torch_dtype=torch.bfloat16,
                    device_map=self.args.device_map,
                    attn_implementation=self.args.attn_implementation,
                    local_files_only=self.args.local_files_only,
                ).eval()
                qwen.requires_grad_(False)
            else:
                latent_student, qwen_processor = self._load_latent_student(spec, cfg)

            if upstream_ema:
                upstream_config = load_upstream_checkpoint_config(checkpoint)
                model = build_upstream_policy(
                    upstream_config,
                    checkpoint,
                    resolve_path(self.args.rdt_repo),
                )
            else:
                model = SFTConditionedRDT(
                    cfg,
                    load_pretrained=True,
                    base_artifact=str(resolve_path(self.args.base_artifact)),
                )
                load_trainable_artifact(model, checkpoint, trainable=False)
            if upstream_ema:
                model.to(device=self.device, dtype=torch.bfloat16).eval()
            else:
                model.to(self.device).eval()
            model.requires_grad_(False)
            self.bundle = LoadedBundle(
                spec=spec,
                checkpoint=checkpoint,
                cfg=cfg,
                benchmark=self.catalogs[suite],
                model=model,
                t5_tokenizer=t5_tokenizer,
                t5=t5,
                siglip_processor=siglip_processor,
                siglip=siglip,
                qwen_processor=qwen_processor,
                qwen=qwen,
                latent_student=latent_student,
            )
            details = {
                "suite": suite,
                "variant": variant,
                "checkpoint": str(checkpoint),
                "checkpoint_step": artifact_metadata.get("global_step"),
                "qwen_fusion": cfg.model.qwen_fusion,
                "controller": spec.controller,
                "conditioning": spec.conditioning,
                "state_mapping": cfg.model.native_rdt_128_mapping,
                "default_diffusion_steps": cfg.noise_scheduler.num_inference_timesteps,
                "action_horizon": cfg.model.pred_horizon,
            }
            suffix = (
                " Qwen waypoints are visualization-only and are not passed to the controller."
                if spec.conditioning == "visualization_only"
                else ""
            )
            return (
                f"Loaded **{variant.upper()} / {suite_label}** from `{checkpoint}`."
                + suffix,
                details,
            )

    def request_stop(self) -> str:
        self.stop_requested.set()
        return "Stop requested; finishing the current model call/action."

    def _language_features(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.bundle is not None
        if instruction not in self.language_cache:
            bundle = self.bundle
            self.language_cache[instruction] = extract_t5_features(
                {"instructions": [instruction]},
                bundle.t5_tokenizer,
                bundle.t5,
                max_lang_tokens=bundle.cfg.model.max_lang_tokens,
                expected_dim=bundle.cfg.model.lang_token_dim,
                device=t5_device_from_encoder(bundle.t5, self.device),
            )
        return self.language_cache[instruction]

    def _plan(
        self,
        observation: dict[str, Any],
        previous: dict[str, Any] | None,
        instruction: str,
        diffusion_steps: int,
        plan_index: int,
        task_id: int,
        init_state_index: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        assert self.bundle is not None
        bundle = self.bundle
        cfg = bundle.cfg
        sample = rollout_sample(
            observation,
            previous,
            dataset_id=bundle.spec.suite,
            instruction=instruction,
            horizon=cfg.model.pred_horizon,
        )
        encoded = standardized_collate_fn(
            [sample],
            max_images_per_sample=6,
            image_history_size=2,
            image_jpeg_quality=100,
            skip_no_image=True,
            encode_image_slots=False,
        )
        if encoded is None:
            raise RuntimeError("Live observation did not produce an image sample")

        if bundle.spec.variant == "b0":
            features = extract_qwen_kv(
                encoded,
                bundle.qwen_processor,
                bundle.qwen,
                device=self.device,
                layer_index=self.args.qwen_layer_index,
                max_new_tokens=self.args.qwen_max_new_tokens,
                expected_dim=cfg.model.qwen_kv_dim,
                stop_at_think_end=True,
                prompt_template=load_feature_metadata(bundle.spec.cache_root).get(
                    "qwen_trajectory_prompt_template"
                ),
                enable_thinking=False,
                return_hidden_state=True,
                think_token_selector="think_end",
            )
            qwen_kv, qwen_hidden_states = features
            latent_waypoints = None
        else:
            visualization_token_count = (
                self.args.spatial_token_count
                if bundle.spec.conditioning == "visualization_only"
                else cfg.model.spatial_token_count
            )
            qwen_kv, qwen_hidden_states, latent_waypoints = (
                extract_latent_student_spatial_kv(
                    encoded,
                    student=bundle.latent_student,
                    processor=bundle.qwen_processor,
                    device=self.device,
                    layer_index=self.args.qwen_layer_index,
                    expected_dim=cfg.model.qwen_kv_dim,
                    spatial_token_count=visualization_token_count,
                    prompt_template=B2_TRAJECTORY_PROMPT,
                )
            )

        img_tokens, img_mask = extract_siglip_features(
            encoded,
            bundle.siglip_processor,
            bundle.siglip,
            max_img_tokens=cfg.model.image_tokens,
            expected_dim=cfg.model.img_token_dim,
            device=self.device,
            encode_invalid_slots=True,
        )
        lang_tokens, lang_mask = self._language_features(instruction)
        action_dim_mask = torch.ones(1, 7, dtype=encoded["action_dim_mask"].dtype)
        state, state_dim_mask, action_dim_mask = native_rdt_policy_inputs(
            encoded["state"],
            encoded["state_dim_mask"],
            action_dim_mask,
            mapping=cfg.model.native_rdt_128_mapping,
            joint_state=encoded.get("joint_state"),
        )
        batch = {
            "state": state.to(self.device),
            "state_dim_mask": state_dim_mask.to(self.device),
            "action_dim_mask": action_dim_mask.to(self.device),
            "ctrl_freq": encoded["ctrl_freq"].to(self.device),
            "lang_tokens": lang_tokens.to(self.device),
            "lang_mask": lang_mask.to(self.device),
            "img_tokens": img_tokens,
            "img_mask": img_mask,
            "qwen_kv": qwen_kv,
            "qwen_hidden_states": qwen_hidden_states,
            "plan_mask": torch.ones(
                qwen_hidden_states.shape[:2], dtype=torch.bool, device=self.device
            ),
        }
        visual_waypoints = None
        if latent_waypoints is not None:
            batch["latent_waypoints"] = latent_waypoints
            visual_waypoints = (
                latent_waypoints[0].detach().float().cpu().numpy().copy()
            )
        torch.manual_seed(
            self.args.seed + task_id * 100_000 + init_state_index * 1_000 + plan_index
        )
        with torch.inference_mode():
            if bundle.spec.controller == "upstream_ema":
                # The Long B2 demo uses TJ-chen's released EMA controller.
                # LatentStudent runs above only to draw its trajectory; no Qwen
                # hidden state, waypoint, or KV tensor enters this prediction.
                bundle.model.num_inference_timesteps = int(diffusion_steps)
                output = bundle.model.predict_action(
                    lang_tokens=batch["lang_tokens"],
                    lang_attn_mask=batch["lang_mask"],
                    img_tokens=batch["img_tokens"],
                    state_tokens=batch["state"].to(torch.bfloat16).unsqueeze(1),
                    action_mask=batch["action_dim_mask"].unsqueeze(1).to(
                        dtype=torch.bfloat16
                    ),
                    ctrl_freqs=batch["ctrl_freq"],
                ).float().cpu().numpy()
            else:
                bundle.model.runner.num_inference_timesteps = int(diffusion_steps)
                output = bundle.model.sample_actions(batch).float().cpu().numpy()
        actions = native_rdt_action_to_libero_7d(output)[0]
        if not np.isfinite(actions).all():
            raise FloatingPointError("RDT produced NaN/Inf actions")
        return actions, visual_waypoints

    def rollout(
        self,
        suite_label: str,
        variant_label: str,
        task_label: str,
        instruction: str,
        init_state_index: float,
        diffusion_steps: float,
        action_chunk: float,
        max_steps: float,
    ) -> Iterator[tuple[str, Image.Image | None, dict[str, Any], str | None, dict[str, Any]]]:
        with self.lock:
            if self.busy:
                yield "Another rollout is already active.", None, {}, None, {}
                return
            if self.bundle is None:
                yield "Load a model first.", None, {}, None, {}
                return
            suite = SUITE_LABELS[suite_label]
            variant = VARIANT_LABELS[variant_label]
            if self.bundle.spec.key != (suite, variant):
                yield "Selection differs from the loaded model. Click Load model.", None, {}, None, {}
                return
            self.busy = True

        try:
            yield from self._rollout_impl(
                suite_label,
                variant_label,
                task_label,
                instruction,
                init_state_index,
                diffusion_steps,
                action_chunk,
                max_steps,
            )
        finally:
            with self.lock:
                self.busy = False

    def _rollout_impl(
        self,
        suite_label: str,
        variant_label: str,
        task_label: str,
        instruction: str,
        init_state_index: float,
        diffusion_steps: float,
        action_chunk: float,
        max_steps: float,
    ) -> Iterator[tuple[str, Image.Image | None, dict[str, Any], str | None, dict[str, Any]]]:
        suite = SUITE_LABELS[suite_label]
        variant = VARIANT_LABELS[variant_label]

        bundle = self.bundle
        assert bundle is not None
        task_id = int(task_label.split(":", 1)[0])
        task = bundle.benchmark.get_task(task_id)
        instruction = instruction.strip() or task.language
        init_state_index = max(0, min(49, int(init_state_index)))
        diffusion_steps = max(1, min(50, int(diffusion_steps)))
        action_chunk = max(1, min(bundle.cfg.model.pred_horizon, int(action_chunk)))
        max_steps = max(1, min(2000, int(max_steps)))
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
        run_dir = resolve_path(self.args.output_dir) / (
            f"{suite}_{variant}_task{task_id:02d}_init{init_state_index:02d}_{run_id}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        video_path = run_dir / "rollout.mp4"
        steps_path = run_dir / "actions.jsonl"
        init_states = torch.load(
            self.args.libero_root
            / "libero"
            / "libero"
            / "init_files"
            / task.problem_folder
            / task.init_states_file,
            map_location="cpu",
            weights_only=False,
        )
        if init_state_index >= len(init_states):
            raise IndexError(
                f"Initial state {init_state_index} unavailable; task has {len(init_states)}"
            )
        env = self.OffScreenRenderEnv(
            bddl_file_name=bundle.benchmark.get_task_bddl_file_path(task_id),
            camera_heights=128,
            camera_widths=128,
            horizon=max_steps + 10,
        )
        writer = imageio.get_writer(
            video_path,
            format="FFMPEG",
            fps=self.args.video_fps,
            codec="libx264",
            quality=8,
        )
        self.stop_requested.clear()
        simulator_step = 0
        plan_index = 0
        success = False
        stopped = False
        previous = None
        started = time.perf_counter()
        last_action: dict[str, Any] = {}
        try:
            # Match the evaluation jobs' deterministic simulator seed. The
            # official initial-state vector is still installed explicitly
            # below; seeding also stabilizes reset-time simulator randomness.
            env.seed(self.args.seed + task_id)
            observation = env.reset()
            observation = env.set_init_state(init_states[init_state_index])
            for _ in range(5):
                observation, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
            with steps_path.open("w", encoding="utf-8") as step_file:
                while simulator_step < max_steps and not success:
                    if self.stop_requested.is_set():
                        stopped = True
                        break
                    actions, visual_waypoints = self._plan(
                        observation,
                        previous,
                        instruction,
                        diffusion_steps,
                        plan_index,
                        task_id,
                        init_state_index,
                    )
                    execute = min(action_chunk, len(actions), max_steps - simulator_step)
                    for offset in range(execute):
                        if self.stop_requested.is_set():
                            stopped = True
                            break
                        action = actions[offset]
                        previous = observation
                        observation, reward, done, _ = env.step(action)
                        simulator_step += 1
                        success = bool(done) or bool(env.check_success())
                        last_action = {
                            "step": simulator_step,
                            "plan": plan_index,
                            "offset": offset,
                            "xyz": action[:3].astype(float).tolist(),
                            "rotation": action[3:6].astype(float).tolist(),
                            "gripper": float(action[6]),
                            "reward": float(reward),
                            "success": success,
                        }
                        if visual_waypoints is not None:
                            last_action["visual_waypoints"] = (
                                visual_waypoints.astype(float).tolist()
                            )
                        step_file.write(json.dumps(last_action) + "\n")
                        step_file.flush()
                        frame = env.env.sim.render(
                            width=self.args.video_resolution,
                            height=self.args.video_resolution,
                            camera_name="agentview",
                        )
                        label = (
                            f"{variant.upper()} task={task_id} step={simulator_step}/{max_steps} "
                            f"diff={diffusion_steps} chunk={action_chunk} success={int(success)}"
                        )
                        rendered = frame_for_video(frame, label)
                        if variant in {"b2", "b3"}:
                            rendered = overlay_visual_trajectory(
                                rendered,
                                visual_waypoints,
                            )
                        writer.append_data(rendered)
                        # Stream at the actual end of each executed action
                        # chunk. Previously this was hard-coded to every five
                        # simulator steps, making the UI appear to ignore the
                        # action-chunk slider even though replanning was right.
                        if offset == execute - 1 or success or stopped:
                            yield (
                                f"Running: step {simulator_step}/{max_steps}, plan {plan_index}",
                                Image.fromarray(rendered),
                                last_action,
                                None,
                                {},
                            )
                        if success or stopped:
                            break
                    plan_index += 1
                    if stopped:
                        break
        finally:
            writer.close()
            env.close()

        summary = {
            "suite": suite,
            "variant": variant,
            "checkpoint": str(bundle.checkpoint),
            "task_id": task_id,
            "task": task.name,
            "default_instruction": task.language,
            "instruction_used": instruction,
            "init_state_index": init_state_index,
            "diffusion_steps": diffusion_steps,
            "action_chunk": action_chunk,
            "max_steps": max_steps,
            "steps_executed": simulator_step,
            "plans": plan_index,
            "success": success,
            "stopped_by_user": stopped,
            "elapsed_sec": time.perf_counter() - started,
            "video": str(video_path.resolve()),
            "actions": str(steps_path.resolve()),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        status = "Success" if success else ("Stopped by user" if stopped else "Finished")
        yield status, None, last_action, str(video_path), summary


def build_app(runtime: RDTDemoRuntime) -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "Gradio is required. Install it with: "
            "uv pip install --python .venv/bin/python 'gradio>=5,<7'"
        ) from exc

    suite_default = "LIBERO Spatial"
    variant_default = "B2"
    choices = runtime.task_choices(suite_default)
    initial_state_update = runtime.state_ui(
        suite_default, variant_default, choices[0]
    )
    with gr.Blocks(title="ThinkFlow RDT LIBERO Demo") as app:
        gr.Markdown(
            "# ThinkFlow RDT LIBERO Demo\n"
            "Load one B0/B2/B3 checkpoint at a time, then run an interactive "
            "LIBERO Spatial or Long episode. Switching models explicitly frees "
            "the previous GPU bundle first."
        )
        with gr.Row():
            with gr.Column(scale=2):
                suite = gr.Dropdown(
                    choices=list(SUITE_LABELS), value=suite_default, label="Suite"
                )
                variant = gr.Dropdown(
                    choices=runtime.variant_choices(suite_default),
                    value=variant_default,
                    label="Model",
                )
                with gr.Row():
                    load = gr.Button("Load model", variant="primary")
                    unload = gr.Button("Unload model")
                model_status = gr.Markdown("No model loaded.")
                model_details = gr.JSON(label="Loaded model")
                task = gr.Dropdown(choices=choices, value=choices[0], label="Task")
                instruction = gr.Textbox(
                    value=runtime.task_instruction(suite_default, choices[0]),
                    label="Instruction (blank uses benchmark instruction)",
                    lines=3,
                    visible=False,
                    interactive=False,
                )
                init_state = gr.Dropdown(
                    choices=initial_state_update["choices"],
                    value=initial_state_update["value"],
                    label="Initial state",
                )
                refresh_seed_results = gr.Button("Refresh seed evaluation", size="sm")
                with gr.Row():
                    diffusion = gr.Slider(1, 20, value=5, step=1, label="Diffusion steps")
                    action_chunk = gr.Slider(1, 64, value=10, step=1, label="Action chunk")
                with gr.Row():
                    max_steps = gr.Slider(1, 1000, value=300, step=1, label="Max steps")
                with gr.Row():
                    run = gr.Button("Run rollout", variant="primary")
                    stop = gr.Button("Stop")
                rollout_status = gr.Markdown("Ready")
            with gr.Column(scale=3):
                live_frame = gr.Image(label="Live agent view", type="pil")
                video = gr.Video(label="Saved rollout", autoplay=False)
        with gr.Row():
            action = gr.JSON(label="Latest executed 7D action")
            summary = gr.JSON(label="Rollout summary")

        def update_suite(value: str):
            task_choices = runtime.task_choices(value)
            variant_choices = runtime.variant_choices(value)
            selected_task = task_choices[0]
            selected_variant = "B2" if "B2" in variant_choices else variant_choices[0]
            selected_max_steps = 600 if SUITE_LABELS[value] == "libero_10" else 300
            state_update = runtime.state_ui(value, selected_variant, selected_task)
            return (
                gr.Dropdown(choices=variant_choices, value=selected_variant),
                gr.Dropdown(choices=task_choices, value=selected_task),
                runtime.default_instruction(value, selected_task),
                gr.Dropdown(**state_update),
                selected_max_steps,
                "Selection changed. Click **Load model** before running.",
            )

        def update_task(value: str, selected_variant: str, selected_task: str):
            state_update = runtime.state_ui(value, selected_variant, selected_task)
            return (
                runtime.default_instruction(value, selected_task),
                gr.Dropdown(**state_update),
            )

        def update_variant(value: str, selected_variant: str, selected_task: str):
            state_update = runtime.state_ui(value, selected_variant, selected_task)
            return (
                gr.Dropdown(**state_update),
                "Selection changed. Click **Load model** before running.",
            )

        def refresh_state_results(
            selected_suite: str, selected_variant: str, selected_task: str
        ):
            runtime.reload_success_map(force=True)
            return gr.Dropdown(
                **runtime.state_ui(selected_suite, selected_variant, selected_task)
            )

        suite.change(
            update_suite,
            inputs=suite,
            outputs=[
                variant,
                task,
                instruction,
                init_state,
                max_steps,
                model_status,
            ],
        )
        task.change(
            update_task,
            inputs=[suite, variant, task],
            outputs=[instruction, init_state],
        )
        variant.change(
            update_variant,
            inputs=[suite, variant, task],
            outputs=[init_state, model_status],
        )
        refresh_seed_results.click(
            refresh_state_results,
            inputs=[suite, variant, task],
            outputs=init_state,
            queue=False,
        )
        result_refresh_timer = gr.Timer(30.0)
        result_refresh_timer.tick(
            refresh_state_results,
            inputs=[suite, variant, task],
            outputs=init_state,
            queue=False,
        )
        load.click(runtime.load_model, inputs=[suite, variant], outputs=[model_status, model_details])
        unload.click(runtime.unload, outputs=[model_status, model_details])
        run.click(
            runtime.rollout,
            inputs=[
                suite,
                variant,
                task,
                instruction,
                init_state,
                diffusion,
                action_chunk,
                max_steps,
            ],
            outputs=[rollout_status, live_frame, action, video, summary],
        )
        stop.click(runtime.request_stop, outputs=rollout_status, queue=False)
    return app


def main() -> None:
    args = parse_args()
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.list_models:
        registry = load_registry(args.registry)
        for key, spec in sorted(registry.items()):
            try:
                if spec.controller == "upstream_ema":
                    checkpoint = resolve_path(spec.checkpoint)
                    load_upstream_checkpoint_config(checkpoint)
                else:
                    checkpoint = resolve_checkpoint(spec.checkpoint)
                state = str(checkpoint)
            except FileNotFoundError:
                state = f"unavailable ({spec.checkpoint})"
            print(f"{key[0]}/{key[1]}: {state}")
        return
    runtime = RDTDemoRuntime(args)
    app = build_app(runtime)
    app.queue(default_concurrency_limit=2).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
