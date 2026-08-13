"""Official LeRobot LIBERO A/B evaluation with Qwen fusion off versus on.

Both branches use LeRobot's native LIBERO environment, observation processor,
policy processor, action postprocessor, seeds, and evaluator. The only changed
variable is whether live Qwen K/V is attached to the trained custom policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

if "--local-files-only" in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/smolvla-qwen-kv-cache")

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
for extra_path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed

from precompute_all_features import QWEN_TRAJECTORY_PROMPT_TEMPLATE, extract_qwen_kv
from precompute_latent_student_kv import (
    extract_latent_student_spatial_kv,
    load_student_and_processor,
)

from .configuration import KVSmolVLAConfig
from .evaluate_checkpoint import load_suite_metadata, resolve_policy_checkpoint
from .modeling import KVSmolVLAPolicy


LIBERO_SUITES = ("libero_10", "libero_spatial", "libero_goal", "libero_object")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/lerobot_smolvla_qwen_kv_fresh/checkpoints/last"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("cache_features_libero_b0_raw_ortho6d"),
        help="Cache provenance only; observations come from the live official environment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lerobot_smolvla_qwen_kv_fresh/fusion_ablation"),
    )
    parser.add_argument("--suites", nargs="+", choices=LIBERO_SUITES, default=["libero_10"])
    parser.add_argument("--task-ids", nargs="+", type=int, default=[0])
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-videos", type=int, default=2)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("disabled", "enabled"),
        default=["disabled", "enabled"],
    )
    parser.add_argument("--qwen-model-id", default=None)
    parser.add_argument("--qwen-processor-id", default=None)
    parser.add_argument("--qwen-device-map", default="cuda")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    parser.add_argument("--latent-student-code-dir", type=Path, default=None)
    parser.add_argument("--spatial-parameters-path", type=Path, default=None)
    parser.add_argument("--latent-count", type=int, default=None)
    parser.add_argument(
        "--latent-student-attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def tensor_batch_to_pil(images: torch.Tensor) -> list[Image.Image]:
    values = images.detach().float().cpu()
    if values.ndim != 4:
        raise ValueError(f"Expected batched images, got {tuple(values.shape)}")
    if values.shape[1] in (1, 3, 4):
        values = values.permute(0, 2, 3, 1)
    elif values.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Cannot identify image channels in {tuple(values.shape)}")
    if float(values.max()) <= 1.5:
        values = values * 255.0
    arrays = values.clamp(0, 255).byte().numpy()
    return [Image.fromarray(array[..., :3], mode="RGB") for array in arrays]


def primary_image_key(config: KVSmolVLAConfig) -> str:
    visual_keys = [
        key
        for key, feature in config.input_features.items()
        if feature.type == FeatureType.VISUAL
    ]
    preferred = [
        "observation.images.image",
        "observation.images.camera1",
    ]
    for key in preferred:
        if key in visual_keys:
            return key
    if not visual_keys:
        raise ValueError("Checkpoint declares no visual input features")
    return visual_keys[0]


def camera_mapping(config: KVSmolVLAConfig) -> dict[str, str] | None:
    keys = set(config.input_features)
    if "observation.images.camera1" in keys:
        return {
            "agentview_image": "camera1",
            "robot0_eye_in_hand_image": "camera2",
        }
    return None


def resolve_model_reference(value: str) -> str:
    """Resolve local model paths recorded relative to an extraction workspace.

    Some cache metadata was produced with values such as
    ``model/model/stage1_unsloth`` even though the remote workspace stores the
    model at ``/workspace/model/stage1_unsloth``. If no local candidate exists,
    preserve the value unchanged so normal Hugging Face repository IDs continue
    to work.
    """

    raw = Path(value).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend((Path.cwd() / raw, REPO_ROOT / raw, REPO_ROOT.parent / raw))
        parts = raw.parts
        if len(parts) >= 3 and parts[:2] == ("model", "model"):
            candidates.append(REPO_ROOT.parent / "model" / Path(*parts[2:]))
        elif parts and parts[0] == "model":
            candidates.append(REPO_ROOT.parent / raw)
    for candidate in candidates:
        if candidate.exists():
            resolved = str(candidate.resolve())
            if resolved != value:
                print(f"Resolved model reference {value!r} -> {resolved!r}")
            return resolved
    return value


class LiveKVPreprocessor:
    """Add live Qwen K/V before the saved policy processor runs."""

    def __init__(
        self,
        base: Any,
        policy: KVSmolVLAPolicy,
        extractor: Any,
        image_key: str,
    ) -> None:
        self.base = base
        self.policy = policy
        self.extractor = extractor
        self.image_key = image_key
        self.last_kv: torch.Tensor | None = None

    def __call__(self, observation: dict[str, Any]) -> dict[str, Any]:
        # SmolVLA queues n_action_steps actions. Re-extract only when that queue
        # is empty and the policy is about to produce a new action chunk.
        queue = self.policy._queues.get(ACTION)
        if queue is None or len(queue) == 0:
            tasks = observation.get("task")
            if tasks is None:
                raise KeyError("Official LIBERO observation has no task descriptions")
            images = tensor_batch_to_pil(observation[self.image_key])
            self.last_kv = self.extractor(list(tasks), images)
        processed = self.base(observation)
        if self.last_kv is None:
            raise RuntimeError("Qwen K/V was not initialized before policy inference")
        processed[self.policy.config.external_kv_key] = self.last_kv.to(
            self.policy.config.device
        )
        return processed


def validate_metadata(
    cache_root: Path,
    suites: list[str],
    token_count: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    metadata = {suite: load_suite_metadata(cache_root, suite) for suite in suites}
    first = metadata[suites[0]]
    fields = (
        ("student_model_id", "processor_id", "layer_index", "spatial_token_count")
        if token_count > 1
        else (
            "qwen_model_id",
            "qwen_processor_id",
            "qwen_layer_index",
            "qwen_enable_thinking",
            "qwen_stop_at_think",
            "qwen_trajectory_prompt_template",
        )
    )
    for suite, item in metadata.items():
        mismatches = [name for name in fields if item.get(name) != first.get(name)]
        if mismatches:
            raise ValueError(f"Qwen provenance differs for {suite}: {mismatches}")
    return first, metadata


def make_live_extractor(
    args: argparse.Namespace,
    config: KVSmolVLAConfig,
    metadata: dict[str, Any],
    device: torch.device,
):
    if config.external_kv_token_count > 1:
        spatial_count = int(metadata.get("spatial_token_count", 5))
        if spatial_count != config.external_kv_token_count:
            raise ValueError(
                f"Metadata has {spatial_count} spatial tokens; checkpoint expects "
                f"{config.external_kv_token_count}"
            )
        student_id = resolve_model_reference(
            args.qwen_model_id or metadata["student_model_id"]
        )
        processor_id = resolve_model_reference(
            args.qwen_processor_id or metadata.get("processor_id", student_id)
        )
        student_args = argparse.Namespace(
            student_model_id=student_id,
            processor_id=processor_id,
            latent_student_code_dir=args.latent_student_code_dir,
            spatial_parameters_path=args.spatial_parameters_path,
            latent_count=(
                args.latent_count
                if args.latent_count is not None
                else int(metadata.get("latent_count", 6))
            ),
            spatial_token_count=spatial_count,
            attn_implementation=args.latent_student_attn_implementation,
        )
        student, processor = load_student_and_processor(student_args, device)

        def extract(instructions: list[str], images: list[Image.Image]) -> torch.Tensor:
            batch = {
                "instructions": instructions,
                "qwen_images": [[image] for image in images],
            }
            kv, _ = extract_latent_student_spatial_kv(
                batch,
                student=student,
                processor=processor,
                device=device,
                layer_index=int(metadata["layer_index"]),
                expected_dim=config.external_kv_width,
                spatial_token_count=spatial_count,
                prompt_template=metadata.get(
                    "prompt_template", QWEN_TRAJECTORY_PROMPT_TEMPLATE
                ),
            )
            return kv

        return extract

    model_id = resolve_model_reference(args.qwen_model_id or metadata["qwen_model_id"])
    processor_id = resolve_model_reference(
        args.qwen_processor_id or metadata.get("qwen_processor_id", model_id)
    )
    processor = AutoProcessor.from_pretrained(
        processor_id,
        local_files_only=args.local_files_only,
    )
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.qwen_device_map,
        attn_implementation="sdpa",
        local_files_only=args.local_files_only,
    ).eval()

    def extract(instructions: list[str], images: list[Image.Image]) -> torch.Tensor:
        return extract_qwen_kv(
            {
                "instructions": instructions,
                "qwen_images": [[image] for image in images],
            },
            processor,
            model,
            device=device,
            layer_index=int(metadata["qwen_layer_index"]),
            max_new_tokens=args.qwen_max_new_tokens,
            expected_dim=config.external_kv_width,
            stop_at_think_end=bool(metadata.get("qwen_stop_at_think", True)),
            prompt_template=metadata.get("qwen_trajectory_prompt_template"),
            enable_thinking=bool(metadata.get("qwen_enable_thinking", False)),
        )

    return extract


def run_mode(
    args: argparse.Namespace,
    mode: str,
    policy: KVSmolVLAPolicy,
    base_preprocessor: Any,
    postprocessor: Any,
    extractor: Any | None,
) -> dict[str, Any]:
    enabled = mode == "enabled"
    policy.config.external_kv_required = enabled
    policy.reset()
    env_config = LiberoEnv(
        task=",".join(args.suites),
        task_ids=args.task_ids,
        camera_name_mapping=camera_mapping(policy.config),
        max_parallel_tasks=1,
    )
    envs = make_env(env_config, n_envs=args.batch_size, use_async_envs=False)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_config,
        policy_cfg=policy.config,
    )
    preprocessor = (
        LiveKVPreprocessor(
            base_preprocessor,
            policy,
            extractor,
            primary_image_key(policy.config),
        )
        if enabled
        else base_preprocessor
    )
    mode_dir = args.output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = mode_dir / "videos" if args.max_videos > 0 else None
    context = (
        torch.autocast(device_type=torch.device(args.device).type)
        if policy.config.use_amp
        else nullcontext()
    )
    set_seed(args.seed)
    with torch.no_grad(), context:
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=args.episodes_per_task,
            max_episodes_rendered=args.max_videos,
            videos_dir=videos_dir,
            return_episode_data=False,
            start_seed=args.seed,
            max_parallel_tasks=1,
        )
    (mode_dir / "eval_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    return info


def main() -> None:
    args = parse_args()
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if any(task_id < 0 or task_id > 9 for task_id in args.task_ids):
        raise ValueError("--task-ids must be in [0, 9]")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_policy_checkpoint(args.checkpoint)
    cache_root = args.cache_root.expanduser().resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

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
    base_preprocessor, postprocessor = make_pre_post_processors(
        config,
        str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    extractor = None
    metadata = None
    if "enabled" in args.modes:
        metadata, _ = validate_metadata(
            cache_root,
            args.suites,
            config.external_kv_token_count,
        )

    results: dict[str, Any] = {}
    for mode in args.modes:
        if mode == "enabled" and extractor is None:
            if metadata is None:
                raise RuntimeError("Enabled evaluation has no Qwen provenance metadata")
            extractor = make_live_extractor(args, config, metadata, device)
        print(f"\nEvaluating trained checkpoint with Qwen fusion {mode}")
        results[mode] = run_mode(
            args,
            mode,
            policy,
            base_preprocessor,
            postprocessor,
            extractor,
        )
        print(f"{mode}: {results[mode]['overall']}")

    comparison = {
        "checkpoint": str(checkpoint),
        "suites": args.suites,
        "task_ids": args.task_ids,
        "episodes_per_task": args.episodes_per_task,
        "seed": args.seed,
        "results": {mode: info["overall"] for mode, info in results.items()},
    }
    if "disabled" in results and "enabled" in results:
        comparison["enabled_minus_disabled_pc_success"] = (
            results["enabled"]["overall"]["pc_success"]
            - results["disabled"]["overall"]["pc_success"]
        )
    output_path = args.output_dir / "comparison.json"
    output_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
