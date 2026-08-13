"""Closed-loop LIBERO evaluation for a Qwen-KV SmolVLA checkpoint.

At every re-plan this script reproduces the cache's Qwen extraction from the
current agent-view frame, injects the resulting [K | V] vector, samples a raw
LIBERO action chunk, executes it, and records success/video output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

# Hugging Face reads its offline flag while modules are imported, so honor the
# command-line switch before importing transformers/LeRobot.
if "--local-files-only" in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/smolvla-qwen-kv-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/smolvla-qwen-kv-matplotlib")

import imageio.v2 as imageio
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
for extra_path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins

from precompute_all_features import (
    QWEN_TRAJECTORY_PROMPT_TEMPLATE,
    extract_qwen_kv,
    standardized_collate_fn,
)
from precompute_latent_student_kv import (
    extract_latent_student_spatial_kv,
    load_student_and_processor,
)
from .libero_rollout import (
    frame_for_video,
    install_robosuite_mujoco_compatibility,
    rollout_sample,
)

from .cached_libero import cached_state_to_libero_state
from .configuration import KVSmolVLAConfig
from .modeling import KVSmolVLAPolicy


LIBERO_SUITES = ("libero_10", "libero_spatial", "libero_goal", "libero_object")
DEFAULT_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/smolvla_base_qwen_kv_all_suites/checkpoint-014000"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("cache_features_libero_b0_raw_ortho6d"),
        help="Used for Qwen extraction metadata only; rollout observations are live.",
    )
    parser.add_argument("--libero-root", type=Path, default=Path("/home/ubuntu/LIBERO"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/smolvla_qwen_kv_checkpoint_014000_eval"),
    )
    parser.add_argument("--suites", nargs="+", choices=LIBERO_SUITES, default=list(LIBERO_SUITES))
    parser.add_argument(
        "--task-id",
        type=int,
        action="append",
        choices=range(10),
        help="May be repeated. Omit to evaluate all ten tasks in every selected suite.",
    )
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--env-batch-size", type=int, default=2)
    parser.add_argument("--action-chunk", type=int, default=4)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override suite defaults (280 spatial/object, 300 goal, 520 LIBERO-10).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qwen-device-map", default="cuda")
    parser.add_argument("--qwen-model-id", default=None)
    parser.add_argument("--qwen-processor-id", default=None)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    parser.add_argument("--latent-student-code-dir", type=Path, default=None)
    parser.add_argument("--spatial-parameters-path", type=Path, default=None)
    parser.add_argument("--latent-count", type=int, default=None)
    parser.add_argument(
        "--latent-student-attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
        help="Attention backend used only while loading LatentStudent for live rollout.",
    )
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--video-resolution", type=int, default=512)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_suite_metadata(cache_root: Path, suite: str) -> dict[str, Any]:
    path = cache_root / suite / "precompute_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing cache provenance metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_policy_checkpoint(path: Path) -> Path:
    """Accept a policy dir, official checkpoint dir, or train_config.json."""

    resolved = path.expanduser().resolve()
    if resolved.is_file():
        if resolved.name != "train_config.json":
            raise ValueError(
                f"Checkpoint file must be train_config.json, got {resolved.name!r}"
            )
        resolved = resolved.parent
    if (resolved / "pretrained_model").is_dir():
        resolved = resolved / "pretrained_model"
    required = ("config.json", "model.safetensors", "policy_preprocessor.json")
    missing = [name for name in required if not (resolved / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Resolved policy directory {resolved} is missing required files: {missing}"
        )
    return resolved


def pil_images_to_float_batch(images) -> torch.Tensor:
    arrays = [np.asarray(image.convert("RGB"), dtype=np.uint8).copy() for image in images]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float().div_(255.0)


def _high_resolution_render(
    env: Any,
    *,
    width: int,
    height: int,
    camera_name: str,
) -> np.ndarray:
    return env.env.sim.render(width=width, height=height, camera_name=camera_name)


def make_recordable_env(env_args: dict[str, Any]) -> Any:
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(**env_args)
    env.render = MethodType(_high_resolution_render, env)
    return env


def render_vector_parallel(env: Any, **kwargs: Any) -> list[np.ndarray]:
    for worker in env.workers:
        worker.parent_remote.send(["render", kwargs])
    return [worker.parent_remote.recv() for worker in env.workers]


def completed_episode_keys(path: Path) -> set[tuple[str, int, int]]:
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {
        (str(row["suite"]), int(row["task_id"]), int(row["init_state_index"]))
        for row in rows
    }


def write_summary(results_path: Path, summary_path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    suites: dict[str, dict[str, Any]] = {}
    for row in rows:
        suite = suites.setdefault(
            row["suite"],
            {"suite": row["suite"], "episodes": 0, "successes": 0, "tasks": {}},
        )
        suite["episodes"] += 1
        suite["successes"] += int(row["success"])
        task = suite["tasks"].setdefault(
            str(row["task_id"]),
            {
                "task_id": row["task_id"],
                "instruction": row["instruction"],
                "episodes": 0,
                "successes": 0,
            },
        )
        task["episodes"] += 1
        task["successes"] += int(row["success"])
    suite_rows = []
    for suite in suites.values():
        task_rows = []
        for task in suite.pop("tasks").values():
            task["success_rate"] = task["successes"] / max(task["episodes"], 1)
            task_rows.append(task)
        suite["success_rate"] = suite["successes"] / max(suite["episodes"], 1)
        suite["tasks"] = sorted(task_rows, key=lambda row: row["task_id"])
        suite_rows.append(suite)
    successes = sum(int(row["success"]) for row in rows)
    summary = {
        "checkpoint": str(rows[0]["checkpoint"]) if rows else None,
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / max(len(rows), 1),
        "suites": sorted(suite_rows, key=lambda row: row["suite"]),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_evaluation(
    args: argparse.Namespace,
    *,
    policy_override: KVSmolVLAPolicy | None = None,
    preprocessor_override=None,
    postprocessor_override=None,
) -> dict[str, Any]:
    """Evaluate a saved policy or an in-memory policy from the training loop."""

    if args.local_files_only:
        # SmolVLA constructs its nested AutoProcessor internally without a
        # local_files_only argument. Offline mode applies the CLI contract to
        # that nested load as well and avoids unnecessary Hub HEAD requests.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    requested_checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint = resolve_policy_checkpoint(requested_checkpoint)
    cache_root = args.cache_root.expanduser().resolve()
    libero_root = args.libero_root.expanduser().resolve()
    if not 1 <= args.episodes_per_task <= 50:
        raise ValueError("--episodes-per-task must be in [1, 50]")
    if args.env_batch_size <= 0 or args.action_chunk <= 0:
        raise ValueError("--env-batch-size and --action-chunk must be positive")
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))

    register_third_party_plugins()
    install_robosuite_mujoco_compatibility()
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    if policy_override is None:
        config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        if not isinstance(config, KVSmolVLAConfig):
            raise TypeError(
                f"Expected a smolvla_qwen_kv checkpoint, got {getattr(config, 'type', None)!r}"
            )
        config.device = str(device)
        # A LeRobot pretrained_model directory is a full policy checkpoint. Build
        # the architecture without separately loading SmolVLM Hub weights; strict
        # safetensor loading below then verifies that the checkpoint supplies every
        # parameter, including the frozen VLM.
        config.load_vlm_weights = False
    else:
        if preprocessor_override is None or postprocessor_override is None:
            raise ValueError(
                "In-memory policy evaluation requires matching preprocessor and postprocessor"
            )
        config = policy_override.config
        if not isinstance(config, KVSmolVLAConfig):
            raise TypeError(f"Expected KVSmolVLAPolicy config, got {type(config).__name__}")
    if args.action_chunk > config.chunk_size:
        raise ValueError(f"--action-chunk exceeds checkpoint chunk_size={config.chunk_size}")

    if policy_override is None:
        print(f"Loading Qwen-KV SmolVLA policy from {checkpoint}")
        policy = KVSmolVLAPolicy.from_pretrained(
            checkpoint,
            config=config,
            local_files_only=True,
            strict=True,
        ).eval()
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
    else:
        policy = policy_override
        preprocessor = preprocessor_override
        postprocessor = postprocessor_override
        policy.eval()

    metadata_by_suite = {suite: load_suite_metadata(cache_root, suite) for suite in args.suites}
    first_metadata = metadata_by_suite[args.suites[0]]
    is_latent_student = config.external_kv_token_count > 1
    provenance_fields = (
        (
            "student_model_id",
            "processor_id",
            "spatial_token_count",
            "layer_index",
            "latent_count",
            "prompt_template",
        )
        if is_latent_student
        else (
            "qwen_model_id",
            "qwen_processor_id",
            "qwen_layer_index",
            "qwen_stop_at_think",
            "qwen_enable_thinking",
            "qwen_trajectory_prompt_template",
            "qwen_image_source",
        )
    )
    for suite, metadata in metadata_by_suite.items():
        mismatched = [
            key for key in provenance_fields if metadata.get(key) != first_metadata.get(key)
        ]
        if mismatched:
            raise ValueError(f"Qwen cache provenance differs for {suite}: {mismatched}")
    if is_latent_student:
        student_id = args.qwen_model_id or first_metadata["student_model_id"]
        processor_id = args.qwen_processor_id or first_metadata.get("processor_id", student_id)
        spatial_token_count = int(first_metadata.get("spatial_token_count", 5))
        if spatial_token_count != config.external_kv_token_count:
            raise ValueError(
                f"Cache metadata has {spatial_token_count} spatial tokens but checkpoint "
                f"expects {config.external_kv_token_count}"
            )
        student_args = argparse.Namespace(
            student_model_id=student_id,
            processor_id=processor_id,
            latent_student_code_dir=args.latent_student_code_dir,
            spatial_parameters_path=args.spatial_parameters_path,
            latent_count=(
                args.latent_count
                if args.latent_count is not None
                else int(first_metadata.get("latent_count", 6))
            ),
            spatial_token_count=spatial_token_count,
            attn_implementation=args.latent_student_attn_implementation,
        )
        print(
            f"Loading LatentStudent extractor {student_id} at layer "
            f"{first_metadata['layer_index']} with {spatial_token_count} spatial tokens"
        )
        qwen, qwen_processor = load_student_and_processor(student_args, device)
    else:
        qwen_id = args.qwen_model_id or first_metadata["qwen_model_id"]
        qwen_processor_id = args.qwen_processor_id or first_metadata.get(
            "qwen_processor_id", qwen_id
        )
        print(
            f"Loading Qwen extractor {qwen_id} at layer {first_metadata['qwen_layer_index']} "
            f"(image_source={first_metadata.get('qwen_image_source')})"
        )
        qwen_processor = AutoProcessor.from_pretrained(
            qwen_processor_id,
            local_files_only=args.local_files_only,
        )
        qwen_processor.tokenizer.padding_side = "left"
        qwen = AutoModelForImageTextToText.from_pretrained(
            qwen_id,
            torch_dtype=torch.bfloat16,
            device_map=args.qwen_device_map,
            attn_implementation="sdpa",
            local_files_only=args.local_files_only,
        ).eval()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "episodes.jsonl"
    summary_path = output_dir / "summary.json"
    completed = completed_episode_keys(results_path)
    task_ids = sorted(set(args.task_id)) if args.task_id else list(range(10))

    with results_path.open("a", encoding="utf-8") as results_file:
        for suite in args.suites:
            benchmark = get_benchmark(suite)(0)
            max_steps = args.max_steps or DEFAULT_MAX_STEPS[suite]
            metadata = metadata_by_suite[suite]
            for task_id in task_ids:
                task = benchmark.get_task(task_id)
                init_states = torch.load(
                    libero_root
                    / "libero"
                    / "libero"
                    / "init_files"
                    / task.problem_folder
                    / task.init_states_file,
                    map_location="cpu",
                    weights_only=False,
                )
                pending = [
                    index
                    for index in range(args.episodes_per_task)
                    if (suite, task_id, index) not in completed
                ]
                for batch_start in range(0, len(pending), args.env_batch_size):
                    indices = pending[batch_start : batch_start + args.env_batch_size]
                    if not indices:
                        continue
                    env_args = {
                        "bddl_file_name": benchmark.get_task_bddl_file_path(task_id),
                        "camera_heights": 128,
                        "camera_widths": 128,
                        "horizon": max_steps + 10,
                    }
                    env_fns = [
                        (lambda env_args=env_args: make_recordable_env(env_args))
                        if args.save_videos
                        else (lambda env_args=env_args: OffScreenRenderEnv(**env_args))
                        for _ in indices
                    ]
                    env = SubprocVectorEnv(env_fns)
                    env.reset()
                    observations = list(env.set_init_state(init_states[indices]))
                    for _ in range(5):
                        observations, _, _, _ = env.step(
                            np.zeros((len(indices), 7), dtype=np.float32)
                        )
                        observations = list(observations)

                    previous: list[dict[str, Any] | None] = [None] * len(indices)
                    done = np.zeros(len(indices), dtype=bool)
                    success_step = np.full(len(indices), max_steps, dtype=np.int32)
                    simulator_step = 0
                    plan_index = 0
                    started = time.perf_counter()
                    writers: list[Any | None] = [None] * len(indices)
                    video_paths: list[Path | None] = [None] * len(indices)
                    if args.save_videos:
                        video_dir = output_dir / "videos" / suite
                        video_dir.mkdir(parents=True, exist_ok=True)
                        for local_index, init_index in enumerate(indices):
                            path = video_dir / f"task{task_id:02d}_init{init_index:02d}.mp4"
                            video_paths[local_index] = path
                            writers[local_index] = imageio.get_writer(
                                path,
                                format="FFMPEG",
                                fps=args.video_fps,
                                codec="libx264",
                                quality=8,
                            )

                    while simulator_step < max_steps and not bool(done.all()):
                        active = np.flatnonzero(~done).tolist()
                        samples = [
                            rollout_sample(
                                observations[index],
                                previous[index],
                                dataset_id=suite,
                                instruction=task.language,
                                horizon=config.chunk_size,
                            )
                            for index in active
                        ]
                        encoded = standardized_collate_fn(
                            samples,
                            max_images_per_sample=6,
                            image_history_size=2,
                            image_jpeg_quality=100,
                            skip_no_image=True,
                            encode_image_slots=False,
                        )
                        if encoded is None:
                            raise RuntimeError("No rollout samples remained after image collation")
                        if is_latent_student:
                            qwen_kv, _ = extract_latent_student_spatial_kv(
                                encoded,
                                student=qwen,
                                processor=qwen_processor,
                                device=device,
                                layer_index=int(metadata["layer_index"]),
                                expected_dim=config.external_kv_width,
                                spatial_token_count=config.external_kv_token_count,
                                prompt_template=metadata.get(
                                    "prompt_template",
                                    QWEN_TRAJECTORY_PROMPT_TEMPLATE,
                                ),
                            )
                        else:
                            qwen_kv = extract_qwen_kv(
                                encoded,
                                qwen_processor,
                                qwen,
                                device=device,
                                layer_index=int(metadata["qwen_layer_index"]),
                                max_new_tokens=args.qwen_max_new_tokens,
                                expected_dim=config.external_kv_width,
                                stop_at_think_end=bool(
                                    metadata.get("qwen_stop_at_think", True)
                                ),
                                prompt_template=metadata.get(
                                    "qwen_trajectory_prompt_template"
                                ),
                                enable_thinking=bool(
                                    metadata.get("qwen_enable_thinking", False)
                                ),
                            )

                        raw_batch = {
                            "observation.state": cached_state_to_libero_state(encoded["state"]),
                            "observation.images.image": pil_images_to_float_batch(
                                [sample["images"]["primary"] for sample in samples]
                            ),
                            "observation.images.image2": pil_images_to_float_batch(
                                [sample["images"]["wrist"] for sample in samples]
                            ),
                            "task": [task.language] * len(active),
                        }
                        policy_batch = preprocessor(raw_batch)
                        # The checkpoint's saved processor predates plugin metadata
                        # registration, so explicitly attach the live tensor too.
                        policy_batch[config.external_kv_key] = qwen_kv.to(device)
                        generator = torch.Generator(device=device).manual_seed(
                            args.seed
                            + LIBERO_SUITES.index(suite) * 1_000_000
                            + task_id * 10_000
                            + batch_start * 100
                            + plan_index
                        )
                        noise = torch.randn(
                            len(active),
                            config.chunk_size,
                            config.max_action_dim,
                            generator=generator,
                            device=device,
                        )
                        with torch.inference_mode():
                            normalized_chunk = policy.predict_action_chunk(
                                policy_batch,
                                noise=noise,
                            )
                            predicted = postprocessor(normalized_chunk).float().cpu().numpy()
                        if not np.isfinite(predicted).all():
                            raise FloatingPointError("SmolVLA produced NaN/Inf actions")

                        execute_count = min(
                            args.action_chunk,
                            predicted.shape[1],
                            max_steps - simulator_step,
                        )
                        for action_offset in range(execute_count):
                            done_before = done.copy()
                            actions = np.zeros((len(indices), 7), dtype=np.float32)
                            for active_position, env_index in enumerate(active):
                                actions[env_index] = np.clip(
                                    predicted[active_position, action_offset], -1.0, 1.0
                                )
                                previous[env_index] = observations[env_index]
                            next_observations, _, step_done, _ = env.step(actions)
                            observations = list(next_observations)
                            simulator_step += 1
                            newly_done = (~done) & np.asarray(step_done, dtype=bool)
                            success_step[newly_done] = simulator_step
                            done |= np.asarray(step_done, dtype=bool)
                            if args.save_videos:
                                rendered = render_vector_parallel(
                                    env,
                                    width=args.video_resolution,
                                    height=args.video_resolution,
                                    camera_name="agentview",
                                )
                                for local_index, writer in enumerate(writers):
                                    if writer is None or done_before[local_index]:
                                        continue
                                    label = (
                                        f"{suite} task={task_id} init={indices[local_index]} "
                                        f"step={simulator_step} success={int(done[local_index])}"
                                    )
                                    writer.append_data(
                                        frame_for_video(rendered[local_index], label)
                                    )
                            if bool(done.all()):
                                break
                        plan_index += 1
                        if plan_index % 10 == 0:
                            print(
                                f"suite={suite} task={task_id} init={indices} "
                                f"step={simulator_step}/{max_steps} "
                                f"successes={int(done.sum())}/{len(done)}",
                                flush=True,
                            )

                    elapsed = time.perf_counter() - started
                    for writer in writers:
                        if writer is not None:
                            writer.close()
                    env.close()
                    for local_index, init_index in enumerate(indices):
                        row = {
                            "checkpoint": str(checkpoint),
                            "suite": suite,
                            "task_id": task_id,
                            "task_name": task.name,
                            "instruction": task.language,
                            "init_state_index": init_index,
                            "success": bool(done[local_index]),
                            "success_step": int(success_step[local_index]),
                            "simulator_steps": simulator_step,
                            "action_chunk": args.action_chunk,
                            "plans": plan_index,
                            "elapsed_sec": elapsed,
                            "video": None
                            if video_paths[local_index] is None
                            else str(video_paths[local_index]),
                        }
                        results_file.write(json.dumps(row) + "\n")
                        results_file.flush()
                    summary = write_summary(results_path, summary_path)
                    print(json.dumps(summary, indent=2), flush=True)

    summary = write_summary(results_path, summary_path)
    return summary


def main() -> None:
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
