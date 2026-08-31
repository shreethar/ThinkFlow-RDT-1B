#!/usr/bin/env python
"""Evaluate a TJ-chen RDT-1B LIBERO checkpoint on LIBERO Goal.

This is a standalone integration of the checkpoint author's LIBERO policy
contract with this repository's installed LIBERO environment.  It deliberately
does not use ThinkFlow's 10-D ortho6D action adapter: the released checkpoint
stores LIBERO's native 7-D delta controller command in unified RDT slots
39:45 (EEF xyz/rpy delta) and 10 (gripper).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from types import MethodType
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/thinkflow-cache")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/thinkflow-matplotlib")

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_model
from transformers import (
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
    T5Tokenizer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RDT_REPO = Path("/home/ubuntu/RoboticsDiffusionTransformer")
LIBERO_ROOT = Path("/home/ubuntu/LIBERO")
for path in (REPO_ROOT / "scripts", RDT_REPO, LIBERO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.rdt_runner import RDTRunner  # noqa: E402
from rollout_libero_rdt import (  # noqa: E402
    frame_for_video,
    install_robosuite_mujoco_compatibility,
)

STATE_INDICES = (0, 1, 2, 3, 4, 5, 6, 10, 11)
ACTION_INDICES = (39, 40, 41, 42, 43, 44, 10)
STATE_DIM = 128
PRED_HORIZON = 64
IMAGE_HISTORY = 2
NUM_CAMERAS = 3
CTRL_FREQUENCY = 20.0
GRIPPER_MIN = -0.04245
GRIPPER_MAX = 0.05185


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("output_3/checkpoints/RDT-1B-LIBERO-Base"),
    )
    parser.add_argument("--model-id", default="TJ-chen/RDT-1B-LIBERO-Base")
    parser.add_argument("--output-dir", type=Path, default=Path("output_3/libero_goal_base"))
    parser.add_argument("--libero-root", type=Path, default=LIBERO_ROOT)
    parser.add_argument("--rdt-repo", type=Path, default=RDT_REPO)
    parser.add_argument("--t5-model", default="google/t5-v1_1-xxl")
    parser.add_argument("--siglip-model", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--action-chunk", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument("--seed", type=int, default=20241201)
    parser.add_argument("--video-resolution", type=int, default=512)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--task-id", type=int, action="append", choices=range(10))
    return parser.parse_args()


def _render(env: Any, *, width: int, height: int, camera_name: str) -> np.ndarray:
    return env.env.sim.render(width=width, height=height, camera_name=camera_name)


def make_recordable_env(env_args: dict[str, Any]) -> Any:
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(**env_args)
    env.render = MethodType(_render, env)
    return env


def render_parallel(env: Any, **kwargs: Any) -> list[np.ndarray]:
    for worker in env.workers:
        worker.parent_remote.send(["render", kwargs])
    return [worker.parent_remote.recv() for worker in env.workers]


def load_checkpoint_config(checkpoint_dir: Path) -> dict[str, Any]:
    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "ema" / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            "Expected inference-only checkpoint files at "
            f"{config_path} and {weights_path}."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_language_features(
    instructions: dict[int, str], model_id: str, device: torch.device
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Encode ten task strings, then release T5 before loading the policy."""
    print(f"Loading T5 encoder {model_id}...", flush=True)
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    encoder = T5EncoderModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device).eval()
    result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.inference_mode():
        for task_id, instruction in instructions.items():
            tokens = tokenizer(
                instruction,
                max_length=1024,
                truncation=True,
                padding="longest",
                return_attention_mask=True,
                return_tensors="pt",
            )
            mask = tokens["attention_mask"].to(device=device, dtype=torch.bool)
            features = encoder(
                input_ids=tokens["input_ids"].to(device),
                attention_mask=mask,
            ).last_hidden_state
            result[task_id] = (features.cpu(), mask.cpu())
    del encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def build_policy(config: dict[str, Any], checkpoint_dir: Path, rdt_repo: Path) -> RDTRunner:
    if not (rdt_repo / "models" / "rdt_runner.py").is_file():
        raise FileNotFoundError(f"Missing RDT source tree: {rdt_repo}")
    runner_config = {
        "lang_adaptor": config["lang_adaptor"],
        "img_adaptor": config["img_adaptor"],
        "state_adaptor": config["state_adaptor"],
        "rdt": config["rdt"],
        "noise_scheduler": config["noise_scheduler"],
    }
    policy = RDTRunner(
        action_dim=int(config["action_dim"]),
        pred_horizon=int(config["pred_horizon"]),
        config=runner_config,
        lang_token_dim=int(config["lang_token_dim"]),
        img_token_dim=int(config["img_token_dim"]),
        state_token_dim=int(config["state_token_dim"]),
        max_lang_cond_len=int(config["max_lang_cond_len"]),
        img_cond_len=int(config["img_cond_len"]),
        lang_pos_embed_config=config["lang_pos_embed_config"],
        img_pos_embed_config=config["img_pos_embed_config"],
        dtype=torch.bfloat16,
    )
    weights = checkpoint_dir / "ema" / "model.safetensors"
    load_model(policy, str(weights), strict=True)
    return policy


def process_images(
    histories: list[tuple[deque[np.ndarray], deque[np.ndarray]]],
    active: list[int],
    processor: SiglipImageProcessor,
    encoder: SiglipVisionModel,
    device: torch.device,
) -> torch.Tensor:
    mean_color = tuple(int(value * 255) for value in processor.image_mean)
    background = np.full(
        (processor.size["height"], processor.size["width"], 3),
        np.asarray(mean_color, dtype=np.uint8),
        dtype=np.uint8,
    )
    tensors: list[torch.Tensor] = []
    for env_index in active:
        agent, wrist = histories[env_index]
        for history_index in range(IMAGE_HISTORY):
            for frame in (agent[history_index], wrist[history_index], background):
                tensors.append(
                    processor.preprocess(
                        Image.fromarray(np.asarray(frame)), return_tensors="pt"
                    )["pixel_values"][0]
                )
    pixels = torch.stack(tensors).to(device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        features = encoder(pixel_values=pixels).last_hidden_state
    return features.reshape(len(active), -1, features.shape[-1])


def format_state(observations: list[dict[str, Any]], active: list[int], device: torch.device) -> torch.Tensor:
    values = []
    for env_index in active:
        obs = observations[env_index]
        proprio = np.concatenate(
            [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]], axis=-1
        ).astype(np.float32)
        proprio[-2:] = (proprio[-2:] - GRIPPER_MIN) / (GRIPPER_MAX - GRIPPER_MIN)
        unified = np.zeros(STATE_DIM, dtype=np.float32)
        unified[list(STATE_INDICES)] = proprio
        values.append(unified)
    return torch.from_numpy(np.stack(values)[:, None]).to(device=device, dtype=torch.bfloat16)


def summarize(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    tasks: dict[int, dict[str, Any]] = {}
    for row in rows:
        block = tasks.setdefault(
            int(row["task_id"]),
            {"task_id": int(row["task_id"]), "instruction": row["instruction"], "episodes": 0, "successes": 0},
        )
        block["episodes"] += 1
        block["successes"] += int(row["success"])
    for block in tasks.values():
        block["success_rate"] = block["successes"] / block["episodes"]
    successes = sum(int(row["success"]) for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / max(len(rows), 1),
        "tasks": [tasks[key] for key in sorted(tasks)],
    }


def existing_keys(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    return {
        (int(row["task_id"]), int(row["init_state_index"]))
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RDT-1B rollout")
    if not 1 <= args.episodes_per_task <= 50:
        raise ValueError("--episodes-per-task must be in [1, 50]")
    if not 1 <= args.action_chunk <= PRED_HORIZON:
        raise ValueError(f"--action-chunk must be in [1, {PRED_HORIZON}]")

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    results_path = output_dir / "episodes.jsonl"
    summary_path = output_dir / "summary.json"
    run_config_path = output_dir / "run_config.json"
    device = torch.device("cuda")

    install_robosuite_mujoco_compatibility()
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import SubprocVectorEnv

    benchmark = get_benchmark("libero_goal")(0)
    task_ids = sorted(set(args.task_id)) if args.task_id else list(range(10))
    instructions = {task_id: benchmark.get_task(task_id).language for task_id in task_ids}
    config = load_checkpoint_config(checkpoint_dir)
    if int(config["action_dim"]) != STATE_DIM or int(config["pred_horizon"]) != PRED_HORIZON:
        raise ValueError("Checkpoint is not the expected 128-D, 64-step LIBERO RDT")

    run_config = {
        "model": args.model_id,
        "checkpoint_dir": str(checkpoint_dir),
        "benchmark": "libero_goal",
        "task_ids": task_ids,
        "episodes_per_task": args.episodes_per_task,
        "action_chunk": args.action_chunk,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "state_indices": STATE_INDICES,
        "action_indices": ACTION_INDICES,
    }
    run_config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    language = load_language_features(instructions, args.t5_model, device)
    print(f"Loading SigLIP encoder {args.siglip_model}...", flush=True)
    processor = SiglipImageProcessor.from_pretrained(args.siglip_model)
    vision = SiglipVisionModel.from_pretrained(
        args.siglip_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device).eval()
    print(f"Loading EMA policy from {checkpoint_dir}...", flush=True)
    policy = build_policy(config, checkpoint_dir, args.rdt_repo).to(
        device=device, dtype=torch.bfloat16
    ).eval()

    action_mask = torch.zeros((1, 1, STATE_DIM), device=device, dtype=torch.bfloat16)
    action_mask[:, :, list(ACTION_INDICES)] = 1
    completed = existing_keys(results_path)
    total = len(task_ids) * args.episodes_per_task
    print(f"Evaluation target: {total} episodes; {len(completed)} already complete", flush=True)

    with results_path.open("a", encoding="utf-8") as output:
        for task_id in task_ids:
            task = benchmark.get_task(task_id)
            init_states = benchmark.get_task_init_states(task_id)
            pending = [
                index for index in range(args.episodes_per_task)
                if (task_id, index) not in completed
            ]
            if not pending:
                continue
            env_args = {
                "bddl_file_name": benchmark.get_task_bddl_file_path(task_id),
                "camera_heights": 128,
                "camera_widths": 128,
                "horizon": args.max_steps + 10,
            }
            env = SubprocVectorEnv([
                lambda env_args=env_args: make_recordable_env(env_args) for _ in pending
            ])
            env.seed(args.seed + task_id)
            env.reset()
            observations = list(env.set_init_state(init_states[pending]))
            for _ in range(5):
                observations, _, _, _ = env.step(np.zeros((len(pending), 7), dtype=np.float32))
                observations = list(observations)

            histories: list[tuple[deque[np.ndarray], deque[np.ndarray]]] = []
            for obs in observations:
                agent = np.asarray(obs["agentview_image"])
                wrist = np.asarray(obs["robot0_eye_in_hand_image"])
                histories.append((deque([agent, agent], maxlen=2), deque([wrist, wrist], maxlen=2)))
            writers = []
            video_paths = []
            for init_index in pending:
                path = video_dir / f"task{task_id:02d}_init{init_index:02d}.mp4"
                video_paths.append(path)
                writers.append(imageio.get_writer(path, format="FFMPEG", fps=args.video_fps, codec="libx264", quality=8))

            done = np.zeros(len(pending), dtype=bool)
            success_steps = np.full(len(pending), args.max_steps, dtype=np.int32)
            buffers: list[np.ndarray | None] = [None] * len(pending)
            offsets = np.zeros(len(pending), dtype=np.int32)
            step = 0
            plan = 0
            started = time.perf_counter()
            try:
                while step < args.max_steps and not bool(done.all()):
                    replan = [index for index in np.flatnonzero(~done) if buffers[index] is None or offsets[index] >= args.action_chunk]
                    if replan:
                        images = process_images(histories, replan, processor, vision, device)
                        states = format_state(observations, replan, device)
                        lang, lang_mask = language[task_id]
                        torch.manual_seed(args.seed + task_id * 100_000 + plan)
                        with torch.inference_mode():
                            predicted = policy.predict_action(
                                lang_tokens=lang.expand(len(replan), -1, -1).to(device),
                                lang_attn_mask=lang_mask.expand(len(replan), -1).to(device),
                                img_tokens=images,
                                state_tokens=states,
                                action_mask=action_mask.expand(len(replan), -1, -1),
                                ctrl_freqs=torch.full((len(replan),), CTRL_FREQUENCY, device=device),
                            ).float().cpu().numpy()
                        commands = predicted[:, :, list(ACTION_INDICES)]
                        commands[:, :, -1] = np.where(commands[:, :, -1] < 0, -1.0, 1.0)
                        if not np.isfinite(commands).all():
                            raise FloatingPointError("Policy produced NaN/Inf actions")
                        for batch_index, env_index in enumerate(replan):
                            buffers[env_index] = commands[batch_index]
                            offsets[env_index] = 0
                        plan += 1

                    actions = np.zeros((len(pending), 7), dtype=np.float32)
                    done_before = done.copy()
                    for env_index in np.flatnonzero(~done):
                        assert buffers[env_index] is not None
                        actions[env_index] = buffers[env_index][offsets[env_index]]
                        offsets[env_index] += 1
                    next_obs, _, step_done, _ = env.step(actions)
                    observations = list(next_obs)
                    step += 1
                    newly_done = (~done) & np.asarray(step_done, dtype=bool)
                    success_steps[newly_done] = step
                    done |= np.asarray(step_done, dtype=bool)
                    rendered = render_parallel(env, width=args.video_resolution, height=args.video_resolution, camera_name="agentview")
                    for env_index, obs in enumerate(observations):
                        if done_before[env_index]:
                            continue
                        histories[env_index][0].append(np.asarray(obs["agentview_image"]))
                        histories[env_index][1].append(np.asarray(obs["robot0_eye_in_hand_image"]))
                        label = f"goal task={task_id} init={pending[env_index]} step={step} success={int(done[env_index])}"
                        writers[env_index].append_data(frame_for_video(rendered[env_index], label))
                    if step % 100 == 0:
                        print(f"task={task_id} step={step}/{args.max_steps} successes={int(done.sum())}/{len(done)}", flush=True)
            finally:
                for writer in writers:
                    writer.close()
                env.close()

            elapsed = time.perf_counter() - started
            for local_index, init_index in enumerate(pending):
                row = {
                    "benchmark": "libero_goal",
                    "task_id": task_id,
                    "task_name": task.name,
                    "instruction": task.language,
                    "init_state_index": init_index,
                    "success": bool(done[local_index]),
                    "steps": int(success_steps[local_index]),
                    "action_chunk": args.action_chunk,
                    "model": args.model_id,
                    "checkpoint": str(checkpoint_dir),
                    "video": str(video_paths[local_index]),
                    "elapsed_batch_seconds": elapsed,
                }
                output.write(json.dumps(row) + "\n")
                output.flush()
                completed.add((task_id, init_index))
            summary = summarize(results_path)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(f"task={task_id} successes={int(done.sum())}/{len(done)} overall={summary['successes']}/{summary['episodes']}", flush=True)

    summary = summarize(results_path)
    summary.update({"requested_episodes": total, "complete": summary["episodes"] >= total})
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
