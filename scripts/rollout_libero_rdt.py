#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/thinkflow-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/thinkflow-matplotlib")

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    SiglipImageProcessor,
    SiglipVisionModel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
LIBERO_ROOT_DEFAULT = Path("/home/ubuntu/LIBERO")
for path in (SRC_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from precompute_all_features import (  # noqa: E402
    extract_qwen_kv,
    extract_siglip_features,
    standardized_collate_fn,
)
from thinkflow_rdt.adapters.action_stats import load_action_stats  # noqa: E402
from thinkflow_rdt.adapters.libero import (  # noqa: E402
    libero_observation_to_rdt,
    rdt_action_to_libero,
)
from thinkflow_rdt.checkpoint import load_trainable_artifact  # noqa: E402
from thinkflow_rdt.config import load_config  # noqa: E402
from thinkflow_rdt.model import SFTConditionedRDT  # noqa: E402


def install_robosuite_mujoco_compatibility() -> None:
    """Adapt robosuite 1.4's sole old-style MuJoCo mass-matrix call."""
    from robosuite.controllers.base_controller import Controller

    def update(controller, force: bool = False) -> None:
        if not (controller.new_update or force):
            return
        sim = controller.sim
        sim.forward()
        site_id = sim.model.site_name2id(controller.eef_name)
        controller.ee_pos = np.asarray(sim.data.site_xpos[site_id]).copy()
        controller.ee_ori_mat = np.asarray(sim.data.site_xmat[site_id]).reshape(3, 3).copy()
        controller.ee_pos_vel = np.asarray(sim.data.get_site_xvelp(controller.eef_name)).copy()
        controller.ee_ori_vel = np.asarray(sim.data.get_site_xvelr(controller.eef_name)).copy()
        controller.joint_pos = np.asarray(sim.data.qpos[controller.qpos_index]).copy()
        controller.joint_vel = np.asarray(sim.data.qvel[controller.qvel_index]).copy()
        controller.J_pos = np.asarray(
            sim.data.get_site_jacp(controller.eef_name).reshape((3, -1))[:, controller.qvel_index]
        ).copy()
        controller.J_ori = np.asarray(
            sim.data.get_site_jacr(controller.eef_name).reshape((3, -1))[:, controller.qvel_index]
        ).copy()
        controller.J_full = np.vstack([controller.J_pos, controller.J_ori])
        mass_matrix = np.empty((sim.model.nv, sim.model.nv), dtype=np.float64, order="C")
        try:
            mujoco.mj_fullM(sim.model._model, sim.data._data, mass_matrix)
        except TypeError:
            mujoco.mj_fullM(sim.model._model, mass_matrix, sim.data.qM)
        controller.mass_matrix = mass_matrix[controller.qvel_index, :][:, controller.qvel_index]
        controller.new_update = False

    Controller.update = update


def load_feature_metadata(cache_root: Path) -> dict[str, Any]:
    path = cache_root / "precompute_metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def free_model(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_cached_language_features(
    task_name: str,
    *,
    cache_root: Path,
    cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    for split in ("train", "validation", "test"):
        manifest = cache_root / split / "manifest.jsonl"
        if not manifest.exists():
            continue
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if isinstance(item, dict) and task_name not in str(item.get("episode_id", "")):
                    continue
                record_path = Path(item["path"] if isinstance(item, dict) else item)
                if not record_path.is_absolute():
                    record_path = manifest.parent / record_path
                record = torch.load(record_path, map_location="cpu", weights_only=True)
                if not isinstance(item, dict) and task_name not in str(record.get("episode_id", "")):
                    continue
                tokens = torch.as_tensor(record["lang_tokens"], dtype=torch.bfloat16)
                source_mask = torch.as_tensor(record.get("lang_mask", torch.ones(len(tokens))), dtype=torch.bool)
                output = torch.zeros(
                    1,
                    cfg.model.max_lang_tokens,
                    cfg.model.lang_token_dim,
                    dtype=torch.bfloat16,
                )
                mask = torch.zeros(1, cfg.model.max_lang_tokens, dtype=torch.bool)
                valid = min(len(tokens), cfg.model.max_lang_tokens)
                output[0, :valid] = tokens[:valid]
                mask[0, :valid] = source_mask[:valid]
                print(f"Loaded cached T5 features from {record_path}")
                return output, mask
    raise FileNotFoundError(
        f"No cached language features for task {task_name!r} below {cache_root}"
    )


def rollout_sample(
    observation: dict[str, Any],
    previous_observation: dict[str, Any] | None,
    *,
    instruction: str,
    horizon: int,
) -> dict[str, Any]:
    converted = libero_observation_to_rdt(observation)
    current = {
        "primary": Image.fromarray(converted["primary"]).convert("RGB"),
        "wrist": None if converted["wrist"] is None else Image.fromarray(converted["wrist"]).convert("RGB"),
        "secondary": None,
    }
    if previous_observation is None:
        previous = current
        previous_mask = {"primary": 0, "wrist": 0, "secondary": 0}
    else:
        old = libero_observation_to_rdt(previous_observation)
        previous = {
            "primary": Image.fromarray(old["primary"]).convert("RGB"),
            "wrist": None if old["wrist"] is None else Image.fromarray(old["wrist"]).convert("RGB"),
            "secondary": None,
        }
        previous_mask = {
            "primary": 1,
            "wrist": int(previous["wrist"] is not None),
            "secondary": 0,
        }
    current_mask = {
        "primary": 1,
        "wrist": int(current["wrist"] is not None),
        "secondary": 0,
    }
    return {
        "dataset_id": "libero_object",
        "episode_id": "rollout",
        "step_idx": "0",
        "instruction": instruction,
        "images": current,
        "image_mask": current_mask,
        "image_history": [previous, current],
        "image_history_mask": [previous_mask, current_mask],
        "state": converted["state"],
        "state_mask": np.ones(7, dtype=np.float32),
        "actions": np.zeros((horizon, 7), dtype=np.float32),
        "actions_mask": np.ones(horizon, dtype=np.float32),
        "ctrl_freq": 20.0,
    }


def frame_for_video(frame: np.ndarray, text: str) -> np.ndarray:
    # LIBERO camera arrays use OpenGL's bottom-up convention.
    frame = np.asarray(frame)[::-1].copy()
    width = int(frame.shape[1])
    cv2.rectangle(frame, (0, 0), (width, 34), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out a trained RDT artifact in LIBERO Object and record MP4.")
    parser.add_argument("--config", default="configs/b0_rdt1b_lora.yaml")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/libero_object_full/checkpoint-1000"))
    parser.add_argument("--cache-root", type=Path, default=Path("cache_features/libero_object/full"))
    parser.add_argument("--action-stats", type=Path, default=Path("dataset/LIBERO/Object/datasets/libero_object/audit.json"))
    parser.add_argument("--libero-root", type=Path, default=LIBERO_ROOT_DEFAULT)
    parser.add_argument("--task-id", type=int, default=0, choices=range(10))
    parser.add_argument("--init-state-index", type=int, default=0)
    parser.add_argument(
        "--demo-hdf5",
        type=Path,
        help="Start from the first recorded simulator state of this HDF5 demo instead of a benchmark init state.",
    )
    parser.add_argument("--demo-name", default="demo_0")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--action-chunk", type=int, default=8)
    parser.add_argument("--qwen-refresh-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--video-resolution",
        type=int,
        default=512,
        help="True simulator render resolution for the MP4; policy observations remain 128x128.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/libero_object_rollout/task0_checkpoint1000.mp4"))
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.libero_root) not in sys.path:
        sys.path.insert(0, str(args.libero_root))
    install_robosuite_mujoco_compatibility()
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RDT rollout")
    device = torch.device("cuda")
    cfg = load_config(args.config)
    stats = load_action_stats(args.action_stats)
    metadata = load_feature_metadata(args.cache_root)
    qwen_id = metadata.get("qwen_model_id", "shreethar/stage1_unsloth")
    qwen_processor_id = metadata.get("qwen_processor_id", qwen_id)
    siglip_id = metadata.get("siglip_model_id", "google/siglip-so400m-patch14-384")

    benchmark = get_benchmark("libero_object")(0)
    task = benchmark.get_task(args.task_id)
    instruction = task.language
    lang_tokens, lang_mask = load_cached_language_features(
        task.name,
        cache_root=args.cache_root,
        cfg=cfg,
    )

    print("Loading Qwen and SigLIP encoders...")
    qwen_processor = AutoProcessor.from_pretrained(qwen_processor_id)
    qwen_processor.tokenizer.padding_side = "left"
    qwen = AutoModelForImageTextToText.from_pretrained(
        qwen_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        attn_implementation="sdpa",
    ).eval()
    siglip_processor = SiglipImageProcessor.from_pretrained(siglip_id)
    siglip = SiglipVisionModel.from_pretrained(
        siglip_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
    ).eval()

    print(f"Loading RDT artifact {args.checkpoint}...")
    model = SFTConditionedRDT(cfg, load_pretrained=True)
    load_trainable_artifact(model, args.checkpoint, trainable=False)
    model.to(device).eval()

    env = OffScreenRenderEnv(
        bddl_file_name=benchmark.get_task_bddl_file_path(args.task_id),
        camera_heights=128,
        camera_widths=128,
        horizon=args.max_steps + 10,
    )
    observation = env.reset()
    if args.demo_hdf5 is not None:
        import h5py

        with h5py.File(args.demo_hdf5, "r") as handle:
            demo = handle["data"][args.demo_name]
            recorded_state = np.asarray(demo["states"][0], dtype=np.float64)
        state_index = -1
        observation = env.set_init_state(recorded_state)
    else:
        init_states = torch.load(
            args.libero_root / "libero" / "libero" / "init_files" / task.problem_folder / task.init_states_file,
            map_location="cpu",
            weights_only=False,
        )
        state_index = args.init_state_index % len(init_states)
        observation = env.set_init_state(init_states[state_index])
        for _ in range(5):
            observation, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.output,
        format="FFMPEG",
        fps=args.fps,
        codec="libx264",
        quality=8,
    )
    previous_observation = None
    cached_qwen: torch.Tensor | None = None
    success = False
    simulator_step = 0
    plan_index = 0
    start = time.perf_counter()
    try:
        while simulator_step < args.max_steps and not success:
            sample = rollout_sample(
                observation,
                previous_observation,
                instruction=instruction,
                horizon=cfg.model.pred_horizon,
            )
            encoded = standardized_collate_fn(
                [sample],
                max_images_per_sample=6,
                image_history_size=2,
                image_jpeg_quality=90,
                skip_no_image=True,
                encode_image_slots=False,
            )
            assert encoded is not None
            if cached_qwen is None or plan_index % args.qwen_refresh_every == 0:
                cached_qwen = extract_qwen_kv(
                    encoded,
                    qwen_processor,
                    qwen,
                    device=device,
                    layer_index=int(metadata.get("qwen_layer_index", 7)),
                    max_new_tokens=args.qwen_max_new_tokens,
                    expected_dim=cfg.model.qwen_kv_dim,
                    stop_at_think_end=bool(metadata.get("qwen_stop_at_think", True)),
                    prompt_template=metadata.get("qwen_trajectory_prompt_template"),
                    enable_thinking=bool(metadata.get("qwen_enable_thinking", False)),
                )
            img_tokens, img_mask = extract_siglip_features(
                encoded,
                siglip_processor,
                siglip,
                max_img_tokens=cfg.model.image_tokens,
                expected_dim=cfg.model.img_token_dim,
                device=device,
            )
            batch = {
                "state": encoded["state"].to(device),
                "action_dim_mask": encoded["action_dim_mask"].to(device),
                "ctrl_freq": encoded["ctrl_freq"].to(device),
                "lang_tokens": lang_tokens.to(device),
                "lang_mask": lang_mask.to(device),
                "img_tokens": img_tokens,
                "img_mask": img_mask,
                "qwen_kv": cached_qwen,
            }
            torch.manual_seed(args.seed + plan_index)
            normalized = model.sample_actions(batch)[0].float().cpu().numpy()
            actions = rdt_action_to_libero(normalized, stats)
            if not np.isfinite(actions).all():
                raise FloatingPointError("RDT produced NaN/Inf actions")

            chunk = min(args.action_chunk, args.max_steps - simulator_step)
            for action_index in range(chunk):
                # Keep adjacent t-1/t frames for the next SigLIP history, matching
                # the feature-precomputation contract even when actions are chunked.
                previous_observation = observation
                observation, reward, done, _ = env.step(actions[action_index])
                simulator_step += 1
                success = bool(done) or bool(env.check_success())
                label = f"task={args.task_id} step={simulator_step} plan={plan_index} success={int(success)}"
                video_frame = env.env.sim.render(
                    width=args.video_resolution,
                    height=args.video_resolution,
                    camera_name="agentview",
                )
                writer.append_data(frame_for_video(video_frame, label))
                if success:
                    break
            plan_index += 1
            print(
                f"plan={plan_index} simulator_step={simulator_step} "
                f"success={success} elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
    finally:
        writer.close()
        env.close()

    summary = {
        "task_id": args.task_id,
        "instruction": instruction,
        "checkpoint": str(args.checkpoint.resolve()),
        "init_state_index": state_index,
        "steps": simulator_step,
        "plans": plan_index,
        "success": success,
        "video": str(args.output.resolve()),
    }
    if args.demo_hdf5 is not None:
        summary["demo_hdf5"] = str(args.demo_hdf5.resolve())
        summary["demo_name"] = args.demo_name
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
