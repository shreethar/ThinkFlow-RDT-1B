#!/usr/bin/env python
"""Interactive LIBERO-Object demo for the one-token Qwen-KV SmolVLA policy."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/smolvla-qwen-demo-cache")

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parent
for extra_path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

from experiments.smolvla_qwen_kv.cached_libero import (  # noqa: E402
    cached_state_to_libero_state,
)
from experiments.smolvla_qwen_kv.configuration import KVSmolVLAConfig  # noqa: E402
from experiments.smolvla_qwen_kv.libero_rollout import (  # noqa: E402
    frame_for_video,
    install_robosuite_mujoco_compatibility,
    rollout_sample,
)
from experiments.smolvla_qwen_kv.modeling import KVSmolVLAPolicy  # noqa: E402
from scripts.precompute_all_features import (  # noqa: E402
    QWEN_TRAJECTORY_PROMPT_TEMPLATE,
    extract_qwen_kv,
    standardized_collate_fn,
)

DEFAULT_POLICY = "shreethar/smolvla-qwen-b2-libero-22000"
DEFAULT_QWEN = "shreethar/stage1_unsloth"
SUITE = "libero_object"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN)
    parser.add_argument("--qwen-processor", default=DEFAULT_QWEN)
    parser.add_argument("--libero-root", type=Path, default=Path("/workspace/LIBERO"))
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo_application"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--video-resolution", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    return parser.parse_args()


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def find_dataset_dir(explicit: Path | None) -> Path:
    candidates = [
        explicit,
        REPO_ROOT / "libero-dataset" / "datasets" / SUITE,
        REPO_ROOT / "dataset" / "datasets" / SUITE,
        REPO_ROOT / "dataset" / SUITE,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.expanduser().is_dir():
            return candidate.expanduser().resolve()
    searched = "\n".join(f"  {path}" for path in candidates if path is not None)
    raise FileNotFoundError(
        "Could not find the LIBERO-Object HDF5 directory. Pass --dataset-dir. "
        f"Searched:\n{searched}"
    )


def find_demo_file(dataset_dir: Path, task_name: str) -> Path:
    files = sorted(dataset_dir.rglob("*.hdf5")) + sorted(dataset_dir.rglob("*.h5"))
    normalized = task_name.lower().replace(" ", "_")
    exact = [path for path in files if normalized in path.stem.lower()]
    if not exact:
        raise FileNotFoundError(
            f"No demonstration HDF5 matching task {task_name!r} below {dataset_dir}"
        )
    return exact[0]


def demo_payload(path: Path, demo_index: int) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        root = handle["data"] if "data" in handle else handle
        names = sorted(
            (name for name in root if isinstance(root[name], h5py.Group)),
            key=natural_key,
        )
        if not names:
            raise RuntimeError(f"No demonstration groups in {path}")
        selected = names[demo_index % len(names)]
        demo = root[selected]
        if "states" not in demo or "actions" not in demo:
            raise KeyError(f"{demo.name} must contain states and actions")
        payload: dict[str, Any] = {
            "name": selected,
            "count": len(names),
            "initial_state": np.asarray(demo["states"][0], dtype=np.float64),
            "actions": np.asarray(demo["actions"], dtype=np.float32)[:, :7],
            "reference_states": None,
        }
        if "obs" in demo:
            payload["reference_states"] = native_state_series(demo["obs"])
        return payload


def first_array(container: Any, keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in container:
            return np.asarray(container[key], dtype=np.float32)
    return None


def orientation_as_rotvec(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] == 3:
        return values.astype(np.float32)
    if values.shape[-1] == 4:
        return Rotation.from_quat(values.reshape(-1, 4)).as_rotvec().reshape(
            *values.shape[:-1], 3
        ).astype(np.float32)
    raise ValueError(f"Unsupported orientation shape: {values.shape}")


def native_state_series(observations: Any) -> np.ndarray | None:
    position = first_array(observations, ("ee_pos", "robot0_eef_pos"))
    orientation = first_array(observations, ("ee_ori", "robot0_eef_quat"))
    gripper = first_array(observations, ("gripper_states", "robot0_gripper_qpos"))
    if position is None or orientation is None or gripper is None:
        return None
    orientation = orientation_as_rotvec(orientation)
    length = min(len(position), len(orientation), len(gripper))
    return np.concatenate(
        [position[:length, :3], orientation[:length, :3], gripper[:length, :2]],
        axis=-1,
    ).astype(np.float32)


def live_native_state(observation: dict[str, Any]) -> np.ndarray:
    position = first_array(observation, ("ee_pos", "robot0_eef_pos"))
    orientation = first_array(observation, ("ee_ori", "robot0_eef_quat"))
    gripper = first_array(observation, ("gripper_states", "robot0_gripper_qpos"))
    if position is None or orientation is None or gripper is None:
        raise KeyError("LIBERO observation is missing end-effector state fields")
    rotvec = orientation_as_rotvec(np.asarray(orientation).reshape(1, -1))[0]
    return np.concatenate(
        [np.asarray(position).reshape(-1)[:3], rotvec, np.asarray(gripper).reshape(-1)[:2]]
    ).astype(np.float32)


def tensor_images(images: list[Image.Image]) -> torch.Tensor:
    arrays = [np.asarray(image.convert("RGB"), dtype=np.uint8).copy() for image in images]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float().div_(255.0)


def diagnostics(
    predicted: np.ndarray,
    reference_action: np.ndarray | None,
    live_state: np.ndarray,
    reference_state: np.ndarray | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "predicted_action": predicted.astype(float).tolist(),
        "predicted_translation_norm": float(np.linalg.norm(predicted[:3])),
        "predicted_rotation_norm": float(np.linalg.norm(predicted[3:6])),
        "live_state": live_state.astype(float).tolist(),
    }
    if reference_action is not None:
        difference = predicted - reference_action
        result.update(
            {
                "reference_demo_action": reference_action.astype(float).tolist(),
                "action_error_l2": float(np.linalg.norm(difference)),
                "action_xyz_error_l2": float(np.linalg.norm(difference[:3])),
                "action_rotation_error_l2": float(np.linalg.norm(difference[3:6])),
                "action_gripper_error_abs": float(abs(difference[6])),
            }
        )
    if reference_state is not None:
        difference = live_state - reference_state
        result.update(
            {
                "reference_demo_state": reference_state.astype(float).tolist(),
                "state_error_l2": float(np.linalg.norm(difference)),
                "state_xyz_error_l2": float(np.linalg.norm(difference[:3])),
                "state_rotation_vector_error_l2": float(np.linalg.norm(difference[3:6])),
                "state_gripper_error_l2": float(np.linalg.norm(difference[6:8])),
            }
        )
    return result


@dataclass
class TaskChoice:
    task_id: int
    name: str
    instruction: str

    @property
    def label(self) -> str:
        return f"{self.task_id}: {self.instruction}"


class DemoRuntime:
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

        self.OffScreenRenderEnv = OffScreenRenderEnv
        self.benchmark = get_benchmark(SUITE)(0)
        self.tasks = [
            TaskChoice(index, self.benchmark.get_task(index).name, self.benchmark.get_task(index).language)
            for index in range(10)
        ]
        self.task_by_label = {task.label: task for task in self.tasks}
        self.dataset_dir = find_dataset_dir(args.dataset_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        self.stop_requested = threading.Event()

        register_third_party_plugins()
        config = PreTrainedConfig.from_pretrained(
            args.policy, local_files_only=args.local_files_only
        )
        if not isinstance(config, KVSmolVLAConfig):
            raise TypeError(f"Expected a Qwen-KV SmolVLA checkpoint, got {type(config).__name__}")
        if config.external_kv_token_count != 1:
            raise ValueError(
                f"{args.policy} declares {config.external_kv_token_count} Qwen tokens. "
                "This presentation model is expected to be the one-token B0 policy."
            )
        if not 1 <= args.action_steps <= config.chunk_size:
            raise ValueError(f"--action-steps must be in [1, {config.chunk_size}]")
        config.n_action_steps = args.action_steps
        config.device = str(self.device)
        config.load_vlm_weights = False
        self.config = config
        self.policy = KVSmolVLAPolicy.from_pretrained(
            args.policy,
            config=config,
            local_files_only=args.local_files_only,
            strict=True,
        ).eval()
        policy_device = next(self.policy.parameters()).device
        if policy_device.type != self.device.type:
            raise RuntimeError(f"Policy loaded on {policy_device}, requested {self.device}")
        print(f"SmolVLA loaded on {policy_device}")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            args.policy,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        self.qwen_processor = AutoProcessor.from_pretrained(
            args.qwen_processor,
            local_files_only=args.local_files_only,
        )
        self.qwen_processor.tokenizer.padding_side = "left"
        self.qwen = AutoModelForImageTextToText.from_pretrained(
            args.qwen_model,
            torch_dtype=torch.bfloat16,
            device_map=str(self.device),
            attn_implementation="sdpa",
            local_files_only=args.local_files_only,
        ).eval()
        print(f"Qwen loaded on {next(self.qwen.parameters()).device}")

    def default_instruction(self, task_label: str) -> str:
        return self.task_by_label[task_label].instruction

    def request_stop(self) -> str:
        self.stop_requested.set()
        return "Stopping after the current model call..."

    def _plan(
        self,
        observation: dict[str, Any],
        previous: dict[str, Any] | None,
        instruction: str,
        plan_index: int,
    ) -> np.ndarray:
        sample = rollout_sample(
            observation,
            previous,
            dataset_id=SUITE,
            instruction=instruction,
            horizon=self.config.chunk_size,
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
            raise RuntimeError("The live observation did not produce an image sample")
        qwen_kv = extract_qwen_kv(
            encoded,
            self.qwen_processor,
            self.qwen,
            device=self.device,
            layer_index=7,
            max_new_tokens=128,
            expected_dim=self.config.external_kv_width,
            stop_at_think_end=True,
            prompt_template=QWEN_TRAJECTORY_PROMPT_TEMPLATE,
            enable_thinking=False,
        )
        raw_batch = {
            "observation.state": cached_state_to_libero_state(encoded["state"]),
            "observation.images.image": tensor_images([sample["images"]["primary"]]),
            "observation.images.image2": tensor_images([sample["images"]["wrist"]]),
            "task": [instruction],
        }
        policy_batch = self.preprocessor(raw_batch)
        policy_batch = {
            key: value.to(self.device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in policy_batch.items()
        }
        policy_batch[self.config.external_kv_key] = qwen_kv.to(self.device)
        generator = torch.Generator(device=self.device).manual_seed(42 + plan_index)
        noise = torch.randn(
            1,
            self.config.chunk_size,
            self.config.max_action_dim,
            generator=generator,
            device=self.device,
        )
        with torch.inference_mode():
            normalized = self.policy.predict_action_chunk(policy_batch, noise=noise)
            actions = self.postprocessor(normalized).float().cpu().numpy()[0]
        if not np.isfinite(actions).all():
            raise FloatingPointError("Policy produced NaN/Inf actions")
        return actions

    def rollout(
        self,
        task_label: str,
        instruction: str,
        demo_index: float,
        max_steps: float,
        auto_stop_xyz_error: float,
    ) -> Iterator[tuple[str, Image.Image | None, dict[str, Any], str | None, dict[str, Any]]]:
        task_choice = self.task_by_label[task_label]
        instruction = instruction.strip() or task_choice.instruction
        demo_path = find_demo_file(self.dataset_dir, task_choice.name)
        demo = demo_payload(demo_path, int(demo_index))
        max_steps = min(1000, max(1, int(max_steps)))
        run_id = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.args.output_dir / f"task{task_choice.task_id:02d}_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=False)
        video_path = run_dir / "rollout.mp4"
        diagnostics_path = run_dir / "steps.jsonl"

        env = self.OffScreenRenderEnv(
            bddl_file_name=self.benchmark.get_task_bddl_file_path(task_choice.task_id),
            camera_heights=128,
            camera_widths=128,
            horizon=max_steps + 10,
        )
        writer = imageio.get_writer(
            video_path,
            format="FFMPEG",
            fps=self.args.fps,
            codec="libx264",
            quality=8,
        )
        success = False
        stopped_for_drift = False
        stopped_by_user = False
        simulator_step = 0
        plan_index = 0
        previous = None
        last_metrics: dict[str, Any] = {}
        started = time.perf_counter()
        observation = env.reset()
        observation = env.set_init_state(demo["initial_state"])
        self.policy.reset()
        self.stop_requested.clear()

        try:
            with diagnostics_path.open("w", encoding="utf-8") as step_file:
                while simulator_step < max_steps and not success and not stopped_by_user:
                    actions = self._plan(observation, previous, instruction, plan_index)
                    execute_count = min(
                        self.args.action_steps,
                        len(actions),
                        max_steps - simulator_step,
                    )
                    for offset in range(execute_count):
                        if self.stop_requested.is_set():
                            stopped_by_user = True
                            break
                        action = np.clip(actions[offset, :7], -1.0, 1.0)
                        reference_action = (
                            demo["actions"][simulator_step]
                            if simulator_step < len(demo["actions"])
                            else None
                        )
                        previous = observation
                        observation, reward, done, _ = env.step(action)
                        simulator_step += 1
                        success = bool(done) or bool(env.check_success())
                        live_state = live_native_state(observation)
                        reference_states = demo["reference_states"]
                        reference_state = (
                            reference_states[min(simulator_step, len(reference_states) - 1)]
                            if reference_states is not None and len(reference_states)
                            else None
                        )
                        last_metrics = diagnostics(
                            action, reference_action, live_state, reference_state
                        )
                        last_metrics.update(
                            {
                                "step": simulator_step,
                                "plan": plan_index,
                                "success": success,
                                "reward": float(reward),
                                "comparison": "time-aligned demonstration reference",
                            }
                        )
                        step_file.write(json.dumps(last_metrics) + "\n")
                        step_file.flush()

                        xyz_error = last_metrics.get("state_xyz_error_l2")
                        if auto_stop_xyz_error > 0 and xyz_error is not None:
                            stopped_for_drift = xyz_error > auto_stop_xyz_error
                        label = (
                            f"step={simulator_step}/{max_steps} success={int(success)} "
                            f"xyz_ref_err={xyz_error:.3f}"
                            if xyz_error is not None
                            else f"step={simulator_step}/{max_steps} success={int(success)}"
                        )
                        frame = env.env.sim.render(
                            width=self.args.video_resolution,
                            height=self.args.video_resolution,
                            camera_name="agentview",
                        )
                        rendered = frame_for_video(frame, label)
                        writer.append_data(rendered)
                        if simulator_step == 1 or simulator_step % 5 == 0 or success:
                            yield (
                                f"Running task {task_choice.task_id}, step {simulator_step}/{max_steps}",
                                Image.fromarray(rendered),
                                last_metrics,
                                None,
                                {},
                            )
                        if success or stopped_for_drift:
                            break
                    plan_index += 1
        finally:
            writer.close()
            env.close()

        summary = {
            "model": self.args.policy,
            "suite": SUITE,
            "task_id": task_choice.task_id,
            "task_name": task_choice.name,
            "default_instruction": task_choice.instruction,
            "instruction_used": instruction,
            "demo_file": str(demo_path),
            "demo_name": demo["name"],
            "steps": simulator_step,
            "plans": plan_index,
            "success": success,
            "stopped_for_drift": stopped_for_drift,
            "stopped_by_user": stopped_by_user,
            "elapsed_sec": time.perf_counter() - started,
            "video": str(video_path.resolve()),
            "step_diagnostics": str(diagnostics_path.resolve()),
            "last_diagnostics": last_metrics,
            "reference_note": (
                "The action/state comparison is against the time-aligned expert demo. "
                "It is not an oracle action for policy-visited states after trajectories diverge."
            ),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        final_status = (
            "Success"
            if success
            else (
                "Stopped by user"
                if stopped_by_user
                else (
                    "Stopped: state drift threshold exceeded"
                    if stopped_for_drift
                    else "Finished"
                )
            )
        )
        yield final_status, None, last_metrics, str(video_path), summary


def build_app(runtime: DemoRuntime):
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "Gradio is required. Install it with: "
            "uv pip install --python .venv-smolvla/bin/python 'gradio>=5,<7'"
        ) from exc

    first = runtime.tasks[0]
    with gr.Blocks(title="SmolVLA Qwen LIBERO-Object Demo") as app:
        gr.Markdown("# SmolVLA Qwen LIBERO-Object Demo")
        with gr.Row():
            with gr.Column(scale=2):
                task = gr.Dropdown(
                    choices=[item.label for item in runtime.tasks],
                    value=first.label,
                    label="LIBERO-Object task",
                )
                instruction = gr.Textbox(
                    value=first.instruction,
                    label="Language instruction",
                    lines=2,
                )
                with gr.Row():
                    demo_index = gr.Number(value=0, precision=0, label="Demo/scene index")
                    max_steps = gr.Slider(1, 1000, value=runtime.args.max_steps, step=1, label="Max steps")
                auto_stop = gr.Number(
                    value=0.0,
                    label="Auto-stop XYZ reference error (metres; 0 disables)",
                )
                with gr.Row():
                    run = gr.Button("Run inference", variant="primary")
                    stop = gr.Button("Stop")
                status = gr.Markdown("Ready")
            with gr.Column(scale=3):
                live_frame = gr.Image(label="Live scene", type="pil")
        with gr.Row():
            metrics = gr.JSON(label="Live predicted vs demonstration reference")
            summary = gr.JSON(label="Run summary")
        video = gr.Video(label="Saved rollout", autoplay=False)

        task.change(runtime.default_instruction, inputs=task, outputs=instruction)
        run.click(
            runtime.rollout,
            inputs=[task, instruction, demo_index, max_steps, auto_stop],
            outputs=[status, live_frame, metrics, video, summary],
        )
        stop.click(runtime.request_stop, outputs=status)
    return app


def main() -> None:
    args = parse_args()
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    runtime = DemoRuntime(args)
    print("\nLIBERO-Object tasks:")
    for task in runtime.tasks:
        print(f"  [{task.task_id}] {task.instruction}")
    if args.list_tasks:
        return
    app = build_app(runtime)
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
