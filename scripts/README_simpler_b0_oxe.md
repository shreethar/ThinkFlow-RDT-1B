# B0 OXE evaluation in SimplerEnv

This adapter evaluates `output_2/checkpoint-20000` without training. It keeps
the ThinkFlow policy and SimplerEnv in separate processes because their Python,
PyTorch, and SAPIEN requirements are incompatible.

## Data contract

The observation sent to B0 is:

- one current third-person RGB image for Qwen and SigLIP;
- one prior RGB image for SigLIP history (masked on the first step);
- the language instruction for T5 XXL and Qwen;
- TCP pose relative to the robot base as `[x, y, z, roll, pitch, yaw]`;
- gripper closedness in `[0, 1]`.

The 7-D state is packed into native RDT-128 slots. Gripper-open uses slot 10,
XYZ uses slots 30–32, and the first two columns of the Euler-derived rotation
matrix use slots 33–38. The remaining slots and masks are zero.

RDT returns `[64, 128]`. The adapter extracts the same ten supervised slots:

- normalized XYZ is de-normalized with Fractal q01/q99 for Google Robot or
  Bridge q01/q99 for WidowX;
- orthogonal-6D is projected onto SO(3) and decoded to Euler XYZ;
- Fractal's three learned rotation numbers are supplied numerically as its
  axis-angle delta; Bridge Euler XYZ is converted to a true axis-angle vector;
- RDT gripper-open is thresholded at zero. It is inverted for Google Robot
  (`+1=close`) and used directly for WidowX (`+1=open`). For Google Robot,
  `--google-gripper-sticky-steps N` converts the absolute target into a
  stateful relative command and holds each transition for `N` controller
  steps.
- `--rotation-scale` multiplies the decoded rotation command after conversion;
  use `0` for the no-rotation ablation or `0.25` for quarter-strength rotation.

Every closed-loop step records raw native output, decoded and executed action,
TCP state before/after, achieved TCP displacement, joint state, gripper state,
object poses, reward, termination, simulator info, and policy timings.

## Commands

### One-line native-Linux setup

From the ThinkFlow repository root, run:

```bash
bash scripts/setup_simpler_env_native_linux.sh
```

The script installs Ubuntu Vulkan/EGL prerequisites, verifies the NVIDIA
driver, synchronizes ThinkFlow from `uv.lock`, checks out a known-good
SimplerEnv revision with submodules and LFS assets, creates a separate Python
3.11 simulator environment, pins NumPy/SAPIEN-compatible dependencies, and
runs import, action-contract, Vulkan, and headless-render checks. It is
idempotent and does not mix SimplerEnv packages into ThinkFlow's `.venv`.

Common overrides include:

```bash
SIMPLER_ROOT=/workspace/SimplerEnv \
THINKFLOW_ROOT=/workspace/ThinkFlow-RDT-1B \
bash scripts/setup_simpler_env_native_linux.sh
```

Set `SKIP_SYSTEM_PACKAGES=1` only when the system dependencies are already
installed, or `SKIP_RENDER_TEST=1` when preparing an image on a machine that
does not currently expose the final NVIDIA/Vulkan device.

Run the deterministic conversion tests:

```bash
uv run --no-sync python scripts/evaluate_simpler_b0_oxe.py \
  --mode contract --task google_robot_pick_coke_can
```

Run a static end-to-end B0 probe using SimplerEnv's example scene image:

```bash
uv run --no-sync python scripts/evaluate_simpler_b0_oxe.py \
  --mode probe --task google_robot_pick_coke_can \
  --instruction "pick up the coke can"
```

Run a real episode on a native-Linux machine with a Vulkan-capable NVIDIA
driver. No X server or physical display is required when
`--renderer-offscreen` is supplied. `--action-chunk 1` matches SimplerEnv's
normal policy-step cadence; also test `--action-chunk 10` because this RDT was
developed around ten-step execution before replanning.

```bash
RDT_REPO=/path/to/RoboticsDiffusionTransformer \
uv run --no-sync python scripts/evaluate_simpler_b0_oxe.py \
  --mode rollout --task google_robot_pick_coke_can \
  --renderer-offscreen \
  --action-chunk 4 --max-steps 80 \
  --google-gripper-sticky-steps 15 --rotation-scale 0.25 \
  --output-dir output_2/simpler_b0_oxe/google_coke_chunk4_sticky15_rot025
```

`RDT_REPO` overrides a machine-specific `rdt_repo` path from the YAML when
that configured path is unavailable.

Full-fine-tune artifacts such as `output_2/checkpoint-20000` do not require
PEFT at inference time. PEFT is imported only when a LoRA artifact is selected.

The local WSL2 host cannot run the renderer. SAPIEN reports that WSL is not
supported and Vulkan device creation fails with `ErrorExtensionNotPresent`.
The evaluator intentionally initializes the environment before loading the
large policy stack, so this failure is fast and saved to
`environment_error.json` and `simpler_worker.log`.

## B2 five-spatial-token rollout

B2 rollout uses the same LatentStudent procedure as feature precomputation:
six latent reasoning iterations are followed by `</think>` and five learned
spatial tokens, then layer-7 K/V for those five tokens is flattened to
`[5, 2048]`. It therefore requires the LatentStudent checkpoint, its stage-2
Python implementation, and the original Qwen processor—not merely the base
Qwen model.

```bash
RDT_REPO=/workspace/RoboticsDiffusionTransformer \
uv run --no-sync python scripts/evaluate_simpler_b0_oxe.py \
  --mode rollout --task google_robot_pick_coke_can \
  --renderer-offscreen --qwen-extraction b2 \
  --checkpoint output_2/fractal_b2_from_oxe5k_native128_full_10k/checkpoint-5000 \
  --config configs/fractal_b2_from_oxe5k_native128_full.yaml \
  --student-model-id model/LatentStudent-ckpt-400-fixed \
  --processor-id model/model/stage1_unsloth \
  --latent-student-code-dir /workspace/VLA-FYP/train/stage2 \
  --qwen-layer-index 7 --latent-count 6 --spatial-token-count 5 \
  --action-chunk 4 --max-steps 80 \
  --google-gripper-sticky-steps 15 --rotation-scale 0.25 \
  --simpler-root /workspace/SimplerEnv \
  --simpler-python /workspace/SimplerEnv/.venv/bin/python \
  --output-dir output_2/simpler_b2_fractal/google_coke_seed42_chunk4
```

## Full Fractal success-rate evaluation

The suite runner evaluates the five canonical Google Robot task families used
by SimplerEnv's Fractal comparison: pick coke can, move near, open drawer,
close drawer, and place an apple in the closed top drawer. By default it runs
10 deterministic seeds per task (50 rollouts), saves every video and action
trajectory, and reports both pooled (micro) and equal-task-weighted (macro)
success rates. The policy models remain loaded between episodes, and a rerun
reuses completed episode summaries.

```bash
bash scripts/run_simpler_fractal_full_eval.sh
```

Common overrides are environment variables:

```bash
CHECKPOINT=output_2/fractal_b2_from_oxe5k_native128_full_10k/checkpoint-5000 \
EPISODES_PER_TASK=20 ACTION_CHUNK=4 MAX_STEPS=300 \
OUTPUT_DIR=output_2/simpler_b2_fractal/full_eval_checkpoint5000 \
bash scripts/run_simpler_fractal_full_eval.sh
```

The aggregate report is written to `suite_summary.json`; `suite_results.json`
contains every trial. Each task/seed directory contains `rollout.mp4`,
`trajectory.jsonl`, `environment_contract.json`, and `summary.json`.

## Translation diagnostics

Run the controlled chunk-1 versus chunk-4 experiment on identical tasks and
seeds:

```bash
CHECKPOINT=/workspace/ThinkFlow-RDT-1B/b0_oxe_5k_fractal_10k \
bash scripts/run_simpler_fractal_translation_ablation.sh
```

The default is five seeds across the same five Fractal tasks (25 rollouts per
chunk). Set `EPISODES_PER_TASK` to change the budget. Each chunk directory gets
a `translation_diagnostics.json` containing requested-versus-executed clipping,
requested-versus-achieved per-axis controller slopes and correlations,
object-direction cosine, distance progress, and per-episode minimum TCP/object
distance.

Cache-only target analysis does not load the model or simulator:

```bash
uv run --no-sync python scripts/analyze_fractal_translation.py \
  --mode cache \
  --manifest output_2/fractal_b2_from_oxe5k_native128_full_10k/manifests/train_manifest.jsonl \
  --action-stats dataset/mock_dataset/fractal_dataset/audit.json \
  --max-packs 2000 \
  --output output_2/fractal_b2_from_oxe5k_native128_full_10k/fractal_translation_cache_analysis.json
```
