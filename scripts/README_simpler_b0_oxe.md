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
  (`+1=close`) and used directly for WidowX (`+1=open`).

Every closed-loop step records raw native output, decoded and executed action,
TCP state before/after, achieved TCP displacement, joint state, gripper state,
object poses, reward, termination, simulator info, and policy timings.

## Commands

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
  --action-chunk 10 --max-steps 80 \
  --output-dir output_2/simpler_b0_oxe/google_coke_chunk10
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
