#!/usr/bin/env bash
set -euo pipefail

# Canonical SimplerEnv Google/Fractal evaluation: five task families x ten seeds.
# The Python evaluator retains T5, Qwen, SigLIP, and RDT in memory across all
# episodes and writes resumable per-episode videos, trajectories, and summaries.
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RDT_REPO=${RDT_REPO:-/workspace/RoboticsDiffusionTransformer}
SIMPLER_ROOT=${SIMPLER_ROOT:-/workspace/SimplerEnv}
SIMPLER_PYTHON=${SIMPLER_PYTHON:-$SIMPLER_ROOT/.venv/bin/python}
CHECKPOINT=${CHECKPOINT:-output_2/fractal_b2_from_oxe5k_native128_full_10k/checkpoint-10000}
CONFIG=${CONFIG:-configs/fractal_b2_from_oxe5k_native128_full.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-output_2/simpler_b2_fractal/full_eval_checkpoint10000}
STUDENT_MODEL_ID=${STUDENT_MODEL_ID:-shreethar/LatentStudent-ckpt-400-fixed}
PROCESSOR_ID=${PROCESSOR_ID:-shreethar/stage1_unsloth}
LATENT_STUDENT_CODE_DIR=${LATENT_STUDENT_CODE_DIR:-/workspace/VLA-FYP/train/stage2}
EPISODES_PER_TASK=${EPISODES_PER_TASK:-10}
BASE_SEED=${BASE_SEED:-42}
ACTION_CHUNK=${ACTION_CHUNK:-4}
MAX_STEPS=${MAX_STEPS:-300}
GRIPPER_STICKY_STEPS=${GRIPPER_STICKY_STEPS:-15}
ROTATION_SCALE=${ROTATION_SCALE:-0.25}

if [[ -z "${VK_ICD_FILENAMES:-}" ]]; then
  VK_ICD_FILENAMES=$(find \
    /etc/vulkan/icd.d \
    /usr/share/vulkan/icd.d \
    /usr/local/share/vulkan/icd.d \
    -type f -iname '*nvidia*.json' -print -quit 2>/dev/null || true)
fi
[[ -n "${VK_ICD_FILENAMES:-}" ]] || {
  echo "ERROR: NVIDIA Vulkan ICD was not found; set VK_ICD_FILENAMES explicitly." >&2
  exit 1
}
export VK_ICD_FILENAMES
export DISPLAY=''
export RDT_REPO

cd "$REPO_ROOT"
uv run --no-sync python scripts/evaluate_simpler_b0_oxe.py \
  --mode suite \
  --renderer-offscreen \
  --dataset fractal \
  --qwen-extraction b2 \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --student-model-id "$STUDENT_MODEL_ID" \
  --processor-id "$PROCESSOR_ID" \
  --latent-student-code-dir "$LATENT_STUDENT_CODE_DIR" \
  --qwen-layer-index 7 \
  --latent-count 6 \
  --spatial-token-count 5 \
  --episodes-per-task "$EPISODES_PER_TASK" \
  --seed "$BASE_SEED" \
  --action-chunk "$ACTION_CHUNK" \
  --max-steps "$MAX_STEPS" \
  --google-gripper-sticky-steps "$GRIPPER_STICKY_STEPS" \
  --rotation-scale "$ROTATION_SCALE" \
  --simpler-root "$SIMPLER_ROOT" \
  --simpler-python "$SIMPLER_PYTHON" \
  --output-dir "$OUTPUT_DIR"
