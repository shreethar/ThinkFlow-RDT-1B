#!/usr/bin/env bash
set -euo pipefail

# Fast B2/B3 evaluation through LeRobot's official LIBERO eval_policy_all loop.
# Qwen fusion is enabled only; this does not run the disabled ablation branch.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

CHECKPOINT=${1:-outputs/lerobot_smolvla_libero_qwen_kv_staged_b2/checkpoints/last}
if [[ $# -gt 0 ]]; then
  shift
fi
CACHE_ROOT=${CACHE_ROOT:-cache_features_libero_b2_native}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/lerobot_smolvla_libero_qwen_kv_staged_b2/official_eval}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
TASK_IDS=${TASK_IDS:-"0 1 2 3 4 5 6 7 8 9"}
EPISODES=${EPISODES:-5}
# Episodes of the current task are evaluated together. Raise this only if the
# combined LatentStudent + SmolVLA inference comfortably fits in GPU memory.
BATCH_SIZE=${BATCH_SIZE:-5}
N_ACTION_STEPS=${N_ACTION_STEPS:-10}
MAX_VIDEOS=${MAX_VIDEOS:-0}
DEVICE=${DEVICE:-cuda}
LATENT_STUDENT_CODE_DIR=${LATENT_STUDENT_CODE_DIR:-/workspace/VLA-FYP/train/stage2}
LATENT_ATTN_IMPLEMENTATION=${LATENT_ATTN_IMPLEMENTATION:-sdpa}
LATENT_STUDENT_PRECISION=${LATENT_STUDENT_PRECISION:-bf16}

if [[ "$DEVICE" != cuda* ]]; then
  echo "Warning: DEVICE=$DEVICE; this launcher is intended to run on CUDA." >&2
fi

echo "Official LeRobot B2/B3 evaluation:"
echo "  checkpoint: $CHECKPOINT"
echo "  cache metadata: $CACHE_ROOT"
echo "  output: $OUTPUT_DIR"
echo "  suites: $SUITES"
echo "  tasks: $TASK_IDS"
echo "  episodes per task: $EPISODES"
echo "  parallel environments: $BATCH_SIZE"
echo "  executed actions per plan: $N_ACTION_STEPS"
echo "  device: $DEVICE"
echo "  LatentStudent precision: $LATENT_STUDENT_PRECISION"

CHECKPOINT="$CHECKPOINT" \
CACHE_ROOT="$CACHE_ROOT" \
OUTPUT_DIR="$OUTPUT_DIR" \
SUITES="$SUITES" \
TASK_IDS="$TASK_IDS" \
EPISODES="$EPISODES" \
BATCH_SIZE="$BATCH_SIZE" \
N_ACTION_STEPS="$N_ACTION_STEPS" \
MAX_VIDEOS="$MAX_VIDEOS" \
bash experiments/smolvla_qwen_kv/evaluate_fusion_ablation.sh \
  --modes enabled \
  --device "$DEVICE" \
  --latent-student-code-dir "$LATENT_STUDENT_CODE_DIR" \
  --latent-student-attn-implementation "$LATENT_ATTN_IMPLEMENTATION" \
  --latent-student-precision "$LATENT_STUDENT_PRECISION" \
  "$@"
