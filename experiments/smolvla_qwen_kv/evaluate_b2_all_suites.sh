#!/usr/bin/env bash
set -euo pipefail

# Full B2 closed-loop evaluation:
#   4 LIBERO suites x 10 tasks x 20 initial states = 800 episodes
#   Each episode is capped at 1,000 simulator steps.
#
# Usage:
#   bash experiments/smolvla_qwen_kv/evaluate_b2_all_suites.sh
#   bash experiments/smolvla_qwen_kv/evaluate_b2_all_suites.sh \
#     outputs/lerobot_smolvla_qwen_kv_b2/checkpoints/050000
#
# Useful overrides:
#   CACHE_ROOT=/path/to/cache_features_libero_b2_native
#   LIBERO_ROOT=/workspace/LIBERO
#   EPISODES_PER_TASK=20
#   MAX_STEPS=1000
#   ENV_BATCH_SIZE=2
#   ACTION_CHUNK=4
#   SAVE_VIDEOS=0
#   LOCAL_FILES_ONLY=1

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

CHECKPOINT=${1:-outputs/lerobot_smolvla_qwen_kv_b2/checkpoints/last}
CACHE_ROOT=${CACHE_ROOT:-cache_features_libero_b2_native}
LIBERO_ROOT=${LIBERO_ROOT:-"$REPO_ROOT/../LIBERO"}
EPISODES_PER_TASK=${EPISODES_PER_TASK:-20}
MAX_STEPS=${MAX_STEPS:-1000}
ENV_BATCH_SIZE=${ENV_BATCH_SIZE:-2}
ACTION_CHUNK=${ACTION_CHUNK:-4}
SAVE_VIDEOS=${SAVE_VIDEOS:-0}
VIDEO_RESOLUTION=${VIDEO_RESOLUTION:-512}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-1}
LATENT_STUDENT_CODE_DIR=${LATENT_STUDENT_CODE_DIR:-/workspace/VLA-FYP/train/stage2}
LATENT_ATTN_IMPLEMENTATION=${LATENT_ATTN_IMPLEMENTATION:-sdpa}
LATENT_STUDENT_PRECISION=${LATENT_STUDENT_PRECISION:-bf16}

SMOLVLA_VENV=${SMOLVLA_VENV:-}
if [[ -z "$SMOLVLA_VENV" ]]; then
  for candidate in "$REPO_ROOT/.venv-smolvla" "$REPO_ROOT/.venv"; do
    if [[ -x "$candidate/bin/python" ]] && "$candidate/bin/python" -c 'import lerobot' >/dev/null 2>&1; then
      SMOLVLA_VENV="$candidate"
      break
    fi
  done
fi
if [[ -z "$SMOLVLA_VENV" || ! -x "$SMOLVLA_VENV/bin/python" ]]; then
  echo "No LeRobot-capable virtual environment found." >&2
  echo "Set SMOLVLA_VENV or run scripts/setup_smolvla_libero_env.sh." >&2
  exit 2
fi
PYTHON="$SMOLVLA_VENV/bin/python"

if [[ ! -d "$LIBERO_ROOT/libero/libero" ]]; then
  echo "LIBERO source tree not found at $LIBERO_ROOT" >&2
  exit 2
fi
if [[ ! -d "$CACHE_ROOT" ]]; then
  echo "B2 cache root not found at $CACHE_ROOT" >&2
  exit 2
fi

for suite in libero_10 libero_spatial libero_goal libero_object; do
  if [[ ! -f "$CACHE_ROOT/$suite/precompute_metadata.json" ]]; then
    echo "Missing B2 cache metadata: $CACHE_ROOT/$suite/precompute_metadata.json" >&2
    exit 2
  fi
done

if ! "$PYTHON" -c 'import peft' >/dev/null 2>&1; then
  echo "B2 evaluation requires PEFT for LatentStudent loading." >&2
  echo "Install: uv pip install --python $SMOLVLA_VENV/bin/python 'peft==0.19.1'" >&2
  exit 2
fi

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export PYTHONPATH="$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

case "${LOCAL_FILES_ONLY,,}" in
  1|true|yes|on)
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    LOCAL_ARGS=(--local-files-only)
    ;;
  *)
    export HF_HUB_OFFLINE=0
    export TRANSFORMERS_OFFLINE=0
    LOCAL_ARGS=()
    ;;
esac

CHECKPOINT_PATH=$(readlink -f "$CHECKPOINT")
if [[ ! -e "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint does not exist: $CHECKPOINT" >&2
  exit 2
fi
CHECKPOINT_NAME=$(basename "$CHECKPOINT_PATH")
if [[ "$CHECKPOINT_NAME" == "pretrained_model" ]]; then
  CHECKPOINT_NAME=$(basename "$(dirname "$CHECKPOINT_PATH")")
fi
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/lerobot_smolvla_qwen_kv_b2/evaluation_${CHECKPOINT_NAME}_all_suites_${EPISODES_PER_TASK}x${MAX_STEPS}"}

VIDEO_ARGS=()
case "${SAVE_VIDEOS,,}" in
  1|true|yes|on)
    VIDEO_ARGS+=(--save-videos --video-resolution "$VIDEO_RESOLUTION")
    ;;
esac

echo "B2 full LIBERO evaluation:"
echo "  checkpoint: $CHECKPOINT_PATH"
echo "  cache root: $CACHE_ROOT"
echo "  output: $OUTPUT_DIR"
echo "  protocol: 4 suites, 10 tasks/suite, $EPISODES_PER_TASK episodes/task"
echo "  max simulator steps: $MAX_STEPS"
echo "  environment batch: $ENV_BATCH_SIZE"
echo "  videos: $SAVE_VIDEOS"

exec "$PYTHON" -m experiments.smolvla_qwen_kv.evaluate_checkpoint \
  --checkpoint "$CHECKPOINT_PATH" \
  --cache-root "$CACHE_ROOT" \
  --libero-root "$LIBERO_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --suites libero_10 libero_spatial libero_goal libero_object \
  --episodes-per-task "$EPISODES_PER_TASK" \
  --env-batch-size "$ENV_BATCH_SIZE" \
  --action-chunk "$ACTION_CHUNK" \
  --max-steps "$MAX_STEPS" \
  --latent-student-code-dir "$LATENT_STUDENT_CODE_DIR" \
  --latent-student-attn-implementation "$LATENT_ATTN_IMPLEMENTATION" \
  --latent-student-precision "$LATENT_STUDENT_PRECISION" \
  "${VIDEO_ARGS[@]}" \
  "${LOCAL_ARGS[@]}"
