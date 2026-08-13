#!/usr/bin/env bash
set -euo pipefail

# Convert the official LIBERO-finetuned SmolVLA checkpoint into the custom
# Qwen-KV policy class, then evaluate it with external K/V fully bypassed.
# This isolates checkpoint conversion from Qwen extraction and fusion.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

BASE_MODEL=${BASE_MODEL:-lerobot/smolvla_libero}
QWEN_TOKEN_COUNT=${QWEN_TOKEN_COUNT:-5}
CONVERTED_DIR=${CONVERTED_DIR:-outputs/smolvla_libero_qwen_kv_step0_bypass}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/smolvla_libero_qwen_kv_step0_bypass_eval}
LIBERO_ROOT=${LIBERO_ROOT:-"$REPO_ROOT/../LIBERO"}
SUITE=${SUITE:-libero_10}
EPISODES_PER_TASK=${EPISODES_PER_TASK:-20}
ENV_BATCH_SIZE=${ENV_BATCH_SIZE:-2}
ACTION_CHUNK=${ACTION_CHUNK:-4}
MAX_STEPS=${MAX_STEPS:-1000}
SAVE_VIDEOS=${SAVE_VIDEOS:-1}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-0}

SMOLVLA_VENV=${SMOLVLA_VENV:-$REPO_ROOT/.venv-smolvla}
PYTHON="$SMOLVLA_VENV/bin/python"
if [[ ! -x "$PYTHON" ]] || ! "$PYTHON" -c 'import lerobot' >/dev/null 2>&1; then
  echo "LeRobot is not available from $PYTHON" >&2
  echo "Set SMOLVLA_VENV to the environment containing lerobot." >&2
  exit 2
fi
if [[ ! -d "$LIBERO_ROOT/libero/libero" ]]; then
  echo "LIBERO source tree not found at $LIBERO_ROOT" >&2
  exit 2
fi
if [[ "$QWEN_TOKEN_COUNT" != "1" && "$QWEN_TOKEN_COUNT" != "5" ]]; then
  echo "QWEN_TOKEN_COUNT must be 1 or 5" >&2
  exit 2
fi

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export PYTHONPATH="$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

LOCAL_ARGS=()
case "${LOCAL_FILES_ONLY,,}" in
  1|true|yes|on)
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    LOCAL_ARGS+=(--local-files-only)
    ;;
  *)
    export HF_HUB_OFFLINE=0
    export TRANSFORMERS_OFFLINE=0
    ;;
esac

if [[ ! -f "$CONVERTED_DIR/model.safetensors" ]]; then
  "$PYTHON" -m experiments.smolvla_qwen_kv.create_base_checkpoint \
    --base "$BASE_MODEL" \
    --output-dir "$CONVERTED_DIR" \
    --external-kv-token-count "$QWEN_TOKEN_COUNT" \
    --preserve-base-processors \
    --device cpu \
    --seed 42 \
    "${LOCAL_ARGS[@]}"
fi

VIDEO_ARGS=()
case "${SAVE_VIDEOS,,}" in
  1|true|yes|on) VIDEO_ARGS+=(--save-videos) ;;
esac

echo "Evaluating converted smolvla_libero control"
echo "  converted checkpoint: $CONVERTED_DIR"
echo "  suite: $SUITE"
echo "  protocol: 10 tasks x $EPISODES_PER_TASK episodes, max $MAX_STEPS steps"
echo "  Qwen fusion: disabled"

exec "$PYTHON" -m experiments.smolvla_qwen_kv.evaluate_checkpoint \
  --checkpoint "$CONVERTED_DIR" \
  --libero-root "$LIBERO_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --suites "$SUITE" \
  --episodes-per-task "$EPISODES_PER_TASK" \
  --env-batch-size "$ENV_BATCH_SIZE" \
  --action-chunk "$ACTION_CHUNK" \
  --max-steps "$MAX_STEPS" \
  --disable-qwen-fusion \
  "${VIDEO_ARGS[@]}" \
  "${LOCAL_ARGS[@]}"
