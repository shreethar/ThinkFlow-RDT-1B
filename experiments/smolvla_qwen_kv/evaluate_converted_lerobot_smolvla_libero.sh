#!/usr/bin/env bash
set -euo pipefail

# Convert lerobot/smolvla_libero into the custom Qwen-KV class, then evaluate
# it with qwen_kv absent. This tests whether the architecture override alone
# preserves the source checkpoint's behavior before any training.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

VENV=${VENV:-$REPO_ROOT/.venv-smolvla}
BASE_MODEL=${BASE_MODEL:-lerobot/smolvla_libero}
QWEN_TOKEN_COUNT=${QWEN_TOKEN_COUNT:-5}
CONVERTED_DIR=${CONVERTED_DIR:-outputs/lerobot_smolvla_libero_qwen_kv_untrained}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/lerobot_smolvla_libero_qwen_kv_untrained_eval}
SUITES=${SUITES:-libero_spatial}
TASK_IDS=${TASK_IDS:-'[0]'}
EPISODES=${EPISODES:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-0}

PYTHON="$VENV/bin/python"
if [[ ! -x "$PYTHON" ]] || ! "$PYTHON" -c 'import lerobot' >/dev/null 2>&1; then
  echo "LeRobot is not available from $PYTHON" >&2
  exit 2
fi

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
    --external-kv-optional \
    --preserve-base-processors \
    --device cpu \
    --seed 42 \
    "${LOCAL_ARGS[@]}"
fi

echo "Converted step-zero control:"
echo "  source: $BASE_MODEL"
echo "  converted: $CONVERTED_DIR"
echo "  Qwen fusion: bypassed (qwen_kv absent)"

exec env \
  VENV="$VENV" \
  POLICY="$CONVERTED_DIR" \
  SUITES="$SUITES" \
  TASK_IDS="$TASK_IDS" \
  EPISODES="$EPISODES" \
  BATCH_SIZE="$BATCH_SIZE" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  bash scripts/evaluate_lerobot_smolvla_libero.sh
