#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LIBERO_ROOT=${SMOLVLA_QWEN_EVAL_LIBERO_ROOT:-"$REPO_ROOT/../LIBERO"}
if [[ ! -d "$LIBERO_ROOT/libero/libero" ]]; then
  echo "LIBERO source tree not found at: $LIBERO_ROOT" >&2
  echo "Set SMOLVLA_QWEN_EVAL_LIBERO_ROOT=/absolute/path/to/LIBERO" >&2
  exit 2
fi

# Prefer the isolated SmolVLA environment created by
# scripts/setup_smolvla_libero_env.sh, while remaining compatible with older
# workspaces that installed LeRobot into .venv.
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
  echo "Run: bash scripts/setup_smolvla_libero_env.sh" >&2
  echo "Or set SMOLVLA_VENV=/absolute/path/to/a/venv containing lerobot." >&2
  exit 2
fi
if ! "$SMOLVLA_VENV/bin/python" -c 'import lerobot' >/dev/null 2>&1; then
  echo "LeRobot is not importable from $SMOLVLA_VENV/bin/python" >&2
  exit 2
fi
if [[ ! -x "$SMOLVLA_VENV/bin/lerobot-train" ]]; then
  echo "Missing $SMOLVLA_VENV/bin/lerobot-train" >&2
  exit 2
fi
PYTHON="$SMOLVLA_VENV/bin/python"
LEROBOT_TRAIN="$SMOLVLA_VENV/bin/lerobot-train"

# Respect caller-provided online/offline settings. The bootstrap needs Hub
# access on a fresh machine, while subsequent runs can opt into offline mode.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-$HF_HUB_OFFLINE}
export ACCELERATE_USE_DEEPSPEED=false
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export SMOLVLA_QWEN_EVAL_ENABLE=${SMOLVLA_QWEN_EVAL_ENABLE:-true}
export SMOLVLA_QWEN_EVAL_LIBERO_ROOT="$LIBERO_ROOT"
# The local LIBERO repository has a nested libero/libero package, so its
# repository root must be present on sys.path for `import libero.libero`.
export PYTHONPATH="$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SMOLVLA_QWEN_EVAL_TASK_IDS=${SMOLVLA_QWEN_EVAL_TASK_IDS:-0}
export SMOLVLA_QWEN_EVAL_EPISODES_PER_TASK=${SMOLVLA_QWEN_EVAL_EPISODES_PER_TASK:-2}
export SMOLVLA_QWEN_EVAL_ACTION_CHUNK=${SMOLVLA_QWEN_EVAL_ACTION_CHUNK:-4}
export SMOLVLA_QWEN_EVAL_SAVE_VIDEOS=${SMOLVLA_QWEN_EVAL_SAVE_VIDEOS:-true}

if [[ "$SMOLVLA_QWEN_EVAL_ENABLE" == "true" ]] && ! ldconfig -p 2>/dev/null | grep -q 'libOpenGL\.so\.0'; then
  echo "Missing system library libOpenGL.so.0 required by LIBERO EGL rollout." >&2
  echo "Install it with: apt-get update && apt-get install -y libopengl0 libegl1 libgl1" >&2
  exit 2
fi

MODE=${1:-}
case "$MODE" in
  "") QWEN_TOKEN_COUNT=${QWEN_TOKEN_COUNT:-1} ;;
  b0) QWEN_TOKEN_COUNT=1 ;;
  b2) QWEN_TOKEN_COUNT=5 ;;
  *)
    echo "Usage: $0 [b0|b2]" >&2
    echo "Alternatively export QWEN_TOKEN_COUNT=1 or 5." >&2
    exit 2
    ;;
esac
if [[ "$QWEN_TOKEN_COUNT" != "1" && "$QWEN_TOKEN_COUNT" != "5" ]]; then
  echo "QWEN_TOKEN_COUNT must be 1 (B0) or 5 (B2), got: $QWEN_TOKEN_COUNT" >&2
  exit 2
fi

if [[ "$QWEN_TOKEN_COUNT" == "1" ]]; then
  DEFAULT_CACHE_ROOT=cache_features_libero_b0_raw_ortho6d
  DEFAULT_BOOTSTRAP_DIR=outputs/smolvla_base_qwen_kv_init
  DEFAULT_STATS_PATH=outputs/smolvla_base_qwen_kv_all_suites/cache_stats.pt
  DEFAULT_OUTPUT_DIR=outputs/lerobot_smolvla_qwen_kv_fresh
  DEFAULT_JOB_NAME=smolvla-base-qwen-kv-fresh
  DEFAULT_WANDB_PROJECT=thinkflow-smolvla-b0-libero
else
  DEFAULT_CACHE_ROOT=cache_features_libero_b2_native
  DEFAULT_BOOTSTRAP_DIR=outputs/smolvla_base_qwen_kv_init_b2
  DEFAULT_STATS_PATH=outputs/smolvla_b2_native_cache_stats.pt
  DEFAULT_OUTPUT_DIR=outputs/lerobot_smolvla_qwen_kv_b2
  DEFAULT_JOB_NAME=smolvla-base-qwen-kv-b2
  DEFAULT_WANDB_PROJECT=thinkflow-smolvla-b2-libero
  export SMOLVLA_QWEN_EVAL_LATENT_STUDENT_CODE_DIR=${SMOLVLA_QWEN_EVAL_LATENT_STUDENT_CODE_DIR:-/workspace/VLA-FYP/train/stage2}
fi

if [[ "$QWEN_TOKEN_COUNT" == "5" && "$SMOLVLA_QWEN_EVAL_ENABLE" == "true" ]]; then
  if ! "$PYTHON" -c 'import peft' >/dev/null 2>&1; then
    echo "B2 live evaluation requires PEFT because LatentStudent imports it." >&2
    echo "Install it with:" >&2
    echo "  uv pip install --python $SMOLVLA_VENV/bin/python 'peft==0.19.1'" >&2
    exit 2
  fi
fi

CACHE_ROOT=${CACHE_ROOT:-$DEFAULT_CACHE_ROOT}
BASE_MODEL=${BASE_MODEL:-lerobot/smolvla_base}
BOOTSTRAP_DIR=${BOOTSTRAP_DIR:-$DEFAULT_BOOTSTRAP_DIR}
STATS_PATH=${STATS_PATH:-$DEFAULT_STATS_PATH}
OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}
JOB_NAME=${JOB_NAME:-$DEFAULT_JOB_NAME}
WANDB_PROJECT=${WANDB_PROJECT:-$DEFAULT_WANDB_PROJECT}
STEPS=${STEPS:-100000}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-8}
SAVE_FREQ=${SAVE_FREQ:-1000}
ENV_EVAL_FREQ=${ENV_EVAL_FREQ:-5000}
N_ACTION_STEPS=${N_ACTION_STEPS:-}
EXTERNAL_LOGIT_BIAS_INIT=${EXTERNAL_LOGIT_BIAS_INIT:--1.0}
EXTERNAL_KV_RANKING_WEIGHT=${EXTERNAL_KV_RANKING_WEIGHT:-0.1}
EXTERNAL_KV_RANKING_MARGIN=${EXTERNAL_KV_RANKING_MARGIN:-0.01}

echo "SmolVLA Qwen-KV training configuration:"
echo "  mode: $([[ "$QWEN_TOKEN_COUNT" == "5" ]] && echo B2 || echo B0)"
echo "  Qwen KV tokens: $QWEN_TOKEN_COUNT"
echo "  source model: $BASE_MODEL"
echo "  cache root: $CACHE_ROOT"
echo "  bootstrap: $BOOTSTRAP_DIR"
echo "  output: $OUTPUT_DIR"
echo "  external logit bias init: $EXTERNAL_LOGIT_BIAS_INIT"
echo "  fusion ranking weight: $EXTERNAL_KV_RANKING_WEIGHT"
echo "  fusion ranking margin: $EXTERNAL_KV_RANKING_MARGIN"

LOCAL_FILES_ARGS=()
case "${HF_HUB_OFFLINE,,}" in
  1|true|yes|on) LOCAL_FILES_ARGS+=(--local-files-only) ;;
esac

ACTION_STEP_ARGS=()
if [[ -n "$N_ACTION_STEPS" ]]; then
  ACTION_STEP_ARGS+=(--n-action-steps "$N_ACTION_STEPS")
fi

# Build the custom policy once from the requested native SmolVLA checkpoint.
# Only the external Qwen K/V adapters are absent from the source checkpoint and
# therefore newly initialized.
if [[ ! -f "$BOOTSTRAP_DIR/model.safetensors" ]]; then
  "$PYTHON" -m experiments.smolvla_qwen_kv.create_base_checkpoint \
    --base "$BASE_MODEL" \
    --output-dir "$BOOTSTRAP_DIR" \
    --stats "$STATS_PATH" \
    --cache-root "$CACHE_ROOT" \
    --external-kv-token-count "$QWEN_TOKEN_COUNT" \
    --external-logit-bias-init "$EXTERNAL_LOGIT_BIAS_INIT" \
    --external-kv-ranking-weight "$EXTERNAL_KV_RANKING_WEIGHT" \
    --external-kv-ranking-margin "$EXTERNAL_KV_RANKING_MARGIN" \
    --device cpu \
    --seed 42 \
    "${ACTION_STEP_ARGS[@]}" \
    "${LOCAL_FILES_ARGS[@]}"
fi

exec "$LEROBOT_TRAIN" \
  --policy.path="$BOOTSTRAP_DIR" \
  --policy.load_vlm_weights=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=cached_libero_qwen:all \
  --dataset.root="$CACHE_ROOT" \
  --dataset.streaming=true \
  --dataset.eval_split=0 \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --log_freq=10 \
  --save_freq="$SAVE_FREQ" \
  --env_eval_freq="$ENV_EVAL_FREQ" \
  --eval_steps=0 \
  --wandb.enable=true \
  --wandb.project="$WANDB_PROJECT"
