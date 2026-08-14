#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV=${SMOLVLA_VENV:-$REPO_ROOT/.venv-smolvla}
PYTHON=${PYTHON:-$VENV/bin/python}
LIBERO_ROOT=${LIBERO_ROOT:-$REPO_ROOT/../LIBERO}

CHECKPOINT=${CHECKPOINT:-outputs/lerobot_smolvla_qwen_kv_fresh/checkpoints/last}
CACHE_ROOT=${CACHE_ROOT:-cache_features_libero_b0_raw_ortho6d}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/lerobot_smolvla_qwen_kv_fresh/fusion_ablation}
SUITES=${SUITES:-libero_10}
TASK_IDS=${TASK_IDS:-0}
EPISODES=${EPISODES:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_VIDEOS=${MAX_VIDEOS:-2}
N_ACTION_STEPS=${N_ACTION_STEPS:-}

if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found: $PYTHON" >&2
  exit 2
fi

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export PYTHONPATH="$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

read -r -a SUITE_ARGS <<< "${SUITES//,/ }"
read -r -a TASK_ARGS <<< "${TASK_IDS//,/ }"

EXTRA_ARGS=()
if [[ -n "$N_ACTION_STEPS" ]]; then
  EXTRA_ARGS+=(--n-action-steps "$N_ACTION_STEPS")
fi

exec "$PYTHON" -m experiments.smolvla_qwen_kv.evaluate_fusion_ablation \
  --checkpoint "$CHECKPOINT" \
  --cache-root "$CACHE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --suites "${SUITE_ARGS[@]}" \
  --task-ids "${TASK_ARGS[@]}" \
  --episodes-per-task "$EPISODES" \
  --batch-size "$BATCH_SIZE" \
  --max-videos "$MAX_VIDEOS" \
  "${EXTRA_ARGS[@]}" \
  "$@"
