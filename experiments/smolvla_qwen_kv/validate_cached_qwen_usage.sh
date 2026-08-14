#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

MODE=${1:-}
if [[ "$MODE" != "b0" && "$MODE" != "b2" ]]; then
  echo "Usage: $0 {b0|b2} [extra validate_cached_qwen_usage.py arguments...]" >&2
  exit 2
fi
shift

PYTHON_BIN=${PYTHON_BIN:-$REPO_ROOT/.venv-smolvla/bin/python}
SUITES=${SUITES:-"libero_10 libero_spatial libero_goal libero_object"}
NUM_SAMPLES=${NUM_SAMPLES:-256}
BATCH_SIZE=${BATCH_SIZE:-8}
PREDICTION_SAMPLES=${PREDICTION_SAMPLES:-32}

if [[ "$MODE" == "b0" ]]; then
  CHECKPOINT=${CHECKPOINT:-outputs/lerobot_smolvla_libero_qwen_kv_fusion_rank_b0/checkpoints/last}
  CACHE_ROOT=${CACHE_ROOT:-cache_features_libero_b0_raw_ortho6d}
  REFERENCE_CHECKPOINT=${REFERENCE_CHECKPOINT:-outputs/smolvla_libero_qwen_kv_fusion_rank_init_b0}
  OUTPUT=${OUTPUT:-outputs/validation_cached_qwen_usage_b0.json}
else
  CHECKPOINT=${CHECKPOINT:-outputs/lerobot_smolvla_libero_qwen_kv_fusion_rank_b2/checkpoints/last}
  CACHE_ROOT=${CACHE_ROOT:-cache_features_libero_b2_native}
  REFERENCE_CHECKPOINT=${REFERENCE_CHECKPOINT:-outputs/smolvla_libero_qwen_kv_fusion_rank_init_b2}
  OUTPUT=${OUTPUT:-outputs/validation_cached_qwen_usage_b2.json}
fi

REFERENCE_ARGS=()
if [[ -n "$REFERENCE_CHECKPOINT" && -d "$REFERENCE_CHECKPOINT" ]]; then
  REFERENCE_ARGS=(--reference-checkpoint "$REFERENCE_CHECKPOINT")
fi

# shellcheck disable=SC2206
SUITE_ARGS=($SUITES)

exec "$PYTHON_BIN" -m experiments.smolvla_qwen_kv.validate_cached_qwen_usage \
  --checkpoint "$CHECKPOINT" \
  --cache-root "$CACHE_ROOT" \
  --suites "${SUITE_ARGS[@]}" \
  --num-samples "$NUM_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --prediction-samples "$PREDICTION_SAMPLES" \
  --output "$OUTPUT" \
  "${REFERENCE_ARGS[@]}" \
  "$@"
