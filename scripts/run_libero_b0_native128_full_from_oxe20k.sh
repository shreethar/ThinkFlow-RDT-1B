#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=${CACHE_ROOT:-cache_features_libero_b0_raw_ortho6d}
CONFIG=${CONFIG:-configs/libero_b0_native128_full.yaml}
INIT_ARTIFACT=${INIT_ARTIFACT:-output_2/checkpoint-20000}
OUTPUT_DIR=${OUTPUT_DIR:-output_2/libero_b0_from_oxe20k}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}

MAX_STEPS=${MAX_STEPS:-20000}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE:-32}
VALIDATION_SAMPLES=${VALIDATION_SAMPLES:-256}
QUALITATIVE_VALIDATION_EXAMPLES=${QUALITATIVE_VALIDATION_EXAMPLES:-32}
VALIDATE_EVERY=${VALIDATE_EVERY:-500}
SAVE_EVERY=${SAVE_EVERY:-1000}
NUM_WORKERS=${NUM_WORKERS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-ThinkLite B0 LIBERO}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-libero-b0-native128-from-oxe20k-full}
SIGLIP_MODEL_ID=${SIGLIP_MODEL_ID:-/home/ubuntu/models/siglip-so400m-patch14-384}
SIGLIP_FALLBACK_MODEL_ID=${SIGLIP_FALLBACK_MODEL_ID:-google/siglip-so400m-patch14-384}
# A 200-rollout job needs a GPU separate from active training.  Set
# ROLLOUT_EVAL=1 ROLLOUT_GPU=<spare gpu id> to start the watcher below.
ROLLOUT_EVAL=${ROLLOUT_EVAL:-0}
ROLLOUT_GPU=${ROLLOUT_GPU:-}
ROLLOUT_SAVE_VIDEOS=${ROLLOUT_SAVE_VIDEOS:-0}

CACHE_ARGS=()
for suite in $SUITES; do
  suite_root="$CACHE_ROOT/$suite"
  for split in train validation; do
    if [[ ! -f "$suite_root/$split/manifest.jsonl" ]]; then
      echo "Missing manifest: $suite_root/$split/manifest.jsonl" >&2
      exit 1
    fi
  done
  CACHE_ARGS+=(--cache-root "$suite_root")
done

if [[ ! -f "$INIT_ARTIFACT/rdt_full.pt" ]] || [[ ! -f "$INIT_ARTIFACT/interfaces.pt" ]]; then
  echo "Incomplete OXE initialization artifact: $INIT_ARTIFACT" >&2
  exit 1
fi

if [[ "$ROLLOUT_EVAL" == "1" ]]; then
  if [[ -z "$ROLLOUT_GPU" ]]; then
    echo "ROLLOUT_EVAL=1 requires ROLLOUT_GPU to avoid contending with training" >&2
    exit 2
  fi
  mkdir -p "$OUTPUT_DIR/rollout_evaluations"
  OUTPUT_DIR="$OUTPUT_DIR" \
    MAX_STEPS="$MAX_STEPS" \
    EVAL_EVERY=2000 \
    ROLLOUT_GPU="$ROLLOUT_GPU" \
    CONFIG="$CONFIG" \
    CACHE_PARENT="$CACHE_ROOT" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_RUN_NAME="$WANDB_RUN_NAME" \
    SAVE_VIDEOS="$ROLLOUT_SAVE_VIDEOS" \
    bash scripts/watch_libero_rollout_evaluations.sh \
      >"$OUTPUT_DIR/rollout_evaluations/watcher.log" 2>&1 &
  echo "Started 2,000-step rollout watcher on GPU $ROLLOUT_GPU (PID $!)"
fi

uv run --no-sync python scripts/train_b0_cached_features.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  "${CACHE_ARGS[@]}" \
  --init-artifact "$INIT_ARTIFACT" \
  --online-siglip \
  --siglip-model-id "$SIGLIP_MODEL_ID" \
  --siglip-fallback-model-id "$SIGLIP_FALLBACK_MODEL_ID" \
  --max-steps "$MAX_STEPS" \
  --micro-batch-size "$MICRO_BATCH_SIZE" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --validation-batch-size "$VALIDATION_BATCH_SIZE" \
  --validation-samples "$VALIDATION_SAMPLES" \
  --sample-validation-batches 1 \
  --qualitative-validation-examples "$QUALITATIVE_VALIDATION_EXAMPLES" \
  --validate-every "$VALIDATE_EVERY" \
  --save-every "$SAVE_EVERY" \
  --mask-noisy-gripper-input \
  --gripper-bce-weight 1.0 \
  --gripper-bce-logit-scale 5.0 \
  --rotation-geodesic-weight 1.0 \
  --num-workers "$NUM_WORKERS" \
  --pin-memory \
  --persistent-workers \
  --report-to wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$WANDB_RUN_NAME"
