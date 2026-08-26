#!/usr/bin/env bash
set -euo pipefail

# Matched short forks from the completed B0 OXE checkpoint. These are
# deliberately weights-only restarts: changing the training objective is not a
# bit-exact resume, and the checkpoint contract correctly refuses to label it
# as one.

CONFIG=${CONFIG:-configs/part3_rdt1b.yaml}
BASE_ARTIFACT=${BASE_ARTIFACT:-output_2/checkpoint-20000}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_2/gripper_experiments}
STEPS=${STEPS:-500}
WARMUP_STEPS=${WARMUP_STEPS:-50}
BCE_WEIGHT=${BCE_WEIGHT:-0.05}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
SIGLIP_MODEL_ID=${SIGLIP_MODEL_ID:-/home/ubuntu/models/siglip-so400m-patch14-384}
SIGLIP_FALLBACK_MODEL_ID=${SIGLIP_FALLBACK_MODEL_ID:-google/siglip-so400m-patch14-384}

CACHE_ROOTS=(
  cache_features/part_1_32frame_per_sample_qwen
  cache_features/part_2_32frame_per_sample_qwen
  cache_features/part_3_32frame_per_sample_qwen
)

CACHE_ARGS=()
for cache_root in "${CACHE_ROOTS[@]}"; do
  CACHE_ARGS+=(--cache-root "$cache_root")
done

run_arm() {
  local arm_name=$1
  shift
  local arm_dir="$OUTPUT_ROOT/$arm_name"
  if [[ -f "$arm_dir/final/metadata.json" ]]; then
    echo "Skipping completed arm: $arm_name"
    return
  fi
  mkdir -p "$arm_dir"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    uv run --no-sync python scripts/train_b0_cached_features.py \
      --config "$CONFIG" \
      --output-dir "$arm_dir" \
      "${CACHE_ARGS[@]}" \
      --online-siglip \
      --siglip-model-id "$SIGLIP_MODEL_ID" \
      --siglip-fallback-model-id "$SIGLIP_FALLBACK_MODEL_ID" \
      --init-artifact "$BASE_ARTIFACT" \
      --max-steps "$STEPS" \
      --warmup-steps "$WARMUP_STEPS" \
      --micro-batch-size "$MICRO_BATCH_SIZE" \
      --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
      --global-batch-size "$GLOBAL_BATCH_SIZE" \
      --learning-rate 1e-4 \
      --validate-every "$STEPS" \
      --save-every 0 \
      --validation-samples 256 \
      --sample-validation-batches 1 \
      --qualitative-validation-examples 0 \
      --skip-nonfinite-updates \
      --log-gradient-stats \
      --report-to none \
      "$@" 2>&1 | tee "$arm_dir/train.log"
}

# The control distinguishes objective changes from the fresh-optimizer restart.
run_arm control_bce0_maskfalse_500 \
  --gripper-bce-weight 0.0 \
  --no-mask-noisy-gripper-input

# This isolates the auxiliary classification objective.
run_arm bce005_maskfalse_500 \
  --gripper-bce-weight "$BCE_WEIGHT" \
  --no-mask-noisy-gripper-input

# This tests the requested combination: BCE supervision plus no noisy-gripper
# shortcut in both training and diffusion sampling.
run_arm bce005_masktrue_500 \
  --gripper-bce-weight "$BCE_WEIGHT" \
  --mask-noisy-gripper-input
