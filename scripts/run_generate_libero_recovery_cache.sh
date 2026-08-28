#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/libero_b0_native128_recovery_continue.yaml}
LIBERO_ROOT=${LIBERO_ROOT:-/home/ubuntu/LIBERO}
DATASET_ROOT=${DATASET_ROOT:-libero-dataset}
OUTPUT_DIR=${OUTPUT_DIR:-cache_features_libero_b0_recovery}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-1024}
VALIDATION_SAMPLES=${VALIDATION_SAMPLES:-128}
TRAIN_REPEAT=${TRAIN_REPEAT:-64}
BATCH_SIZE=${BATCH_SIZE:-32}
QWEN_MODEL_ID=${QWEN_MODEL_ID:-model/model/stage1_unsloth}
QWEN_PROCESSOR_ID=${QWEN_PROCESSOR_ID:-}
T5_MODEL_ID=${T5_MODEL_ID:-/home/ubuntu/RoboticsDiffusionTransformer/google/t5-v1_1-xxl}
T5_FALLBACK_MODEL_ID=${T5_FALLBACK_MODEL_ID:-google/t5-v1_1-xxl}
REUSE_T5_CACHE_ROOT=${REUSE_T5_CACHE_ROOT:-cache_features_libero_b0_raw_ortho6d}
IMAGE_CODEC=${IMAGE_CODEC:-png}

args=(
  --config "$CONFIG"
  --libero-root "$LIBERO_ROOT"
  --dataset-root "$DATASET_ROOT"
  --output-dir "$OUTPUT_DIR"
  --train-samples "$TRAIN_SAMPLES"
  --validation-samples "$VALIDATION_SAMPLES"
  --train-repeat "$TRAIN_REPEAT"
  --batch-size "$BATCH_SIZE"
  --image-codec "$IMAGE_CODEC"
  --qwen-model-id "$QWEN_MODEL_ID"
  --t5-model-id "$T5_MODEL_ID"
  --t5-fallback-model-id "$T5_FALLBACK_MODEL_ID"
  --reuse-t5-cache-root "$REUSE_T5_CACHE_ROOT"
)

if [[ -n "$QWEN_PROCESSOR_ID" ]]; then
  args+=(--qwen-processor-id "$QWEN_PROCESSOR_ID")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi
if [[ "${LOAD_T5_ONLINE:-0}" == "1" ]]; then
  args+=(--load-t5-online)
fi

exec uv run --no-sync python scripts/generate_libero_recovery_cache.py "${args[@]}"
