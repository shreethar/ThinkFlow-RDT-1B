#!/usr/bin/env bash
set -euo pipefail

# Extract B0/Qwen </think> cached features for one LIBERO suite.
#
# Usage:
#   scripts/run_libero_b0_feature_extraction.sh libero_object
#
# Common overrides:
#   DATA_ROOT=dataset/datasets
#   OUTPUT_ROOT=cache_features_libero_b0_absolute
#   GCS_DEST=gs://my-bucket/libero_b0_absolute
#   ALL_SAMPLES_PER_EPISODE=1
#   MAX_SAMPLES_PER_EPISODE=128  # used only when ALL_SAMPLES_PER_EPISODE=0
#   OVERWRITE=1

SUITE=${1:?suite is required, e.g. libero_object/libero_spatial/libero_goal/libero_10/libero_90}

case "$SUITE" in
  libero_object|libero_spatial|libero_goal|libero_10|libero_90) ;;
  *)
    echo "Unsupported LIBERO suite: $SUITE" >&2
    exit 2
    ;;
esac

CONFIG=${CONFIG:-configs/b0_rdt1b_lora.yaml}
DATA_ROOT=${DATA_ROOT:-dataset/datasets}
OUTPUT_ROOT=${OUTPUT_ROOT:-cache_features_libero_b0_absolute}
OUT_DIR="${OUTPUT_ROOT}/${SUITE}"

QWEN_MODEL_ID=${QWEN_MODEL_ID:-/workspace/model/stage1_unsloth}
QWEN_PROCESSOR_ID=${QWEN_PROCESSOR_ID:-}
T5_MODEL_ID=${T5_MODEL_ID:-/home/ubuntu/RoboticsDiffusionTransformer/google/t5-v1_1-xxl}
T5_PRECISION=${T5_PRECISION:-bf16}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
ALL_SAMPLES_PER_EPISODE=${ALL_SAMPLES_PER_EPISODE:-1}
MAX_SAMPLES_PER_EPISODE=${MAX_SAMPLES_PER_EPISODE:-128}
OPEN_TO_CLOSE_BEFORE=${OPEN_TO_CLOSE_BEFORE:-10}
OPEN_TO_CLOSE_AFTER=${OPEN_TO_CLOSE_AFTER:-11}
CLOSE_TO_OPEN_BEFORE=${CLOSE_TO_OPEN_BEFORE:-10}
CLOSE_TO_OPEN_AFTER=${CLOSE_TO_OPEN_AFTER:-11}
IMAGE_HISTORY_SIZE=${IMAGE_HISTORY_SIZE:-2}
MAX_IMAGES_PER_SAMPLE=${MAX_IMAGES_PER_SAMPLE:-6}
IMAGE_JPEG_QUALITY=${IMAGE_JPEG_QUALITY:-85}

ARGS=(
  --config "$CONFIG"
  --root "$DATA_ROOT"
  --dataset "$SUITE"
  --output-dir "$OUT_DIR"
  --split train
  --split validation
  --split test
  --feature-set qwen_t5
  --cache-layout sample_shards
  --qwen-cache-scope per_sample
  --action-target-mode absolute_state
  --gripper-change-scope directional
  --open-to-close-before "$OPEN_TO_CLOSE_BEFORE"
  --open-to-close-after "$OPEN_TO_CLOSE_AFTER"
  --close-to-open-before "$CLOSE_TO_OPEN_BEFORE"
  --close-to-open-after "$CLOSE_TO_OPEN_AFTER"
  --qwen-model-id "$QWEN_MODEL_ID"
  --qwen-layer-index 7
  --no-qwen-enable-thinking
  --qwen-stop-at-think
  --qwen-max-new-tokens 128
  --t5-model-id "$T5_MODEL_ID"
  --t5-precision "$T5_PRECISION"
  --image-history-size "$IMAGE_HISTORY_SIZE"
  --max-images-per-sample "$MAX_IMAGES_PER_SAMPLE"
  --image-jpeg-quality "$IMAGE_JPEG_QUALITY"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --pin-memory
)

if [[ "$ALL_SAMPLES_PER_EPISODE" == "1" ]]; then
  ARGS+=(--all-samples-per-episode)
else
  ARGS+=(--max-samples-per-episode "$MAX_SAMPLES_PER_EPISODE")
fi

if [[ -n "$QWEN_PROCESSOR_ID" ]]; then
  ARGS+=(--qwen-processor-id "$QWEN_PROCESSOR_ID")
fi

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

uv run --no-sync python scripts/precompute_all_features.py "${ARGS[@]}"

if [[ -n "${GCS_DEST:-}" ]]; then
  gsutil -m rsync -r "$OUT_DIR" "${GCS_DEST%/}/${SUITE}"
fi
