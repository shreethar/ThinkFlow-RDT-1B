#!/usr/bin/env bash
set -euo pipefail

# Full Fast-ThinkAct/RDT-1B post-training from cached Qwen/T5/image records.
# All three completed cache parts are used by default. Override CACHE_ROOTS to
# train on a subset.

CONFIG=${CONFIG:-configs/part3_rdt1b.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-output_2}
CACHE_ROOTS=${CACHE_ROOTS:-}
if [[ -z "$CACHE_ROOTS" ]]; then
  CACHE_ROOTS="cache_features/part_1_32frame_per_sample_qwen
cache_features/part_2_32frame_per_sample_qwen
cache_features/part_3_32frame_per_sample_qwen"
fi
SIGLIP_MODEL_ID=${SIGLIP_MODEL_ID:-/home/ubuntu/models/siglip-so400m-patch14-384}
SIGLIP_FALLBACK_MODEL_ID=${SIGLIP_FALLBACK_MODEL_ID:-google/siglip-so400m-patch14-384}
WANDB_PROJECT=${WANDB_PROJECT:-"ThinkLite B0 OXE"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-oxe-rdt1b-native128-fastthinkact-full}
WANDB_ENTITY=${WANDB_ENTITY:-}
ACCELERATE=${ACCELERATE:-0}
RESUME_FROM=${RESUME_FROM:-}

CACHE_ARGS=()
for cache_root in $CACHE_ROOTS; do
  if [[ ! -f "$cache_root/train/manifest.jsonl" ]]; then
    echo "Missing train manifest: $cache_root/train/manifest.jsonl" >&2
    exit 1
  fi
  if [[ ! -f "$cache_root/validation/manifest.jsonl" ]]; then
    echo "Missing validation manifest: $cache_root/validation/manifest.jsonl" >&2
    exit 1
  fi
  CACHE_ARGS+=(--cache-root "$cache_root")
done

if [[ "$ACCELERATE" == "1" ]]; then
  LAUNCH=(uv run --no-sync accelerate launch)
else
  LAUNCH=(uv run --no-sync python)
fi

if [[ -n "$WANDB_ENTITY" ]]; then
  export WANDB_ENTITY
fi

RESUME_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
  RESUME_ARGS+=(--resume-from "$RESUME_FROM")
fi

"${LAUNCH[@]}" scripts/train_b0_cached_features.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  "${CACHE_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --online-siglip \
  --siglip-model-id "$SIGLIP_MODEL_ID" \
  --siglip-fallback-model-id "$SIGLIP_FALLBACK_MODEL_ID" \
  --learning-rate 1e-4 \
  --max-steps 20000 \
  --micro-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --global-batch-size 32 \
  --validate-every 500 \
  --validation-batch-size 32 \
  --validation-samples 256 \
  --save-every 1000 \
  --sample-validation-batches 1 \
  --qualitative-validation-examples 32 \
  --skip-nonfinite-updates \
  --log-gradient-stats \
  --report-to wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$WANDB_RUN_NAME"
