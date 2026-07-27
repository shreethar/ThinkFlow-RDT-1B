#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_libero_object_training.sh prepare CONFIG DATA_ROOT
#   scripts/run_libero_object_training.sh overfit CONFIG DATA_ROOT
#   scripts/run_libero_object_training.sh full CONFIG DATA_ROOT
#   scripts/run_libero_object_training.sh train CONFIG DATA_ROOT
# DATA_ROOT layout: DATA_ROOT/libero_object/data/**/*.hdf5

MODE=${1:?mode must be prepare, overfit, or full}
CONFIG=${2:?config yaml is required}
DATA_ROOT=${3:?dataset root is required}
CHECKPOINT=${CHECKPOINT:-outputs/b0_cached_sft/checkpoint-1000}
CACHE_ROOT=${CACHE_ROOT:-cache_features/libero_object}
PRECOMPUTE_BATCH_SIZE=${PRECOMPUTE_BATCH_SIZE:-8}
PRECOMPUTE_WORKERS=${PRECOMPUTE_WORKERS:-4}
PRECOMPUTE_OVERWRITE=${PRECOMPUTE_OVERWRITE:-0}
TRAIN_MICRO_BATCH_SIZE=${TRAIN_MICRO_BATCH_SIZE:-128}
TRAIN_GRADIENT_ACCUMULATION_STEPS=${TRAIN_GRADIENT_ACCUMULATION_STEPS:-2}
DISABLE_GRADIENT_CHECKPOINTING=${DISABLE_GRADIENT_CHECKPOINTING:-0}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-2}
TRAIN_PIN_MEMORY=${TRAIN_PIN_MEMORY:-0}
OVERWRITE_ARGS=()
if [[ "$PRECOMPUTE_OVERWRITE" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi
CHECKPOINTING_ARGS=()
if [[ "$DISABLE_GRADIENT_CHECKPOINTING" == "1" ]]; then
  CHECKPOINTING_ARGS+=(--no-gradient-checkpointing)
fi
TRAIN_MEMORY_ARGS=(--num-workers "$TRAIN_NUM_WORKERS")
if [[ "$TRAIN_PIN_MEMORY" == "1" ]]; then
  TRAIN_MEMORY_ARGS+=(--pin-memory --persistent-workers)
else
  TRAIN_MEMORY_ARGS+=(--no-pin-memory --no-persistent-workers)
fi

if [[ "$MODE" == "prepare" ]]; then
  python scripts/prepare_libero_object.py \
    --data-dir "$DATA_ROOT/libero_object/data" \
    --output "$DATA_ROOT/libero_object/audit.json"
elif [[ "$MODE" == "overfit" ]]; then
  python scripts/precompute_all_features.py \
    --config "$CONFIG" --root "$DATA_ROOT" --dataset libero_object \
    --split train --max-episodes 2 --max-samples-per-split 64 \
    --batch-size "$PRECOMPUTE_BATCH_SIZE" \
    --num-workers "$PRECOMPUTE_WORKERS" --pin-memory \
    "${OVERWRITE_ARGS[@]}" \
    --output-dir "$CACHE_ROOT/overfit"
  python scripts/train_b0_cached_features.py \
    --config "$CONFIG" --init-artifact "$CHECKPOINT" \
    --train-manifest "$CACHE_ROOT/overfit/train/manifest.jsonl" \
    --val-manifest "$CACHE_ROOT/overfit/train/manifest.jsonl" \
    --output-dir outputs/libero_object_overfit \
    --max-steps 500 --micro-batch-size 4 \
    --gradient-accumulation-steps 1 --warmup-steps 20 \
    --validate-every 50 --save-every 250 --report-to wandb
elif [[ "$MODE" == "full" ]]; then
  python scripts/precompute_all_features.py \
    --config "$CONFIG" --root "$DATA_ROOT" --dataset libero_object \
    --split train --split validation --output-dir "$CACHE_ROOT/full" \
    --batch-size "$PRECOMPUTE_BATCH_SIZE" \
    --num-workers "$PRECOMPUTE_WORKERS" --pin-memory \
    "${OVERWRITE_ARGS[@]}" \
    --profile-timing --profile-every-episodes 100
  python scripts/train_b0_cached_features.py \
    --config "$CONFIG" --init-artifact "$CHECKPOINT" \
    --cache-root "$CACHE_ROOT/full" \
    --output-dir outputs/libero_object_full \
    --max-steps 20000 \
    --micro-batch-size "$TRAIN_MICRO_BATCH_SIZE" \
    --gradient-accumulation-steps "$TRAIN_GRADIENT_ACCUMULATION_STEPS" \
    "${CHECKPOINTING_ARGS[@]}" \
    "${TRAIN_MEMORY_ARGS[@]}" \
    --report-to wandb
elif [[ "$MODE" == "train" ]]; then
  python scripts/train_b0_cached_features.py \
    --config "$CONFIG" --init-artifact "$CHECKPOINT" \
    --cache-root "$CACHE_ROOT/full" \
    --output-dir outputs/libero_object_full \
    --max-steps 2000 \
    --micro-batch-size "$TRAIN_MICRO_BATCH_SIZE" \
    --gradient-accumulation-steps "$TRAIN_GRADIENT_ACCUMULATION_STEPS" \
    "${CHECKPOINTING_ARGS[@]}" \
    "${TRAIN_MEMORY_ARGS[@]}" \
    --report-to wandb
else
  echo "unknown mode: $MODE (expected prepare, overfit, full, or train)" >&2
  exit 2
fi
