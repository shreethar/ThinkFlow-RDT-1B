#!/usr/bin/env bash
set -euo pipefail

# Train on several precomputed LIBERO cache roots in one run.
#
# Expected cache layout:
#   CACHE_ROOT/libero_spatial/train/manifest.jsonl
#   CACHE_ROOT/libero_spatial/validation/manifest.jsonl
#   CACHE_ROOT/libero_object/train/manifest.jsonl
#   ...
#
# For OXE -> LIBERO continuation:
#   1. Merge the OXE LoRA checkpoint with scripts/merge_lora_adapter.py.
#   2. Pass BASE_ARTIFACT=/path/to/merged_artifact here. The merged RDT core is
#      loaded before a fresh LIBERO LoRA adapter is created.

CACHE_ROOT=${1:-${CACHE_ROOT:-cache_features_libero_b0_raw_ortho6d}}
CONFIG=${CONFIG:-configs/b0_rdt1b_lora.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/libero_b0_4suites_mask_aligned}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
BASE_ARTIFACT=${BASE_ARTIFACT:-oxe_b0_merged_for_libero}
INIT_ARTIFACT=${INIT_ARTIFACT:-}

MAX_STEPS=${MAX_STEPS:-2000}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
WARMUP_STEPS=${WARMUP_STEPS:-100}
LOG_EVERY=${LOG_EVERY:-10}
VALIDATE_EVERY=${VALIDATE_EVERY:-100}
SAVE_EVERY=${SAVE_EVERY:-200}
VALIDATION_BATCHES=${VALIDATION_BATCHES:-50}
SAMPLE_VALIDATION_BATCHES=${SAMPLE_VALIDATION_BATCHES:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
REPORT_TO=${REPORT_TO:-wandb}
WANDB_PROJECT=${WANDB_PROJECT:-thinkflow-rdt-b0-libero}
WANDB_ENTITY=${WANDB_ENTITY:-shreethar2004-universiti-teknikal-malaysia}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-libero-b0-4suites-mask-aligned-full}
SIGLIP_MODEL_ID=${SIGLIP_MODEL_ID:-/home/ubuntu/models/siglip-so400m-patch14-384}
SIGLIP_FALLBACK_MODEL_ID=${SIGLIP_FALLBACK_MODEL_ID:-google/siglip-so400m-patch14-384}
DISABLE_GRADIENT_CHECKPOINTING=${DISABLE_GRADIENT_CHECKPOINTING:-0}
PIN_MEMORY=${PIN_MEMORY:-1}
ACCELERATE=${ACCELERATE:-0}
HORIZON_LOSS_SCHEDULE=${HORIZON_LOSS_SCHEDULE:-"1-4:5,5-8:3,9-16:2,17-64:1"}
MASK_NOISY_GRIPPER_INPUT=${MASK_NOISY_GRIPPER_INPUT:-1}
GRIPPER_BCE_WEIGHT=${GRIPPER_BCE_WEIGHT:-1.0}
GRIPPER_BCE_LOGIT_SCALE=${GRIPPER_BCE_LOGIT_SCALE:-5.0}
ROTATION_GEODESIC_WEIGHT=${ROTATION_GEODESIC_WEIGHT:-1.0}

CACHE_ARGS=()
for suite in $SUITES; do
  suite_root="$CACHE_ROOT/$suite"
  train_manifest="$suite_root/train/manifest.jsonl"
  val_manifest="$suite_root/validation/manifest.jsonl"
  if [[ ! -f "$train_manifest" ]]; then
    echo "Missing train manifest: $train_manifest" >&2
    exit 1
  fi
  if [[ ! -f "$val_manifest" ]]; then
    echo "Missing validation manifest: $val_manifest" >&2
    exit 1
  fi
  CACHE_ARGS+=(--cache-root "$suite_root")
done

if [[ ! -f "$BASE_ARTIFACT/rdt_full.pt" ]]; then
  echo "Missing merged OXE base artifact: $BASE_ARTIFACT/rdt_full.pt" >&2
  exit 1
fi

BASE_ARGS=()
if [[ -n "${BASE_ARTIFACT:-}" ]]; then
  BASE_ARGS+=(--base-artifact "$BASE_ARTIFACT")
fi

INIT_ARGS=()
if [[ -n "${INIT_ARTIFACT:-}" ]]; then
  INIT_ARGS+=(--init-artifact "$INIT_ARTIFACT")
fi

CHECKPOINTING_ARGS=()
if [[ "$DISABLE_GRADIENT_CHECKPOINTING" == "1" ]]; then
  CHECKPOINTING_ARGS+=(--no-gradient-checkpointing)
fi

GRIPPER_MASK_ARGS=()
if [[ "$MASK_NOISY_GRIPPER_INPUT" == "1" ]]; then
  GRIPPER_MASK_ARGS+=(--mask-noisy-gripper-input)
else
  GRIPPER_MASK_ARGS+=(--no-mask-noisy-gripper-input)
fi

MEMORY_ARGS=(--num-workers "$NUM_WORKERS")
if [[ "$PIN_MEMORY" == "1" ]]; then
  MEMORY_ARGS+=(--pin-memory --persistent-workers)
else
  MEMORY_ARGS+=(--no-pin-memory --no-persistent-workers)
fi

if [[ "$ACCELERATE" == "1" ]]; then
  LAUNCH=(uv run --no-sync accelerate launch)
else
  LAUNCH=(uv run --no-sync python)
fi

export WANDB_ENTITY

"${LAUNCH[@]}" scripts/train_b0_cached_features.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  "${CACHE_ARGS[@]}" \
  "${BASE_ARGS[@]}" \
  "${INIT_ARGS[@]}" \
  --online-siglip \
  --siglip-model-id "$SIGLIP_MODEL_ID" \
  --siglip-fallback-model-id "$SIGLIP_FALLBACK_MODEL_ID" \
  --max-steps "$MAX_STEPS" \
  --micro-batch-size "$MICRO_BATCH_SIZE" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --warmup-steps "$WARMUP_STEPS" \
  --log-every "$LOG_EVERY" \
  --validate-every "$VALIDATE_EVERY" \
  --save-every "$SAVE_EVERY" \
  --validation-batches "$VALIDATION_BATCHES" \
  --sample-validation-batches "$SAMPLE_VALIDATION_BATCHES" \
  --horizon-loss-schedule "$HORIZON_LOSS_SCHEDULE" \
  "${GRIPPER_MASK_ARGS[@]}" \
  --gripper-bce-weight "$GRIPPER_BCE_WEIGHT" \
  --gripper-bce-logit-scale "$GRIPPER_BCE_LOGIT_SCALE" \
  --rotation-geodesic-weight "$ROTATION_GEODESIC_WEIGHT" \
  "${CHECKPOINTING_ARGS[@]}" \
  "${MEMORY_ARGS[@]}" \
  --report-to "$REPORT_TO" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$WANDB_RUN_NAME"
