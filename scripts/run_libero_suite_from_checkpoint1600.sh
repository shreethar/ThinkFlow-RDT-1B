#!/usr/bin/env bash
set -euo pipefail

# Fine-tune one LIBERO suite from the transformer learned by root
# checkpoint-1600/.  checkpoint-1600 is a legacy 7D LoRA artifact, so it must
# first be reconstructed with its original architecture and merged.  The
# resulting transformer is then loaded into the current 11D-state/10D-action
# LIBERO model; only the incompatible 7D final output projection is
# reinitialized by load_full_rdt_base().

SUITE=${SUITE:-libero_goal}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-checkpoint-1600}
MERGE_CONFIG=${MERGE_CONFIG:-configs/checkpoint1600_legacy_7d_merge.yaml}
MERGED_BASE=${MERGED_BASE:-outputs/checkpoint1600_merged_base_for_libero}
CONFIG=${CONFIG:-configs/b0_rdt1b_lora.yaml}
CACHE_PARENT=${CACHE_PARENT:-cache_features_libero_b0_raw_ortho6d}
CACHE_ROOT=${CACHE_ROOT:-$CACHE_PARENT/$SUITE}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/${SUITE}_from_checkpoint1600}

MAX_STEPS=${MAX_STEPS:-4000}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
WARMUP_STEPS=${WARMUP_STEPS:-100}
LOG_EVERY=${LOG_EVERY:-10}
VALIDATE_EVERY=${VALIDATE_EVERY:-200}
SAVE_EVERY=${SAVE_EVERY:-200}
VALIDATION_BATCHES=${VALIDATION_BATCHES:-50}
SAMPLE_VALIDATION_BATCHES=${SAMPLE_VALIDATION_BATCHES:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
PIN_MEMORY=${PIN_MEMORY:-1}
ACCELERATE=${ACCELERATE:-0}
DISABLE_GRADIENT_CHECKPOINTING=${DISABLE_GRADIENT_CHECKPOINTING:-0}

HORIZON_LOSS_SCHEDULE=${HORIZON_LOSS_SCHEDULE:-"1-4:5,5-8:3,9-16:2,17-64:1"}
MASK_NOISY_GRIPPER_INPUT=${MASK_NOISY_GRIPPER_INPUT:-1}
GRIPPER_BCE_WEIGHT=${GRIPPER_BCE_WEIGHT:-1.0}
GRIPPER_BCE_LOGIT_SCALE=${GRIPPER_BCE_LOGIT_SCALE:-5.0}

REPORT_TO=${REPORT_TO:-wandb}
WANDB_PROJECT=${WANDB_PROJECT:-thinkflow-rdt-b0-libero}
WANDB_ENTITY=${WANDB_ENTITY:-shreethar2004-universiti-teknikal-malaysia}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${SUITE}-checkpoint1600-suite-only}
SIGLIP_MODEL_ID=${SIGLIP_MODEL_ID:-/home/ubuntu/models/siglip-so400m-patch14-384}
SIGLIP_FALLBACK_MODEL_ID=${SIGLIP_FALLBACK_MODEL_ID:-google/siglip-so400m-patch14-384}

case "$SUITE" in
  libero_10|libero_spatial|libero_goal|libero_object) ;;
  *)
    echo "Unsupported single suite: $SUITE" >&2
    echo "Choose one of: libero_10 libero_spatial libero_goal libero_object" >&2
    exit 2
    ;;
esac

for required in \
  "$SOURCE_CHECKPOINT/metadata.json" \
  "$SOURCE_CHECKPOINT/rdt_lora/adapter_config.json" \
  "$SOURCE_CHECKPOINT/rdt_lora/adapter_model.safetensors" \
  "$MERGE_CONFIG" \
  "$CONFIG" \
  "$CACHE_ROOT/train/manifest.jsonl" \
  "$CACHE_ROOT/validation/manifest.jsonl"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

if [[ ! -f "$MERGED_BASE/rdt_full.pt" ]]; then
  echo "Merging legacy checkpoint into full RDT base: $MERGED_BASE"
  uv run --no-sync python scripts/merge_lora_adapter.py \
    --config "$MERGE_CONFIG" \
    --checkpoint "$SOURCE_CHECKPOINT" \
    --output-dir "$MERGED_BASE" \
    --rdt-only
else
  echo "Using existing merged checkpoint-1600 base: $MERGED_BASE/rdt_full.pt"
fi

if [[ ! -f "$MERGED_BASE/rdt_full.pt" ]]; then
  echo "Merge did not produce $MERGED_BASE/rdt_full.pt" >&2
  exit 1
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

echo "Starting suite-specific LIBERO fine-tuning"
echo "  suite:             $SUITE"
echo "  cache:             $CACHE_ROOT"
echo "  source checkpoint: $SOURCE_CHECKPOINT"
echo "  merged RDT base:   $MERGED_BASE"
echo "  output:            $OUTPUT_DIR"

"${LAUNCH[@]}" scripts/train_b0_cached_features.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --cache-root "$CACHE_ROOT" \
  --base-artifact "$MERGED_BASE" \
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
  "${CHECKPOINTING_ARGS[@]}" \
  "${MEMORY_ARGS[@]}" \
  --report-to "$REPORT_TO" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$WANDB_RUN_NAME"
