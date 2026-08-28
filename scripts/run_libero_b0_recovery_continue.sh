#!/usr/bin/env bash
set -euo pipefail

# Continue the original 20K LIBERO B0 model with simulator-labelled spatial
# recovery examples. Five thousand updates is the default decision point. Set
# MAX_STEPS=10000 only after checkpoint-5000 improves fixed online rollouts.

CACHE_ROOT=${CACHE_ROOT:-cache_features_libero_b0_raw_ortho6d}
RECOVERY_CACHE_ROOT=${RECOVERY_CACHE_ROOT:-cache_features_libero_b0_recovery}
CONFIG=${CONFIG:-configs/libero_b0_native128_recovery_continue.yaml}
INIT_ARTIFACT=${INIT_ARTIFACT:-output_2/libero_b0_from_oxe20k_v2/checkpoint-20000}
OUTPUT_DIR=${OUTPUT_DIR:-output_2/libero_b0_recovery_from_libero20k_5k}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}

MAX_STEPS=${MAX_STEPS:-5000}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
LEARNING_RATE_INTERFACES=${LEARNING_RATE_INTERFACES:-5e-6}
XYZ_LOSS_WEIGHT=${XYZ_LOSS_WEIGHT:-1.0}
HORIZON_LOSS_SCHEDULE=${HORIZON_LOSS_SCHEDULE:-"1-10:4,11-20:2,21-64:0.5"}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE:-32}
VALIDATION_SAMPLES=${VALIDATION_SAMPLES:-256}
QUALITATIVE_VALIDATION_EXAMPLES=${QUALITATIVE_VALIDATION_EXAMPLES:-32}
VALIDATE_EVERY=${VALIDATE_EVERY:-500}
SAVE_EVERY=${SAVE_EVERY:-1000}
WARMUP_STEPS=${WARMUP_STEPS:-250}
NUM_WORKERS=${NUM_WORKERS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-ThinkLite B0 LIBERO Recovery}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-libero-b0-20k-recovery-plus5k}
SIGLIP_MODEL_ID=${SIGLIP_MODEL_ID:-/home/ubuntu/models/siglip-so400m-patch14-384}
SIGLIP_FALLBACK_MODEL_ID=${SIGLIP_FALLBACK_MODEL_ID:-google/siglip-so400m-patch14-384}
RESUME_FROM=${RESUME_FROM:-}

if [[ "$MAX_STEPS" != "5000" && "$MAX_STEPS" != "10000" ]]; then
  echo "Warning: expected MAX_STEPS=5000 or 10000; got $MAX_STEPS" >&2
fi

CACHE_ARGS=()
for suite in $SUITES; do
  suite_root="$CACHE_ROOT/$suite"
  for split in train validation; do
    if [[ ! -f "$suite_root/$split/manifest.jsonl" ]]; then
      echo "Missing original cache manifest: $suite_root/$split/manifest.jsonl" >&2
      exit 1
    fi
  done
  CACHE_ARGS+=(--cache-root "$suite_root")
done
for split in train validation; do
  if [[ ! -f "$RECOVERY_CACHE_ROOT/$split/manifest.jsonl" ]]; then
    echo "Missing recovery cache manifest: $RECOVERY_CACHE_ROOT/$split/manifest.jsonl" >&2
    echo "Run scripts/run_generate_libero_recovery_cache.sh first." >&2
    exit 1
  fi
done
if [[ ! -f "$RECOVERY_CACHE_ROOT/precompute_metadata.json" ]]; then
  echo "Recovery cache is incomplete (missing precompute_metadata.json): $RECOVERY_CACHE_ROOT" >&2
  exit 1
fi
CACHE_ARGS+=(--cache-root "$RECOVERY_CACHE_ROOT")

for required in rdt_full.pt interfaces.pt metadata.json; do
  if [[ ! -f "$INIT_ARTIFACT/$required" ]]; then
    echo "Incomplete initialization artifact: $INIT_ARTIFACT/$required" >&2
    exit 1
  fi
done

ARTIFACT_ARGS=(--init-artifact "$INIT_ARTIFACT")
if [[ -n "$RESUME_FROM" ]]; then
  ARTIFACT_ARGS=(--resume-from "$RESUME_FROM")
fi

uv run --no-sync python scripts/train_b0_cached_features.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  "${CACHE_ARGS[@]}" \
  --online-siglip \
  --siglip-model-id "$SIGLIP_MODEL_ID" \
  --siglip-fallback-model-id "$SIGLIP_FALLBACK_MODEL_ID" \
  --max-steps "$MAX_STEPS" \
  --micro-batch-size "$MICRO_BATCH_SIZE" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --learning-rate-interfaces "$LEARNING_RATE_INTERFACES" \
  --warmup-steps "$WARMUP_STEPS" \
  --horizon-loss-schedule "$HORIZON_LOSS_SCHEDULE" \
  --xyz-loss-weight "$XYZ_LOSS_WEIGHT" \
  --mask-noisy-gripper-input \
  --gripper-bce-weight 1.0 \
  --gripper-bce-logit-scale 5.0 \
  --rotation-geodesic-weight 1.0 \
  --validation-batch-size "$VALIDATION_BATCH_SIZE" \
  --validation-samples "$VALIDATION_SAMPLES" \
  --sample-validation-batches 1 \
  --qualitative-validation-examples "$QUALITATIVE_VALIDATION_EXAMPLES" \
  --validate-every "$VALIDATE_EVERY" \
  --save-every "$SAVE_EVERY" \
  --log-every 10 \
  --num-workers "$NUM_WORKERS" \
  --pin-memory \
  --persistent-workers \
  --skip-nonfinite-updates \
  --log-gradient-stats \
  --report-to wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  "${ARTIFACT_ARGS[@]}"
