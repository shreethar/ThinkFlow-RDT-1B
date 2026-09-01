#!/usr/bin/env bash
set -euo pipefail

# Fine-tune B0/B2/B3 on one LIBERO suite using hidden conditioning
# and the Libero_RDT native action/state slot assignment.
#
# By default this starts from the public Libero_RDT LIBERO-Base EMA runner,
# including its RDT core and trained language/image/state adaptors. Override:
#   BASE_ARTIFACT=/path/to/RDT-1B-LIBERO-Base
#
# Examples:
#   VARIANT=b2 SUITE=libero_spatial MAX_STEPS=5000 bash scripts/run_libero_hidden_waypoint_native128_full.sh
#   VARIANT=b2 SUITE=libero_10 MAX_STEPS=5000 bash scripts/run_libero_hidden_waypoint_native128_full.sh
#   VARIANT=b3 SUITE=libero_spatial MAX_STEPS=5000 bash scripts/run_libero_hidden_waypoint_native128_full.sh

VARIANT=${VARIANT:-b2}
SUITE=${SUITE:-libero_spatial}
case "$VARIANT" in
  b0|b2|b3) ;;
  *) echo "VARIANT must be b0, b2 or b3, got '$VARIANT'" >&2; exit 2 ;;
esac
case "$SUITE" in
  libero_spatial|libero_10) ;;
  *) echo "SUITE must be libero_spatial or libero_10, got '$SUITE'" >&2; exit 2 ;;
esac
if [[ "$VARIANT" == "b3" && "$SUITE" != "libero_spatial" ]]; then
  echo "The current experiment matrix defines B3 only for libero_spatial" >&2
  exit 2
fi

if [[ "$VARIANT" == "b0" ]]; then
  DEFAULT_CONFIG=configs/libero_b0_hidden_native128_full.yaml
  DEFAULT_CACHE_ROOT=cache_features_libero_b0_hidden_native/${SUITE}
  CONDITION_LABEL=hidden
  DEFAULT_WARMUP_STEPS=500
  DEFAULT_CONDITIONING_WARMUP_STEPS=0
  DEFAULT_QWEN_FUSION_LOSS_WEIGHT=0.5
else
  DEFAULT_CONFIG=configs/libero_b2_hidden_waypoint_native128_full.yaml
  DEFAULT_CACHE_ROOT=cache_features_libero_${VARIANT}_native/${SUITE}
  CONDITION_LABEL=hidden_waypoint
  DEFAULT_WARMUP_STEPS=100
  DEFAULT_CONDITIONING_WARMUP_STEPS=250
  DEFAULT_QWEN_FUSION_LOSS_WEIGHT=10.0
fi
CONFIG=${CONFIG:-$DEFAULT_CONFIG}
CACHE_ROOT=${CACHE_ROOT:-$DEFAULT_CACHE_ROOT}
INIT_ARTIFACT=${INIT_ARTIFACT:-}
RESUME_FROM=${RESUME_FROM:-}
BASE_ARTIFACT=${BASE_ARTIFACT:-output_3/checkpoints/RDT-1B-LIBERO-Base}
OUTPUT_DIR=${OUTPUT_DIR:-output_2/${SUITE}_${VARIANT}_${CONDITION_LABEL}_native128_full}
MAX_STEPS=${MAX_STEPS:-20000}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
LEARNING_RATE_INTERFACES=${LEARNING_RATE_INTERFACES:-1e-4}
LEARNING_RATE_RDT_BACKBONE=${LEARNING_RATE_RDT_BACKBONE:-1e-4}
LEARNING_RATE_RDT_CROSS_ATTENTION=${LEARNING_RATE_RDT_CROSS_ATTENTION:-1e-4}
LEARNING_RATE_PLAN_PROJECTOR=${LEARNING_RATE_PLAN_PROJECTOR:-1e-4}
CONDITIONING_WARMUP_CROSS_ATTENTION_LEARNING_RATE=${CONDITIONING_WARMUP_CROSS_ATTENTION_LEARNING_RATE:-1e-4}
WARMUP_STEPS=${WARMUP_STEPS:-$DEFAULT_WARMUP_STEPS}
CONDITIONING_WARMUP_STEPS=${CONDITIONING_WARMUP_STEPS:-$DEFAULT_CONDITIONING_WARMUP_STEPS}
VALIDATE_EVERY=${VALIDATE_EVERY:-500}
SAVE_EVERY=${SAVE_EVERY:-1000}
VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE:-32}
VALIDATION_SAMPLES=${VALIDATION_SAMPLES:-256}
QUALITATIVE_VALIDATION_EXAMPLES=${QUALITATIVE_VALIDATION_EXAMPLES:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
QWEN_FUSION_LOSS_WEIGHT=${QWEN_FUSION_LOSS_WEIGHT:-$DEFAULT_QWEN_FUSION_LOSS_WEIGHT}
CONDITIONING_WARMUP_FUSION_LOSS_WEIGHT=${CONDITIONING_WARMUP_FUSION_LOSS_WEIGHT:-}
QWEN_FUSION_LOSS_MARGIN=${QWEN_FUSION_LOSS_MARGIN:-0.002}
# Differentiable training counterpart of rollout's sign threshold. BCE treats
# target>=0 as class 1 and prediction=0 as the exact decision boundary; rollout
# then converts the final scalar to {-1,+1}.
GRIPPER_BCE_WEIGHT=${GRIPPER_BCE_WEIGHT:-1.0}
GRIPPER_BCE_LOGIT_SCALE=${GRIPPER_BCE_LOGIT_SCALE:-5.0}
WANDB_PROJECT=${WANDB_PROJECT:-ThinkLite ${VARIANT^^} LIBERO}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${SUITE}-${VARIANT}-${CONDITION_LABEL}-native128-full}
SIGLIP_MODEL_ID=${SIGLIP_MODEL_ID:-/home/ubuntu/models/siglip-so400m-patch14-384}
SIGLIP_FALLBACK_MODEL_ID=${SIGLIP_FALLBACK_MODEL_ID:-google/siglip-so400m-patch14-384}

for split in train validation; do
  manifest="$CACHE_ROOT/$split/manifest.jsonl"
  if [[ ! -f "$manifest" ]]; then
    echo "Missing $split manifest: $manifest" >&2
    exit 1
  fi
done
if [[ -n "$INIT_ARTIFACT" ]]; then
  for file in rdt_full.pt interfaces.pt metadata.json; do
    if [[ ! -f "$INIT_ARTIFACT/$file" ]]; then
      echo "Incomplete initialization artifact: $INIT_ARTIFACT/$file" >&2
      exit 1
    fi
  done
fi
if [[ -n "$INIT_ARTIFACT" && -n "$RESUME_FROM" ]]; then
  echo "INIT_ARTIFACT and RESUME_FROM are mutually exclusive" >&2
  exit 2
fi
if [[ -n "$RESUME_FROM" ]]; then
  for file in rdt_full.pt interfaces.pt metadata.json trainer_state.pt rng_state_rank_00000.pt; do
    if [[ ! -f "$RESUME_FROM/$file" ]]; then
      echo "Incomplete exact-resume artifact: $RESUME_FROM/$file" >&2
      exit 1
    fi
  done
fi
if [[ -n "$BASE_ARTIFACT" ]]; then
  if [[ ! -f "$BASE_ARTIFACT/rdt_full.pt" && \
        ! -f "$BASE_ARTIFACT/ema/model.safetensors" && \
        ! -f "$BASE_ARTIFACT/model.safetensors" ]]; then
    echo "Unsupported or incomplete base artifact: $BASE_ARTIFACT" >&2
    echo "Expected rdt_full.pt or ema/model.safetensors" >&2
    exit 1
  fi
fi

uv run --no-sync python scripts/preflight_hidden_waypoint_cache.py \
  --manifest "$CACHE_ROOT/train/manifest.jsonl" \
  --expected-variant "$VARIANT" \
  --expected-dataset "$SUITE"
uv run --no-sync python scripts/preflight_hidden_waypoint_cache.py \
  --manifest "$CACHE_ROOT/validation/manifest.jsonl" \
  --expected-variant "$VARIANT" \
  --expected-dataset "$SUITE"

ARGS=(
  --config "$CONFIG"
  --output-dir "$OUTPUT_DIR"
  --cache-root "$CACHE_ROOT"
  --online-siglip
  --siglip-model-id "$SIGLIP_MODEL_ID"
  --siglip-fallback-model-id "$SIGLIP_FALLBACK_MODEL_ID"
  --max-steps "$MAX_STEPS"
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --conditioning-variant "$VARIANT"
  --learning-rate "$LEARNING_RATE"
  --learning-rate-interfaces "$LEARNING_RATE_INTERFACES"
  --learning-rate-rdt-backbone "$LEARNING_RATE_RDT_BACKBONE"
  --learning-rate-rdt-cross-attention "$LEARNING_RATE_RDT_CROSS_ATTENTION"
  --learning-rate-plan-projector "$LEARNING_RATE_PLAN_PROJECTOR"
  --conditioning-warmup-cross-attention-learning-rate "$CONDITIONING_WARMUP_CROSS_ATTENTION_LEARNING_RATE"
  --warmup-steps "$WARMUP_STEPS"
  --conditioning-warmup-steps "$CONDITIONING_WARMUP_STEPS"
  --validate-every "$VALIDATE_EVERY"
  --save-every "$SAVE_EVERY"
  --validation-batch-size "$VALIDATION_BATCH_SIZE"
  --validation-samples "$VALIDATION_SAMPLES"
  --sample-validation-batches 1
  --qualitative-validation-examples "$QUALITATIVE_VALIDATION_EXAMPLES"
  --qwen-fusion-loss-weight "$QWEN_FUSION_LOSS_WEIGHT"
  --qwen-fusion-loss-margin "$QWEN_FUSION_LOSS_MARGIN"
  --mask-noisy-gripper-input
  --gripper-bce-weight "$GRIPPER_BCE_WEIGHT"
  --gripper-bce-logit-scale "$GRIPPER_BCE_LOGIT_SCALE"
  --rotation-geodesic-weight 0.0
  --num-workers "$NUM_WORKERS"
  --pin-memory
  --persistent-workers
  --report-to wandb
  --wandb-project "$WANDB_PROJECT"
  --wandb-run-name "$WANDB_RUN_NAME"
)
if [[ -n "$CONDITIONING_WARMUP_FUSION_LOSS_WEIGHT" ]]; then
  ARGS+=(
    --conditioning-warmup-fusion-loss-weight \
    "$CONDITIONING_WARMUP_FUSION_LOSS_WEIGHT"
  )
fi
if [[ -n "$INIT_ARTIFACT" ]]; then
  ARGS+=(--init-artifact "$INIT_ARTIFACT")
fi
if [[ -n "$RESUME_FROM" ]]; then
  ARGS+=(--resume-from "$RESUME_FROM")
fi
if [[ -n "$BASE_ARTIFACT" ]]; then
  ARGS+=(--base-artifact "$BASE_ARTIFACT")
fi

uv run --no-sync python scripts/train_b0_cached_features.py "${ARGS[@]}"
