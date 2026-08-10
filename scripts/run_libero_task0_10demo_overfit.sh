#!/usr/bin/env bash
set -euo pipefail

# Overfit all 1,576 timesteps from ten complete demonstrations of LIBERO
# Object task 0. Unlike the normal trainer, this uses the real diffusion
# sampler throughout training and writes decoded command/rollout diagnostics.

CACHE_ROOT=${CACHE_ROOT:-cache_features_libero_object_task0_10demos}
CONFIG=${CONFIG:-configs/b0_rdt1b_lora.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/libero_object_task0_10demo_overfit_release_bce_so3}
BASE_ARTIFACT=${BASE_ARTIFACT:-oxe_b0_merged_for_libero}
INIT_ARTIFACT=${INIT_ARTIFACT:-}
SIGLIP_MODEL_ID=${SIGLIP_MODEL_ID:-google/siglip-so400m-patch14-384}

STEPS=${STEPS:-800}
BATCH_SIZE=${BATCH_SIZE:-32}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
LEARNING_RATE_LORA=${LEARNING_RATE_LORA:-1e-4}
LEARNING_RATE_INTERFACES=${LEARNING_RATE_INTERFACES:-1e-4}
SAMPLE_EVERY=${SAMPLE_EVERY:-100}
SAMPLING_NUM_SAMPLES=${SAMPLING_NUM_SAMPLES:-32}
SAMPLING_REPEATS=${SAMPLING_REPEATS:-1}
INFERENCE_STEPS=${INFERENCE_STEPS:-5}
GRIPPER_BCE_WEIGHT=${GRIPPER_BCE_WEIGHT:-1.0}
GRIPPER_BCE_LOGIT_SCALE=${GRIPPER_BCE_LOGIT_SCALE:-5.0}
MASK_NOISY_GRIPPER_INPUT=${MASK_NOISY_GRIPPER_INPUT:-1}
RELEASE_GRIPPER_WEIGHT=${RELEASE_GRIPPER_WEIGHT:-5.0}
RELEASE_OVERSAMPLE_FACTOR=${RELEASE_OVERSAMPLE_FACTOR:-4.0}
RELEASE_OVERSAMPLE_HORIZON=${RELEASE_OVERSAMPLE_HORIZON:-8}
ROTATION_GEODESIC_WEIGHT=${ROTATION_GEODESIC_WEIGHT:-1.0}
RUN_ROLLOUT=${RUN_ROLLOUT:-1}
ROLLOUT_EPISODES=${ROLLOUT_EPISODES:-2}
ROLLOUT_ACTION_CHUNK=${ROLLOUT_ACTION_CHUNK:-4}
ROLLOUT_SAVE_VIDEOS=${ROLLOUT_SAVE_VIDEOS:-1}
WANDB_PROJECT=${WANDB_PROJECT:-thinkflow-rdt-b0-libero}
WANDB_ENTITY=${WANDB_ENTITY:-shreethar2004-universiti-teknikal-malaysia}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-libero-object-task0-10demo-overfit-release-bce-so3-test}

if [[ ! -f "$CACHE_ROOT/train/manifest.jsonl" ]]; then
  echo "Missing filtered training cache: $CACHE_ROOT/train/manifest.jsonl" >&2
  exit 1
fi
if [[ ! -f "$BASE_ARTIFACT/rdt_full.pt" ]]; then
  echo "Missing merged base artifact: $BASE_ARTIFACT/rdt_full.pt" >&2
  exit 1
fi

ROLLOUT_ARGS=()
INIT_ARGS=()
GRIPPER_MASK_ARGS=()
if [[ -n "$INIT_ARTIFACT" ]]; then
  if [[ ! -f "$INIT_ARTIFACT/interfaces.pt" ]]; then
    echo "Missing initialization artifact: $INIT_ARTIFACT/interfaces.pt" >&2
    exit 1
  fi
  INIT_ARGS+=(--init-artifact "$INIT_ARTIFACT")
fi
if [[ "$MASK_NOISY_GRIPPER_INPUT" == "1" ]]; then
  GRIPPER_MASK_ARGS+=(--mask-noisy-gripper-input)
fi
if [[ "$RUN_ROLLOUT" == "1" ]]; then
  ROLLOUT_ARGS+=(
    --run-rollout
    --rollout-task-id 0
    --rollout-episodes-per-task "$ROLLOUT_EPISODES"
    --rollout-action-chunk "$ROLLOUT_ACTION_CHUNK"
  )
  if [[ "$ROLLOUT_SAVE_VIDEOS" == "1" ]]; then
    ROLLOUT_ARGS+=(--rollout-save-videos --rollout-video-resolution 512)
  fi
fi

uv run --no-sync python scripts/overfit_libero_cached.py \
  --config "$CONFIG" \
  --cache-root "$CACHE_ROOT" \
  --suite libero_object \
  --output-dir "$OUTPUT_DIR" \
  --base-artifact "$BASE_ARTIFACT" \
  "${INIT_ARGS[@]}" \
  --all-samples \
  --batch-size "$BATCH_SIZE" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --learning-rate-lora "$LEARNING_RATE_LORA" \
  --learning-rate-interfaces "$LEARNING_RATE_INTERFACES" \
  --steps "$STEPS" \
  --sample-every "$SAMPLE_EVERY" \
  --sampling-num-samples "$SAMPLING_NUM_SAMPLES" \
  --sampling-repeats "$SAMPLING_REPEATS" \
  --inference-steps "$INFERENCE_STEPS" \
  --gripper-bce-weight "$GRIPPER_BCE_WEIGHT" \
  --gripper-bce-logit-scale "$GRIPPER_BCE_LOGIT_SCALE" \
  "${GRIPPER_MASK_ARGS[@]}" \
  --release-gripper-weight "$RELEASE_GRIPPER_WEIGHT" \
  --release-oversample-factor "$RELEASE_OVERSAMPLE_FACTOR" \
  --release-oversample-horizon "$RELEASE_OVERSAMPLE_HORIZON" \
  --rotation-geodesic-weight "$ROTATION_GEODESIC_WEIGHT" \
  --siglip-model-id "$SIGLIP_MODEL_ID" \
  --siglip-fallback-model-id google/siglip-so400m-patch14-384 \
  --report-to wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  "${ROLLOUT_ARGS[@]}"
