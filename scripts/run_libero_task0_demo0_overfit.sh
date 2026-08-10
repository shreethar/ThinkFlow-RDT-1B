#!/usr/bin/env bash
set -euo pipefail

# Strict memorization test: all timesteps from task-0 demo_0 only. The final
# rollout starts from that same demonstration's exact MuJoCo state.

SOURCE_CACHE_ROOT=${SOURCE_CACHE_ROOT:-cache_features_libero_object_task0_10demos}
CACHE_ROOT=${CACHE_ROOT:-cache_features_libero_object_task0_demo0}
CONFIG=${CONFIG:-configs/b0_rdt1b_lora.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/libero_object_task0_demo0_overfit_mask_aligned}
BASE_ARTIFACT=${BASE_ARTIFACT:-oxe_b0_merged_for_libero}
INIT_ARTIFACT=${INIT_ARTIFACT:-}
DEMO_HDF5=${DEMO_HDF5:-libero-dataset/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5}
DEMO_NAME=${DEMO_NAME:-demo_0}
SIGLIP_MODEL_ID=${SIGLIP_MODEL_ID:-google/siglip-so400m-patch14-384}

STEPS=${STEPS:-200}
BATCH_SIZE=${BATCH_SIZE:-32}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
LEARNING_RATE_LORA=${LEARNING_RATE_LORA:-1e-4}
LEARNING_RATE_INTERFACES=${LEARNING_RATE_INTERFACES:-1e-4}
SAMPLE_EVERY=${SAMPLE_EVERY:-25}
SAMPLING_NUM_SAMPLES=${SAMPLING_NUM_SAMPLES:-64}
SAMPLING_REPEATS=${SAMPLING_REPEATS:-1}
INFERENCE_STEPS=${INFERENCE_STEPS:-5}

# The executed chunk is four commands, so release oversampling is deliberately
# restricted to samples whose first four targets contain the open phase.
GRIPPER_BCE_WEIGHT=${GRIPPER_BCE_WEIGHT:-5.0}
GRIPPER_BCE_LOGIT_SCALE=${GRIPPER_BCE_LOGIT_SCALE:-5.0}
RELEASE_GRIPPER_WEIGHT=${RELEASE_GRIPPER_WEIGHT:-10.0}
# With 148 demo-0 samples and batch size 32, factor 4 produces an audited
# ~50/50 mix of release-relevant and other sampled conditions.
RELEASE_OVERSAMPLE_FACTOR=${RELEASE_OVERSAMPLE_FACTOR:-4.0}
RELEASE_OVERSAMPLE_HORIZON=${RELEASE_OVERSAMPLE_HORIZON:-4}
ROTATION_GEODESIC_WEIGHT=${ROTATION_GEODESIC_WEIGHT:-1.0}

WANDB_PROJECT=${WANDB_PROJECT:-thinkflow-rdt-b0-libero}
WANDB_ENTITY=${WANDB_ENTITY:-shreethar2004-universiti-teknikal-malaysia}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-libero-object-task0-demo0-overfit-mask-aligned}

if [[ ! -f "$SOURCE_CACHE_ROOT/train/manifest.jsonl" ]]; then
  echo "Missing source cache: $SOURCE_CACHE_ROOT/train/manifest.jsonl" >&2
  exit 1
fi
if [[ ! -f "$CACHE_ROOT/train/manifest.jsonl" ]]; then
  uv run --no-sync python scripts/extract_libero_task_cache.py \
    --source-cache-root "$SOURCE_CACHE_ROOT" \
    --output-cache-root "$CACHE_ROOT" \
    --suite libero_object \
    --task-id 0 \
    --num-demos 1
fi
if [[ ! -f "$BASE_ARTIFACT/rdt_full.pt" ]]; then
  echo "Missing merged base artifact: $BASE_ARTIFACT/rdt_full.pt" >&2
  exit 1
fi
if [[ ! -f "$DEMO_HDF5" ]]; then
  echo "Missing demonstration: $DEMO_HDF5" >&2
  exit 1
fi

INIT_ARGS=()
if [[ -n "$INIT_ARTIFACT" ]]; then
  INIT_ARGS+=(--init-artifact "$INIT_ARTIFACT")
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
  --mask-noisy-gripper-input \
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
  --run-rollout \
  --rollout-task-id 0 \
  --rollout-episodes-per-task 1 \
  --rollout-env-batch-size 1 \
  --rollout-action-chunk 4 \
  --rollout-max-steps 200 \
  --rollout-save-videos \
  --rollout-video-resolution 512 \
  --rollout-demo-hdf5 "$DEMO_HDF5" \
  --rollout-demo-name "$DEMO_NAME"
