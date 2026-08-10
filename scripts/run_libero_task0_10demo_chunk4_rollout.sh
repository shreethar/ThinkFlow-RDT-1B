#!/usr/bin/env bash
set -euo pipefail

# Roll out the overfit task-0 policy from the exact initial simulator states
# belonging to the ten demonstrations used to build its training cache.

CONFIG=${CONFIG:-configs/b0_rdt1b_lora.yaml}
CHECKPOINT=${CHECKPOINT:-outputs/libero_object_task0_10demo_overfit/artifact}
CACHE_ROOT=${CACHE_ROOT:-cache_features_libero_object_task0_10demos}
BASE_ARTIFACT=${BASE_ARTIFACT:-oxe_b0_merged_for_libero}
LIBERO_ROOT=${LIBERO_ROOT:-/home/ubuntu/LIBERO}
DEMO_HDF5=${DEMO_HDF5:-libero-dataset/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/libero_object_task0_10demo_overfit/exact_demo_chunk4}
ENV_BATCH_SIZE=${ENV_BATCH_SIZE:-2}
MAX_STEPS=${MAX_STEPS:-600}
SEED=${SEED:-42}

uv run --no-sync python scripts/evaluate_libero_rdt.py \
  --config "$CONFIG" \
  --benchmark libero_object \
  --checkpoint "$CHECKPOINT" \
  --cache-root "$CACHE_ROOT" \
  --base-artifact "$BASE_ARTIFACT" \
  --libero-root "$LIBERO_ROOT" \
  --demo-hdf5 "$DEMO_HDF5" \
  --demo-name demo_0 \
  --demo-name demo_4 \
  --demo-name demo_5 \
  --demo-name demo_6 \
  --demo-name demo_7 \
  --demo-name demo_8 \
  --demo-name demo_9 \
  --demo-name demo_10 \
  --demo-name demo_11 \
  --demo-name demo_12 \
  --task-id 0 \
  --output-dir "$OUTPUT_DIR" \
  --env-batch-size "$ENV_BATCH_SIZE" \
  --action-chunk 4 \
  --max-steps "$MAX_STEPS" \
  --seed "$SEED" \
  --save-videos \
  --video-resolution 512
