#!/usr/bin/env bash
set -euo pipefail

# Goal-only B0 full fine-tune. On one GPU, training advances in bit-exact 2K
# segments, unloads, evaluates all 10 Goal tasks, then resumes the same run.
CACHE_ROOT=${CACHE_ROOT:-cache_features_libero_b0_raw_ortho6d}
CONFIG=${CONFIG:-configs/libero_goal_b0_native128_xyz25_10k.yaml}
INIT_ARTIFACT=${INIT_ARTIFACT:-output_2/checkpoint-20000}
OUTPUT_DIR=${OUTPUT_DIR:-output_2/libero_goal_b0_from_oxe20k_xyz25_10k}
WANDB_PROJECT=${WANDB_PROJECT:-ThinkLite B0 LIBERO Goal XYZ}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-libero-goal-b0-from-oxe20k-xyz25-10k}

CACHE_ROOT="$CACHE_ROOT" \
CONFIG="$CONFIG" \
INIT_ARTIFACT="$INIT_ARTIFACT" \
OUTPUT_DIR="$OUTPUT_DIR" \
SUITES="libero_goal" \
MAX_STEPS=10000 \
MICRO_BATCH_SIZE=8 \
GLOBAL_BATCH_SIZE=32 \
LEARNING_RATE=1e-4 \
XYZ_LOSS_WEIGHT=2.5 \
VALIDATE_EVERY=500 \
SAVE_EVERY=1000 \
WANDB_PROJECT="$WANDB_PROJECT" \
WANDB_RUN_NAME="$WANDB_RUN_NAME" \
ROLLOUT_EVAL=interleaved \
ROLLOUT_EVERY=2000 \
ROLLOUT_EPISODES_PER_TASK=2 \
ROLLOUT_ENV_BATCH_SIZE=2 \
ROLLOUT_TASK_IDS="0 1 2 3 4 5 6 7 8 9" \
ROLLOUT_ACTION_CHUNK=1 \
ROLLOUT_SAVE_VIDEOS=1 \
bash scripts/run_libero_b0_native128_full_from_oxe20k.sh
