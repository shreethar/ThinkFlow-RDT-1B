#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# Start from the already LIBERO-finetuned SmolVLA policy. The bootstrap keeps
# its learned model weights, replaces only the cache-facing feature contract
# with native state8/action7 statistics, and initializes the new Qwen adapters.
export BASE_MODEL=${BASE_MODEL:-lerobot/smolvla_base}
export QWEN_TOKEN_COUNT=1
export CACHE_ROOT=${CACHE_ROOT:-cache_features_libero_b0_raw_ortho6d}
export BOOTSTRAP_DIR=${BOOTSTRAP_DIR:-outputs/smolvla_libero_qwen_kv_init_b0}
export STATS_PATH=${STATS_PATH:-outputs/smolvla_base_qwen_kv_all_suites/cache_stats.pt}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/lerobot_smolvla_libero_qwen_kv_b0}
export JOB_NAME=${JOB_NAME:-smolvla-libero-qwen-kv-b0}
export WANDB_PROJECT=${WANDB_PROJECT:-thinkflow-smolvla-libero-b0}

export STEPS=${STEPS:-30000}
export SAVE_FREQ=${SAVE_FREQ:-10000}
export ENV_EVAL_FREQ=${ENV_EVAL_FREQ:-10000}
export N_ACTION_STEPS=${N_ACTION_STEPS:-10}

# The periodic rollout executes ten actions from each sampled chunk before
# extracting a fresh live Qwen K/V token and replanning. By default it evaluates
# task 0 with two initial states in every cached LIBERO suite at 10k/20k/30k.
export SMOLVLA_QWEN_EVAL_ACTION_CHUNK=${SMOLVLA_QWEN_EVAL_ACTION_CHUNK:-10}
export SMOLVLA_QWEN_EVAL_TASK_IDS=${SMOLVLA_QWEN_EVAL_TASK_IDS:-0}
export SMOLVLA_QWEN_EVAL_EPISODES_PER_TASK=${SMOLVLA_QWEN_EVAL_EPISODES_PER_TASK:-2}
export SMOLVLA_QWEN_EVAL_QWEN_MODEL=${SMOLVLA_QWEN_EVAL_QWEN_MODEL:-shreethar/stage1_unsloth}
export SMOLVLA_QWEN_EVAL_QWEN_PROCESSOR=${SMOLVLA_QWEN_EVAL_QWEN_PROCESSOR:-shreethar/stage1_unsloth}
# Stop after preserving the matching checkpoint if evaluation is misconfigured;
# otherwise a failed 10k rollout could go unnoticed until the 30k run finishes.
export SMOLVLA_QWEN_EVAL_FAIL_OPEN=${SMOLVLA_QWEN_EVAL_FAIL_OPEN:-false}

exec bash "$REPO_ROOT/experiments/smolvla_qwen_kv/run_lerobot_cached_fresh.sh" b0
