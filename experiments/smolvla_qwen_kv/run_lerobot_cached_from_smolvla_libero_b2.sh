#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# B2 is an independent five-token experiment initialized from the same native
# LIBERO-finetuned policy as B0. Do not resume the one-token B0 checkpoint: that
# would confound the B0/B2 comparison with additional prior training.
export BASE_MODEL=${BASE_MODEL:-lerobot/smolvla_libero}
export QWEN_TOKEN_COUNT=5
export CACHE_ROOT=${CACHE_ROOT:-cache_features_libero_b2_native}
export BOOTSTRAP_DIR=${BOOTSTRAP_DIR:-outputs/smolvla_libero_qwen_kv_staged_init_b2}
export STATS_PATH=${STATS_PATH:-outputs/smolvla_libero_qwen_kv_b2_cache_stats.pt}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/lerobot_smolvla_libero_qwen_kv_staged_b2}
export JOB_NAME=${JOB_NAME:-smolvla-libero-qwen-kv-staged-b2}
export WANDB_PROJECT=${WANDB_PROJECT:-thinkflow-smolvla-libero-b2}

export STEPS=${STEPS:-30000}
export BATCH_SIZE=${BATCH_SIZE:-64}
export SAVE_FREQ=${SAVE_FREQ:-10000}
export ENV_EVAL_FREQ=${ENV_EVAL_FREQ:-10000}
export N_ACTION_STEPS=${N_ACTION_STEPS:-10}
export EXTERNAL_LOGIT_BIAS_INIT=${EXTERNAL_LOGIT_BIAS_INIT:--1.0}
export EXTERNAL_KV_RANKING_WEIGHT=${EXTERNAL_KV_RANKING_WEIGHT:-1.0}
export EXTERNAL_KV_RANKING_MARGIN=${EXTERNAL_KV_RANKING_MARGIN:-0.01}
export EXTERNAL_KV_ADAPTER_WARMUP_STEPS=${EXTERNAL_KV_ADAPTER_WARMUP_STEPS:-1000}
export EXTERNAL_KV_ADAPTER_LR=${EXTERNAL_KV_ADAPTER_LR:-1.0e-4}
export ACTION_EXPERT_LR=${ACTION_EXPERT_LR:-1.0e-5}

# The live rollout reproduces the B2 cache extractor: five layer-7 spatial
# tokens from LatentStudent, with stage1_unsloth supplying its processor.
export SMOLVLA_QWEN_EVAL_ACTION_CHUNK=${SMOLVLA_QWEN_EVAL_ACTION_CHUNK:-10}
export SMOLVLA_QWEN_EVAL_TASK_IDS=${SMOLVLA_QWEN_EVAL_TASK_IDS:-0}
export SMOLVLA_QWEN_EVAL_EPISODES_PER_TASK=${SMOLVLA_QWEN_EVAL_EPISODES_PER_TASK:-2}
export SMOLVLA_QWEN_EVAL_QWEN_MODEL=${SMOLVLA_QWEN_EVAL_QWEN_MODEL:-/workspace/model/LatentStudent-ckpt-400-fixed}
export SMOLVLA_QWEN_EVAL_QWEN_PROCESSOR=${SMOLVLA_QWEN_EVAL_QWEN_PROCESSOR:-/workspace/model/stage1_unsloth}
export SMOLVLA_QWEN_EVAL_LATENT_STUDENT_CODE_DIR=${SMOLVLA_QWEN_EVAL_LATENT_STUDENT_CODE_DIR:-/workspace/VLA-FYP/train/stage2}
export SMOLVLA_QWEN_EVAL_LATENT_ATTN_IMPLEMENTATION=${SMOLVLA_QWEN_EVAL_LATENT_ATTN_IMPLEMENTATION:-sdpa}
export SMOLVLA_QWEN_EVAL_FAIL_OPEN=${SMOLVLA_QWEN_EVAL_FAIL_OPEN:-false}

exec bash "$REPO_ROOT/experiments/smolvla_qwen_kv/run_lerobot_cached_fresh.sh" b2
