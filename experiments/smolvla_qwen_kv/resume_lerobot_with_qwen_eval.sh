#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ACCELERATE_USE_DEEPSPEED=false
export SMOLVLA_QWEN_EVAL_ENABLE=true
export SMOLVLA_QWEN_EVAL_QWEN_MODEL=shreethar/stage1_unsloth
export SMOLVLA_QWEN_EVAL_QWEN_PROCESSOR=shreethar/stage1_unsloth
export SMOLVLA_QWEN_EVAL_TASK_IDS=0
export SMOLVLA_QWEN_EVAL_EPISODES_PER_TASK=2
export SMOLVLA_QWEN_EVAL_ENV_BATCH_SIZE=2
export SMOLVLA_QWEN_EVAL_ACTION_CHUNK=4
export SMOLVLA_QWEN_EVAL_SAVE_VIDEOS=true
export SMOLVLA_QWEN_EVAL_LOCAL_FILES_ONLY=true

# Resume all optimizer/scheduler/model/RNG state. The CLI override turns on the
# custom callback every 5,000 steps even though the original run stored
# env_eval_freq=0 in its train_config.json.
exec .venv/bin/lerobot-train \
  --config_path=outputs/lerobot_smolvla_qwen_kv_fresh/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  --env_eval_freq=5000
