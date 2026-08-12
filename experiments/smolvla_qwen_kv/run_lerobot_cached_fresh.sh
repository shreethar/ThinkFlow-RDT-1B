#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LIBERO_ROOT=${SMOLVLA_QWEN_EVAL_LIBERO_ROOT:-"$REPO_ROOT/../LIBERO"}
if [[ ! -d "$LIBERO_ROOT/libero/libero" ]]; then
  echo "LIBERO source tree not found at: $LIBERO_ROOT" >&2
  echo "Set SMOLVLA_QWEN_EVAL_LIBERO_ROOT=/absolute/path/to/LIBERO" >&2
  exit 2
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ACCELERATE_USE_DEEPSPEED=false
export SMOLVLA_QWEN_EVAL_ENABLE=true
export SMOLVLA_QWEN_EVAL_LIBERO_ROOT="$LIBERO_ROOT"
export SMOLVLA_QWEN_EVAL_QWEN_MODEL=shreethar/stage1_unsloth
export SMOLVLA_QWEN_EVAL_QWEN_PROCESSOR=shreethar/stage1_unsloth
export SMOLVLA_QWEN_EVAL_TASK_IDS=0
export SMOLVLA_QWEN_EVAL_EPISODES_PER_TASK=2
export SMOLVLA_QWEN_EVAL_ACTION_CHUNK=4
export SMOLVLA_QWEN_EVAL_SAVE_VIDEOS=true

# Build the custom policy once from native lerobot/smolvla_base. This does not
# load checkpoint-014000 or any prior LIBERO fine-tuning.
if [[ ! -f outputs/smolvla_base_qwen_kv_init/model.safetensors ]]; then
  .venv/bin/python -m experiments.smolvla_qwen_kv.create_base_checkpoint \
    --base lerobot/smolvla_base \
    --output-dir outputs/smolvla_base_qwen_kv_init \
    --stats outputs/smolvla_base_qwen_kv_all_suites/cache_stats.pt \
    --device cpu \
    --seed 42 \
    --local-files-only
fi

exec .venv/bin/lerobot-train \
  --policy.path=outputs/smolvla_base_qwen_kv_init \
  --policy.load_vlm_weights=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=cached_libero_qwen:all \
  --dataset.root=cache_features_libero_b0_raw_ortho6d \
  --dataset.streaming=true \
  --dataset.eval_split=0 \
  --output_dir=outputs/lerobot_smolvla_qwen_kv_fresh \
  --job_name=smolvla-base-qwen-kv-fresh \
  --steps=100000 \
  --batch_size=128 \
  --num_workers=8 \
  --log_freq=10 \
  --save_freq=1000 \
  --env_eval_freq=5000 \
  --eval_steps=0 \
  --wandb.enable=true \
  --wandb.project=thinkflow-smolvla-b0-libero
