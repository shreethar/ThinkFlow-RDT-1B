#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# This uses LeRobot's official trainer and checkpoint format. checkpoint-014000
# is a MODEL warm-start: its legacy training_state.pt is not LeRobot's
# Accelerate checkpoint layout, so optimizer/scheduler state starts fresh.
exec .venv/bin/lerobot-train \
  --policy.path=outputs/smolvla_base_qwen_kv_all_suites/checkpoint-014000 \
  --policy.push_to_hub=false \
  --dataset.repo_id=cached_libero_qwen:all \
  --dataset.root=cache_features_libero_b0_raw_ortho6d \
  --dataset.streaming=true \
  --dataset.eval_split=0 \
  --output_dir=outputs/lerobot_smolvla_qwen_kv_from_014000 \
  --job_name=smolvla-qwen-kv-from-014000 \
  --steps=100000 \
  --batch_size=256 \
  --num_workers=8 \
  --log_freq=10 \
  --save_freq=1000 \
  --env_eval_freq=0 \
  --eval_steps=0 \
  --wandb.enable=true \
  --wandb.project=thinkflow-rdt-b0-libero
