#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

exec .venv/bin/python -m experiments.smolvla_qwen_kv.train_cached \
  --pretrained lerobot/smolvla_base \
  --cache-root cache_features_libero_b0_raw_ortho6d \
  --suites libero_goal \
  --output-dir outputs/smolvla_base_qwen_kv_libero_goal \
  --batch-size 16 \
  --gradient-accumulation 2 \
  --steps 30000 \
  --n-action-steps 4 \
  --save-every 1000 \
  --sample-every 500 \
  --local-files-only \
  --bf16 \
  --wandb-project thinkflow-smolvla-qwen-kv \
  --wandb-name smolvla-base-qwen-kv-libero-goal
