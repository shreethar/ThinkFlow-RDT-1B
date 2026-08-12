#!/usr/bin/env bash
set -euo pipefail

# Eight rollout smoke test: two initial states from task 0 in each suite.
exec .venv/bin/python -m experiments.smolvla_qwen_kv.evaluate_checkpoint \
  --checkpoint outputs/smolvla_base_qwen_kv_all_suites/checkpoint-014000 \
  --cache-root cache_features_libero_b0_raw_ortho6d \
  --libero-root /home/ubuntu/LIBERO \
  --output-dir outputs/smolvla_qwen_kv_checkpoint_014000_eval \
  --suites libero_10 libero_spatial libero_goal libero_object \
  --task-id 0 \
  --episodes-per-task 2 \
  --env-batch-size 2 \
  --action-chunk 4 \
  --save-videos \
  --video-resolution 512 \
  --local-files-only
