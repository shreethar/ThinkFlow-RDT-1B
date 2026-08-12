#!/usr/bin/env bash
set -euo pipefail

# Closed-loop smoke evaluation: task 0, two initial states, all four suites.
# The evaluator accepts LeRobot's train_config.json and resolves the associated
# pretrained_model directory automatically.
exec .venv/bin/python -m experiments.smolvla_qwen_kv.evaluate_checkpoint \
  --checkpoint outputs/lerobot_smolvla_qwen_kv_fresh/checkpoints/004000/pretrained_model/train_config.json \
  --cache-root cache_features_libero_b0_raw_ortho6d \
  --libero-root /home/ubuntu/LIBERO \
  --output-dir outputs/lerobot_smolvla_qwen_kv_fresh/evaluation_step_004000 \
  --suites libero_10 libero_spatial libero_goal libero_object \
  --task-id 0 \
  --episodes-per-task 2 \
  --env-batch-size 2 \
  --action-chunk 4 \
  --save-videos \
  --video-resolution 512 \
  --local-files-only \
  --qwen-model-id shreethar/stage1_unsloth \
  --qwen-processor-id shreethar/stage1_unsloth
