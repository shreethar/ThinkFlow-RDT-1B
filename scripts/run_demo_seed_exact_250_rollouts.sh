#!/usr/bin/env bash
set -euo pipefail

# Rebuild the Gradio demo's state-success map using exactly the website's
# deterministic seed schedule. Every model prediction has batch size one:
#   42 + task_id * 100000 + init_state_index * 1000 + plan_index

cd "$(dirname "$0")/.."
export RDT_REPO="${RDT_REPO:-/home/ubuntu/RoboticsDiffusionTransformer}"

PYTHON="${PYTHON:-.venv/bin/python}"
LIBERO_ROOT="${LIBERO_ROOT:-/home/ubuntu/LIBERO}"
BASE_ARTIFACT="${BASE_ARTIFACT:-output_3/checkpoints/RDT-1B-LIBERO-Base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output_2/demo_seed_exact_250}"

common_thinkflow=(
  --base-artifact "$BASE_ARTIFACT"
  --libero-root "$LIBERO_ROOT"
  --episodes-per-task 5
  --env-batch-size 1
  --action-chunk 10
  --inference-steps 5
  --seed 42
  --t5-precision bf16
  --attn-implementation flash_attention_2
  --require-qwen-fusion
)

"$PYTHON" -u scripts/evaluate_libero_rdt.py \
  --config configs/libero_b0_hidden_native128_full.yaml \
  --benchmark libero_spatial \
  --checkpoint output_2/libero_spatial_b0_hidden/checkpoint-10000 \
  --cache-root cache_features_libero_b0_raw_ortho6d/libero_spatial \
  --output-dir "$OUTPUT_ROOT/libero_spatial_b0" \
  --max-steps 300 \
  --qwen-extraction b0 \
  "${common_thinkflow[@]}"

"$PYTHON" -u scripts/evaluate_libero_rdt.py \
  --config configs/libero_b2_hidden_waypoint_native128_full.yaml \
  --benchmark libero_spatial \
  --checkpoint output_2/libero_spatial_b2_hidden_waypoint_fusion10_warm250_v2/checkpoint-5000 \
  --cache-root cache_features_libero_b2_native/libero_spatial \
  --output-dir "$OUTPUT_ROOT/libero_spatial_b2" \
  --max-steps 300 \
  --qwen-extraction b2 \
  --student-model-id model/LatentStudent-ckpt-400-fixed \
  --processor-id model/model/stage1_unsloth \
  --latent-student-code-dir /home/ubuntu/VLA-FYP/train/stage2 \
  --qwen-layer-index 7 \
  --latent-count 6 \
  --spatial-token-count 5 \
  --student-precision bf16 \
  "${common_thinkflow[@]}"

"$PYTHON" -u scripts/evaluate_libero_rdt.py \
  --config configs/libero_b2_hidden_waypoint_native128_full.yaml \
  --benchmark libero_spatial \
  --checkpoint output_2/libero_spatial_b3_hidden_waypoint_fusion10_warm250_v1/checkpoint-5000 \
  --cache-root LIBERO_Spatial_B3 \
  --output-dir "$OUTPUT_ROOT/libero_spatial_b3" \
  --max-steps 300 \
  --qwen-extraction b3 \
  --student-model-id shreethar/Latent-Student-Spatial-Forcing \
  --processor-id shreethar/Latent-Student-Spatial-Forcing \
  --latent-student-code-dir /home/ubuntu/VLA-FYP/train/stage2 \
  --qwen-layer-index 7 \
  --latent-count 6 \
  --spatial-token-count 5 \
  --student-precision bf16 \
  "${common_thinkflow[@]}"

"$PYTHON" -u scripts/evaluate_libero_rdt.py \
  --config configs/libero_b0_hidden_native128_full.yaml \
  --benchmark libero_10 \
  --checkpoint output_2/libero_long_b0_hidden/checkpoint-10000 \
  --cache-root cache_features_libero_b0_raw_ortho6d/libero_10 \
  --output-dir "$OUTPUT_ROOT/libero_10_b0" \
  --max-steps 600 \
  --qwen-extraction b0 \
  "${common_thinkflow[@]}"

"$PYTHON" -u scripts/evaluate_hf_rdt_libero_goal.py \
  --checkpoint-dir output_3/checkpoints/RDT-1B-LIBERO-Long \
  --model-id TJ-chen/RDT-1B-LIBERO-Long \
  --benchmark libero_10 \
  --output-dir "$OUTPUT_ROOT/libero_10_b2" \
  --libero-root "$LIBERO_ROOT" \
  --rdt-repo "$RDT_REPO" \
  --episodes-per-task 5 \
  --action-chunk 10 \
  --max-steps 600 \
  --seed 42
