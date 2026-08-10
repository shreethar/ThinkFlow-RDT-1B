#!/usr/bin/env bash
set -euo pipefail

# Compare closed-loop replanning intervals for one trained task-0 artifact.
# Chunk 1 is omitted by default because the overfit launcher already evaluates it.

CONFIG=${CONFIG:-configs/b0_rdt1b_lora.yaml}
CHECKPOINT=${CHECKPOINT:-outputs/libero_object_task0_10demo_overfit/artifact}
CACHE_ROOT=${CACHE_ROOT:-cache_features_libero_object_task0_10demos}
BASE_ARTIFACT=${BASE_ARTIFACT:-oxe_b0_merged_for_libero}
LIBERO_ROOT=${LIBERO_ROOT:-/home/ubuntu/LIBERO}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/libero_object_task0_10demo_overfit/chunk_comparison}
EPISODES=${EPISODES:-2}
ENV_BATCH_SIZE=${ENV_BATCH_SIZE:-2}
MAX_STEPS=${MAX_STEPS:-600}
SEED=${SEED:-42}
CHUNKS=${CHUNKS:-"4 8 64"}

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "Missing trained artifact: $CHECKPOINT" >&2
  exit 1
fi

for chunk in $CHUNKS; do
  output_dir="$OUTPUT_ROOT/chunk_$chunk"
  echo "Evaluating action chunk $chunk -> $output_dir"
  uv run --no-sync python scripts/evaluate_libero_rdt.py \
    --config "$CONFIG" \
    --benchmark libero_object \
    --checkpoint "$CHECKPOINT" \
    --cache-root "$CACHE_ROOT" \
    --output-dir "$output_dir" \
    --libero-root "$LIBERO_ROOT" \
    --episodes-per-task "$EPISODES" \
    --env-batch-size "$ENV_BATCH_SIZE" \
    --action-chunk "$chunk" \
    --max-steps "$MAX_STEPS" \
    --seed "$SEED" \
    --task-id 0 \
    --base-artifact "$BASE_ARTIFACT" \
    --save-videos \
    --video-resolution 512
done

echo "Chunk comparison complete. Summaries:"
for chunk in $CHUNKS; do
  summary="$OUTPUT_ROOT/chunk_$chunk/summary.json"
  if [[ -f "$summary" ]]; then
    echo "chunk=$chunk"
    sed -n '1,80p' "$summary"
  fi
done
