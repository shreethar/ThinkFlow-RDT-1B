#!/usr/bin/env bash
set -euo pipefail

# One output shard per physical episode. Each retained episode contributes
# exactly 32 samples, except short Kuka episodes which retain every available
# step. Each episode is processed in one Qwen call.
#
# Required override for a newly downloaded dataset:
#   ROOT=/path/to/raw/hf-layout
#
# Common optional overrides:
#   OUTPUT_DIR=cache_features/oxe_32frame_per_sample_qwen
#   CONFIG=configs/part3_rdt1b_lora32.yaml
#   DATASETS="bc_z bridge droid fractal kuka"
#   SPLITS="train validation test"

ROOT=${ROOT:-dataset/hf_parts/part_1}
OUTPUT_DIR=${OUTPUT_DIR:-cache_features/part_1_32frame_per_sample_qwen}
CONFIG=${CONFIG:-configs/part3_rdt1b_lora32.yaml}
SPLITS=${SPLITS:-"train validation test"}
if [[ -n "${DATASETS:-}" ]]; then
  SELECTED_DATASETS=$DATASETS
elif [[ -n "${DATASET:-}" ]]; then
  # Accept the singular spelling as a convenience, including a space-separated
  # list, while retaining DATASETS as the documented variable.
  SELECTED_DATASETS=$DATASET
else
  SELECTED_DATASETS="bc_z bridge droid fractal kuka"
fi

if [[ ! -d "$ROOT" ]]; then
  echo "Dataset root does not exist: $ROOT" >&2
  exit 1
fi

for dataset in $SELECTED_DATASETS; do
  if [[ ! -d "$ROOT/$dataset/data" ]]; then
    echo "Dataset data directory does not exist: $ROOT/$dataset/data" >&2
    echo "Set DATASETS to the datasets actually present under $ROOT." >&2
    exit 1
  fi
  if [[ ! -f "$ROOT/$dataset/audit.json" ]]; then
    echo "Action-normalization audit is missing: $ROOT/$dataset/audit.json" >&2
    exit 1
  fi
done

args=(
  --config "$CONFIG"
  --root "$ROOT"
  --output-dir "$OUTPUT_DIR"
  --batch-size 32
  --episode-batch-size 1
  --max-samples-per-episode 32
  --require-exact-samples-per-episode
  --allow-short-episode-dataset kuka
  --gripper-change-scope first_directional
  --open-to-close-before 4
  --open-to-close-after 4
  --close-to-open-before 4
  --close-to-open-after 4
  --action-target-mode delta
  --feature-set qwen_t5
  --cache-layout episode_packs
  --qwen-cache-scope per_sample
  --no-qwen-enable-thinking
  --image-codec jpeg
  --image-jpeg-quality 90
  --episode-shards-per-directory 500
  --image-history-size 2
  --max-images-per-sample 6
)

for dataset in $SELECTED_DATASETS; do
  args+=(--dataset "$dataset")
done
for split in $SPLITS; do
  args+=(--split "$split")
done

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi
if [[ -n "${MAX_EPISODES:-}" ]]; then
  args+=(--max-episodes "$MAX_EPISODES")
fi
if [[ -n "${EPISODE_PREFETCH_SIZE:-}" ]]; then
  args+=(--episode-prefetch-size "$EPISODE_PREFETCH_SIZE")
fi
if [[ -n "${ASYNC_WRITE_WORKERS:-}" ]]; then
  args+=(--async-write-workers "$ASYNC_WRITE_WORKERS")
fi

exec uv run --no-sync python scripts/precompute_all_features.py "${args[@]}"
