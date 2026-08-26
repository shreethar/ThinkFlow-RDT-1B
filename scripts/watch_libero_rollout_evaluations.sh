#!/usr/bin/env bash
# Watch immutable training checkpoints and evaluate every N steps on a separate
# GPU.  Run this in a second terminal or enable it from the training launcher.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: OUTPUT_DIR=<training-output> MAX_STEPS=<train-steps> ROLLOUT_GPU=<spare-gpu> \
  bash scripts/watch_libero_rollout_evaluations.sh

Watches checkpoint-2000, checkpoint-4000, ... and runs 200 closed-loop
LIBERO rollouts per checkpoint (5 fixed initial states x 10 tasks x 4 suites).
EOF
  exit 0
fi

OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR to the active training output directory}
MAX_STEPS=${MAX_STEPS:-20000}
EVAL_EVERY=${EVAL_EVERY:-2000}
ROLLOUT_GPU=${ROLLOUT_GPU:?Set ROLLOUT_GPU to a GPU not used by training}
CONFIG=${CONFIG:-configs/libero_b0_native128_full.yaml}
CACHE_PARENT=${CACHE_PARENT:-cache_features_libero_b0_raw_ortho6d}
LIBERO_ROOT=${LIBERO_ROOT:-/home/ubuntu/LIBERO}
ROLLOUT_ROOT=${ROLLOUT_ROOT:-$OUTPUT_DIR/rollout_evaluations}
EPISODES_PER_TASK=${EPISODES_PER_TASK:-5}
ENV_BATCH_SIZE=${ENV_BATCH_SIZE:-5}
ACTION_CHUNK=${ACTION_CHUNK:-10}
MAX_ROLLOUT_STEPS=${MAX_ROLLOUT_STEPS:-600}
POLL_SECONDS=${POLL_SECONDS:-60}
WANDB_PROJECT=${WANDB_PROJECT:-ThinkLite B0 LIBERO}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-libero-b0-native128-from-oxe20k-full}
SAVE_VIDEOS=${SAVE_VIDEOS:-0}
VIDEO_RESOLUTION=${VIDEO_RESOLUTION:-512}

if (( EVAL_EVERY <= 0 || MAX_STEPS <= 0 || MAX_STEPS < EVAL_EVERY )); then
  echo "Invalid MAX_STEPS/EVAL_EVERY: $MAX_STEPS/$EVAL_EVERY" >&2
  exit 2
fi
if (( MAX_STEPS % EVAL_EVERY != 0 )); then
  echo "MAX_STEPS must be divisible by EVAL_EVERY so no requested checkpoint is skipped" >&2
  exit 2
fi

mkdir -p "$ROLLOUT_ROOT"
for ((step=EVAL_EVERY; step<=MAX_STEPS; step+=EVAL_EVERY)); do
  checkpoint="$OUTPUT_DIR/checkpoint-$step"
  eval_dir="$ROLLOUT_ROOT/checkpoint-$step"
  marker="$eval_dir/summary.json"
  while [[ ! -f "$checkpoint/rdt_full.pt" || ! -f "$checkpoint/interfaces.pt" || ! -f "$checkpoint/trainer_state.pt" ]]; do
    echo "[$(date -Is)] waiting for complete checkpoint-$step" >&2
    sleep "$POLL_SECONDS"
  done
  if [[ -f "$marker" ]]; then
    echo "[$(date -Is)] rollout evaluation already complete: checkpoint-$step" >&2
    continue
  fi

  video_args=()
  if [[ "$SAVE_VIDEOS" == "1" ]]; then
    video_args=(--save-videos --video-resolution "$VIDEO_RESOLUTION")
  fi
  log="$eval_dir/job.log"
  mkdir -p "$eval_dir"
  echo "[$(date -Is)] evaluating checkpoint-$step on GPU $ROLLOUT_GPU" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$ROLLOUT_GPU" \
    uv run --no-sync python scripts/evaluate_libero_checkpoint_grid.py \
      --config "$CONFIG" \
      --checkpoint "$checkpoint" \
      --checkpoint-step "$step" \
      --cache-parent "$CACHE_PARENT" \
      --libero-root "$LIBERO_ROOT" \
      --output-dir "$eval_dir" \
      --episodes-per-task "$EPISODES_PER_TASK" \
      --env-batch-size "$ENV_BATCH_SIZE" \
      --action-chunk "$ACTION_CHUNK" \
      --max-steps "$MAX_ROLLOUT_STEPS" \
      --wandb-project "$WANDB_PROJECT" \
      --wandb-run-name "$WANDB_RUN_NAME" \
      "${video_args[@]}" 2>&1 | tee -a "$log"
done
