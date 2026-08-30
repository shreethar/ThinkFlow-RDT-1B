#!/usr/bin/env bash
# Watch immutable training checkpoints and evaluate every N steps. This can run
# on a spare GPU during training or on the training GPU after training exits.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: OUTPUT_DIR=<training-output> MAX_STEPS=<train-steps> ROLLOUT_GPU=<gpu-id> \
  bash scripts/watch_libero_rollout_evaluations.sh

Watches checkpoint-5000, checkpoint-10000, ... and runs 40 online-Qwen,
closed-loop LIBERO rollouts per checkpoint (2 fixed initial states x 5 fixed
tasks x 4 suites).
EOF
  exit 0
fi

OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR to the active training output directory}
MAX_STEPS=${MAX_STEPS:-20000}
EVAL_EVERY=${EVAL_EVERY:-5000}
ROLLOUT_GPU=${ROLLOUT_GPU:?Set ROLLOUT_GPU to the evaluation GPU ID}
CONFIG=${CONFIG:-configs/libero_b0_native128_full.yaml}
CACHE_PARENT=${CACHE_PARENT:-cache_features_libero_b0_raw_ortho6d}
LIBERO_ROOT=${LIBERO_ROOT:-/home/ubuntu/LIBERO}
ROLLOUT_ROOT=${ROLLOUT_ROOT:-$OUTPUT_DIR/rollout_evaluations}
EPISODES_PER_TASK=${EPISODES_PER_TASK:-2}
ENV_BATCH_SIZE=${ENV_BATCH_SIZE:-2}
TASK_IDS=${TASK_IDS:-"0 2 4 6 8"}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
ACTION_CHUNK=${ACTION_CHUNK:-10}
MAX_ROLLOUT_STEPS=${MAX_ROLLOUT_STEPS:-600}
POLL_SECONDS=${POLL_SECONDS:-60}
WANDB_PROJECT=${WANDB_PROJECT:-ThinkLite B0 LIBERO}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-libero-b0-native128-from-oxe20k-full-v2}
SAVE_VIDEOS=${SAVE_VIDEOS:-0}
VIDEO_RESOLUTION=${VIDEO_RESOLUTION:-512}
SIGLIP_MODEL_ID=${SIGLIP_MODEL_ID:-/home/ubuntu/models/siglip-so400m-patch14-384}
SIGLIP_FALLBACK_MODEL_ID=${SIGLIP_FALLBACK_MODEL_ID:-google/siglip-so400m-patch14-384}

if (( EVAL_EVERY <= 0 || MAX_STEPS <= 0 || MAX_STEPS < EVAL_EVERY )); then
  echo "Invalid MAX_STEPS/EVAL_EVERY: $MAX_STEPS/$EVAL_EVERY" >&2
  exit 2
fi
if (( MAX_STEPS % EVAL_EVERY != 0 )); then
  echo "MAX_STEPS must be divisible by EVAL_EVERY so no requested checkpoint is skipped" >&2
  exit 2
fi

mkdir -p "$ROLLOUT_ROOT"
task_args=()
declare -A seen_task_ids=()
task_count=0
for task_id in $TASK_IDS; do
  if [[ ! "$task_id" =~ ^[0-9]+$ ]] || (( task_id < 0 || task_id > 9 )); then
    echo "Invalid LIBERO task ID '$task_id'; expected an integer in [0, 9]" >&2
    exit 2
  fi
  if [[ -n "${seen_task_ids[$task_id]:-}" ]]; then
    echo "Duplicate LIBERO task ID '$task_id' in TASK_IDS='$TASK_IDS'" >&2
    exit 2
  fi
  seen_task_ids[$task_id]=1
  task_args+=(--task-id "$task_id")
  ((task_count += 1))
done
if (( task_count == 0 )); then
  echo "TASK_IDS must contain at least one task ID" >&2
  exit 2
fi
suite_args=()
declare -A seen_suites=()
suite_count=0
for suite in $SUITES; do
  case "$suite" in
    libero_spatial|libero_object|libero_goal|libero_10) ;;
    *)
      echo "Invalid LIBERO suite '$suite'" >&2
      exit 2
      ;;
  esac
  if [[ -n "${seen_suites[$suite]:-}" ]]; then
    echo "Duplicate LIBERO suite '$suite' in SUITES='$SUITES'" >&2
    exit 2
  fi
  seen_suites[$suite]=1
  suite_args+=(--suite "$suite")
  ((suite_count += 1))
done
if (( suite_count == 0 )); then
  echo "SUITES must contain at least one suite" >&2
  exit 2
fi
expected_rollouts=$((suite_count * task_count * EPISODES_PER_TASK))
echo "Online-Qwen rollout grid: $EPISODES_PER_TASK initial states x $task_count tasks x $suite_count suites = $expected_rollouts rollouts/checkpoint"

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
      --siglip-model-id "$SIGLIP_MODEL_ID" \
      --siglip-fallback-model-id "$SIGLIP_FALLBACK_MODEL_ID" \
      "${suite_args[@]}" \
      "${task_args[@]}" \
      --wandb-project "$WANDB_PROJECT" \
      --wandb-run-name "$WANDB_RUN_NAME" \
      "${video_args[@]}" 2>&1 | tee -a "$log"
done
