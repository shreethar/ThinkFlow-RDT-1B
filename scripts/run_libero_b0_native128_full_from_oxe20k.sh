#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=${CACHE_ROOT:-cache_features_libero_b0_raw_ortho6d}
CONFIG=${CONFIG:-configs/libero_b0_native128_full.yaml}
INIT_ARTIFACT=${INIT_ARTIFACT:-output_2/checkpoint-20000}
OUTPUT_DIR=${OUTPUT_DIR:-output_2/libero_b0_from_oxe20k_v2}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}

MAX_STEPS=${MAX_STEPS:-20000}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE:-32}
VALIDATION_SAMPLES=${VALIDATION_SAMPLES:-256}
QUALITATIVE_VALIDATION_EXAMPLES=${QUALITATIVE_VALIDATION_EXAMPLES:-32}
VALIDATE_EVERY=${VALIDATE_EVERY:-500}
SAVE_EVERY=${SAVE_EVERY:-1000}
NUM_WORKERS=${NUM_WORKERS:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
XYZ_LOSS_WEIGHT=${XYZ_LOSS_WEIGHT:-0.0}
HORIZON_LOSS_SCHEDULE=${HORIZON_LOSS_SCHEDULE:-}
WANDB_PROJECT=${WANDB_PROJECT:-ThinkLite B0 LIBERO}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-libero-b0-native128-from-oxe20k-full-v2}
SIGLIP_MODEL_ID=${SIGLIP_MODEL_ID:-/home/ubuntu/models/siglip-so400m-patch14-384}
SIGLIP_FALLBACK_MODEL_ID=${SIGLIP_FALLBACK_MODEL_ID:-google/siglip-so400m-patch14-384}
# On one GPU, finish training first and then evaluate all saved 5,000-step
# checkpoints. Other supported modes are interleaved, parallel (requires a
# spare GPU), and 0 (disabled).
ROLLOUT_EVAL=${ROLLOUT_EVAL:-post_training}
ROLLOUT_GPU=${ROLLOUT_GPU:-}
ROLLOUT_SAVE_VIDEOS=${ROLLOUT_SAVE_VIDEOS:-0}
ROLLOUT_EVERY=${ROLLOUT_EVERY:-5000}
ROLLOUT_EPISODES_PER_TASK=${ROLLOUT_EPISODES_PER_TASK:-2}
ROLLOUT_ENV_BATCH_SIZE=${ROLLOUT_ENV_BATCH_SIZE:-2}
ROLLOUT_TASK_IDS=${ROLLOUT_TASK_IDS:-"0 2 4 6 8"}
ROLLOUT_ACTION_CHUNK=${ROLLOUT_ACTION_CHUNK:-10}

CACHE_ARGS=()
for suite in $SUITES; do
  suite_root="$CACHE_ROOT/$suite"
  for split in train validation; do
    if [[ ! -f "$suite_root/$split/manifest.jsonl" ]]; then
      echo "Missing manifest: $suite_root/$split/manifest.jsonl" >&2
      exit 1
    fi
  done
  CACHE_ARGS+=(--cache-root "$suite_root")
done

if [[ ! -f "$INIT_ARTIFACT/rdt_full.pt" ]] || [[ ! -f "$INIT_ARTIFACT/interfaces.pt" ]]; then
  echo "Incomplete OXE initialization artifact: $INIT_ARTIFACT" >&2
  exit 1
fi

if (( MAX_STEPS % ROLLOUT_EVERY != 0 )); then
  echo "MAX_STEPS must be divisible by ROLLOUT_EVERY: $MAX_STEPS/$ROLLOUT_EVERY" >&2
  exit 2
fi
if (( ROLLOUT_EVERY % SAVE_EVERY != 0 )); then
  echo "ROLLOUT_EVERY must be divisible by SAVE_EVERY so each boundary has a checkpoint" >&2
  exit 2
fi

run_rollout_watcher() {
  local available_steps=${1:-$MAX_STEPS}
  OUTPUT_DIR="$OUTPUT_DIR" \
    MAX_STEPS="$available_steps" \
    EVAL_EVERY="$ROLLOUT_EVERY" \
    ROLLOUT_GPU="$ROLLOUT_GPU" \
    CONFIG="$CONFIG" \
    CACHE_PARENT="$CACHE_ROOT" \
    EPISODES_PER_TASK="$ROLLOUT_EPISODES_PER_TASK" \
    ENV_BATCH_SIZE="$ROLLOUT_ENV_BATCH_SIZE" \
    TASK_IDS="$ROLLOUT_TASK_IDS" \
    SUITES="$SUITES" \
    ACTION_CHUNK="$ROLLOUT_ACTION_CHUNK" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_RUN_NAME="$WANDB_RUN_NAME" \
    SIGLIP_MODEL_ID="$SIGLIP_MODEL_ID" \
    SIGLIP_FALLBACK_MODEL_ID="$SIGLIP_FALLBACK_MODEL_ID" \
    SAVE_VIDEOS="$ROLLOUT_SAVE_VIDEOS" \
    bash scripts/watch_libero_rollout_evaluations.sh
}

if [[ "$ROLLOUT_EVAL" != "parallel" && "$ROLLOUT_EVAL" != "1" && -z "$ROLLOUT_GPU" ]]; then
  ROLLOUT_GPU=0
fi

TRAIN_ARGS=(
  --config "$CONFIG"
  --output-dir "$OUTPUT_DIR"
  "${CACHE_ARGS[@]}"
  --online-siglip
  --siglip-model-id "$SIGLIP_MODEL_ID"
  --siglip-fallback-model-id "$SIGLIP_FALLBACK_MODEL_ID"
  --max-steps "$MAX_STEPS"
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --xyz-loss-weight "$XYZ_LOSS_WEIGHT"
  --validation-batch-size "$VALIDATION_BATCH_SIZE"
  --validation-samples "$VALIDATION_SAMPLES"
  --sample-validation-batches 1
  --qualitative-validation-examples "$QUALITATIVE_VALIDATION_EXAMPLES"
  --validate-every "$VALIDATE_EVERY"
  --save-every "$SAVE_EVERY"
  --mask-noisy-gripper-input
  --gripper-bce-weight 1.0
  --gripper-bce-logit-scale 5.0
  --rotation-geodesic-weight 1.0
  --num-workers "$NUM_WORKERS"
  --pin-memory
  --persistent-workers
  --report-to wandb
  --wandb-project "$WANDB_PROJECT"
  --wandb-run-name "$WANDB_RUN_NAME"
)
if [[ -n "$HORIZON_LOSS_SCHEDULE" ]]; then
  TRAIN_ARGS+=(--horizon-loss-schedule "$HORIZON_LOSS_SCHEDULE")
fi

mkdir -p "$OUTPUT_DIR/rollout_evaluations"
TRAIN_WANDB_RUN_ID_FILE="$OUTPUT_DIR/.training_wandb_run_id"
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  TRAIN_WANDB_RUN_ID="$WANDB_RUN_ID"
elif [[ -s "$TRAIN_WANDB_RUN_ID_FILE" ]]; then
  IFS= read -r TRAIN_WANDB_RUN_ID < "$TRAIN_WANDB_RUN_ID_FILE"
else
  TRAIN_WANDB_RUN_ID=$(uv run --no-sync python -c 'import wandb; print(wandb.util.generate_id())')
  printf '%s\n' "$TRAIN_WANDB_RUN_ID" > "$TRAIN_WANDB_RUN_ID_FILE"
fi

checkpoint_complete() {
  local checkpoint=$1
  [[ -f "$checkpoint/rdt_full.pt" \
    && -f "$checkpoint/interfaces.pt" \
    && -f "$checkpoint/metadata.json" \
    && -f "$checkpoint/trainer_state.pt" \
    && -f "$checkpoint/rng_state_rank_00000.pt" ]]
}

latest_checkpoint_before() {
  local target_step=$1
  local latest_step=0
  local latest_path=""
  local candidate step
  for candidate in "$OUTPUT_DIR"/checkpoint-*; do
    [[ -d "$candidate" ]] || continue
    step=${candidate##*-}
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    if (( step < target_step && step > latest_step )) && checkpoint_complete "$candidate"; then
      latest_step=$step
      latest_path=$candidate
    fi
  done
  printf '%s' "$latest_path"
}

run_training_until() {
  local target_step=$1
  local target_checkpoint="$OUTPUT_DIR/checkpoint-$target_step"
  if checkpoint_complete "$target_checkpoint"; then
    echo "Complete checkpoint-$target_step already exists; skipping its training segment"
    return
  fi

  local resume_checkpoint
  local artifact_args=()
  resume_checkpoint=$(latest_checkpoint_before "$target_step")
  if [[ -n "$resume_checkpoint" ]]; then
    artifact_args=(--resume-from "$resume_checkpoint")
    echo "Bit-exact training resume: $resume_checkpoint -> checkpoint-$target_step"
  else
    artifact_args=(--init-artifact "$INIT_ARTIFACT")
    echo "Starting LIBERO fine-tuning from $INIT_ARTIFACT -> checkpoint-$target_step"
  fi

  WANDB_RUN_ID="$TRAIN_WANDB_RUN_ID" WANDB_RESUME=allow \
    uv run --no-sync python scripts/train_b0_cached_features.py \
      "${TRAIN_ARGS[@]}" \
      "${artifact_args[@]}" \
      --stop-after-step "$target_step"

  if ! checkpoint_complete "$target_checkpoint"; then
    echo "Training segment ended without a complete checkpoint-$target_step" >&2
    exit 3
  fi
}

case "$ROLLOUT_EVAL" in
  interleaved)
    for ((target_step=ROLLOUT_EVERY; target_step<=MAX_STEPS; target_step+=ROLLOUT_EVERY)); do
      run_training_until "$target_step"
      echo "Training paused at step $target_step; releasing GPU for 40 online-Qwen rollouts"
      run_rollout_watcher "$target_step" 2>&1 | tee -a "$OUTPUT_DIR/rollout_evaluations/watcher.log"
      if (( target_step < MAX_STEPS )); then
        echo "Rollouts complete at step $target_step; resuming bit-exact training"
      fi
    done
    ;;
  post_training)
    run_training_until "$MAX_STEPS"
    echo "Training complete; starting checkpoint rollout evaluations on GPU $ROLLOUT_GPU"
    run_rollout_watcher "$MAX_STEPS" 2>&1 | tee -a "$OUTPUT_DIR/rollout_evaluations/watcher.log"
    ;;
  parallel|1)
    if [[ -z "$ROLLOUT_GPU" ]]; then
      echo "ROLLOUT_EVAL=parallel requires ROLLOUT_GPU=<spare gpu id>" >&2
      exit 2
    fi
    mkdir -p "$OUTPUT_DIR/rollout_evaluations"
    run_rollout_watcher "$MAX_STEPS" >"$OUTPUT_DIR/rollout_evaluations/watcher.log" 2>&1 &
    echo "Started online-Qwen rollout watcher on spare GPU $ROLLOUT_GPU (PID $!)"
    run_training_until "$MAX_STEPS"
    ;;
  0)
    run_training_until "$MAX_STEPS"
    ;;
  *)
    echo "ROLLOUT_EVAL must be interleaved, post_training, parallel (or 1), or 0" >&2
    exit 2
    ;;
esac
