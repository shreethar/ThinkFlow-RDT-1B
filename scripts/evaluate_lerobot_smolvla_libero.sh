#!/usr/bin/env bash
set -euo pipefail

# Evaluate the untouched lerobot/smolvla_libero checkpoint with LeRobot's
# native evaluator. The checkpoint expects camera1/camera2 plus a masked third
# camera, while the LIBERO environment exposes agentview and wrist images.

VENV=${VENV:-.venv-smolvla}
POLICY=${POLICY:-lerobot/smolvla_libero}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/lerobot_smolvla_libero_eval}
SUITES=${SUITES:-libero_spatial,libero_object,libero_goal,libero_10}
EPISODES=${EPISODES:-10}
BATCH_SIZE=${BATCH_SIZE:-1}
TASK_IDS=${TASK_IDS:-}
N_ACTION_STEPS=${N_ACTION_STEPS:-}
CAMERA_NAME_MAPPING=${CAMERA_NAME_MAPPING:-'{"agentview_image":"camera1","robot0_eye_in_hand_image":"camera2"}'}

export MUJOCO_GL=${MUJOCO_GL:-egl}

if [[ ! -x "$VENV/bin/lerobot-eval" ]]; then
  echo "Missing $VENV/bin/lerobot-eval" >&2
  echo "Set VENV to the environment containing LeRobot." >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ -n "$TASK_IDS" ]]; then
  EXTRA_ARGS+=(--env.task_ids="$TASK_IDS")
fi
if [[ -n "$N_ACTION_STEPS" ]]; then
  EXTRA_ARGS+=(--policy.n_action_steps="$N_ACTION_STEPS")
fi

exec "$VENV/bin/lerobot-eval" \
  --policy.path="$POLICY" \
  --policy.empty_cameras=1 \
  --env.type=libero \
  --env.task="$SUITES" \
  --env.camera_name_mapping="$CAMERA_NAME_MAPPING" \
  --eval.batch_size="$BATCH_SIZE" \
  --eval.n_episodes="$EPISODES" \
  --env.max_parallel_tasks=1 \
  --output_dir="$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
