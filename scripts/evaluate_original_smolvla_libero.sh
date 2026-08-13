#!/usr/bin/env bash
set -euo pipefail

# Clean upstream baseline from the LeRobot LIBERO documentation.
#
# This deliberately contains no ThinkFlow/Qwen imports, checkpoint conversion,
# camera remapping, feature overrides, or custom rollout implementation.
# Reference: https://huggingface.co/docs/lerobot/main/en/libero#evaluation

VENV=${VENV:-.venv-smolvla}
POLICY=${POLICY:-HuggingFaceVLA/smolvla_libero}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/original_smolvla_libero_eval}
SUITES=${SUITES:-libero_spatial,libero_object,libero_goal,libero_10}
EPISODES=${EPISODES:-10}
BATCH_SIZE=${BATCH_SIZE:-1}

export MUJOCO_GL=${MUJOCO_GL:-egl}

if [[ ! -x "$VENV/bin/lerobot-eval" ]]; then
  echo "Missing $VENV/bin/lerobot-eval" >&2
  echo "Set VENV to the unmodified LeRobot environment." >&2
  exit 2
fi

exec "$VENV/bin/lerobot-eval" \
  --policy.path="$POLICY" \
  --env.type=libero \
  --env.task="$SUITES" \
  --eval.batch_size="$BATCH_SIZE" \
  --eval.n_episodes="$EPISODES" \
  --env.max_parallel_tasks=1 \
  --output_dir="$OUTPUT_DIR"
