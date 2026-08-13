#!/usr/bin/env bash
set -euo pipefail

# Official SmolVLA checkpoint fine-tuned on the LeRobot LIBERO dataset.
VENV=${VENV:-.venv-smolvla}
POLICY=${POLICY:-lerobot/smolvla_libero}
SUITE=${SUITE:-libero_goal}
TASK_IDS=${TASK_IDS:-}
N_EPISODES=${N_EPISODES:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_STEPS=${MAX_STEPS:-300}
N_ACTION_STEPS=${N_ACTION_STEPS:-}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/smolvla_${SUITE}_eval}
# Match LeRobot's official smolvla_libero CI contract exactly. The environment
# emits camera1/camera2 directly, and one masked empty slot preserves the three
# visual positions used when this checkpoint was trained.
CAMERA_NAME_MAPPING=${CAMERA_NAME_MAPPING:-'{"agentview_image":"camera1","robot0_eye_in_hand_image":"camera2"}'}
EMPTY_CAMERAS=${EMPTY_CAMERAS:-1}

LEROBOT_EVAL="$VENV/bin/lerobot-eval"
if [[ ! -x "$LEROBOT_EVAL" ]]; then
  echo "Missing $LEROBOT_EVAL" >&2
  echo "Install the isolated environment first:" >&2
  echo "  bash scripts/setup_smolvla_libero_env.sh" >&2
  exit 1
fi

"$VENV/bin/python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import sys

problems = []
for package, wanted in (("lerobot", "0.6.0"), ("hf-libero", "0.1.4"), ("robosuite", "1.4.0")):
    try:
        actual = version(package)
    except PackageNotFoundError:
        problems.append(f"{package} is missing")
        continue
    if actual != wanted:
        problems.append(f"{package} must be {wanted}, found {actual}")
try:
    actual_mujoco = version("mujoco")
    mujoco_version = tuple(int(part) for part in actual_mujoco.split(".")[:2])
    if not ((3, 0) <= mujoco_version < (3, 9)):
        problems.append(f"mujoco must be >=3.0,<3.9, found {actual_mujoco}")
except PackageNotFoundError:
    problems.append("mujoco is missing")

if problems:
    print("Incompatible SmolVLA/LIBERO environment:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    print("Run: bash scripts/setup_smolvla_libero_env.sh", file=sys.stderr)
    raise SystemExit(2)
PY

case "$SUITE" in
  libero_10|libero_spatial|libero_goal|libero_object) ;;
  *)
    echo "Unsupported suite: $SUITE" >&2
    exit 2
    ;;
esac

ARGS=(
  --policy.path="$POLICY"
  --policy.device="$DEVICE"
  --policy.empty_cameras="$EMPTY_CAMERAS"
  --env.type=libero
  --env.task="$SUITE"
  --env.control_mode=relative
  --env.camera_name_mapping="$CAMERA_NAME_MAPPING"
  --env.init_states=true
  --env.episode_length="$MAX_STEPS"
  --env.max_parallel_tasks=1
  --eval.n_episodes="$N_EPISODES"
  --eval.batch_size="$BATCH_SIZE"
  --eval.use_async_envs=false
  --output_dir="$OUTPUT_DIR"
  --seed="$SEED"
)

if [[ -n "$TASK_IDS" ]]; then
  # Draccus list syntax, for example TASK_IDS='[0]' or TASK_IDS='[0,1]'.
  ARGS+=(--env.task_ids="$TASK_IDS")
fi
if [[ -n "$N_ACTION_STEPS" ]]; then
  # The checkpoint predicts 50 actions. This controls how many are executed
  # before SmolVLA is called again; leave empty for the checkpoint default.
  ARGS+=(--policy.n_action_steps="$N_ACTION_STEPS")
fi

echo "Running official SmolVLA LIBERO evaluation"
echo "  policy:   $POLICY"
echo "  suite:    $SUITE"
echo "  task IDs: ${TASK_IDS:-all}"
echo "  episodes: $N_EPISODES per selected task"
echo "  max steps: $MAX_STEPS"
echo "  actions/replan: ${N_ACTION_STEPS:-checkpoint default}"
echo "  camera map: $CAMERA_NAME_MAPPING"
echo "  empty cameras: $EMPTY_CAMERAS"
echo "  output:   $OUTPUT_DIR"

exec "$LEROBOT_EVAL" "${ARGS[@]}"
