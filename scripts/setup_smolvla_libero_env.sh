#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Keep LeRobot's current Torch/NumPy stack isolated from ThinkFlow-RDT.
VENV=${VENV:-.venv-smolvla}
LEROBOT_VERSION=${LEROBOT_VERSION:-0.6.0}
UV_CACHE_DIR=${UV_CACHE_DIR:-.uv-cache-smolvla}

if [[ "$VENV" == ".venv" ]]; then
  echo "Refusing to install SmolVLA into ThinkFlow-RDT's .venv." >&2
  echo "Use the default .venv-smolvla or set VENV to another isolated path." >&2
  exit 2
fi

uv venv "$VENV" --python 3.12
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install \
  --python "$VENV/bin/python" \
  "lerobot[smolvla,libero,evaluation]==$LEROBOT_VERSION" \
  "hf-libero==0.1.4" \
  "robosuite==1.4.0" \
  "mujoco>=3.0.0,<3.9.0"

UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install \
  --python "$VENV/bin/python" \
  --no-deps \
  --no-build-isolation \
  -e "$REPO_ROOT/experiments/lerobot_policy_smolvla_qwen_kv"

"$VENV/bin/python" - <<'PY'
from importlib.metadata import version

expected = {
    "lerobot": "0.6.0",
    "hf-libero": "0.1.4",
    "robosuite": "1.4.0",
}
for package, wanted in expected.items():
    actual = version(package)
    if actual != wanted:
        raise RuntimeError(f"{package}: expected {wanted}, found {actual}")
mujoco_version = tuple(int(part) for part in version("mujoco").split(".")[:2])
if not ((3, 0) <= mujoco_version < (3, 9)):
    raise RuntimeError(f"mujoco must be >=3.0,<3.9; found {version('mujoco')}")
print("Validated SmolVLA/LIBERO simulator dependencies")
PY

echo "SmolVLA environment ready: $VENV"
echo "Run training with: SMOLVLA_VENV=$VENV bash experiments/smolvla_qwen_kv/run_lerobot_cached_fresh.sh"
