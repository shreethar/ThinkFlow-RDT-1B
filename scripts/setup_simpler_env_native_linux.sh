#!/usr/bin/env bash
set -euo pipefail

# Idempotent native-Linux bootstrap for ThinkFlow-RDT + SimplerEnv rollouts.
# Override paths/revision with environment variables; no activation is needed.
THINKFLOW_ROOT=${THINKFLOW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SIMPLER_ROOT=${SIMPLER_ROOT:-$(dirname "$THINKFLOW_ROOT")/SimplerEnv}
SIMPLER_REPO=${SIMPLER_REPO:-https://github.com/simpler-env/SimplerEnv.git}
# Known-good main-branch revision used by this adapter.
SIMPLER_REV=${SIMPLER_REV:-06accaca93535902d408da4855f21cece12bceb7}
SIMPLER_PYTHON_VERSION=${SIMPLER_PYTHON_VERSION:-3.11}
SKIP_SYSTEM_PACKAGES=${SKIP_SYSTEM_PACKAGES:-0}
SKIP_THINKFLOW_SYNC=${SKIP_THINKFLOW_SYNC:-0}
SKIP_RENDER_TEST=${SKIP_RENDER_TEST:-0}

log() { printf '[ThinkFlow SimplerEnv setup] %s\n' "$*"; }
die() { printf '[ThinkFlow SimplerEnv setup] ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || die "SimplerEnv rollout requires native Linux."
if uname -r | grep -qi microsoft; then
  die "WSL/WSL2 is unsupported by the SAPIEN Vulkan renderer; use native Linux."
fi
[[ -f "$THINKFLOW_ROOT/pyproject.toml" ]] || die "Invalid THINKFLOW_ROOT=$THINKFLOW_ROOT"

if [[ "$SKIP_SYSTEM_PACKAGES" != "1" ]]; then
  if [[ "$(id -u)" -eq 0 ]]; then
    APT=(apt-get)
  elif command -v sudo >/dev/null 2>&1; then
    APT=(sudo apt-get)
  else
    die "Install sudo or rerun as root (or set SKIP_SYSTEM_PACKAGES=1)."
  fi
  log "Installing Ubuntu Vulkan/EGL and media prerequisites"
  "${APT[@]}" update
  DEBIAN_FRONTEND=noninteractive "${APT[@]}" install -y --no-install-recommends \
    build-essential ca-certificates curl ffmpeg git git-lfs pkg-config \
    libegl1 libgl1 libgles2 libglib2.0-0 libsm6 libvulkan1 libx11-6 \
    libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxxf86vm1 \
    mesa-vulkan-drivers vulkan-tools
fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable; install the host NVIDIA driver."
nvidia-smi >/dev/null || die "The NVIDIA driver is not responding."

if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv installation failed."

if [[ "$SKIP_THINKFLOW_SYNC" != "1" ]]; then
  log "Synchronizing the pinned ThinkFlow environment"
  (cd "$THINKFLOW_ROOT" && uv sync --frozen)
fi

if [[ ! -d "$SIMPLER_ROOT/.git" ]]; then
  [[ ! -e "$SIMPLER_ROOT" ]] || die "$SIMPLER_ROOT exists but is not a Git checkout."
  log "Cloning SimplerEnv and its ManiSkill2 submodule"
  git clone --recurse-submodules "$SIMPLER_REPO" "$SIMPLER_ROOT"
fi

actual_origin=$(git -C "$SIMPLER_ROOT" remote get-url origin)
[[ "$actual_origin" == "$SIMPLER_REPO" ]] || log "Using existing SimplerEnv origin: $actual_origin"
if [[ -n "$(git -C "$SIMPLER_ROOT" status --porcelain --untracked-files=no)" ]]; then
  current_rev=$(git -C "$SIMPLER_ROOT" rev-parse HEAD)
  [[ "$current_rev" == "$SIMPLER_REV" ]] || die \
    "SimplerEnv has tracked local changes and cannot switch from $current_rev to $SIMPLER_REV."
else
  log "Checking out known-good SimplerEnv revision $SIMPLER_REV"
  git -C "$SIMPLER_ROOT" fetch origin "$SIMPLER_REV"
  git -C "$SIMPLER_ROOT" checkout --detach "$SIMPLER_REV"
fi
git -C "$SIMPLER_ROOT" submodule sync --recursive
git -C "$SIMPLER_ROOT" submodule update --init --recursive
git -C "$SIMPLER_ROOT" lfs install --local
git -C "$SIMPLER_ROOT" lfs pull

log "Creating isolated SimplerEnv Python $SIMPLER_PYTHON_VERSION environment"
uv python install "$SIMPLER_PYTHON_VERSION"
if [[ ! -x "$SIMPLER_ROOT/.venv/bin/python" ]]; then
  uv venv --python "$SIMPLER_PYTHON_VERSION" "$SIMPLER_ROOT/.venv"
fi
SIMPLER_PYTHON="$SIMPLER_ROOT/.venv/bin/python"

# Keep simulator dependencies isolated from RDT's Torch/Transformers stack.
# NumPy 2 breaks SimplerEnv's Pinocchio/IK path; the OpenCV pin avoids pulling it.
# SAPIEN 2.2.2 imports pkg_resources, which setuptools 81+ no longer ships.
log "Installing pinned minimal simulator dependencies"
uv pip install --python "$SIMPLER_PYTHON" --upgrade \
  "setuptools==80.9.0" \
  "numpy==1.24.4" "scipy==1.11.4" "opencv-python==4.8.1.78"
uv pip install --python "$SIMPLER_PYTHON" -e "$SIMPLER_ROOT/ManiSkill2_real2sim"
uv pip install --python "$SIMPLER_PYTHON" -e "$SIMPLER_ROOT"
uv pip install --python "$SIMPLER_PYTHON" --upgrade \
  "setuptools==80.9.0" \
  "numpy==1.24.4" "scipy==1.11.4" "opencv-python==4.8.1.78"

log "Checking package and adapter contracts"
"$SIMPLER_PYTHON" - <<'PY'
from importlib.metadata import version

import numpy
import sapien.core as sapien
import simpler_env
import mani_skill2_real2sim

assert numpy.__version__ == "1.24.4", numpy.__version__
assert version("sapien") == "2.2.2", version("sapien")
assert "google_robot_pick_coke_can" in simpler_env.ENVIRONMENTS
print("Simulator imports OK:", "numpy", numpy.__version__, "sapien", version("sapien"))
PY
(cd "$THINKFLOW_ROOT" && uv run --no-sync python scripts/evaluate_simpler_b0_oxe.py \
  --mode contract --task google_robot_pick_coke_can)

if [[ "$SKIP_RENDER_TEST" != "1" ]]; then
  command -v vulkaninfo >/dev/null 2>&1 || die "vulkaninfo is unavailable."
  log "Checking Vulkan device visibility"
  vulkaninfo --summary >/dev/null 2>&1 || die \
    "Vulkan cannot initialize. Verify the native NVIDIA driver and Vulkan ICD."
  log "Resetting and rendering one headless SimplerEnv scene"
  DISPLAY='' timeout 180 "$SIMPLER_PYTHON" - <<'PY'
import numpy as np
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

env = simpler_env.make(
    "google_robot_pick_coke_can",
    renderer_kwargs={"offscreen_only": True},
)
try:
    observation, _ = env.reset(seed=42)
    image = np.asarray(get_image_from_maniskill2_obs_dict(env, observation))
    assert image.ndim == 3 and image.shape[-1] == 3 and image.size > 0, image.shape
    assert np.isfinite(image).all()
    print("Headless render OK:", image.shape, image.dtype)
finally:
    env.close()
PY
fi

cat <<EOF

Setup complete.

Run a rollout from $THINKFLOW_ROOT with:

RDT_REPO=/path/to/RoboticsDiffusionTransformer uv run --no-sync python \\
  scripts/evaluate_simpler_b0_oxe.py --mode rollout \\
  --task google_robot_pick_coke_can --renderer-offscreen \\
  --simpler-root "$SIMPLER_ROOT" \\
  --simpler-python "$SIMPLER_PYTHON" \\
  --checkpoint output_2/checkpoint-20000 --config configs/part3_rdt1b.yaml \\
  --action-chunk 4 --max-steps 80 \\
  --google-gripper-sticky-steps 15 --rotation-scale 0.25 \\
  --output-dir output_2/simpler_b0_oxe/google_coke_chunk4
EOF
