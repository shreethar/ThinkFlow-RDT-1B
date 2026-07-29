#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_ROOT="${MODEL_ROOT:-/workspace/model}"
DATASET_ROOT="${DATASET_ROOT:-dataset/hf_parts}"
LATENT_REPO="${LATENT_REPO:-shreethar/LatentStudent-ckpt-400}"
STAGE1_REPO="${STAGE1_REPO:-shreethar/stage1_unsloth}"
LATENT_DIR="${LATENT_DIR:-}"
STAGE1_DIR="${STAGE1_DIR:-}"
FIXED_DIR="${FIXED_DIR:-}"
DATASET_PARTS=()
SKIP_MODELS=0
SKIP_CONVERT=0
SKIP_DATASETS=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_remote_assets.sh [options]

Options:
  --model-root PATH       Default: /workspace/model
  --dataset-root PATH     Default: dataset/hf_parts
  --dataset-part N        N is 1, 2, 3, or all. Repeatable. Default: 3
  --all-datasets          Download parts 1, 2, and 3
  --skip-models
  --skip-convert
  --skip-datasets

Environment overrides:
  MODEL_ROOT, DATASET_ROOT, LATENT_REPO, STAGE1_REPO, LATENT_DIR, STAGE1_DIR, FIXED_DIR
EOF
}

add_dataset_part() {
  case "$1" in
    1|2|3)
      DATASET_PARTS+=("$1")
      ;;
    all)
      DATASET_PARTS+=(1 2 3)
      ;;
    *)
      echo "Dataset part must be 1, 2, 3, or all; got: $1" >&2
      exit 2
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-root)
      MODEL_ROOT="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --dataset-part)
      add_dataset_part "$2"
      shift 2
      ;;
    --all-datasets)
      add_dataset_part all
      shift
      ;;
    --skip-models)
      SKIP_MODELS=1
      shift
      ;;
    --skip-convert)
      SKIP_CONVERT=1
      shift
      ;;
    --skip-datasets)
      SKIP_DATASETS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

LATENT_DIR="${LATENT_DIR:-${MODEL_ROOT}/LatentStudent-ckpt-400}"
STAGE1_DIR="${STAGE1_DIR:-${MODEL_ROOT}/stage1_unsloth}"
FIXED_DIR="${FIXED_DIR:-${MODEL_ROOT}/LatentStudent-ckpt-400-fixed}"

if [[ ${#DATASET_PARTS[@]} -eq 0 ]]; then
  DATASET_PARTS=(3)
fi

cd "${REPO_DIR}"

if [[ "${SKIP_MODELS}" -eq 0 ]]; then
  bash scripts/download_models.sh \
    --model-root "${MODEL_ROOT}" \
    --latent-repo "${LATENT_REPO}" \
    --stage1-repo "${STAGE1_REPO}" \
    --latent-dir "${LATENT_DIR}" \
    --stage1-dir "${STAGE1_DIR}"
fi

if [[ "${SKIP_CONVERT}" -eq 0 ]]; then
  bash scripts/convert_model.sh \
    --src "${LATENT_DIR}" \
    --dst "${FIXED_DIR}" \
    --processor-src "${STAGE1_DIR}"
fi

if [[ "${SKIP_DATASETS}" -eq 0 ]]; then
  dataset_args=(--dataset-root "${DATASET_ROOT}")
  for part in "${DATASET_PARTS[@]}"; do
    dataset_args+=(--part "${part}")
  done
  bash scripts/download_dataset_part.sh "${dataset_args[@]}"
fi

echo "Remote asset setup complete."
echo "  latent model:  ${LATENT_DIR}"
echo "  stage1 model:  ${STAGE1_DIR}"
echo "  fixed model:   ${FIXED_DIR}"
echo "  dataset root:  ${DATASET_ROOT}"
