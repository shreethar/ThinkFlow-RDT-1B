#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/workspace/model}"
LATENT_REPO="${LATENT_REPO:-shreethar/LatentStudent-ckpt-400}"
STAGE1_REPO="${STAGE1_REPO:-shreethar/stage1_unsloth}"
LATENT_DIR="${LATENT_DIR:-}"
STAGE1_DIR="${STAGE1_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-root)
      MODEL_ROOT="$2"
      shift 2
      ;;
    --latent-repo)
      LATENT_REPO="$2"
      shift 2
      ;;
    --stage1-repo)
      STAGE1_REPO="$2"
      shift 2
      ;;
    --latent-dir)
      LATENT_DIR="$2"
      shift 2
      ;;
    --stage1-dir)
      STAGE1_DIR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash scripts/download_models.sh [--model-root PATH] [--latent-repo ID] [--stage1-repo ID]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

LATENT_DIR="${LATENT_DIR:-${MODEL_ROOT}/LatentStudent-ckpt-400}"
STAGE1_DIR="${STAGE1_DIR:-${MODEL_ROOT}/stage1_unsloth}"

mkdir -p "${MODEL_ROOT}"

echo "Downloading latent student model:"
echo "  repo: ${LATENT_REPO}"
echo "  dir:  ${LATENT_DIR}"
uv run hf download "${LATENT_REPO}" --local-dir "${LATENT_DIR}"

echo "Downloading stage1 processor/model:"
echo "  repo: ${STAGE1_REPO}"
echo "  dir:  ${STAGE1_DIR}"
uv run hf download "${STAGE1_REPO}" --local-dir "${STAGE1_DIR}"

echo "Model downloads complete."
