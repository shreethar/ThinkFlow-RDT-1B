#!/usr/bin/env bash
set -euo pipefail

# Download original LIBERO HDF5 demo folders directly from the HF dataset repo.
#
# Usage:
#   scripts/download_libero_hdf5_suites.sh libero_spatial libero_object libero_goal libero_10
#
# The official LIBERO downloader calls the combined long-horizon bundle
# "libero_100", but the downloaded folders are "libero_90" and "libero_10".
# This script downloads the concrete folder names used by the cache extractors.

LOCAL_DIR=${LOCAL_DIR:-dataset/datasets}
REPO_ID=${REPO_ID:-yifengzhu-hf/LIBERO-datasets}

if [[ "$#" -eq 0 ]]; then
  set -- libero_spatial libero_object libero_goal libero_10
fi

mkdir -p "$LOCAL_DIR"

for SUITE in "$@"; do
  case "$SUITE" in
    libero_spatial|libero_object|libero_goal|libero_10|libero_90) ;;
    libero_100)
      echo "libero_100 is a bundle name; downloading libero_90 and libero_10 instead."
      "$0" libero_90 libero_10
      exit 0
      ;;
    *)
      echo "Unsupported suite: $SUITE" >&2
      exit 2
      ;;
  esac

  uv run --no-sync hf download "$REPO_ID" \
    --repo-type dataset \
    --local-dir "$LOCAL_DIR" \
    --include "${SUITE}/**"
done
