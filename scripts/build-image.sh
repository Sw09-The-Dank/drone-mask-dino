#!/usr/bin/env bash
set -euo pipefail

repo_root=$(dirname "$0")/..
orig_dockerfile="$repo_root/Dockerfile.demo"
tmp_dockerfile="$repo_root/Dockerfile.demo.tmp"

usage(){
  cat <<EOF
Usage: $0 [--platform PLATFORM] [--base BASE_IMAGE] [--tag IMAGE_TAG] [--no-cache]

Defaults:
  PLATFORM=linux/amd64
  BASE_IMAGE=pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
  IMAGE_TAG=maskdino-demo:local

This script generates a temporary Dockerfile with an explicit FROM line
so buildx pulls the correct platform variant and avoids ARG-in-FROM resolution issues.
EOF
}

PLATFORM=linux/amd64
BASE_IMAGE=pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
IMAGE_TAG=maskdino-demo:local
NO_CACHE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) PLATFORM="$2"; shift 2;;
    --base) BASE_IMAGE="$2"; shift 2;;
    --tag) IMAGE_TAG="$2"; shift 2;;
    --no-cache) NO_CACHE=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [ ! -f "$orig_dockerfile" ]; then
  echo "Original Dockerfile not found: $orig_dockerfile" >&2
  exit 2
fi

echo "Generating temporary Dockerfile using base image: ${BASE_IMAGE} for ${PLATFORM}"

# Quick check: ensure the requested base image has metadata (manifests) for the registry
if ! docker buildx imagetools inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "ERROR: unable to find metadata for base image: $BASE_IMAGE" >&2
  echo "Run 'docker buildx imagetools inspect $BASE_IMAGE' to view available manifests and platforms." >&2
  echo "If the image has no arm64 variant, choose a different base (vendor aarch64/CUDA image for Jetson) or use an Ubuntu base and install PyTorch for aarch64." >&2
  exit 3
fi

# Create tmp with explicit FROM and append everything after the first FROM in original
printf "FROM --platform=%s %s\n" "$PLATFORM" "$BASE_IMAGE" > "$tmp_dockerfile"
awk 'found==0 && /^FROM /{found=1; next} found==1{print}' "$orig_dockerfile" >> "$tmp_dockerfile"

BUILD_CMD=(docker buildx build --platform "$PLATFORM" --pull --progress=plain -t "$IMAGE_TAG" -f "$tmp_dockerfile" .)
if [ "$NO_CACHE" -eq 1 ]; then
  BUILD_CMD+=(--no-cache)
fi

echo "Running: ${BUILD_CMD[*]}"
"${BUILD_CMD[@]}"

rc=$?
rm -f "$tmp_dockerfile"
exit $rc
