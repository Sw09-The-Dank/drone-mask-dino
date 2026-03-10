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
NO_CACHE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) PLATFORM="$2"; shift 2;;
    --base) BASE_IMAGE="$2"; shift 2;;
    --native) FORCE_NATIVE=1; shift 1;;
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

# Auto-fallback: if target is arm64 and user selected a CUDA/amd64 base, switch
# to a multi-arch Ubuntu base to avoid pulling an incompatible amd64 CUDA image.
if echo "$PLATFORM" | grep -q "arm64" && echo "$BASE_IMAGE" | grep -Ei "cuda|cudnn|nvidia|pytorch.*cuda" >/dev/null 2>&1; then
  echo "Target is arm64 but requested CUDA base image may be amd64; switching base to ubuntu:22.04 as a fallback." >&2
  BASE_IMAGE=ubuntu:22.04
  echo "You should install PyTorch/conda in the resulting image for aarch64 or provide a vendor aarch64 CUDA image via --base." >&2
fi

# Quick check: if buildx imagetools is available, warn when metadata is missing
if command -v docker >/dev/null 2>&1 && docker buildx >/dev/null 2>&1; then
  if ! docker buildx imagetools inspect "$BASE_IMAGE" >/dev/null 2>&1; then
    echo "WARNING: unable to find metadata for base image: $BASE_IMAGE" >&2
    echo "Run 'docker buildx imagetools inspect $BASE_IMAGE' to view available manifests and platforms." >&2
    echo "Continuing, but the build may fail if the image/tag doesn't exist for the target platform." >&2
  fi
fi

# Detect host platform and prefer plain docker build when host matches requested platform
host_uname=$(uname -m)
case "$host_uname" in
  x86_64) host_platform=linux/amd64 ;;
  aarch64|arm64) host_platform=linux/arm64 ;;
  *) host_platform=unknown ;;
esac

# Create tmp with explicit FROM line. Use --platform only when cross-building.
if [ "${FORCE_NATIVE:-0}" = "1" ] || [ "$PLATFORM" = "$host_platform" ]; then
  printf "FROM %s\n" "$BASE_IMAGE" > "$tmp_dockerfile"
else
  printf "FROM --platform=%s %s\n" "$PLATFORM" "$BASE_IMAGE" > "$tmp_dockerfile"
fi

# Provide ARG placeholders so any ${TARGETPLATFORM}/${TARGETARCH}/${BASE_IMAGE}
# expansions in later RUN lines are defined (they will be empty unless build-args
# are provided). These ARG lines are safe after the FROM.
printf "ARG TARGETPLATFORM\nARG TARGETARCH\nARG BASE_IMAGE=%s\n" "$BASE_IMAGE" >> "$tmp_dockerfile"
awk 'found==0 && /^FROM /{found=1; next} found==1{print}' "$orig_dockerfile" >> "$tmp_dockerfile"

if [ "${FORCE_NATIVE:-0}" = "1" ] || [ "$PLATFORM" = "$host_platform" ]; then
  # Use plain docker build (no --platform) when building natively to avoid emulation issues
  BUILD_CMD=(docker build --pull --progress=plain -t "$IMAGE_TAG" -f "$tmp_dockerfile" .)
else
  BUILD_CMD=(docker buildx build --platform "$PLATFORM" --pull --progress=plain -t "$IMAGE_TAG" -f "$tmp_dockerfile" .)
fi
if [ "$NO_CACHE" -eq 1 ]; then
  BUILD_CMD+=(--no-cache)
fi

# Determine correct Miniconda installer for target platform and pass as build-arg
CONDA_INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh"
if echo "$PLATFORM" | grep -q "arm64"; then
  CONDA_INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-aarch64.sh"
fi
BUILD_CMD+=(--build-arg "CONDA_INSTALLER_URL=$CONDA_INSTALLER_URL")

echo "Running: ${BUILD_CMD[*]}"
"${BUILD_CMD[@]}"

rc=$?
rm -f "$tmp_dockerfile"
exit $rc
