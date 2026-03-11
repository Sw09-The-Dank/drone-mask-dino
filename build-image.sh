#!/usr/bin/env bash
set -euo pipefail
# Helper to build Docker image and pick an appropriate BASE_IMAGE for the host arch.
# Usage: ./build-image.sh [--devel] [--tag name]

DEVEL=0
TAG=maskdino-demo
while [[ $# -gt 0 ]]; do
  case "$1" in
    --devel) DEVEL=1; shift ;;
    --tag) TAG="$2"; shift 2 ;;
    -t) TAG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

ARCH=$(uname -m)
echo "Host arch detected: $ARCH"

if [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm64" ]]; then
  # Use an NVIDIA NGC pytorch image that commonly has arm64 builds; you can override
  # by setting BASE_IMAGE environment variable or passing --build-arg BASE_IMAGE=...
  if [[ "$DEVEL" -eq 1 ]]; then
    BASE_IMAGE="nvcr.io/nvidia/pytorch:26.01-py3"
  else
    BASE_IMAGE="nvcr.io/nvidia/pytorch:26.01-py3"
  fi
else
  # x86_64 default PyTorch CUDA runtime
  if [[ "$DEVEL" -eq 1 ]]; then
    BASE_IMAGE="pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel"
  else
    BASE_IMAGE="pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime"
  fi
fi

echo "Using BASE_IMAGE=${BASE_IMAGE}"
docker build -f Dockerfile.demo --build-arg BASE_IMAGE="${BASE_IMAGE}" -t "${TAG}" .
