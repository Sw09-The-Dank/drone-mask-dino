#!/bin/bash
set -euo pipefail
# Entrypoint: optionally build CUDA ops at container start, then run the given command.
# If you need nvcc/toolchain to build extensions, build the image with a devel base:
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel -t maskdino-demo .

WORKDIR=/workspace
cd "$WORKDIR" || true

if [ "${BUILD_OPS:-0}" = "1" ]; then
  echo "BUILD_OPS=1: attempting to build CUDA ops via /workspace/build_ops.sh"
  if [ -x /workspace/build_ops.sh ]; then
    set +e
    /workspace/build_ops.sh
    status=$?
    set -e
    if [ $status -ne 0 ]; then
      echo "Warning: /workspace/build_ops.sh failed with exit code $status."
      echo "If building CUDA ops requires nvcc/toolchain, rebuild image using a devel base or run the devel container manually."
    else
      echo "CUDA ops build completed successfully."
    fi
  else
    echo "No /workspace/build_ops.sh found or not executable; skipping ops build."
  fi
fi

if [ "$#" -eq 0 ]; then
  echo "No command provided; launching interactive shell."
  exec /bin/bash
else
  echo "Executing: $@"
  exec "$@"
fi
