#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_demo.sh [docker-image] -- [demo-args]
# Example: ./run_demo.sh drone-mask-dino:latest -- --config configs/demo.yaml

IMAGE=${1:-drone-mask-dino:latest}
shift || true

# If user passes `--` we will forward remaining args; otherwise all args are forwarded.
DEMO_ARGS=("$@")

DOCKER_SHM_SIZE=${DOCKER_SHM_SIZE:-8g}
DOCKER_ARGS=(--gpus all -it --rm --shm-size "${DOCKER_SHM_SIZE}" -v "$(pwd)":/workspace -w /workspace -e PYTHONPATH=/workspace)

# If X11 is available on the host, forward display for GUI demos
if [ -n "${DISPLAY:-}" ]; then
  DOCKER_ARGS+=( -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix )
fi

docker run "${DOCKER_ARGS[@]}" "$IMAGE" /opt/venv/bin/python MaskDINO/demo/demo.py "${DEMO_ARGS[@]}"
