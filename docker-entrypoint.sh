#!/bin/bash
# Entry point: run training if RUN_TRAIN=1, otherwise run provided command (default: bash)
set -e
if [ "${RUN_TRAIN:-0}" = "1" ]; then
  exec /opt/venv/bin/python train.py
fi

if [ "$#" -eq 0 ]; then
  exec /bin/bash
else
  exec "$@"
fi
