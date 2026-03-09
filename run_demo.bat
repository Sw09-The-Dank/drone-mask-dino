@echo off
REM Usage: run_demo.bat [image] [-- demo-args]
SETLOCAL ENABLEDELAYEDEXPANSION
IF "%~1"=="" (
  set IMAGE=drone-mask-dino:latest
) ELSE (
  set IMAGE=%~1
  shift
)

REM Mount current dir and run MaskDINO demo inside the container
docker run --gpus all -it --rm -v "%cd%":/workspace -w /workspace -e PYTHONPATH=/workspace %IMAGE% /opt/venv/bin/python MaskDINO/demo/demo.py %*
