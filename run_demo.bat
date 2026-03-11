@echo off
REM Usage: run_demo.bat [image] [-- demo-args]
SETLOCAL ENABLEDELAYEDEXPANSION
IF "%~1"=="" (
  set IMAGE=drone-mask-dino:latest
) ELSE (
  set IMAGE=%~1
  shift
)

REM Default shm size if not set
IF "%DOCKER_SHM_SIZE%"=="" (
  set DOCKER_SHM_SIZE=8g
)

REM Mount current dir and run MaskDINO demo inside the container
docker run --gpus all -it --rm --shm-size %DOCKER_SHM_SIZE% -v "%cd%":/workspace -w /workspace -e PYTHONPATH=/workspace %IMAGE% /opt/venv/bin/python MaskDINO/demo/demo.py %*
