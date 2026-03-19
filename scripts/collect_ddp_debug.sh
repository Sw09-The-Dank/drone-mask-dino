#!/usr/bin/env bash
set -euo pipefail

OUTDIR=${PWD}/ddp_debug_logs
mkdir -p "$OUTDIR"

echo "Saving debug output to $OUTDIR"

echo "Example docker run that enables NCCL/Torch debug and captures stdout/stderr.\nAdjust NCCL_SOCKET_IFNAME, master addr, and other args as needed."

docker run --gpus all --rm -it --network=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=ALL -e TORCH_DISTRIBUTED_DEBUG=DETAIL \
  -e TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f0np0} \
  -v "$(pwd):/workspace" -w /workspace \
  maskdino-demo:latest \
  /bin/bash -lc "bash ./scripts/m_ddp.sh 2 1 1 169.254.18.231 29500 --fix-json-root --max-iter 10000 --base-lr 5e-05 --resume --output /workspace/output --config-file maskdino_drone_config.yaml TEST.IMS_PER_BATCH 1 DATALOADER.NUM_WORKERS 0 TEST.DETECTIONS_PER_IMAGE 50 2>&1 | tee /workspace/ddp_debug_logs/docker_run.log"

echo "Docker run finished. Logs are in $OUTDIR/docker_run.log"
