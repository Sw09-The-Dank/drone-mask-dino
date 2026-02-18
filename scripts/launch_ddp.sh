#!/usr/bin/env bash
set -euo pipefail

# launch_ddp.sh
# Simple wrapper to run PyTorch DDP across multiple nodes using torch.distributed.run
# Usage:
#   ./scripts/launch_ddp.sh <nnodes> <nproc_per_node> <node_rank> <master_addr> <master_port> [-- train.py args...]
# Example (on master):
#   ./scripts/launch_ddp.sh 2 8 0 10.0.0.1 29500 -- --config cfg.yaml

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 <nnodes> <nproc_per_node> <node_rank> <master_addr> <master_port> [-- train.py args...]"
  exit 2
fi

NNODES="$1"
NPROC_PER_NODE="$2"
NODE_RANK="$3"
MASTER_ADDR="$4"
MASTER_PORT="$5"
shift 5

# Remaining args after `--` will be forwarded to train.py
EXTRA_ARGS=("$@")

# Recommended NCCL tuning for multi-node GPU training. Adjust interface to match
# your DGX network (e.g. mlx5_0 for InfiniBand, eth0 for ethernet). You can override
# by exporting these variables before running the script.
: "${NCCL_DEBUG:=INFO}"
: "${NCCL_SOCKET_IFNAME:=eth0}"
: "${NCCL_IB_DISABLE:=0}"

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_P2P_LEVEL=SYS

echo "Launching DDP: nnodes=${NNODES}, nproc_per_node=${NPROC_PER_NODE}, node_rank=${NODE_RANK}, master_addr=${MASTER_ADDR}, master_port=${MASTER_PORT}"
echo "NCCL_DEBUG=${NCCL_DEBUG}, NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}, NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"

# Run torch distributed launcher (torch.distributed.run)
python -m torch.distributed.run \
  --nproc_per_node=${NPROC_PER_NODE} \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train.py "${EXTRA_ARGS[@]}"

echo "DDP launcher exited with status $?"
