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
# these by exporting the environment variables before running the script. The Dockerfile
# sets sane defaults for containers; this script will use those if present or fall back
# to the same defaults here so running outside the container still behaves similarly.
: "${NCCL_DEBUG:=INFO}"
: "${NCCL_SOCKET_IFNAME:=enp1s0f1np1}"
: "${NCCL_IB_DISABLE:=0}"
: "${NCCL_P2P_LEVEL:=SYS}"
: "${NCCL_SOCKET_RETRY_CNT:=3}"
: "${NCCL_SOCKET_RETRY_SLEEP_MSEC:=2000}"
: "${NCCL_NET_GDR_LEVEL:=2}"
: "${NCCL_IB_HCA:=mlx5}"

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}
export NCCL_SOCKET_RETRY_CNT=${NCCL_SOCKET_RETRY_CNT:-3}
export NCCL_SOCKET_RETRY_SLEEP_MSEC=${NCCL_SOCKET_RETRY_SLEEP_MSEC:-2000}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5}

# Auto-detect network interface used to reach MASTER_ADDR when the user left the
# default interface (eth0) or did not set NCCL_SOCKET_IFNAME. This helps common
# multi-host setups where the interface name differs between machines.
if [ "${NCCL_SOCKET_IFNAME:-}" = "eth0" ] || [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
  if command -v ip >/dev/null 2>&1; then
    DET_IF=$(ip route get ${MASTER_ADDR} 2>/dev/null | awk '/dev/ {for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' | head -n1)
    if [ -n "${DET_IF}" ]; then
      export NCCL_SOCKET_IFNAME=${DET_IF}
    fi
  fi
fi

echo "Launching DDP: nnodes=${NNODES}, nproc_per_node=${NPROC_PER_NODE}, node_rank=${NODE_RANK}, master_addr=${MASTER_ADDR}, master_port=${MASTER_PORT}"
echo "NCCL_DEBUG=${NCCL_DEBUG}, NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}, NCCL_IB_DISABLE=${NCCL_IB_DISABLE}, NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL}, NCCL_SOCKET_RETRY_CNT=${NCCL_SOCKET_RETRY_CNT}, NCCL_SOCKET_RETRY_SLEEP_MSEC=${NCCL_SOCKET_RETRY_SLEEP_MSEC}, NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL}, NCCL_IB_HCA=${NCCL_IB_HCA}"


# Run torch distributed launcher (torch.distributed.run)
python -m torch.distributed.run \
  --nproc_per_node=${NPROC_PER_NODE} \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train.py "${EXTRA_ARGS[@]}"

echo "DDP launcher exited with status $?"
s