#!/usr/bin/env bash
set -euo pipefail

# scripts/m_ddp.ssh - improved helper to run distributed training via torchrun
# Usage:
#   ./scripts/m_ddp.ssh <nnodes> <nproc_per_node> <node_rank> <master_addr> <master_port> -- [train_m.py args...]
# Example (on master):
#   ./scripts/m_ddp.ssh 2 4 0 master.example.com 29500 -- --config-file maskdino_drone_config.yaml --output /workspace/output

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 <nnodes> <nproc_per_node> <node_rank> <master_addr> <master_port> -- [train_m.py args...]"
  exit 2
fi

NNODES="$1"
NPROC_PER_NODE="$2"
NODE_RANK="$3"
MASTER_ADDR="$4"
MASTER_PORT="$5"
shift 5

# Remaining args after `--` will be forwarded to train_m.py
EXTRA_ARGS=("$@")

# If the caller provided an explicit `--` token, remove it so we don't pass
# duplicate `--` tokens to the training script (which can confuse argparse).
if [ "${#EXTRA_ARGS[@]}" -gt 0 ] && [ "${EXTRA_ARGS[0]}" = "--" ]; then
  EXTRA_ARGS=("${EXTRA_ARGS[@]:1}")
fi

# Recommended NCCL tuning for multi-node GPU training. Override via env if needed.
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
# default interface (eth0) or did not set NCCL_SOCKET_IFNAME. Helps common
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

# Force PYTHONPATH to workspace so repository MaskDINO is used by workers
# export PYTHONPATH=/workspace/MaskDINO:/workspace${PYTHONPATH:+:}$PYTHONPATH
# echo "PYTHONPATH=${PYTHONPATH}"

# Run torch.distributed.run (torchrun) and execute the root-level train wrapper
python -m torch.distributed.run \
  --nproc_per_node=${NPROC_PER_NODE} \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train_m.py "${EXTRA_ARGS[@]}"

EXIT_STATUS=$?
echo "DDP launcher exited with status ${EXIT_STATUS}"
exit ${EXIT_STATUS}
