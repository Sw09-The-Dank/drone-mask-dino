#!/usr/bin/env bash
set -euo pipefail

# Run Mask R-CNN training over each dataset variant inside an ablation folder.
#
# Example (host node):
#   ./scripts/run_maskrcnn_ablation.sh \
#     --role host \
#     --master-addr 169.254.18.231 \
#     --ablation-dir /Documents/drone-mask-dino/dataset/ablation
#
# Example (worker node):
#   ./scripts/run_maskrcnn_ablation.sh \
#     --role worker \
#     --master-addr 169.254.18.231 \
#     --ablation-dir /Documents/drone-mask-dino/dataset/ablation

ROLE=""
MASTER_ADDR=""
MASTER_PORT="29500"
AB_DIR=""
IMAGE="maskdino-demo:latest"
CONFIG_FILE="maskrcnn_r50_fpn_1x_25ep_scale.yaml"
NPROC_PER_NODE="2"
NNODES="1"
USE_GPU="1"
NCCL_IFNAME=""
MEMORY="90g"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  run_maskrcnn_ablation.sh --role <host|worker> --master-addr <ip> --ablation-dir <path> [options]

Required:
  --role <host|worker>     DDP node role for this machine.
  --master-addr <ip>       Master node address (same value on both nodes).
  --ablation-dir <path>    Folder containing variant subfolders (e.g. brightness1, crop2, ...).

Optional:
  --master-port <port>     Default: 29500
  --image <name>           Docker image. Default: maskdino-demo:latest
  --config-file <path>     Config file inside workspace. Default: maskrcnn_r50_fpn_1x_25ep_scale.yaml
  --nproc-per-node <n>     GPUs per node for launch_ddp.sh. Default: 2
  --nnodes <n>             Number of nodes for launch_ddp.sh. Default: 1
  --nccl-ifname <name>     Optional NCCL_SOCKET_IFNAME value.
  --memory <size>          Docker memory/memory-swap. Default: 90g
  --cpu-only               Disable --gpus all.
  -h, --help               Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role)
      ROLE="${2:-}"
      shift 2
      ;;
    --master-addr)
      MASTER_ADDR="${2:-}"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="${2:-}"
      shift 2
      ;;
    --ablation-dir)
      AB_DIR="${2:-}"
      shift 2
      ;;
    --image)
      IMAGE="${2:-}"
      shift 2
      ;;
    --config-file)
      CONFIG_FILE="${2:-}"
      shift 2
      ;;
    --nproc-per-node)
      NPROC_PER_NODE="${2:-}"
      shift 2
      ;;
    --nnodes)
      NNODES="${2:-}"
      shift 2
      ;;
    --nccl-ifname)
      NCCL_IFNAME="${2:-}"
      shift 2
      ;;
    --memory)
      MEMORY="${2:-}"
      shift 2
      ;;
    --cpu-only)
      USE_GPU="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$ROLE" != "host" && "$ROLE" != "worker" ]]; then
  echo "--role must be host or worker" >&2
  exit 1
fi

if [[ -z "$MASTER_ADDR" ]]; then
  echo "--master-addr is required" >&2
  exit 1
fi

if [[ -z "$AB_DIR" ]]; then
  echo "--ablation-dir is required" >&2
  exit 1
fi

if [[ ! -d "$AB_DIR" ]]; then
  echo "ablation directory not found: $AB_DIR" >&2
  exit 1
fi

AB_DIR_ABS="$(cd -- "$AB_DIR" && pwd -P)"
REPO_ROOT_ABS="$(cd -- "$REPO_ROOT" && pwd -P)"

if [[ "$AB_DIR_ABS" != "$REPO_ROOT_ABS"/* ]]; then
  echo "ablation directory must be inside repository root." >&2
  echo "repo root: $REPO_ROOT_ABS" >&2
  echo "ablation:  $AB_DIR_ABS" >&2
  exit 1
fi

REL_AB_DIR="${AB_DIR_ABS#"$REPO_ROOT_ABS"/}"
CONTAINER_AB_DIR="/workspace/$REL_AB_DIR"

NODE_RANK="0"
if [[ "$ROLE" == "worker" ]]; then
  NODE_RANK="1"
fi

mapfile -t variants < <(find "$AB_DIR_ABS" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#variants[@]} -eq 0 ]]; then
  echo "No variant directories found under: $AB_DIR" >&2
  exit 1
fi

echo "Found ${#variants[@]} ablation variants."
echo "Role=$ROLE node_rank=$NODE_RANK master=${MASTER_ADDR}:${MASTER_PORT}"

if [[ ! -f "$REPO_ROOT/$CONFIG_FILE" ]]; then
  echo "Config file not found at: $REPO_ROOT/$CONFIG_FILE" >&2
  exit 1
fi

for variant_path in "${variants[@]}"; do
  variant_name="$(basename "$variant_path")"

  if [[ ! -f "$variant_path/train.json" || ! -f "$variant_path/val.json" ]]; then
    echo "Skipping ${variant_name}: missing train.json or val.json in $variant_path" >&2
    continue
  fi

  train_json="${CONTAINER_AB_DIR}/${variant_name}/train.json"
  val_json="${CONTAINER_AB_DIR}/${variant_name}/val.json"
  images_root="${CONTAINER_AB_DIR}/${variant_name}"
  output_dir="output/maskrcnn/${variant_name}"

  echo "============================================================"
  echo "[$(date '+%F %T')] Running variant: ${variant_name}"
  echo "train_json=${train_json}"
  echo "val_json=${val_json}"
  echo "images_root=${images_root}"
  echo "output=${output_dir}"

  extra_env=()
  if [[ -n "$NCCL_IFNAME" ]]; then
    extra_env+=("-e" "NCCL_SOCKET_IFNAME=${NCCL_IFNAME}")
  fi

  docker_args=(run --rm -it
    --network=host
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    --memory="$MEMORY"
    --memory-swap="$MEMORY"
    -v /tmp/empty:/opt/hpcx/nccl_rdma_sharp_plugin:ro
    -v "$REPO_ROOT:/workspace" -w /workspace
  )

  if [[ "$USE_GPU" == "1" ]]; then
    docker_args=(run --gpus all --rm -it
      --network=host
      --ipc=host
      --ulimit memlock=-1
      --ulimit stack=67108864
      --memory="$MEMORY"
      --memory-swap="$MEMORY"
      -v /tmp/empty:/opt/hpcx/nccl_rdma_sharp_plugin:ro
      -v "$REPO_ROOT:/workspace" -w /workspace
    )
  fi

  docker "${docker_args[@]}" \
    "${extra_env[@]}" \
    "$IMAGE" \
    /bin/bash -lc "
      bash ./scripts/launch_ddp.sh ${NPROC_PER_NODE} ${NNODES} ${NODE_RANK} ${MASTER_ADDR} ${MASTER_PORT} \
      --train-json ${train_json} \
      --val-json ${val_json} \
      --images-root ${images_root} \
      --output ${output_dir} \
      --config-file ${CONFIG_FILE} \
      --no-resume
    "
done

echo "All variants completed for role: $ROLE"