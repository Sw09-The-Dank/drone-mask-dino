#!/usr/bin/env bash
set -uo pipefail  # -e omitted so a failed run does not abort the whole sweep

# ============================================================
# CONFIGURATION — edit these constants to control the sweep
# ============================================================

# --- Docker / DDP infrastructure ---
IMAGE="maskdino-demo:latest"
CONFIG_FILE="maskrcnn_r50_fpn_1x_25ep_scale.yaml"
MASTER_PORT="29500"
NNODES="2"
NPROC_PER_NODE="1"
MEMORY="90g"

# --- Dataset paths (inside the container, /workspace maps to repo root) ---
TRAIN_JSON="/workspace/dataset/scale25/train.json"
VAL_JSON="/workspace/dataset/scale25/val.json"
IMAGES_ROOT="/workspace/dataset/scale25"

# --- Base output directory (under repo root) ---
OUTPUT_BASE="output/aug_sweep"

# --- Gaussian Blur runs: "prob sigma_min sigma_max" ---
# prob is fixed at 0.5 across all runs; only sigma range varies (no overlap).
BLUR_RUN_1="0.5 0.1 1.0"
BLUR_RUN_2="0.5 1.0 2.0"
BLUR_RUN_3="0.5 2.0 3.0"

# --- Gaussian Noise runs: "prob std_min std_max" ---
# prob is fixed at 0.5 across all runs; only std range varies (no overlap).
NOISE_RUN_1="0.5 0.0 10.0"
NOISE_RUN_2="0.5 10.0 20.0"
NOISE_RUN_3="0.5 20.0 30.0"

# --- Motion Blur runs: "prob len_min len_max" ---
# prob is fixed at 0.5 across all runs; only kernel length range varies (no overlap).
MBLUR_RUN_1="0.5 0 10"
MBLUR_RUN_2="0.5 10 20"
MBLUR_RUN_3="0.5 20 30"

# ============================================================
# END OF CONFIGURATION
# ============================================================

ROLE=""
MASTER_ADDR=""
NCCL_IFNAME=""
USE_SUDO_DOCKER="0"
SSH_HOST_USER=""
DROP_CACHES="0"
USE_GPU="1"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  run_augmentation_sweep.sh --role <host|worker> --master-addr <ip> [options]

Required:
  --role <host|worker>     DDP node role for this machine.
  --master-addr <ip>       Master node IP (same value on both nodes).

Optional:
  --master-port <port>     Default: 29500
  --nccl-ifname <name>     NCCL_SOCKET_IFNAME (e.g. enp1s0f1np1).
  --memory <size>          Docker --memory/--memory-swap. Default: 90g
  --sudo-docker            Run docker commands via sudo.
  --ssh-host-user <user>   SSH user for master node (worker only), used to skip
                           already-completed runs. Example: --ssh-host-user spark1gm
  --drop-caches            Drop host kernel page cache between runs.
  --cpu-only               Disable --gpus all.
  -h, --help               Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role)           ROLE="${2:-}";           shift 2 ;;
    --master-addr)    MASTER_ADDR="${2:-}";    shift 2 ;;
    --master-port)    MASTER_PORT="${2:-}";    shift 2 ;;
    --nccl-ifname)    NCCL_IFNAME="${2:-}";    shift 2 ;;
    --memory)         MEMORY="${2:-}";         shift 2 ;;
    --sudo-docker)    USE_SUDO_DOCKER="1";     shift   ;;
    --ssh-host-user)  SSH_HOST_USER="${2:-}";  shift 2 ;;
    --drop-caches)    DROP_CACHES="1";         shift   ;;
    --cpu-only)       USE_GPU="0";             shift   ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$ROLE" != "host" && "$ROLE" != "worker" ]]; then
  echo "--role must be 'host' or 'worker'" >&2
  exit 1
fi
if [[ -z "$MASTER_ADDR" ]]; then
  echo "--master-addr is required" >&2
  exit 1
fi

NODE_RANK="0"
[[ "$ROLE" == "worker" ]] && NODE_RANK="1"

# Build the list of runs as parallel arrays: run_name, aug_type, arg_string
run_names=()
aug_args=()

for i in 1 2 3; do
  var="BLUR_RUN_${i}"
  params="${!var}"
  read -r prob smin smax <<< "$params"
  run_names+=("blur_run${i}_p${prob}_s${smin}-${smax}")
  aug_args+=("--gaussian-blur ${prob} ${smin} ${smax}")
done

for i in 1 2 3; do
  var="NOISE_RUN_${i}"
  params="${!var}"
  read -r prob smin smax <<< "$params"
  run_names+=("noise_run${i}_p${prob}_std${smin}-${smax}")
  aug_args+=("--gaussian-noise ${prob} ${smin} ${smax}")
done

for i in 1 2 3; do
  var="MBLUR_RUN_${i}"
  params="${!var}"
  read -r prob lmin lmax <<< "$params"
  run_names+=("motionblur_run${i}_p${prob}_len${lmin}-${lmax}")
  aug_args+=("--motion-blur ${prob} ${lmin} ${lmax}")
done

echo "Augmentation sweep: ${#run_names[@]} runs total."
echo "Role=${ROLE}  node_rank=${NODE_RANK}  master=${MASTER_ADDR}:${MASTER_PORT}"
echo ""

DOCKER_BIN=(docker)
if [[ "$USE_SUDO_DOCKER" == "1" ]]; then
  DOCKER_BIN=(sudo docker)
fi

if ! "${DOCKER_BIN[@]}" info >/dev/null 2>&1; then
  if [[ "$USE_SUDO_DOCKER" == "0" ]] && command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then
    echo "Docker requires elevated permissions — auto-switching to sudo docker."
    DOCKER_BIN=(sudo docker)
  else
    echo "Cannot access Docker daemon." >&2
    echo "Try: rerun with --sudo-docker, or add your user to the docker group." >&2
    exit 1
  fi
fi

ABORT=0
SUDO_KEEPALIVE_PID=""

cleanup() {
  [[ -n "$SUDO_KEEPALIVE_PID" ]] && kill "$SUDO_KEEPALIVE_PID" >/dev/null 2>&1 || true
}
abort_handler() {
  ABORT=1
  echo "" >&2
  echo "[$(date '+%F %T')] Interrupted — finishing current Docker call then stopping." >&2
}
trap cleanup EXIT
trap abort_handler INT TERM

if [[ "${DOCKER_BIN[0]}" == "sudo" ]]; then
  echo "Authenticating sudo once for the full sweep..."
  sudo -v
  ( while true; do sudo -n true; sleep 50; done ) &
  SUDO_KEEPALIVE_PID="$!"
fi

for idx in "${!run_names[@]}"; do
  run_name="${run_names[$idx]}"
  extra_aug="${aug_args[$idx]}"
  output_dir="${OUTPUT_BASE}/${run_name}"

  # Skip if already finished
  already_done=0
  if [[ "$ROLE" == "worker" && -n "$SSH_HOST_USER" ]]; then
    repo_rel_home="${REPO_ROOT#"${HOME}/"}"
    remote_check="~/${repo_rel_home}/${output_dir}/model_final.pth"
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "${SSH_HOST_USER}@${MASTER_ADDR}" \
         "test -f ${remote_check}" 2>/dev/null; then
      already_done=1
    fi
  elif [[ -f "${REPO_ROOT}/${output_dir}/model_final.pth" ]]; then
    already_done=1
  fi

  if [[ "$already_done" == "1" ]]; then
    echo "[$(date '+%F %T')] Skipping ${run_name}: model_final.pth already exists."
    continue
  fi

  echo "============================================================"
  echo "[$(date '+%F %T')] Starting run: ${run_name}"
  echo "  aug args : ${extra_aug}"
  echo "  output   : ${output_dir}"

  extra_env=()
  if [[ -n "$NCCL_IFNAME" ]]; then
    extra_env+=("-e" "NCCL_SOCKET_IFNAME=${NCCL_IFNAME}")
  fi

  if [[ "$USE_GPU" == "1" ]]; then
    docker_args=(run --gpus all --rm -i
      --network=host --ipc=host
      --ulimit memlock=-1 --ulimit stack=67108864
      --memory="$MEMORY" --memory-swap="$MEMORY"
      -v /tmp/empty:/opt/hpcx/nccl_rdma_sharp_plugin:ro
      -v "$REPO_ROOT:/workspace" -w /workspace
    )
  else
    docker_args=(run --rm -i
      --network=host --ipc=host
      --ulimit memlock=-1 --ulimit stack=67108864
      --memory="$MEMORY" --memory-swap="$MEMORY"
      -v /tmp/empty:/opt/hpcx/nccl_rdma_sharp_plugin:ro
      -v "$REPO_ROOT:/workspace" -w /workspace
    )
  fi

  run_exit=0
  # shellcheck disable=SC2086
  "${DOCKER_BIN[@]}" "${docker_args[@]}" \
    "${extra_env[@]}" \
    "$IMAGE" \
    /bin/bash -lc "
      bash ./scripts/launch_ddp.sh ${NNODES} ${NPROC_PER_NODE} ${NODE_RANK} ${MASTER_ADDR} ${MASTER_PORT} \
        --train-json ${TRAIN_JSON} \
        --val-json   ${VAL_JSON} \
        --images-root ${IMAGES_ROOT} \
        --output ${output_dir} \
        --config-file ${CONFIG_FILE} \
        ${extra_aug} \
        --no-resume
    " || run_exit=$?

  if [[ "$run_exit" -ne 0 ]]; then
    if [[ "$run_exit" -eq 130 || "$run_exit" -eq 143 || "$ABORT" -eq 1 ]]; then
      echo "[$(date '+%F %T')] Aborted by user during run ${run_name}." >&2
      exit 1
    fi
    echo "[$(date '+%F %T')] ERROR: run ${run_name} failed (exit ${run_exit}). Continuing to next run." >&2
  else
    echo "[$(date '+%F %T')] Finished: ${run_name}"
  fi

  [[ "$ABORT" -eq 1 ]] && { echo "[$(date '+%F %T')] Aborted. Stopping." >&2; exit 1; }

  if [[ "$DROP_CACHES" == "1" ]]; then
    echo "[$(date '+%F %T')] Dropping host kernel page cache..."
    sync
    if echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null 2>&1; then
      echo "[$(date '+%F %T')] Page cache dropped."
    else
      echo "[$(date '+%F %T')] WARN: drop_caches requires passwordless sudo. Cache NOT dropped." >&2
    fi
  fi

  sleep 30
done

echo ""
echo "All augmentation sweep runs completed for role: $ROLE"
