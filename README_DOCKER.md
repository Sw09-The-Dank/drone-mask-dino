# Docker / GPU usage for this repository


Build image (from repository root)

- Linux / WSL:

```bash
# If your user requires sudo to access the Docker daemon
sudo docker build -t drone-maskdino:latest .
```

- Windows (PowerShell / CMD):

```powershell
docker build -t drone-maskdino:latest .
```

Pull base image separately (optional, helps diagnose network/auth issues):

```bash
docker pull nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04
```

Run examples
- Linux / WSL (bash):

```bash
# mount current folder and run the image (prints GPU info then runs training)
sudo docker run --gpus all --rm -it \
  -v "$(pwd):/workspace" \
  -w /workspace \
  drone-maskdino:latest \
  /bin/bash -lc "nvidia-smi && python train.py"
```

- PowerShell (Windows):

```powershell
# Use ${PWD} expansion in PowerShell
docker run --gpus all --rm -it `
  -v "${PWD}:/workspace" `
  -w /workspace `
  drone-maskdino:latest `
  /bin/bash -lc "nvidia-smi && python train.py"
```

- CMD (Windows):

```bat
REM Use %%cd%% expansion in CMD
docker run --gpus all --rm -it -v "%cd%:/workspace" -w /workspace drone-maskdino:latest /bin/bash -lc "nvidia-smi && python train.py"
```

- Interactive shell (inspect container, then run commands manually):

```bash
docker run --gpus all --rm -it -v "$(pwd):/workspace" -w /workspace drone-maskdino:latest /bin/bash
# inside container, python and venv are on PATH; run:
python train.py
```

- Run detached in background (write logs to host):

```bash
docker run --gpus all -d --name drone_train \
  -v "$(pwd):/workspace" -w /workspace \
  drone-maskdino:latest \
  /bin/bash -lc "nvidia-smi && python train.py > /workspace/train.log 2>&1"
```

- Run as current host user (preserve file ownership on Linux):

```bash
docker run --gpus all --rm -it -u $(id -u):$(id -g) -v "$(pwd):/workspace" -w /workspace drone-maskdino:latest /bin/bash -lc "nvidia-smi && python train.py"
```

Docker Compose minimal example (docker-compose.yml):

```yaml
version: '3.8'
services:
  trainer:
    image: drone-maskdino:latest
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    volumes:
      - ./:/workspace
    working_dir: /workspace
    command: /bin/bash -lc "nvidia-smi && python train.py"
```

Notes & troubleshooting
- Use the correct path expansion for your shell: `$(pwd)` (bash/WSL), `${PWD}` (PowerShell), `%cd%` (CMD).
- If the container cannot access host files under OneDrive, enable file sharing for that folder in Docker Desktop or move the repo to a non-OneDrive path.
- `--gpus all` requires NVIDIA drivers and NVIDIA Container Toolkit (or Docker Desktop WSL GPU support). If `nvidia-smi` fails, check host `nvidia-smi` first.
- Building step compiles `detectron2` from source — allow several minutes and ensure `cmake`, `gcc`, and CUDA toolkit are present.
- To avoid long builds, you can use prebuilt wheels for `detectron2` matching your `torch`/CUDA versions; ask me to pin specific versions for your host drivers.

Multi-node DDP (2 DGX nodes)
--------------------------------
This repository includes `scripts/launch_ddp.sh`, a small wrapper that runs
PyTorch's torch.distributed launcher (`torch.distributed.run`) and sets common
NCCL tuning environment variables. Build the image with NCCL present (the
`Dockerfile` installs NVIDIA's NCCL packages from the NVIDIA apt repo).

Example (2 nodes, 8 GPUs per node).
- Build the image on both hosts (or push/pull to a shared registry):

```bash
docker build -t drone-maskdino:latest .
```

- Run on the master node (replace `MASTER_IP` with the master node IP):

```bash
docker run --gpus all --rm -it --network=host \
  -v "$(pwd):/workspace" -w /workspace \
  drone-maskdino:latest \
  /bin/bash -lc "./scripts/launch_ddp.sh 2 8 0 MASTER_IP 29500 --"
```

- Run on the worker node:

```bash
docker run --gpus all --rm -it --network=host \
  -v "$(pwd):/workspace" -w /workspace \
  drone-maskdino:latest \
  /bin/bash -lc "./scripts/launch_ddp.sh 2 8 1 MASTER_IP 29500 --"
```
Notes:
- `2` = number of nodes (`--nnodes`), `8` = GPUs per node (`--nproc_per_node`).
- `MASTER_IP` should be reachable from the worker; use a private cluster IP.
- `--network=host` avoids container networking/NAT issues for rendezvous ports.
- You can override NCCL tuning variables by exporting them before running:
Quick single-node master test (1 GPU)
----------------------------------
To quickly verify the distributed setup and NCCL networking without bringing up a second node, run the launcher on a single host with one GPU:

```bash
docker run --gpus "device=0" --rm -it --network=host \
  -v "$(pwd):/workspace" -w /workspace \
  drone-maskdino:latest \
  /bin/bash -lc "./scripts/launch_ddp.sh 1 1 0 169.254.18.231 29500 --"
```
```bash
sudo docker run --gpus "device=0" --rm -it --network=host \
  -v "$(pwd):/workspace" -w /workspace \
  drone-maskdino:latest \
  /bin/bash -lc "bash ./scripts/launch_ddp.sh 1 1 0 169.254.18.231 29500 --"
```



```bash
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0   # or mlx5_0 for InfiniBand
export NCCL_IB_DISABLE=0         # 1 to disable InfiniBand if not used
```

- If your cluster provides an internal container image with NCCL preinstalled
  (DGX images / NVIDIA NGC), prefer that base image and skip the NCCL install.

Run tips: shared memory, IPC, CUDA forward-compatibility, and ulimits
---------------------------------------------------------------

- NOTE: you may see a CUDA forward-compatibility message such as:
  "CUDA Forward Compatibility mode ENABLED. Using CUDA 13.1 driver version X
  with kernel driver version Y." This is usually acceptable; only act if you
  hit runtime CUDA errors. See https://docs.nvidia.com/deploy/cuda-compatibility/.

- PyTorch multiprocessing and NCCL can require more shared memory than the
  Docker default (64MB). If you see hangs, NCCL errors, or process stalls,
  increase shared memory or use the host IPC namespace. Two common options:

  1) Prefer full host IPC (simplest; less isolation):

```bash
docker run --gpus all --rm -it \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$(pwd):/workspace" -w /workspace \
  drone-maskdino:latest \
  /bin/bash -lc "nvidia-smi && bash ./scripts/launch_ddp.sh 1 1 0 169.254.18.231 29500 --"
```

  2) If you prefer to keep IPC isolation, increase `/dev/shm` instead:

```bash
docker run --gpus all --rm -it --shm-size=1g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$(pwd):/workspace" -w /workspace drone-maskdino:latest \
  /bin/bash -lc "python train.py"
```

- `--ulimit memlock=-1` and `--ulimit stack=67108864` are recommended to
  avoid pinned-memory or large-stack failures during training.

- Security/performance tradeoff: `--ipc=host` reduces container isolation.
  Use `--shm-size` when isolation is required or when running on multi-tenant
  hosts.



Not working
Master
```bash
sudo docker run --gpus all --rm -it \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$(pwd):/workspace" -w /workspace \
  drone-maskdino:latest \
  /bin/bash -lc "nvidia-smi && bash ./scripts/launch_ddp.sh 2 1 0 169.254.18.231 29500 --"
```

Worker
```bash
sudo docker run --gpus all --rm -it \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$(pwd):/workspace" -w /workspace \
  drone-maskdino:latest \
  /bin/bash -lc "nvidia-smi && bash ./scripts/launch_ddp.sh 2 1 1 169.254.18.231 29500 --"
```

working!
master
```bash
sudo docker run --gpus all --rm -it --network=host --ipc=host \
  -e NCCL_DEBUG=INFO -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_IB_DISABLE=0 \
  -v "$(pwd):/workspace" -w /workspace \
  drone-maskdino:latest \
  /bin/bash -lc "bash ./scripts/launch_ddp.sh 2 1 0 169.254.18.231 29500 -- --max-iter 3000 --ims-per-batch 32 --base-lr 0.00001 --num-workers 10 --no-resume"
```

worker
```bash
sudo docker run --gpus all --rm -it --network=host --ipc=host \
  -e NCCL_DEBUG=INFO -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_IB_DISABLE=0 \
  -v "$(pwd):/workspace" -w /workspace \
  drone-maskdino:latest \
  /bin/bash -lc "bash ./scripts/launch_ddp.sh 2 1 1 169.254.18.231 29500 -- --max-iter 3000 --ims-per-batch 32 --base-lr 0.00001 --num-workers 10 --no-resume"
```