# Docker / GPU usage for this repository


Build image (from repository root):

```bash
docker build -t drone-maskdino:latest .
```

Pull base image separately (optional, helps diagnose network/auth issues):

```bash
docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
```

Run examples
- Linux / WSL (bash):

```bash
# mount current folder and run the image (prints GPU info then runs training)
docker run --gpus all --rm -it \
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

