# MaskDINO — Quick Start & Examples

This file collects quick commands and examples to build, run, and train MaskDINO in this repository.

**Build Docker demo image**

Linux / macOS:

```bash
docker build -f Dockerfile.demo -t maskdino-demo:local .
```

Windows (PowerShell):

```powershell
docker build -f Dockerfile.demo -t maskdino-demo:local .
```

**Run container (GPU)**

Mount the repo into the container, enter a shell, then compile ops and run demo:

```bash
docker run --gpus all --rm -it -v "%CD%":/workspace maskdino-demo:local /bin/bash
# inside container
export TORCH_CUDA_ARCH_LIST="12.1+PTX;8.6;8.0;7.5"
/workspace/build_ops.sh
python MaskDINO/demo/demo.py --config-file MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml
```

**Compile CUDA ops at runtime (if needed)**

If the in-image compile fails, compile inside a running container or host with matching CUDA toolkit:

```bash
export TORCH_CUDA_ARCH_LIST="12.1+PTX;8.6;8.0;7.5"
bash /workspace/build_ops.sh
```

**Training examples**

- Single-GPU training using the launcher script:

```bash
python train_mask.py -c MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml -g 1 -o output
```

- Multi-GPU training (example: 4 GPUs):

```bash
python train_mask.py -c MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml -g 4 -o output --batch-size 16 --max-iter 90000
```

The launcher wraps `MaskDINO/train_net.py` with `torch.distributed.run` and forwards `--opts KEY VALUE ...` overrides.

Files you may use:

- Configs: [MaskDINO/configs](MaskDINO/configs#L1)
- Training launcher: [train_mask.py](train_mask.py#L1)
- Alternative launcher: [reain_mask.py](reain_mask.py#L1)
- Dockerfile demo: [Dockerfile.demo](Dockerfile.demo#L1)
- Runtime ops build script: [build_ops.sh](build_ops.sh#L1)

**Checkpoint key mismatch (warning)**

If you see a warning such as:

```
WARNING: The checkpoint state_dict contains keys that are not used by the model: stem.fc.{bias, weight}
```

You can either ignore it (if partial loading is expected) or strip those keys before loading. Example Python snippet to remove keys starting with `stem.fc.` and save a cleaned checkpoint:

```python
import torch
ckpt = torch.load('model.pth', map_location='cpu')
state = ckpt.get('model', ckpt)
filtered = {k: v for k, v in state.items() if not k.startswith('stem.fc.')}
if 'model' in ckpt:
    ckpt['model'] = filtered
torch.save(ckpt, 'model_stripped.pth')
```


python train_mask.py   -c MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml   -g 1   --train-json output_annotations/train_polygons.json   --val-json output_annotations/val_polygons.json   --images-root dataset/images   -o output_maskdino/trainer_output   --resume   --max-iter 3000   --batch-size 12   --lr 0.00005   --num-workers 12   --batch-size-per-image 256

**Troubleshooting**

- If CUDA arch errors occur during op compilation, prefer PTX-first arch list (`12.1+PTX`) in `TORCH_CUDA_ARCH_LIST` and fallback to older arches. currrently it will only run 8
- Ensure `git`, `cmake`, `ninja`, and a compatible `gcc` are available inside the build environment when compiling detectron2 or CUDA extensions.
- If detectron2 fails to install from source, run the install commands manually inside the container so you can inspect errors.

If you'd like, I can add an example config-to-opts mapping, a sample minimal dataset config, or a full reproducible CI-friendly Dockerfile. What should I add next?

**Multi-platform builds**

- **Overview:** CUDA-enabled PyTorch images with GPU support are generally provided for `linux/amd64`. On `arm64` hosts (including Apple Silicon) you typically need a CPU-only base image or must build amd64 images via emulation/cross-build. The `Dockerfile.demo` now accepts a build-arg `BASE_IMAGE` so you can pick the appropriate base image for your platform.

- **Build (native/default):**

```bash
docker build -f Dockerfile.demo -t maskdino-demo:local .
```

- **Build for amd64 (recommended for GPU images) using Buildx:**

```bash
docker buildx build --platform linux/amd64 \
    --build-arg BASE_IMAGE=pytorch/pytorch:2.1.0-cuda12.1-cudnn8-latest \
    -t maskdino-demo:local -f Dockerfile.demo .
```

- **Build for arm64 (CPU-only) using Buildx:**

- **Build for arm64 (CPU-only or arm64-GPU) using Buildx:**

- CPU-only example:

```bash
docker buildx build --platform linux/arm64 \
    --build-arg BASE_IMAGE=ubuntu:22.04 \
    -t maskdino-demo:local -f Dockerfile.demo .
```

Note: the Dockerfile helper will automatically switch to a Ubuntu fallback when an arm64 target is requested but a CUDA/amd64 base image is selected. After building the Ubuntu-based image you should install Miniconda and an aarch64-compatible PyTorch build inside the container (or provide a vendor-provided aarch64 CUDA image via `--base`). Example inside the container:

```bash
# inside container
apt-get update && apt-get install -y wget bzip2
wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-aarch64.sh
bash /tmp/miniconda.sh -b -p /opt/conda
/opt/conda/bin/conda create -y -n maskdino python=3.8
/opt/conda/bin/conda run -n maskdino pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

- arm64 host with NVIDIA GPU (example): replace `BASE_IMAGE` with a matching aarch64/CUDA image provided by your platform (Jetson/L4T, NVIDIA NGC, or a distro-specific image). Example placeholder — replace with the exact tag for your device:

```bash
docker buildx build --platform linux/arm64 \
    --build-arg BASE_IMAGE=<your-aarch64-cuda-image> \
    -t maskdino-demo:local -f Dockerfile.demo .
```

- **Guidance:**
    - If your arm64 machine has an NVIDIA GPU (for example, Jetson devices), you must use a CUDA-enabled aarch64 base image that matches the device's CUDA/JetPack version; generic amd64 CUDA images will not provide GPU passthrough on arm64 hosts. 
    - If no suitable aarch64 CUDA image exists for your platform, consider building from source on the target device or using a vendor-provided container image.

**Build helper (recommended)**

If your builder or buildx setup still resolves the wrong platform variant when using `ARG` in the `FROM` line, use the included helper script which generates a temporary Dockerfile with an explicit `FROM --platform=... <image>` line. This avoids ambiguous resolution and matches the exact image you want to pull.

Basic usage:

```bash
./scripts/build-image.sh --platform linux/arm64 --base pytorch/pytorch:2.1.0-cpu --tag maskdino-demo:local
```

Or use the provided `Makefile` targets:

```bash
make build-amd64       # GPU image for amd64
make build-arm64-cpu   # CPU image for arm64
make build-arm64-gpu   # placeholder target (set base image for your device)
```

**DGX systems (NVIDIA DGX / Spark)**

- On DGX hosts (amd64 with NVIDIA GPUs) build the amd64 CUDA image locally — you don't need arm64. Example:

```bash
make build-dgx
# or
./scripts/build-image.sh --platform linux/amd64 \
    --base pytorch/pytorch:2.1.0-cuda12.1-cudnn8-latest \
    --tag maskdino-demo:dgx
```

- Run on DGX with GPU passthrough:

```bash
docker run --gpus all --rm -it -v "$PWD":/workspace maskdino-demo:dgx /bin/bash
```

- **Run (GPU):** Use a machine with NVIDIA drivers and Docker Desktop/WSL2 or Linux with `nvidia-container-toolkit`:

```bash
docker run --gpus all --rm -it -v "$PWD":/workspace maskdino-demo:local /bin/bash
```

- **Run (CPU-only / arm64):** Omit `--gpus all` and use the CPU image you built.

- **Notes:**
    - Building an amd64 CUDA image on an arm64 host requires QEMU-based emulation or cross-building with `docker buildx`; running that amd64 container with GPU passthrough typically only works on Linux hosts with matching NVIDIA drivers. 
    - If you need, I can add a small Makefile or Windows PowerShell helper to simplify common build targets.






Managed to do the ops:

# create patched copy
cp -a MaskDINO /workspace/MaskDINO_patched

# confirm ops dir exists
ls -la /workspace/MaskDINO_patched/maskdino/modeling/pixel_decoder/ops

# apply replacements (safe, in-patched copy)
find /workspace/MaskDINO_patched -type f \( -name "*.cu" -o -name "*.cuh" -o -name "*.h" -o -name "*.cpp" \) -print0 \
  | xargs -0 sed -i 's/\.type()\.is_cuda()/\.is_cuda()/g'
find /workspace/MaskDINO_patched -type f -name "*.cu" -print0 \
  | xargs -0 perl -0777 -pe "s/AT_DISPATCH_FLOATING_TYPES\s*\(\s*value\.type\s*\(\s*\)\s*,/AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),/gs" -i
find /workspace/MaskDINO_patched -type f \( -name "*.cu" -o -name "*.h" -o -name "*.cpp" \) -print0 \
  | xargs -0 sed -i 's/value.type()/value.scalar_type()/g'

# build from patched ops (force CUDA, set PATH to nvcc)
cd /workspace/MaskDINO_patched/maskdino/modeling/pixel_decoder/ops
chmod +x make.sh || true
FORCE_CUDA=1 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=8.0 sh make.sh









sudo docker run --gpus=all --rm -it \
  --ulimit memlock=-1 --shm-size=8g --ulimit stack=67108864 \
  -v "$(pwd)":/workspace --entrypoint /bin/bash maskdino-demo:latest -c "\
    cd /workspace && \
    # Use the project root config by default (maskdino_drone_config.yaml).
    # Example: train from scratch and override common hyperparameters.
    python scripts/launch_maskdino.py \
      --train-json /workspace/output_annotations/train_polygons_clean.json \
      --val-json /workspace/output_annotations/val_polygons_clean.json \
      --images-root /workspace/dataset/images/train \
      --from-scratch \
      --max-iter 100000 --base-lr 5e-05 --num-workers 8 --ims-per-batch 8 \
      --num-gpus 1 --output /workspace/output"





  sudo docker build -f Dockerfile.demo --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3 --build-arg BUILD_OPS_AT_BUILD=1 -t maskdino-demo:latest .



  windows:

docker run --gpus all --rm -it --ulimit memlock=-1 --shm-size=8g --ulimit stack=67108864 -v "%CD%:/workspace" --entrypoint /bin/bash maskdino-demo:latest -c "cd /workspace && python scripts/launch_maskdino.py --train-json /workspace/output_annotations/train_polygons_clean.json --val-json /workspace/output_annotations/val_polygons_clean.json --images-root /workspace/dataset/images/train --val-images-root /workspace/dataset/images/val --from-scratch --max-iter 100 --base-lr 5e-05 --num-workers 2 --ims-per-batch 2 --num-gpus 1 --output /workspace/output"

docker run --gpus all --rm -it --ulimit memlock=-1 --shm-size=8g --ulimit stack=67108864 -v "%CD%:/workspace" --entrypoint /bin/bash maskdino-demo:latest -c "cd /workspace && python scripts/launch_maskdino.py --train-json /workspace/output_annotations/train_polygons_clean.json --val-json /workspace/output_annotations/val_polygons_clean.json --images-root /workspace/dataset/images/train --val-images-root /workspace/dataset/images/val --fix-json-root --from-scratch --max-iter 100 --base-lr 5e-05 --num-workers 0 --ims-per-batch 1 --resume --low-mem-eval --num-gpus 1 --output /workspace/output MODEL.MaskDINO.NUM_OBJECT_QUERIES 100 MODEL.MaskDINO.TRAIN_NUM_POINTS 4096 INPUT.MIN_SIZE_TEST 800 INPUT.MAX_SIZE_TEST 800"

docker build -f Dockerfile.demo --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3 --build-arg BUILD_OPS_AT_BUILD=1 --build-arg TORCH_CUDA_ARCH_LIST="12.0+PTX;8.6;8.0;7.5" -t maskdino-demo:latest .





docker run --gpus all --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v "%CD%:/workspace" --entrypoint /bin/bash maskdino-demo:latest

Notes:
- The launcher will use `maskdino_drone_config.yaml` at the repo root when `--config-file` is not provided.
- To use a specific MaskDINO config, pass `--config-file MaskDINO/configs/.../your_config.yaml`.
- Use `--from-scratch` to clear `MODEL.WEIGHTS` and infer class count from your `--train-json`.
- Hyperparameters can be overridden from the launcher with `--max-iter`, `--base-lr`, `--num-workers`, and `--ims-per-batch`.






old commands:


sudo docker run --gpus=all --rm -it   --ulimit memlock=-1 --shm-size=8g --ulimit stack=67108864   -v "$(pwd)":/workspace --entrypoint /bin/bash maskdino-demo:latest -c "\
    cd /workspace && \
    python scripts/launch_maskdino.py \
  --train-json /workspace/output_annotations/train_polygons_clean.json \
  --val-json   /workspace/output_annotations/val_polygons_clean.json \
  --images-root /workspace/dataset/images/train \
  --config-file MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml \
  --num-gpus 1 --output /workspace/output"





  sudo docker build -f Dockerfile.demo --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3 --build-arg BUILD_OPS_AT_BUILD=1 -t maskdino-demo:latest .



  windows:

  docker run --gpus all --rm -it --ulimit memlock=-1 --shm-size=8g --ulimit stack=67108864 -v "%CD%:/workspace" --entrypoint /bin/bash maskdino-demo:latest -c "cd /workspace && python scripts/launch_maskdino.py --train-json /workspace/output_annotations/train_polygons_clean.json --val-json /workspace/output_annotations/val_polygons_clean.json --images-root /workspace/dataset/images/train --config-file MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml --num-gpus 1 --output /workspace/output"



docker build -f Dockerfile.demo --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3 --build-arg BUILD_OPS_AT_BUILD=1 --build-arg TORCH_CUDA_ARCH_LIST="12.0+PTX;8.6;8.0;7.5" -t maskdino-demo:latest .





docker run --gpus all --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v "%CD%:/workspace" --entrypoint /bin/bash maskdino-demo:latest




















docker run --gpus all --rm -it --ipc=host --ulimit memlock=-1 --shm-size=16g --ulimit stack=67108864 -v "%CD%:/workspace" --entrypoint /bin/bash maskdino-demo:latest -c "cd /workspace && python scripts/launch_maskdino.py --train-json /workspace/output_annotations/train_polygons_clean.json --val-json /workspace/output_annotations/val_polygons_clean.json --images-root /workspace/dataset/images/train --val-images-root /workspace/dataset/images/val --fix-json-root --from-scratch --max-iter 100 --base-lr 5e-05 --num-workers 4 --ims-per-batch 1 --resume --low-mem-eval-aggressive --num-gpus 1 --output /workspace/output MODEL.MaskDINO.NUM_OBJECT_QUERIES 50 MODEL.MaskDINO.TRAIN_NUM_POINTS 1024 INPUT.MIN_SIZE_TEST 600 INPUT.MAX_SIZE_TEST 800 MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 64"



This does not eat ram on eval:

docker run --gpus all --rm -it --ipc=host --ulimit memlock=-1 --shm-size=16g -v "%CD%:/workspace" --entrypoint /bin/bash maskdino-demo:latest -c "cd /workspace && python scripts/launch_maskdino.py --train-json /workspace/output_annotations/train_polygons_clean.json --val-json /workspace/output_annotations/val_polygons_clean.json --images-root /workspace/dataset/images/train --val-images-root /workspace/dataset/images/val --fix-json-root --max-iter 100 --base-lr 5e-05 --resume --output /workspace/output MODEL.MaskDINO.NUM_OBJECT_QUERIES 8 MODEL.MaskDINO.TRAIN_NUM_POINTS 256 MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 32 TEST.IMS_PER_BATCH 1 DATALOADER.NUM_WORKERS 0 MODEL.MaskDINO.NUM_OBJECT_QUERIES 8 TEST.DETECTIONS_PER_IMAGE 8 --config-file maskdino_drone_config.yaml" 