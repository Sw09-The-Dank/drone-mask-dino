FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
LABEL maintainer="" \
    description="drone-mask-dino image with CUDA-enabled PyTorch and detectron2"

# Ensure compiled CUDA extensions include support for recent GPUs (e.g. sm_120)
# Adjust the list if you know specific architectures to include.
ENV TORCH_CUDA_ARCH_LIST="12.0;11.8;11.7;11.6;11.3;11.1;11.0;10.2;10.1;10.0;9.0;9.5;8.6;8.0;7.5"

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common dirmngr gnupg ca-certificates \
    build-essential wget git curl bzip2 \
    libglib2.0-0 libsm6 libxrender1 libxext6 pkg-config cmake \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-venv python3-pip python3-dev python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-lc"]

WORKDIR /workspace

# Copy requirements before the rest of the repository for Docker cache efficiency
COPY requirements.txt /workspace/requirements.txt

# Install NCCL runtime and development packages for multi-node GPU collectives
# Add the official NVIDIA apt repository (Ubuntu 24.04) and install matching
# `libnccl2` and `libnccl-dev`. This ensures NCCL matches the CUDA toolkit
# present in the base image. If your environment already provides NCCL (DGX
# images usually do), this step is safe and will be a no-op.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends gnupg2 curl ca-certificates; \
    mkdir -p /etc/apt/keyrings; \
    # Fetch NVIDIA repo signing key and add it (key may change; fallback to continue)
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
        | gpg --dearmor -o /etc/apt/keyrings/nvidia-archive-keyring.gpg || true; \
    echo "deb [signed-by=/etc/apt/keyrings/nvidia-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
        > /etc/apt/sources.list.d/nvidia-cuda.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends libnccl2 libnccl-dev || true; \
    rm -rf /var/lib/apt/lists/*;

# Create virtualenv and install CUDA-enabled PyTorch (CUDA 13 wheels) and other deps via pip
RUN python3.11 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel \
        # Install PyTorch wheels for CUDA 13 (nightly cu131). Fail build if unavailable.
         # Allow pre-release/nightly wheels and use extra-index-url so pip can find cu131 builds
         && /opt/venv/bin/pip install --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu131 --no-cache-dir \
             torch torchvision torchaudio \
        && /opt/venv/bin/pip install -r /workspace/requirements.txt

# Ensure the virtualenv has setuptools (provides pkg_resources) so imports
# from the venv (used at runtime) don't fail. Also upgrade system pip/setuptools
# as a fallback.
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel || true \
    && python3 -m pip install --upgrade pip setuptools || true

# Install detectron2 from source (will compile against CUDA/toolkit present in base image)
# Use --no-build-isolation so the build can import the already-installed torch from the venv
RUN /opt/venv/bin/pip install --no-build-isolation --no-cache-dir "git+https://github.com/facebookresearch/detectron2.git"

# Copy repo
COPY . /workspace

ENV PATH=/opt/venv/bin:${PATH}

# Make `python` and `pip` point to the virtualenv executables so user commands
# like `python train.py` use the venv even when invoked directly.
RUN ln -sf /opt/venv/bin/python /usr/local/bin/python \
    && ln -sf /opt/venv/bin/pip /usr/local/bin/pip || true

# Force-reinstall a stable setuptools into the venv and verify pkg_resources
# so runtime imports succeed.
RUN /opt/venv/bin/pip install --no-cache-dir --force-reinstall "setuptools==65.6.3" || /opt/venv/bin/pip install --no-cache-dir --force-reinstall setuptools || true \
    && /opt/venv/bin/python -c "import pkg_resources; print('BUILD-CHECK pkg_resources OK', getattr(pkg_resources,'__file__',None))"

# Default command: run training script using the created virtualenv
CMD ["/opt/venv/bin/python", "train.py"]

