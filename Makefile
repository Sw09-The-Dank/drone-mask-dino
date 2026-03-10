# Makefile for building the demo image with convenient targets

.PHONY: build-amd64 build-dgx build-arm64-cpu build-arm64-gpu build-clean-arm64-cuda

build-amd64:
	./scripts/build-image.sh --platform linux/amd64 --tag maskdino-demo:amd64

build-dgx:
	# DGX systems are amd64 with NVIDIA GPUs; build an amd64 CUDA image locally
	./scripts/build-image.sh --native --tag maskdino-demo:dgx

build-arm64-cpu:
	# Use Ubuntu as a multi-arch fallback on arm64; install PyTorch inside the image
	./scripts/build-image.sh --platform linux/arm64 --tag maskdino-demo:arm64-cpu

build-arm64-gpu:
	# Replace <your-aarch64-cuda-image> with a matching aarch64 CUDA base image for your device
	./scripts/build-image.sh --platform linux/arm64 --base <your-aarch64-cuda-image> --tag maskdino-demo:arm64-gpu

build-clean-arm64-cuda:
	# Remove local image (if present) and build a fresh arm64 CUDA-enabled image with no cache
	-docker rmi -f maskdino-demo:arm64-cuda || true
	./scripts/build-image.sh --platform linux/arm64 --tag maskdino-demo:arm64-cuda --no-cache
