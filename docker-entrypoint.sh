#!/bin/bash
set -euo pipefail
# Entrypoint: optionally build CUDA ops at container start, then run the given command.
# If you need nvcc/toolchain to build extensions, build the image with a devel base:
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel -t maskdino-demo .

WORKDIR=/workspace
cd "$WORKDIR" || true

if [ "${BUILD_OPS:-0}" = "1" ]; then
  echo "BUILD_OPS=1: attempting to build CUDA ops via /workspace/build_ops.sh"
  echo "Applying in-image compatibility patches for MaskDINO sources (non-destructive)"
  set +e
  PATCH_DIR="/workspace/MaskDINO_patched"
  rm -rf "$PATCH_DIR" || true
  if [ -d MaskDINO ]; then
    echo "Copying MaskDINO to $PATCH_DIR"
    cp -a MaskDINO "$PATCH_DIR"
  elif [ -d third_party/MaskDINO ]; then
    echo "Copying third_party/MaskDINO to $PATCH_DIR"
    cp -a third_party/MaskDINO "$PATCH_DIR"
  else
    echo "No MaskDINO source directories found to patch."
    PATCH_DIR=""
  fi

  if [ -n "$PATCH_DIR" ] && [ -d "$PATCH_DIR" ]; then
    echo "Patching sources in $PATCH_DIR"
    # Replace deprecated .type().is_cuda() usages
    find "$PATCH_DIR" -type f \( -name "*.cu" -o -name "*.cuh" -o -name "*.h" -o -name "*.cpp" \) -print0 | xargs -0 -r sed -i "s/\\.type()\\.is_cuda()/\\.is_cuda()/g"
    # Replace common AT_DISPATCH pattern variations with scalar_type()
    find "$PATCH_DIR" -type f \( -name "*.cu" -o -name "*.cuh" -o -name "*.h" -o -name "*.cpp" \) -print0 | xargs -0 -r perl -0777 -pe "s/AT_DISPATCH_FLOATING_TYPES\s*\(\s*value\.type\s*\(\s*\)\s*,/AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),/gs" -i
    # Also replace any remaining occurrences of 'value.type()' used to fetch scalar types
    find "$PATCH_DIR" -type f \( -name "*.cu" -o -name "*.cuh" -o -name "*.h" -o -name "*.cpp" \) -print0 | xargs -0 -r sed -i "s/value.type()/value.scalar_type()/g"
  fi
  set -e
  if [ -x /workspace/build_ops.sh ]; then
    # First try building from the patched copy (if present)
    if [ -d /workspace/MaskDINO_patched ]; then
      echo "Attempting build from patched copy: /workspace/MaskDINO_patched"
      set +e
      (cd /workspace/MaskDINO_patched/maskdino/modeling/pixel_decoder/ops && \
        chmod +x make.sh || true && \
        TRY_LISTS=("12.0;12.1+PTX;8.6;8.0;7.5" "12.1+PTX;8.6;8.0;7.5" "8.6;8.0;7.5") && \
        for LIST in "${TRY_LISTS[@]}"; do \
          echo "Attempting build with TORCH_CUDA_ARCH_LIST=${LIST}"; \
          if TORCH_CUDA_ARCH_LIST="$LIST" sh make.sh; then echo "Build succeeded with ${LIST}"; exit 0; else echo "Build failed with ${LIST}"; fi; \
        done; exit 1)
      status=$?
      set -e
      if [ $status -ne 0 ]; then
        echo "Patched build failed (exit $status). Falling back to /workspace/build_ops.sh"
        set +e
        /workspace/build_ops.sh
        status=$?
        set -e
      fi
    else
      /workspace/build_ops.sh
      status=$?
    fi

    if [ $status -ne 0 ]; then
      echo "Warning: CUDA ops build failed with exit code $status."
      echo "If building CUDA ops requires nvcc/toolchain, rebuild image using a devel base or run the devel container manually."
    else
      echo "CUDA ops build completed successfully."
    fi
  else
    echo "No /workspace/build_ops.sh found or not executable; skipping ops build."
  fi
fi

if [ "$#" -eq 0 ]; then
  echo "No command provided; launching interactive shell."
  exec /bin/bash
else
  echo "Executing: $@"
  exec "$@"
fi
