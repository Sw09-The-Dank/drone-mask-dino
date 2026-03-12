#!/usr/bin/env bash
set -ex
# Build the CUDA ops from the patched copy (or repo copy if patched absent)
OPS_DIR=/workspace/MaskDINO_patched/maskdino/modeling/pixel_decoder/ops
if [ ! -d "$OPS_DIR" ]; then
  OPS_DIR=/workspace/MaskDINO/maskdino/modeling/pixel_decoder/ops
fi
if [ ! -d "$OPS_DIR" ]; then
  echo "No ops directory found at expected locations: /workspace/MaskDINO_patched/... or /workspace/MaskDINO/..."
  exit 0
fi

cd "$OPS_DIR"
sed -i 's/\r$//' make.sh || true
chmod +x make.sh || true

# Clean older build artifacts and any Python-3.8 specific outputs so the runtime
# interpreter cannot accidentally load the wrong ABI-specific .so.
rm -rf build build/* || true
find . -type f -name '*cpython-38*' -print -delete || true
find /workspace -type f -name '*cpython-38*' -print -delete || true

# Determine a reasonable TORCH_CUDA_ARCH_LIST and include +PTX for portability
TORCH_CUDA_ARCH_LIST=$(python - <<'PY'
import sys
try:
    import torch
    p = torch.cuda.get_device_properties(0)
    # add +PTX to allow PTX fallback when necessary
    print(f"{p.major}.{p.minor}+PTX")
except Exception:
    # safe default if torch/device inspection fails
    print("12.0+PTX;8.6;8.0;7.5")
    sys.exit(0)
PY
)
echo "Using TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

export FORCE_CUDA=1
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

# Try the preferred list then fallbacks (avoid unsupported 13.* tokens)
TRY_LISTS=("${TORCH_CUDA_ARCH_LIST}" "12.0+PTX;8.6;8.0;7.5" "8.6;8.0;7.5" "8.0;7.5;7.0")
for LIST in "${TRY_LISTS[@]}"; do
  echo "Attempting make.sh with TORCH_CUDA_ARCH_LIST=$LIST"
  set +e
  # run make.sh under the container's default `python` environment; make.sh
  # should use that python for any setup/install steps, preventing builds
  # for other Python versions.
  TORCH_CUDA_ARCH_LIST="$LIST" FORCE_CUDA=1 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH bash -x make.sh
  RC=$?
  set -e
  if [ $RC -eq 0 ]; then
    echo "make.sh succeeded with $LIST"
    # ensure the installed artifact is the one for the active python
    python -m pip uninstall -y MultiScaleDeformableAttention || true
    python -m pip install . || true
    exit 0
  else
    echo "make.sh failed with $LIST (rc=$RC)"
  fi
done

echo "All make.sh attempts failed"
exit 1
