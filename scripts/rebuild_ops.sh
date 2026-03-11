#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-$(pwd)}
ORIG_DIR="$WORKSPACE/MaskDINO"
PATCHED_DIR="${PATCHED_DIR:-$WORKSPACE/MaskDINO_patched}"
OPSDIR="$PATCHED_DIR/maskdino/modeling/pixel_decoder/ops"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found in PATH. Use a CUDA devel image with nvcc to build ops."
  exit 2
fi

if [ ! -d "$ORIG_DIR" ]; then
  echo "Original MaskDINO not found at $ORIG_DIR"
  exit 3
fi

# Create patched copy if missing
if [ ! -d "$PATCHED_DIR" ]; then
  echo "Creating patched copy at $PATCHED_DIR"
  cp -a "$ORIG_DIR" "$PATCHED_DIR"
  echo "Applying safe source patches in patched copy..."
  find "$PATCHED_DIR" -type f \( -name "*.cu" -o -name "*.cuh" -o -name "*.h" -o -name "*.cpp" \) -print0 \
    | xargs -0 sed -i 's/\.type()\.is_cuda()/\.is_cuda()/g' || true
  find "$PATCHED_DIR" -type f -name "*.cu" -print0 \
    | xargs -0 perl -0777 -pe "s/AT_DISPATCH_FLOATING_TYPES\\s*\\(\\s*value\\.type\\s*\\(\\s*\\)\\s*,/AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),/gs" -i || true
  find "$PATCHED_DIR" -type f \( -name "*.cu" -o -name "*.h" -o -name "*.cpp" \) -print0 \
    | xargs -0 sed -i 's/value.type()/value.scalar_type()/g' || true
fi

if [ ! -d "$OPSDIR" ]; then
  echo "Ops directory not found at $OPSDIR"
  exit 4
fi

cd "$OPSDIR"

echo "Cleaning previous builds..."
rm -rf build *.so *.pyd || true

# Detect compute capability as major.minor
ARCH=$(python - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(f"{p.major}.{p.minor}")
PY
)

echo "Detected compute capability: $ARCH"

set +e
# Try single-arch build first
echo "Attempting single-arch build for $ARCH"
TORCH_CUDA_ARCH_LIST="$ARCH" FORCE_CUDA=1 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH sh make.sh
RC=$?
if [ $RC -eq 0 ]; then
  echo "Build succeeded for $ARCH"
  exit 0
fi

echo "Single-arch build failed (rc=$RC). Trying fallback arch lists..."

candidates=(
  "8.6;8.0;7.5;7.0"
  "8.6;8.0;7.5"
  "8.0;7.5;7.0"
  "7.5;7.0;6.1"
  "7.0;6.1;6.0"
  "6.1;6.0;5.2"
)

for LIST in "${candidates[@]}"; do
  echo "Trying TORCH_CUDA_ARCH_LIST=$LIST"
  rm -rf build *.so *.pyd || true
  TORCH_CUDA_ARCH_LIST="$LIST" FORCE_CUDA=1 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH sh make.sh
  if [ $? -eq 0 ]; then
    echo "Build succeeded with $LIST"
    exit 0
  fi
done

echo "All build attempts failed. Inspect stdout/stderr above for errors."
exit 1
