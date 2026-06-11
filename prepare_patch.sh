#!/usr/bin/env bash
set -ex
# Create a patched copy of MaskDINO and apply source-compatible patches
PATCH_DIR=/workspace/MaskDINO_patched
rm -rf "$PATCH_DIR" || true
if [ -d /workspace/MaskDINO ]; then
  cp -a /workspace/MaskDINO "$PATCH_DIR"
elif [ -d /workspace/third_party/MaskDINO ]; then
  cp -a /workspace/third_party/MaskDINO "$PATCH_DIR"
else
  echo "No MaskDINO source found to patch; skipping patch step"
  exit 0
fi

echo "Patching sources in $PATCH_DIR"
# Remove DOS carriage returns if present
find "$PATCH_DIR" -type f -print0 | xargs -0 -r sed -i 's/\r$//' || true

# Remove any stale build artifacts from previous runs (including cpython-38 outputs)
rm -rf "$PATCH_DIR"/maskdino/modeling/pixel_decoder/ops/build* || true
find "$PATCH_DIR" -type f -name '*cpython-38*' -print -delete || true

# Replace deprecated .type().is_cuda() usages
find "$PATCH_DIR" -type f \( -name "*.cu" -o -name "*.cuh" -o -name "*.h" -o -name "*.cpp" \) -print0 \
  | xargs -0 -r sed -i 's/\.type()\.is_cuda()/\.is_cuda()/g' || true

# Replace common AT_DISPATCH pattern variations with scalar_type()
find "$PATCH_DIR" -type f -name "*.cu" -print0 \
  | xargs -0 -r perl -0777 -pe "s/AT_DISPATCH_FLOATING_TYPES\s*\(\s*value\.type\s*\(\s*\)\s*,/AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),/gs" -i || true

# Replace any remaining occurrences of 'value.type()'
find "$PATCH_DIR" -type f \( -name "*.cu" -o -name "*.h" -o -name "*.cpp" \) -print0 \
  | xargs -0 -r sed -i 's/value.type()/value.scalar_type()/g' || true

echo "Patch step complete"
