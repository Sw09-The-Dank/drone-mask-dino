#!/usr/bin/env bash
set -eu

# If a patched copy doesn't exist, create it and apply non-invasive fixes
if [ ! -d /workspace/MaskDINO_patched ]; then
    echo "Creating runtime patched copy: /workspace/MaskDINO_patched"
    cp -a /workspace/MaskDINO /workspace/MaskDINO_patched || true
    # First-pass safe replacements for deprecated PyTorch C++ APIs
    find /workspace/MaskDINO_patched -type f \( -name "*.cu" -o -name "*.cuh" -o -name "*.h" -o -name "*.cpp" \) -print0 | xargs -0 -r sed -i "s/\.type()\.is_cuda()/\.is_cuda()/g" || true
    find /workspace/MaskDINO_patched -type f -name "*.cu" -print0 | xargs -0 -r perl -0777 -pe "s/AT_DISPATCH_FLOATING_TYPES\s*\(\s*value\.type\s*\(\s*\)\s*,/AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),/gs" -i || true
    find /workspace/MaskDINO_patched -type f \( -name "*.cu" -o -name "*.h" -o -name "*.cpp" \) -print0 | xargs -0 -r sed -i "s/value.type()/value.scalar_type()/g" || true
    # Extra aggressive substitution to catch whitespace/linebreak variations
    find /workspace/MaskDINO_patched -type f -name "*.cu" -print0 | xargs -0 -r perl -0777 -pe \
        "s/AT_DISPATCH_FLOATING_TYPES\s*\(\s*[^,\)]*?value\s*\.\s*type\s*\(\s*\)\s*,/AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),/igs" -i || true
    # Normalize make.sh line endings and make executable in patched tree
    sed -i 's/\r$//' /workspace/MaskDINO_patched/maskdino/modeling/pixel_decoder/ops/make.sh || true
    chmod +x /workspace/MaskDINO_patched/maskdino/modeling/pixel_decoder/ops/make.sh || true
fi

# Build from the patched copy to avoid modifying upstream sources
cd /workspace/MaskDINO_patched/maskdino/modeling/pixel_decoder/ops
sed -i 's/\r$//' make.sh || true
chmod +x make.sh
export FORCE_CUDA=1
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=${CUDA_HOME}/bin:$PATH

if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    echo "Using provided TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    echo "Attempting build with TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    if TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" sh make.sh; then
        echo "Build succeeded with TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
        exit 0
    else
        echo "Build failed with TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}. Not attempting fallbacks because an explicit list was provided."
        echo "You can retry inside the container with a different TORCH_CUDA_ARCH_LIST."
        exit 1
    fi
else
    # No explicit arch list provided: try preferred fallback arch lists
    TRY_LISTS=("13.1;13.0;12.0;12.1+PTX;8.6;8.0;7.5" "12.1+PTX;8.6;8.0;7.5" "8.6;8.0;7.5" "8.0")
    for LIST in "${TRY_LISTS[@]}"; do
        echo "Attempting build with TORCH_CUDA_ARCH_LIST=${LIST}"
        if TORCH_CUDA_ARCH_LIST="$LIST" sh make.sh; then
            echo "Build succeeded with TORCH_CUDA_ARCH_LIST=${LIST}"
            exit 0
        else
            echo "Build failed with TORCH_CUDA_ARCH_LIST=${LIST}, trying next option..."
        fi
    done
    echo "All build attempts failed. You can retry inside the container with a different TORCH_CUDA_ARCH_LIST."
    exit 1
fi
