#!/usr/bin/env bash
set -eu
cd /workspace/MaskDINO/maskdino/modeling/pixel_decoder/ops
sed -i 's/\r$//' make.sh || true
chmod +x make.sh
export FORCE_CUDA=1
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=${CUDA_HOME}/bin:$PATH

# Try preferred arch lists (include explicit 12.0, PTX fallback), then fall back
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
