#!/bin/bash
set -eux
# If nvcc is available in the base image, compile the CUDA ops during build.
# If the user explicitly requested BUILD_OPS_AT_BUILD=1 but nvcc is missing,
# fail the build so the mistake is visible. Prefer a devel base image that
# contains nvcc (example: nvcr.io/nvidia/pytorch:26.01-py3-devel) when
# compiling at build time.
if command -v nvcc >/dev/null 2>&1; then
    echo "nvcc detected: attempting in-image build of MultiScaleDeformableAttention"
    # Ensure nvcc is the expected CUDA version when BUILD_OPS_AT_BUILD requested
    if [ "${BUILD_OPS_AT_BUILD:-0}" = "1" ]; then
        NVCC_VER_FULL=$(nvcc --version || true)
        echo "nvcc --version:\n$NVCC_VER_FULL"
        if ! echo "$NVCC_VER_FULL" | grep -q "release 13.1"; then
            echo "Warning: nvcc does not report CUDA 13.1. Detected:" >&2
            echo "$NVCC_VER_FULL" >&2
            echo "Proceeding anyway, but consider using a CUDA 13.1 devel base image." >&2
        fi
    fi

        cp -a /workspace/MaskDINO /workspace/MaskDINO_patched
        # First-pass safe replacements for deprecated PyTorch C++ APIs
        find /workspace/MaskDINO_patched -type f \( -name "*.cu" -o -name "*.cuh" -o -name "*.h" -o -name "*.cpp" \) -print0 | xargs -0 -r sed -i 's/\.type()\.is_cuda()/\.is_cuda()/g' || true
        find /workspace/MaskDINO_patched -type f -name "*.cu" -print0 | xargs -0 -r perl -0777 -pe "s/AT_DISPATCH_FLOATING_TYPES\s*\(\s*value\.type\s*\(\s*\)\s*,/AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),/gs" -i || true
        find /workspace/MaskDINO_patched -type f \( -name "*.cu" -o -name "*.h" -o -name "*.cpp" \) -print0 | xargs -0 -r sed -i 's/value.type()/value.scalar_type()/g' || true

        # Extra aggressive substitutions to catch variations in whitespace/linebreaks
        find /workspace/MaskDINO_patched -type f -name "*.cu" -print0 | xargs -0 -r perl -0777 -pe \
            "s/AT_DISPATCH_FLOATING_TYPES\s*\(\s*[^,\)]*?value\s*\.\s*type\s*\(\s*\)\s*,/AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),/igs" -i || true

        # Report any remaining uses so failures are visible in build logs
        echo "Remaining occurrences of deprecated patterns in MaskDINO_patched (if any):"
        grep -R --line-number --colour=never -E "value\.type\(|\.type\(\)\.is_cuda\(|AT_DISPATCH_FLOATING_TYPES\s*\(" /workspace/MaskDINO_patched || true

    # Build the ops using the ops' own make.sh (this script installs the
    # extension into the Python environment). Run it from the ops directory.
    cd /workspace/MaskDINO_patched/maskdino/modeling/pixel_decoder/ops
    TORCH_CUDA_ARCH_LIST=$(python - <<'PYEND'
import sys
try:
    import torch
    p = torch.cuda.get_device_properties(0)
    print(f"{p.major}.{p.minor}")
except Exception:
    # fallback list
    print("12.1;8.6;8.0;7.5")
    sys.exit(0)
PYEND
)
    echo "Using TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    export FORCE_CUDA=1
    export CUDA_HOME=/usr/local/cuda
    export PATH=/usr/local/cuda/bin:$PATH
    export TORCH_CUDA_ARCH_LIST
    # Make the patched MaskDINO tree importable during image build
    export PYTHONPATH=/workspace/MaskDINO_patched:${PYTHONPATH:-}
    chmod +x make.sh || true
    set +e
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" FORCE_CUDA=1 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH sh make.sh
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then
        echo "make.sh failed (rc=$RC), trying fallback multi-arch lists"
        candidates=("12.1;8.6;8.0;7.5" "8.6;8.0;7.5" "8.0;7.5;7.0")
        for LIST in "${candidates[@]}"; do
            echo "Trying TORCH_CUDA_ARCH_LIST=$LIST"
            rm -rf build || true
            set +e
            TORCH_CUDA_ARCH_LIST="$LIST" FORCE_CUDA=1 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH sh make.sh
            RC=$?
            set -e
            if [ $RC -eq 0 ]; then
                echo "Build succeeded with $LIST"
                break
            fi
        done
        if [ $RC -ne 0 ]; then
            echo "All make.sh build attempts failed. See output above." >&2
            exit 1
        fi
    fi

    # Some make.sh/setup.py installers place the compiled .so at top-level
    # (e.g. MultiScaleDeformableAttention*.so). Ensure it's also available at
    # maskdino.modeling.pixel_decoder.ops.ms_deform_attn_cuda by creating a
    # symlink inside site-packages if needed, then verify importability.
    python - <<'PYEND'
import sys, os, glob, site, importlib


def ensure_symlink():
    so_patterns = ['*MultiScaleDeformableAttention*.so', '*ms_deform_attn_cuda*.so']
    try:
        site_paths = list(site.getsitepackages())
    except Exception:
        site_paths = [sys.prefix + '/lib/python' + sys.version[:3] + '/site-packages']

    # Copy/link the compiled .so into the package directory, but keep the
    # original filename to avoid Python attempting to load it under the
    # `ms_deform_attn_cuda` symbol name (which would fail if the compiled
    # module was built with a different PyInit name). We'll create a
    # Python shim below that imports the original module name and re-exports
    # symbols under `ms_deform_attn_cuda`.
    for sp in site_paths:
        for pat in so_patterns:
            for f in glob.glob(os.path.join(sp, pat)):
                target_dir = os.path.join(sp, 'maskdino', 'modeling', 'pixel_decoder', 'ops')
                os.makedirs(target_dir, exist_ok=True)
                dest = os.path.join(target_dir, os.path.basename(f))
                if not os.path.exists(dest):
                    try:
                        os.symlink(f, dest)
                    except Exception:
                        try:
                            import shutil
                            shutil.copy2(f, dest)
                        except Exception:
                            pass
                print('linked', f, '->', dest)

    # Also link/copy into the in-repo patched MaskDINO tree (on PYTHONPATH)
    repo_target = '/workspace/MaskDINO_patched/maskdino/modeling/pixel_decoder/ops'
    os.makedirs(repo_target, exist_ok=True)
    for sp in site_paths:
        for pat in so_patterns:
            for f in glob.glob(os.path.join(sp, pat)):
                dest = os.path.join(repo_target, os.path.basename(f))
                if not os.path.exists(dest):
                    try:
                        os.symlink(f, dest)
                    except Exception:
                        try:
                            import shutil
                            shutil.copy2(f, dest)
                        except Exception:
                            pass
                print('linked to repo', f, '->', dest)

ensure_symlink()

# If the compiled .so has a different extension module name (e.g. 
# MultiScaleDeformableAttention.*.so), provide a small Python shim named
# ms_deform_attn_cuda.py that imports the compiled module and re-exports
# its public attributes. This avoids needing to rebuild the extension with
# a different name.
try:
    site_paths2 = []
    try:
        site_paths2 = list(site.getsitepackages())
    except Exception:
        site_paths2 = [sys.prefix + '/lib/python' + sys.version[:3] + '/site-packages']

    for sp in site_paths2:
        target_dir = os.path.join(sp, 'maskdino', 'modeling', 'pixel_decoder', 'ops')
        os.makedirs(target_dir, exist_ok=True)
        for so in glob.glob(os.path.join(sp, 'MultiScaleDeformAttENTION*.so')):
            so_base = os.path.basename(so).split('.')[0]
            shim = (
                f"from importlib import import_module as _import_module\n"
                f"_mod = _import_module('{so_base}')\n"
                "for _name in dir(_mod):\n"
                "    if not _name.startswith('_'):\n"
                "        globals()[_name] = getattr(_mod, _name)\n"
            )
            shim_path = os.path.join(target_dir, 'ms_deform_attn_cuda.py')
            if not os.path.exists(shim_path):
                try:
                    with open(shim_path, 'w') as fh:
                        fh.write(shim)
                    print('wrote shim', shim_path)
                except Exception as e:
                    print('failed to write shim', shim_path, e)

    # Also create shim in the patched repo tree so PYTHONPATH import can find it
    repo_target = '/workspace/MaskDINO_patched/maskdino/modeling/pixel_decoder/ops'
    os.makedirs(repo_target, exist_ok=True)
    for so in glob.glob(os.path.join(site_paths2[0], 'MultiScaleDeformAttENTION*.so')):
        so_base = os.path.basename(so).split('.')[0]
        shim_path = os.path.join(repo_target, 'ms_deform_attn_cuda.py')
        if not os.path.exists(shim_path):
            try:
                with open(shim_path, 'w') as fh:
                    fh.write(
                        f"from importlib import import_module as _import_module\n_mod = _import_module('{so_base}')\nfor _name in dir(_mod):\n    if not _name.startswith('_'):\n        globals()[_name] = getattr(_mod, _name)\n"
                    )
                print('wrote repo shim', shim_path)
            except Exception as e:
                print('failed to write repo shim', shim_path, e)
except Exception as e:
    print('shim creation failed', e)
PYEND

# Finally, verify importability of the expected module name used by MaskDINO
try:
    importlib.import_module('maskdino.modeling.pixel_decoder.ops.ms_deform_attn_cuda')
except Exception as e:
    echo "ms_deform_attn_cuda import failed during image build: $e" >&2
    exit 1

echo "ms_deform_attn_cuda import OK"
else
    if [ "${BUILD_OPS_AT_BUILD:-0}" = "1" ]; then
        echo "BUILD_OPS_AT_BUILD=1 but nvcc not found in base image; failing build" >&2
        exit 1
    fi
    echo "Skipping in-image ops build (nvcc not found, BUILD_OPS_AT_BUILD=${BUILD_OPS_AT_BUILD:-0})"
fi
