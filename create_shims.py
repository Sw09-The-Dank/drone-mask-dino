#!/usr/bin/env python3
import sys, os, glob, site, importlib

def ensure_symlink():
    so_patterns = ['*MultiScaleDeformableAttention*.so', '*ms_deform_attn_cuda*.so']
    try:
        site_paths = list(site.getsitepackages())
    except Exception:
        site_paths = [sys.prefix + '/lib/python' + sys.version[:3] + '/site-packages']

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

    # Also link into in-repo patched tree if present
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

def write_shim_for_multiscale():
    try:
        site_paths = list(site.getsitepackages())
    except Exception:
        site_paths = [sys.prefix + '/lib/python' + sys.version[:3] + '/site-packages']
    for sp in site_paths:
        target_dir = os.path.join(sp, 'maskdino', 'modeling', 'pixel_decoder', 'ops')
        os.makedirs(target_dir, exist_ok=True)
        for so in glob.glob(os.path.join(sp, 'MultiScaleDeformableAttention*.so')):
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

    # Also in patched repo tree
    repo_target = '/workspace/MaskDINO_patched/maskdino/modeling/pixel_decoder/ops'
    os.makedirs(repo_target, exist_ok=True)
    for so in glob.glob(os.path.join(site_paths[0], 'MultiScaleDeformableAttention*.so')):
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

if __name__ == '__main__':
    ensure_symlink()
    write_shim_for_multiscale()
    # verify import
    try:
        importlib.import_module('maskdino.modeling.pixel_decoder.ops.ms_deform_attn_cuda')
        print('ms_deform_attn_cuda import OK')
    except Exception as e:
        print('ms_deform_attn_cuda import failed during image build:', e, file=sys.stderr)
        sys.exit(1)
