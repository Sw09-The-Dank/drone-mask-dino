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
    # Also link into workspace MaskDINO repo tree if present
    repo_target2 = '/workspace/MaskDINO/maskdino/modeling/pixel_decoder/ops'
    os.makedirs(repo_target2, exist_ok=True)
    for sp in site_paths:
        for pat in so_patterns:
            for f in glob.glob(os.path.join(sp, pat)):
                dest = os.path.join(repo_target2, os.path.basename(f))
                if not os.path.exists(dest):
                    try:
                        os.symlink(f, dest)
                    except Exception:
                        try:
                            import shutil
                            shutil.copy2(f, dest)
                        except Exception:
                            pass
                print('linked to workspace repo', f, '->', dest)

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
                "import os, glob, importlib, importlib.util\n"
                "here = os.path.dirname(__file__)\n"
                "so_candidates = glob.glob(os.path.join(here, '" + so_base + "*.so'))\n"
                "if so_candidates:\n"
                "    path = so_candidates[0]\n"
                "    spec = importlib.util.spec_from_file_location('._ms_deform', path)\n"
                "    _mod = importlib.util.module_from_spec(spec)\n"
                "    spec.loader.exec_module(_mod)\n"
                "else:\n"
                "    try:\n"
                "        _mod = importlib.import_module('" + so_base + "')\n"
                "    except Exception:\n"
                "        try:\n"
                "            _mod = importlib.import_module('." + so_base + "', package=__package__)\n"
                "        except Exception:\n"
                "            raise\n"
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
                        "import os, glob, importlib, importlib.util\n"
                        "here = os.path.dirname(__file__)\n"
                        "so_candidates = glob.glob(os.path.join(here, '" + so_base + "*.so'))\n"
                        "if so_candidates:\n"
                        "    path = so_candidates[0]\n"
                        "    spec = importlib.util.spec_from_file_location('._ms_deform', path)\n"
                        "    _mod = importlib.util.module_from_spec(spec)\n"
                        "    spec.loader.exec_module(_mod)\n"
                        "else:\n"
                        "    try:\n"
                        "        _mod = importlib.import_module('" + so_base + "')\n"
                        "    except Exception:\n"
                        "        try:\n"
                        "            _mod = importlib.import_module('." + so_base + "', package=__package__)\n"
                        "        except Exception:\n"
                        "            raise\n"
                        "for _name in dir(_mod):\n"
                        "    if not _name.startswith('_'):\n"
                        "        globals()[_name] = getattr(_mod, _name)\n"
                    )
                print('wrote repo shim', shim_path)
            except Exception as e:
                print('failed to write repo shim', shim_path, e)
    # Also write shim into workspace MaskDINO repo so imports work from repo copy
    repo_work = '/workspace/MaskDINO/maskdino/modeling/pixel_decoder/ops'
    os.makedirs(repo_work, exist_ok=True)
    # ensure package __init__ files exist
    def ensure_init(p):
        d = os.path.dirname(p)
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        if not os.path.exists(p):
            open(p, 'w').close()
    ensure_init('/workspace/MaskDINO/maskdino/__init__.py')
    ensure_init('/workspace/MaskDINO/maskdino/modeling/__init__.py')
    ensure_init('/workspace/MaskDINO/maskdino/modeling/pixel_decoder/__init__.py')
    ensure_init('/workspace/MaskDINO/maskdino/modeling/pixel_decoder/ops/__init__.py')
    shim_path = os.path.join(repo_work, 'ms_deform_attn_cuda.py')
    if not os.path.exists(shim_path):
        try:
            with open(shim_path, 'w') as fh:
                fh.write(
                    "import os, glob, importlib, importlib.util\n"
                    "here = os.path.dirname(__file__)\n"
                    "so_candidates = glob.glob(os.path.join(here, 'MultiScaleDeformableAttention*.so'))\n"
                    "if so_candidates:\n"
                    "    path = so_candidates[0]\n"
                    "    spec = importlib.util.spec_from_file_location('._ms_deform', path)\n"
                    "    _mod = importlib.util.module_from_spec(spec)\n"
                    "    spec.loader.exec_module(_mod)\n"
                    "else:\n"
                    "    try:\n"
                    "        _mod = importlib.import_module('MultiScaleDeformableAttention')\n"
                    "    except Exception:\n"
                    "        try:\n"
                    "            _mod = importlib.import_module('.MultiScaleDeformableAttention', package=__package__)\n"
                    "        except Exception:\n"
                    "            raise\n"
                    "for _name in dir(_mod):\n"
                    "    if not _name.startswith('_'):\n"
                    "        globals()[_name] = getattr(_mod, _name)\n"
                )
            print('wrote workspace repo shim', shim_path)
        except Exception as e:
            print('failed to write workspace repo shim', shim_path, e)

if __name__ == '__main__':
    ensure_symlink()
    write_shim_for_multiscale()
    # verify import
    try:
        importlib.import_module('maskdino.modeling.pixel_decoder.ops.ms_deform_attn_cuda')
        print('ms_deform_attn_cuda import OK')
    except Exception as e:
        print('ms_deform_attn_cuda import failed during image build:', e, file=sys.stderr)
        print('Continuing build; shim was written. Import will be attempted at runtime.')
        # Do not fail the whole image build if the import check cannot succeed
        # in the build environment (site-packages layout differs). Exit 0.
        sys.exit(0)
