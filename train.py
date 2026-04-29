import os
import random
import math
import json
import sys
import subprocess
import importlib
import numpy as np
import torch
try:
    import cv2
except Exception:
    cv2 = None

try:
    from detectron2.engine import HookBase
except ModuleNotFoundError as e:
    # Some container images miss setuptools at runtime, which detectron2 imports via pkg_resources.
    if getattr(e, "name", "") == "pkg_resources":
        try:
            print("[INFO] Missing pkg_resources; installing compatible setuptools and retrying detectron2 import...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "setuptools<81"])
            importlib.invalidate_caches()
            from detectron2.engine import HookBase
        except Exception as install_e:
            raise RuntimeError(
                "detectron2 import failed because pkg_resources is missing. "
                "Install setuptools in the runtime environment and retry."
            ) from install_e
    else:
        raise

# standard utilities used across this script
import os
import sys
import time
import socket
import glob
import json
import math
import random
import datetime
import inspect
import argparse

import numpy as np

# torch is used throughout; ensure it's available and provide DDP helpers
import torch

# -----------------------------
# SANITY CHECK: CUDA
# -----------------------------

print("PyTorch version:", getattr(torch, '__version__', 'n/a'))
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", getattr(torch.version, 'cuda', 'n/a'))



def _resolve_detectron2_cfg_file(cfg_key_or_path: str) -> str:
    """Resolve a detectron2 config file path without requiring pkg_resources/model_zoo."""
    if not cfg_key_or_path:
        return cfg_key_or_path

    if os.path.isfile(cfg_key_or_path):
        return cfg_key_or_path

    normalized = cfg_key_or_path.replace("\\", "/").lstrip("/")
    if normalized.startswith("configs/"):
        normalized = normalized[len("configs/"):]

    candidates = [
        os.path.join("detectron2", "configs", normalized),
        os.path.join("/workspace", "detectron2", "configs", normalized),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return cfg_key_or_path


def _ensure_pkg_resources_available() -> bool:
    """Ensure pkg_resources is importable in the current runtime.

    Returns True when importable, otherwise False.
    """
    try:
        import pkg_resources  # noqa: F401
        return True
    except Exception:
        pass

    try:
        print("[INFO] pkg_resources not found; attempting to install compatible setuptools...")
        # Newer setuptools builds may omit pkg_resources in some environments.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "setuptools<81"])
        importlib.invalidate_caches()
    except Exception as e:
        print(f"[WARN] Failed to install setuptools automatically: {e}")
        return False

    try:
        import pkg_resources  # noqa: F401
        return True
    except Exception as e:
        print(f"[WARN] pkg_resources still unavailable after setuptools install: {e}")
        return False


def _get_model_zoo():
    """Return detectron2.model_zoo after ensuring pkg_resources is available."""
    if not _ensure_pkg_resources_available():
        raise RuntimeError("pkg_resources is unavailable; cannot import detectron2.model_zoo")

    try:
        from detectron2 import model_zoo as _model_zoo
        return _model_zoo
    except Exception as e:
        raise RuntimeError(f"Failed to import detectron2.model_zoo: {e}") from e


def _running_in_container() -> bool:
    return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")


def _get_shm_size_bytes():
    try:
        stat = os.statvfs("/dev/shm")
        return int(stat.f_frsize * stat.f_blocks)
    except Exception:
        return None


def _configure_torch_runtime() -> None:
    strategy = os.environ.get("PYTORCH_SHARING_STRATEGY")
    shm_bytes = _get_shm_size_bytes()
    if not strategy and (shm_bytes is not None and shm_bytes < 1024 * 1024 * 1024):
        strategy = "file_system"

    if strategy:
        try:
            torch.multiprocessing.set_sharing_strategy(strategy)
            print(f"[INFO] torch multiprocessing sharing strategy: {strategy}")
        except Exception as e:
            print(f"[WARN] Could not set torch multiprocessing sharing strategy to {strategy}: {e}")


def _choose_default_num_workers(cli_num_workers=None) -> int:
    if cli_num_workers is not None:
        return max(0, int(cli_num_workers))

    env_num_workers = os.environ.get("TRAIN_NUM_WORKERS")
    if env_num_workers is not None:
        try:
            return max(0, int(env_num_workers))
        except Exception:
            print(f"[WARN] Ignoring invalid TRAIN_NUM_WORKERS={env_num_workers!r}")

    shm_bytes = _get_shm_size_bytes()
    if _running_in_container() and (shm_bytes is None or shm_bytes < 1024 * 1024 * 1024):
        return 0

    return 8


_configure_torch_runtime()


class RandomGaussianBlurMapper:
    """Wrap a dataset mapper and apply random Gaussian blur to training images."""

    def __init__(self, mapper, prob: float = 0.5, sigma_min: float = 0.5, sigma_max: float = 2.0):
        self.mapper = mapper
        self.prob = float(prob)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def __call__(self, dataset_dict):
        data = self.mapper(dataset_dict)
        if data is None or cv2 is None:
            return data
        if self.prob <= 0.0 or random.random() >= self.prob:
            return data

        img = data.get("image", None)
        if img is None or not torch.is_tensor(img) or img.ndim != 3:
            return data

        # Detectron2 image tensor is CxHxW; OpenCV expects HxWxC.
        src = img.detach().cpu()
        np_img = src.permute(1, 2, 0).numpy()
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        blurred = cv2.GaussianBlur(np_img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        if blurred.ndim == 2:
            blurred = blurred[:, :, None]

        out = torch.from_numpy(blurred).permute(2, 0, 1)
        if out.dtype != img.dtype:
            out = out.to(dtype=img.dtype)
        data["image"] = out.to(device=img.device)
        return data


def setup_ddp_from_env():
    """Initialize torch.distributed and detectron2 local PG from environment.

    Safe to call multiple times. Reads `WORLD_SIZE`, `RANK`, `LOCAL_RANK`,
    and `LOCAL_WORLD_SIZE` / `LOCAL_SIZE` to determine local process counts.
    """
    try:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    except Exception:
        local_rank = 0
    try:
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
    except Exception:
        world_size = 1

    # set CUDA device for this process
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'not-set')
    print(f"[CUDA] is_available={cuda_available} device_count={cuda_device_count} CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
    
    if cuda_available:
        try:
            torch.cuda.set_device(local_rank)
            print(f"[CUDA] set_device({local_rank}) succeeded")
        except Exception as e:
            print(f"[CUDA] set_device({local_rank}) failed: {e}")
        # Cap GPU memory so the process OOM-crashes instead of stalling
        try:
            frac = float(os.environ.get("CUDA_MEMORY_FRACTION", "0.85"))
            torch.cuda.set_per_process_memory_fraction(frac, local_rank)
            print(f"[MEM] GPU memory fraction capped at {frac:.0%} for device {local_rank}")
        except Exception as e:
            print(f"[MEM] Could not set GPU memory fraction: {e}")
    else:
        print("[WARN] CUDA not available! Will use Gloo backend (SLOW)")

    # init torch.distributed if needed
    if torch.distributed.is_available() and not torch.distributed.is_initialized() and world_size > 1:
        backend = 'nccl' if cuda_available else 'gloo'
        print(f"[DDP] Selecting backend: {backend}")
        torch.distributed.init_process_group(backend=backend, init_method='env://')
        rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
        print(f"DDP INIT: backend={backend} RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world_size}")

    # Ensure detectron2 local process group exists
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            from detectron2.utils import comm as d2comm
            try:
                local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', None) or os.environ.get('LOCAL_SIZE', None) or 0)
            except Exception:
                local_world_size = 0
            try:
                global_world_size = int(os.environ.get('WORLD_SIZE', '1'))
            except Exception:
                global_world_size = 1
            # Infer per-node count from visible CUDA devices when not provided
            if local_world_size <= 0:
                try:
                    local_world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
                except Exception:
                    local_world_size = 1
            should_create = (local_world_size > 1) or (global_world_size > 1)
            if should_create:
                if getattr(d2comm, "_LOCAL_PROCESS_GROUP", None) is None:
                    print(f"[DDP] creating local process group: local_world_size={local_world_size} global_world_size={global_world_size}")
                    d2comm.create_local_process_group(local_world_size)
                    print("[DDP] detectron2 local process group created")
                else:
                    print("[DDP] detectron2 local process group already initialized")
        except Exception as e:
            print(f"[DDP] failed to create detectron2 local process group: {e!r}")
            raise

    # tuning
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        os.environ.setdefault('OMP_NUM_THREADS', '4')
        os.environ.setdefault('MKL_NUM_THREADS', '4')
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    except Exception:
        pass

    try:
        print(f"DDP START host={socket.gethostname()} RANK={os.environ.get('RANK')} LOCAL_RANK={local_rank} WORLD_SIZE={world_size}")
    except Exception:
        pass


def finalize_ddp(wait_seconds: float = 3.0):
    """Attempt a graceful distributed shutdown.

    - Synchronize CUDA on each device to flush kernels.
    - Run a barrier so all ranks reach this point.
    - Sleep briefly to allow outstanding async NCCL ops to complete on the remote side.
    - Run a second barrier and then destroy the process group.
    This helps avoid "remote process exited" NCCL errors when one rank exits earlier than others.
    """
    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
    except Exception:
        pass

    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                pass
            try:
                # give a small grace period for async ops to finish on remote ranks
                time.sleep(float(wait_seconds))
            except Exception:
                pass
            try:
                torch.distributed.barrier()
            except Exception:
                pass
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass
    except Exception:
        pass


def _unwrap_train_loader_sampler(loader):
    """Best-effort extraction of the effective sampler from Detectron2 train loaders."""
    current = loader
    fallback = None
    seen = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        sampler = getattr(current, "sampler", None)
        if sampler is not None:
            module_name = type(sampler).__module__
            class_name = type(sampler).__name__
            if module_name.startswith("detectron2.") or class_name.endswith("TrainingSampler"):
                return sampler
            fallback = sampler
        current = getattr(current, "dataset", None)

    return fallback


def _debug_log_train_loader_sampler(loader, prefix="[DEBUG]"):
    sampler = _unwrap_train_loader_sampler(loader)
    if sampler is None:
        print(f"{prefix} Train loader sampler: None")
        return

    sampler_type = type(sampler).__name__
    sampler_module = type(sampler).__module__
    print(f"{prefix} Train loader sampler: {sampler_module}.{sampler_type}")

    if sampler_module.startswith("detectron2.data.samplers") and sampler_type == "TrainingSampler":
        print(f"{prefix} Train loader uses Detectron2 TrainingSampler: True")
        print(f"{prefix} TrainingSampler is distributed-aware: rank gets indices[rank::world_size]")
        return

    try:
        from torch.utils.data.distributed import DistributedSampler as _DS
        is_dist_sampler = isinstance(sampler, _DS)
        print(f"{prefix} Train loader uses DistributedSampler: {is_dist_sampler}")
    except Exception:
        print(f"{prefix} Could not determine if sampler is distributed-aware")



# (DDP initialization is handled by `setup_ddp_from_env()` where needed)


# -----------------------------
# DATASET REGISTRATION (register only the split needed at each phase)
# -----------------------------
CLASS_NAMES = ["rotor", "frame", "camera", "landinggear", "air2s", "neo", "mavic3m", "mini3pro"]


class CheckpointCleanupHook(HookBase):
    def __init__(self, output_dir, keep=4, period=500):
        self.output_dir = output_dir
        self.keep = keep
        self.period = period
    def after_step(self):
        # Only clean up right after a checkpoint is written (every `period` steps)
        if (self.trainer.iter + 1) % self.period != 0:
            return
        import glob, os
        checkpoint_files = sorted(
            glob.glob(os.path.join(self.output_dir, "model_*.pth")),
            key=os.path.getmtime,
            reverse=True
        )
        to_delete = checkpoint_files[self.keep:]
        for ckpt in to_delete:
            try:
                os.remove(ckpt)
                print(f"Deleted old checkpoint: {ckpt}")
            except Exception as e:
                print(f"Failed to delete {ckpt}: {e}")


print("\n--- TRAINER DEFINITION ---")
# print_cuda_mem("before TrainerWithDebug instantiation")

def run_default_trainer(train_json_path="output_annotations/train_polygons.json",
                        val_json_path="output_annotations/val_polygons.json",
                        images_root="dataset/images",
                        val_images_root=None,
                        output_dir=None,
                        max_iter=None,
                        ims_per_batch=None,
                        base_lr=None,
                        num_workers=None,
                        batch_size_per_image=None,
                        num_classes=None,
                        config_file="maskrcnn_config.yaml",
                        weights=None,
                        extra_cfg=None,
                        resume=True,
                        epochs=None,
                        eval_score_thresh=None,
                        eval_detections_per_image=None,
                        eval_focus_on_box=None,
                        fast_eval=False,
                        eval_bbox_only=False,
                        gaussian_blur_prob=0.0):
    _ensure_pkg_resources_available()
    try:
        from detectron2.data.datasets import register_coco_instances
        from detectron2.engine import DefaultTrainer
        from detectron2.config import get_cfg
    except Exception as e:
        # Retry once after ensuring setuptools/pkg_resources for environments
        # where detectron2's dependency chain imports pkg_resources lazily.
        if "pkg_resources" in str(e):
            if _ensure_pkg_resources_available():
                try:
                    from detectron2.data.datasets import register_coco_instances
                    from detectron2.engine import DefaultTrainer
                    from detectron2.config import get_cfg
                except Exception as e2:
                    print("[ERROR] detectron2 is required to run the trainer:", e2)
                    return
            else:
                print("[ERROR] detectron2 is required to run the trainer:", e)
                return
        else:
            print("[ERROR] detectron2 is required to run the trainer:", e)
            return

    # Ensure DDP/init and detectron2 local PG exist before constructing DefaultTrainer
    try:
        try:
            setup_ddp_from_env()
        except Exception:
            pass
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                from detectron2.utils import comm as d2comm
                # Prefer explicit env var from launcher; fallbacks to 1
                try:
                    n_local = int(os.environ.get('LOCAL_WORLD_SIZE', os.environ.get('LOCAL_SIZE', os.environ.get('NPROC_PER_NODE', '1'))))
                except Exception:
                    n_local = 1
                if n_local < 1:
                    n_local = 1
                try:
                    d2comm.create_local_process_group(n_local)
                except Exception as e:
                    # It's okay if already created or fails; warn for visibility
                    print(f"[WARN] create_local_process_group failed: {e}")
            except Exception:
                pass
    except Exception:
        pass

    # Auto-detect images_root if not provided and common path exists
    if not images_root:
        candidate = os.path.join("dataset", "images")
        if os.path.isdir(candidate):
            images_root = candidate

    # Register datasets if available. Use unique names for polygon-converted JSONs
    train_name = "drone_train_polygons"
    val_name = "drone_val_polygons"

    # If images are organized under images_root/train and images_root/val, prefer those
    train_img_root = images_root
    val_img_root = val_images_root if val_images_root else images_root
    train_candidate = os.path.join(images_root, "train")
    if os.path.isdir(train_candidate):
        train_img_root = train_candidate
    if not val_images_root:
        for subdir in ("val", "test"):
            candidate = os.path.join(images_root, subdir)
            if os.path.isdir(candidate):
                val_img_root = candidate
                break

    # Fallbacks: some setups place images under top-level `images/` instead of `dataset/images/`.
    # If the chosen img roots do not exist or contain few files, try common alternate locations.
    def _choose_existing_root(preferred, alternates):
        if isinstance(preferred, str) and os.path.isdir(preferred) and len(os.listdir(preferred))>0:
            return preferred
        for a in alternates:
            try:
                if os.path.isdir(a) and len(os.listdir(a))>0:
                    return a
            except Exception:
                continue
        # last resort: return preferred even if empty
        return preferred

    train_img_root = _choose_existing_root(train_img_root, [os.path.join('images','train'), 'images', os.path.join('dataset','images','train')])
    val_img_root = _choose_existing_root(val_img_root, [os.path.join('images','val'), os.path.join('images','test'), 'images', os.path.join('dataset','images','val'), os.path.join('dataset','images','test')])

    # Require both train and val JSONs to exist at this point; abort early if missing.
    train_exists = os.path.isfile(train_json_path)
    val_exists = os.path.isfile(val_json_path)
    # If either JSON is missing, emit helpful debug information to locate the problem
    if not train_exists or not val_exists:
        try:
            print(f"[DEBUG] CWD: {os.getcwd()}")
        except Exception:
            pass
        try:
            try:
                print(f"[DEBUG] train_json_path abs: {os.path.abspath(train_json_path)}")
            except Exception:
                print("[DEBUG] Could not compute abs path for train_json_path")
            try:
                print(f"[DEBUG] val_json_path abs: {os.path.abspath(val_json_path)}")
            except Exception:
                print("[DEBUG] Could not compute abs path for val_json_path")

            for name, path, exists in (("train", train_json_path, train_exists), ("val", val_json_path, val_exists)):
                try:
                    parent = os.path.dirname(path) or '.'
                    print(f"[DEBUG] {name} parent dir: {parent} (abs: {os.path.abspath(parent)}) exists={os.path.isdir(parent)}")
                    try:
                        print(f"[DEBUG] {name} parent dir listing: {os.listdir(parent)}")
                    except Exception as e:
                        print(f"[DEBUG] Could not list {parent}: {e}")
                    try:
                        # show any similar files that might hint at naming/casing issues
                        similar = glob.glob(os.path.join(parent, f"*{os.path.basename(path)}")) or glob.glob(os.path.join(parent, f"*{name}*.json"))
                        print(f"[DEBUG] {name} similar files in dir: {similar}")
                    except Exception as e:
                        print(f"[DEBUG] glob failed: {e}")
                except Exception:
                    pass
        except Exception:
            pass

    if not train_exists:
        print(f"[ERROR] Train JSON not found: {train_json_path}")
    if not val_exists:
        print(f"[ERROR] Val JSON not found: {val_json_path}")
    if not (train_exists and val_exists):
        print("[ERROR] Required annotation JSON file(s) missing; aborting.")
        raise SystemExit(1)

    # Both JSONs present — register them.
    register_coco_instances(train_name, {}, train_json_path, train_img_root)
    print(f"Registered {train_name} -> {train_json_path} (images root: {train_img_root})")
    register_coco_instances(val_name, {}, val_json_path, val_img_root)
    print(f"Registered {val_name} -> {val_json_path} (images root: {val_img_root})")

    # Load registered dataset dicts and sanitize segmentation entries; then re-register cleaned datasets
    from detectron2.data import DatasetCatalog, MetadataCatalog
    # Helper: convert binary mask ndarray -> list of polygons (list of list of floats)
    def mask_to_polygons(mask_arr, approx_eps=1.0):
        try:
            import cv2
            has_cv2 = True
        except Exception:
            cv2 = None
            has_cv2 = False
        try:
            from skimage import measure
            has_skimage = True
        except Exception:
            measure = None
            has_skimage = False

        m = np.asarray(mask_arr)
        if m.ndim != 2:
            return []
        m = (m > 0).astype('uint8')
        polys = []
        if has_cv2:
            try:
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cnt = cnt.squeeze()
                    if cnt.ndim != 2 or cnt.shape[0] < 3:
                        continue
                    poly = cnt.flatten().tolist()
                    if len(poly) >= 6:
                        polys.append([float(x) for x in poly])
            except Exception:
                pass
        elif has_skimage:
            try:
                contours = measure.find_contours(m, 0.5)
                for c in contours:
                    # skimage contours are (row, col) -> convert to (x, y)
                    coords = np.fliplr(c)
                    poly = coords.flatten().tolist()
                    if len(poly) >= 6:
                        polys.append([float(x) for x in poly])
            except Exception:
                pass
        return polys
    def sanitize_dataset(name, img_root, orig_json_path=None):
        try:
            dicts = list(DatasetCatalog.get(name))
        except Exception:
            return name
        clean = []
        problematic = []
        for d in dicts:
            dd = dict(d)
            fn = dd.get("file_name", "") or ''
            # Robustly resolve the image file path using several strategies.
            resolved = None
            try:
                fn_norm = fn.replace('\\', '/').lstrip('./')
                img_root_norm = (img_root.replace('\\', '/').rstrip('/')) if isinstance(img_root, str) else ''
                candidates = []
                # If absolute path provided, try it first
                if os.path.isabs(fn_norm):
                    candidates.append(fn_norm)
                # If the fn already contains the img_root, strip duplicate prefixes
                if img_root_norm and img_root_norm in fn_norm:
                    tail = fn_norm.split(img_root_norm)[-1].lstrip('/')
                    candidates.append(os.path.join(img_root_norm, tail))
                    candidates.append(tail)
                # Common joins
                candidates.append(os.path.join(img_root, fn_norm))
                candidates.append(os.path.join(img_root, os.path.basename(fn_norm)))
                candidates.append(fn_norm)

                # Try all candidates (as-is and relative to cwd) and pick the first that exists
                for c in candidates:
                    if not c:
                        continue
                    for test in (c, os.path.join(os.getcwd(), c)):
                        try:
                            test_norm = os.path.normpath(test)
                        except Exception:
                            test_norm = test
                        if os.path.isfile(test_norm):
                            resolved = test_norm
                            break
                    if resolved:
                        break
            except Exception:
                resolved = None

            if resolved:
                dd["file_name"] = resolved
            else:
                # Could not resolve image file; skip this image and its annotations
                problematic.append({"image_file": fn, "reason": "file_not_found"})
                continue
            anns = []
            for ann in dd.get("annotations", []):
                ann2 = dict(ann)
                seg = ann2.get("segmentation")
                # Handle numpy.ndarray segmentations
                try:
                    if isinstance(seg, np.ndarray):
                        # If it's a 2D mask array, convert to polygons
                        if getattr(seg, 'ndim', None) == 2:
                            polys = mask_to_polygons(seg)
                            if polys:
                                ann2['segmentation'] = polys
                                seg = polys
                            else:
                                ann2.pop('segmentation', None)
                                seg = None
                        else:
                            # otherwise convert 1D arrays to lists
                            seg = seg.tolist()
                except Exception:
                    pass

                # Detect obviously-bad segmentation types (numpy arrays nested in segmentation)
                try:
                    is_numpy_seg = isinstance(seg, np.ndarray) or (
                        isinstance(seg, list) and any(isinstance(x, np.ndarray) for x in seg)
                    )
                except Exception:
                    is_numpy_seg = False
                if is_numpy_seg:
                    # If the segmentation is a list containing numpy arrays, try coercing them to lists
                    if isinstance(seg, list):
                        coerced = []
                        changed = False
                        for part in seg:
                            if isinstance(part, np.ndarray):
                                try:
                                    arr_list = np.asarray(part).astype(float).flatten().tolist()
                                    coerced.append([float(x) for x in arr_list])
                                    changed = True
                                except Exception:
                                    coerced.append(part)
                            else:
                                coerced.append(part)
                        if changed:
                            ann2['segmentation'] = coerced
                            seg = coerced
                    # If still contains numpy or otherwise invalid, record and skip
                    try:
                        if isinstance(seg, list) and any(isinstance(x, np.ndarray) for x in seg):
                            problematic.append({
                                "image_file": dd.get("file_name", ""),
                                "ann_id": ann2.get("id", None),
                                "seg_type": str(type(seg)),
                                "seg_preview": repr(seg)[:200]
                            })
                            continue
                    except Exception:
                        problematic.append({
                            "image_file": dd.get("file_name", ""),
                            "ann_id": ann2.get("id", None),
                            "seg_type": str(type(seg)),
                            "seg_preview": repr(seg)[:200]
                        })
                        continue

                if isinstance(seg, list):
                    # COCO allows a single polygon as a flat list; normalize to list of polygons
                    if len(seg) > 0 and all(not isinstance(x, (list, tuple, np.ndarray)) for x in seg) and all(isinstance(x, (int, float)) for x in seg):
                        seg = [seg]
                    new_seg = []
                    for poly in seg:
                        try:
                            # convert numpy arrays or other iterables to plain list of floats
                            poly_arr = np.asarray(poly).astype(float).flatten()
                            poly_list = poly_arr.tolist()
                        except Exception:
                            poly_list = None
                        if poly_list and isinstance(poly_list, list) and len(poly_list) >= 6:
                            new_seg.append([float(x) for x in poly_list])
                    if new_seg:
                        ann2["segmentation"] = new_seg
                    else:
                        ann2.pop("segmentation", None)
                # leave RLE dicts as-is
                anns.append(ann2)
            dd["annotations"] = anns
            clean.append(dd)
        # Write any problematic segmentation entries to disk for inspection
        if problematic:
            os.makedirs("output_annotations", exist_ok=True)
            out_path = os.path.join("output_annotations", "problematic_annotations.json")
            try:
                with open(out_path, "w", encoding="utf-8") as pf:
                    json.dump(problematic, pf, indent=2)
                print(f"[WARN] Found {len(problematic)} problematic segmentation entries; wrote to {out_path}")
            except Exception as e:
                print("[ERROR] Failed to write problematic annotations:", e)

        # If an original COCO JSON path was provided, write a cleaned COCO JSON
        if orig_json_path is not None:
            try:
                cats = []
                try:
                    with open(orig_json_path, 'r', encoding='utf-8') as of:
                        orig = json.load(of)
                        cats = orig.get('categories', [])
                except Exception:
                    cats = []
                images = []
                annotations_out = []
                ann_id = 1
                # Assign stable image ids and set annotation['image_id'] accordingly
                for img_idx, img in enumerate(clean, start=1):
                    img_id = img.get('image_id') or img.get('id') or img_idx
                    # Ensure file_name is relative to the images root used when registering the dataset.
                    raw_fn = img.get('file_name') or ''
                    try:
                        # normalize separators
                        fn_norm = raw_fn.replace('\\', '/').lstrip('./')
                        # if fn_norm already contains the img_root prefix, strip it
                        if isinstance(img_root, str) and img_root:
                            img_root_norm = img_root.replace('\\', '/').rstrip('/')
                        else:
                            img_root_norm = ''
                        rel_fn = None
                        if img_root_norm and fn_norm.startswith(img_root_norm):
                            rel_fn = fn_norm[len(img_root_norm):].lstrip('/')
                        else:
                            # try to compute a relative path if possible
                            try:
                                rel = os.path.relpath(fn_norm, img_root_norm or '.')
                                # if rel does not climb above img_root, use it
                                if not rel.startswith('..'):
                                    rel_fn = rel.replace('\\', '/')
                            except Exception:
                                rel_fn = None
                        if not rel_fn or rel_fn == '.' or rel_fn.startswith('..'):
                            # fallback to basename so load_coco_json will join correctly
                            rel_fn = os.path.basename(fn_norm)
                    except Exception:
                        rel_fn = os.path.basename(raw_fn)

                    images.append({
                        'id': img_id,
                        'file_name': rel_fn,
                        'height': img.get('height'),
                        'width': img.get('width')
                    })
                    for a in img.get('annotations', []):
                        a_copy = dict(a)
                        if a_copy.get('id') is None:
                            a_copy['id'] = ann_id
                            ann_id += 1
                        # ensure annotation references correct image id
                        a_copy['image_id'] = img_id
                        seg = a_copy.get('segmentation')
                        if isinstance(seg, list):
                            safe_segs = []
                            for s in seg:
                                try:
                                    if isinstance(s, (list, tuple)):
                                        s_list = [float(x) for x in s]
                                    else:
                                        s_list = None
                                except Exception:
                                    s_list = None
                                if s_list and len(s_list) >= 6:
                                    safe_segs.append(s_list)
                            if safe_segs:
                                a_copy['segmentation'] = safe_segs
                            else:
                                a_copy.pop('segmentation', None)
                        annotations_out.append(a_copy)
                # Fix category_id offset if annotations use 0 but categories start at 1
                try:
                    cat_ids = {int(c.get('id')) for c in cats if 'id' in c}
                except Exception:
                    cat_ids = set()
                need_increment = False
                if annotations_out:
                    for a in annotations_out:
                        if a.get('category_id') == 0 and 0 not in cat_ids:
                            need_increment = True
                            break
                if need_increment:
                    for a in annotations_out:
                        if 'category_id' in a and isinstance(a['category_id'], (int, float)):
                            try:
                                a['category_id'] = int(a['category_id']) + 1
                            except Exception:
                                pass
                    print("[INFO] Incremented annotation 'category_id' values by 1 to match categories ids")

                cleaned = {'images': images, 'annotations': annotations_out, 'categories': cats}
                base = os.path.splitext(os.path.basename(orig_json_path))[0]
                cleaned_path = os.path.join('output_annotations', f"{base}_clean.json")
                with open(cleaned_path, 'w', encoding='utf-8') as cf:
                    json.dump(cleaned, cf, indent=2)
                print(f"Wrote cleaned COCO JSON to {cleaned_path}")
            except Exception as e:
                print("[ERROR] Failed to write cleaned COCO JSON:", e)

        clean_name = name + "_clean"
        DatasetCatalog.register(clean_name, lambda d=clean: d)
        # copy over basic metadata
        try:
            meta = MetadataCatalog.get(name)
            MetadataCatalog.get(clean_name).set(**{k: getattr(meta, k) for k in ["thing_classes", "evaluator_type"] if hasattr(meta, k)})
        except Exception:
            pass
        return clean_name

    train_dataset_name = sanitize_dataset(train_name, train_img_root, train_json_path) if os.path.isfile(train_json_path) else train_name
    val_dataset_name = sanitize_dataset(val_name, val_img_root, val_json_path) if os.path.isfile(val_json_path) else val_name

    # After writing cleaned JSONs, scan them for any remaining malformed segmentation entries
    def is_valid_polygon_seg(seg):
        # valid: list of polygon(s), each polygon is a list of floats with even length >=6
        if not isinstance(seg, list):
            return False
        if len(seg) == 0:
            return False
        for poly in seg:
            if not isinstance(poly, (list, tuple)):
                return False
            if len(poly) < 6:
                return False
            for x in poly:
                if not isinstance(x, (int, float)):
                    return False
        return True

    def scan_and_filter_clean_json(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                j = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not open cleaned JSON {path}: {e}")
            return None
        imgs = j.get('images', [])
        anns = j.get('annotations', [])
        bad = []
        for a in anns:
            seg = a.get('segmentation')
            if seg is None:
                continue
            if not is_valid_polygon_seg(seg):
                bad.append({'id': a.get('id'), 'image_id': a.get('image_id'), 'seg_type': str(type(seg)), 'seg_preview': repr(seg)[:200]})
        if bad:
            os.makedirs('output_annotations', exist_ok=True)
            report = path.replace('.json', '') + '_problems.json'
            with open(report, 'w', encoding='utf-8') as rf:
                json.dump(bad, rf, indent=2)
            print(f"[WARN] Found {len(bad)} problematic annotations in {path}; wrote {report}")
            # create a filtered JSON removing bad annotations
            filtered = {'images': imgs, 'annotations': [a for a in anns if a.get('segmentation') is None or is_valid_polygon_seg(a.get('segmentation'))], 'categories': j.get('categories', [])}
            out_path = path.replace('.json', '') + '_filtered.json'
            with open(out_path, 'w', encoding='utf-8') as of:
                json.dump(filtered, of, indent=2)
            print(f"Wrote filtered JSON to {out_path}")
            return out_path
        else:
            print(f"No problems found in {path}")
            return path

    # Scan cleaned outputs if present
    train_clean_path = os.path.join('output_annotations', os.path.splitext(os.path.basename(train_json_path))[0] + '_clean.json') if os.path.isfile(train_json_path) else None
    val_clean_path = os.path.join('output_annotations', os.path.splitext(os.path.basename(val_json_path))[0] + '_clean.json') if os.path.isfile(val_json_path) else None
    if train_clean_path and os.path.isfile(train_clean_path):
        train_filtered = scan_and_filter_clean_json(train_clean_path)
        if train_filtered and train_filtered.endswith('_filtered.json'):
            # register filtered JSON instead
            try:
                register_coco_instances(train_name + '_filtered', {}, train_filtered, train_img_root)
                train_dataset_name = sanitize_dataset(train_name + '_filtered', train_img_root, train_filtered)
                cfg.DATASETS.TRAIN = (train_dataset_name,)
                print(f"Using filtered train JSON: {train_filtered}")
            except Exception as e:
                print('[WARN] Could not register filtered train JSON:', e)
    if val_clean_path and os.path.isfile(val_clean_path):
        val_filtered = scan_and_filter_clean_json(val_clean_path)
        if val_filtered and val_filtered.endswith('_filtered.json'):
            try:
                register_coco_instances(val_name + '_filtered', {}, val_filtered, val_img_root)
                val_dataset_name = sanitize_dataset(val_name + '_filtered', val_img_root, val_filtered)
                cfg.DATASETS.TEST = (val_dataset_name,)
                print(f"Using filtered val JSON: {val_filtered}")
            except Exception as e:
                print('[WARN] Could not register filtered val JSON:', e)

    cfg = get_cfg()
    try:
        _mz = _get_model_zoo()
        cfg.merge_from_file(_mz.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = _mz.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    except Exception as e:
        print(f"[WARN] Could not load default model_zoo config/weights: {e}")

   

    # Use the sanitized (clean) dataset names for training/testing if available
    cfg.DATASETS.TRAIN = (train_dataset_name,) if isinstance(train_dataset_name, str) else (train_name,)
    cfg.DATASETS.TEST = (val_dataset_name,) if isinstance(val_dataset_name, str) else (val_name,)
    cfg.DATALOADER.NUM_WORKERS = _choose_default_num_workers()
    # cfg.SOLVER.STEPS = (3000,4000)
    print(f"[INFO] Default DATALOADER.NUM_WORKERS = {cfg.DATALOADER.NUM_WORKERS}")

     # Apply optional config/weight overrides provided by caller (CLI or function args)
    try:
        if config_file:
            try:
                cfg.merge_from_file(config_file)
                print(f"[INFO] Merged config file: {config_file}")
            except Exception:
                try:
                    # maybe a short detectron2 config key under detectron2/configs
                    resolved_cfg = _resolve_detectron2_cfg_file(config_file)
                    cfg.merge_from_file(resolved_cfg)
                    print(f"[INFO] Merged detectron2 config key: {config_file} -> {resolved_cfg}")
                except Exception:
                    print(f"[WARN] Could not load config file: {config_file}")
        if weights:
            cfg.MODEL.WEIGHTS = weights
            print(f"[INFO] Set MODEL.WEIGHTS = {weights}")

        # Simple scalar overrides
        if output_dir:
            cfg.OUTPUT_DIR = output_dir
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            print(f"[INFO] Set OUTPUT_DIR = {cfg.OUTPUT_DIR}")
        if num_workers is not None:
            cfg.DATALOADER.NUM_WORKERS = int(num_workers)
            print(f"[INFO] Set DATALOADER.NUM_WORKERS = {cfg.DATALOADER.NUM_WORKERS}")
        if ims_per_batch is not None:
            cfg.SOLVER.IMS_PER_BATCH = int(ims_per_batch)
            print(f"[INFO] Set SOLVER.IMS_PER_BATCH = {cfg.SOLVER.IMS_PER_BATCH}")
        if base_lr is not None:
            cfg.SOLVER.BASE_LR = float(base_lr)
            print(f"[INFO] Set SOLVER.BASE_LR = {cfg.SOLVER.BASE_LR}")
        if max_iter is not None:
            cfg.SOLVER.MAX_ITER = int(max_iter)
            print(f"[INFO] Set SOLVER.MAX_ITER = {cfg.SOLVER.MAX_ITER}")
        if batch_size_per_image is not None:
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = int(batch_size_per_image)
            print(f"[INFO] Set MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = {cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}")
        if num_classes is not None:
            _num_classes = int(num_classes)
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = _num_classes
            # MaskDINO uses SEM_SEG_HEAD class count for its class embedding.
            if hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
                cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = _num_classes
            print(f"[INFO] Set MODEL.ROI_HEADS.NUM_CLASSES = {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
            if hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
                print(f"[INFO] Set MODEL.SEM_SEG_HEAD.NUM_CLASSES = {cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}")

        # Extra dotted cfg overrides, e.g. "SOLVER.BASE_LR=0.001"
        if extra_cfg:
            for opt in extra_cfg:
                try:
                    if '=' not in opt:
                        print(f"[WARN] Skipping invalid cfg override (no '='): {opt}")
                        continue
                    key, val = opt.split('=', 1)
                    parts = key.split('.')
                    node = cfg
                    for p in parts[:-1]:
                        node = getattr(node, p)
                    attr = parts[-1]
                    # parse value to bool/int/float when possible
                    v_lower = val.lower()
                    if v_lower in ('true', 'false'):
                        parsed = v_lower == 'true'
                    else:
                        try:
                            if '.' in val:
                                parsed = float(val)
                                if parsed.is_integer():
                                    parsed = int(parsed)
                            else:
                                parsed = int(val)
                        except Exception:
                            try:
                                parsed = float(val)
                            except Exception:
                                parsed = val
                    setattr(node, attr, parsed)
                    print(f"[INFO] Set cfg {key} = {parsed}")
                except Exception as e:
                    print(f"[WARN] Failed to apply cfg override '{opt}': {e}")
    except Exception as e:
        print(f"[WARN] Error applying config overrides: {e}")

    # Re-apply sanitized dataset names after config merges unless the user
    # explicitly overrode DATASETS via --set.
    try:
        _datasets_overridden = False
        if extra_cfg:
            for _opt in extra_cfg:
                if not isinstance(_opt, str):
                    continue
                _k = _opt.split('=', 1)[0].strip().upper()
                if _k in ("DATASETS.TRAIN", "DATASETS.TEST"):
                    _datasets_overridden = True
                    break
        if not _datasets_overridden:
            cfg.DATASETS.TRAIN = (train_dataset_name,) if isinstance(train_dataset_name, str) else (train_name,)
            cfg.DATASETS.TEST = (val_dataset_name,) if isinstance(val_dataset_name, str) else (val_name,)
            print(f"[INFO] Enforced sanitized datasets: TRAIN={cfg.DATASETS.TRAIN}, TEST={cfg.DATASETS.TEST}")
        else:
            print("[INFO] Keeping user-overridden DATASETS from --set")
    except Exception as e:
        print(f"[WARN] Failed to enforce sanitized datasets after config merge: {e}")

    # Optional eval-speed tuning. COCO post-processing can dominate runtime when
    # too many low-confidence instances/masks are emitted per image.
    try:
        if bool(fast_eval):
            if eval_score_thresh is None:
                eval_score_thresh = 0.5
            if eval_detections_per_image is None:
                eval_detections_per_image = 100
            if eval_focus_on_box is None:
                eval_focus_on_box = True

        if eval_detections_per_image is not None:
            cfg.TEST.DETECTIONS_PER_IMAGE = int(eval_detections_per_image)
            print(f"[INFO] Set TEST.DETECTIONS_PER_IMAGE = {cfg.TEST.DETECTIONS_PER_IMAGE}")

        if eval_score_thresh is not None:
            _th = float(eval_score_thresh)
            try:
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = _th
                print(f"[INFO] Set MODEL.ROI_HEADS.SCORE_THRESH_TEST = {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
            except Exception:
                pass
            try:
                cfg.MODEL.RETINANET.SCORE_THRESH_TEST = _th
                print(f"[INFO] Set MODEL.RETINANET.SCORE_THRESH_TEST = {cfg.MODEL.RETINANET.SCORE_THRESH_TEST}")
            except Exception:
                pass
            try:
                if hasattr(cfg.MODEL, "MaskDINO") and hasattr(cfg.MODEL.MaskDINO, "TEST"):
                    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = _th
                    print(f"[INFO] Set MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = {cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD}")
            except Exception:
                pass

        if eval_focus_on_box is not None:
            try:
                if hasattr(cfg.MODEL, "MaskDINO") and hasattr(cfg.MODEL.MaskDINO, "TEST"):
                    cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX = bool(eval_focus_on_box)
                    print(f"[INFO] Set MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX = {cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX}")
            except Exception:
                pass
    except Exception as e:
        print(f"[WARN] Failed to apply eval tuning options: {e}")
        
        
    # Infer number of classes from train JSON categories
    try:
        with open(train_json_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        cats = j.get("categories", [])
        if cats:
            _num_classes = len(cats)
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = _num_classes
            if hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
                cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = _num_classes
            print(f"Set NUM_CLASSES = {_num_classes} from train JSON categories")
            if hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
                print(f"[INFO] Set MODEL.SEM_SEG_HEAD.NUM_CLASSES = {cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}")
        else:
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            if hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
                cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
            print("No categories found in train JSON; defaulting NUM_CLASSES=1")
    except Exception:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        if hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
        print("Could not read train JSON to infer NUM_CLASSES; defaulting to 1")

    # If epochs provided, compute SOLVER.MAX_ITER from dataset size and ims_per_batch
    try:
        if epochs is not None:
            try:
                from detectron2.data import DatasetCatalog
                ds = list(DatasetCatalog.get(train_dataset_name))
                num_images = len(ds)
            except Exception:
                # fallback to reading COCO JSON if registered dataset not available
                try:
                    with open(train_json_path, 'r', encoding='utf-8') as _jf:
                        jjj = json.load(_jf)
                        num_images = len(jjj.get('images', []))
                except Exception:
                    num_images = None

            if num_images and num_images > 0:
                imgs_per_iter = getattr(cfg.SOLVER, 'IMS_PER_BATCH', 1) or 1
                try:
                    iters_per_epoch = int(math.ceil(float(num_images) / float(imgs_per_iter)))
                except Exception:
                    iters_per_epoch = int(max(1, num_images))
                cfg.SOLVER.MAX_ITER = int(iters_per_epoch * int(epochs))
                print(f"[INFO] Set SOLVER.MAX_ITER={cfg.SOLVER.MAX_ITER} from epochs={epochs} (images={num_images}, ims_per_batch={imgs_per_iter}, iters_per_epoch={iters_per_epoch})")
            else:
                print("[WARN] Could not determine number of training images; skipping epochs->MAX_ITER conversion")
    except Exception as e:
        print(f"[WARN] Failed to compute MAX_ITER from epochs: {e}")
    if not output_dir:
        cfg.OUTPUT_DIR = "output_maskdino/trainer_output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Match train_m.py behavior: write Detectron2 logs under OUTPUT_DIR/log.txt.
    try:
        from detectron2.utils import comm as d2_comm
        from detectron2.utils.logger import setup_logger as d2_setup_logger

        d2_setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=d2_comm.get_rank())
        d2_setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=d2_comm.get_rank(), name="maskdino")
        print(f"[INFO] Logging to {os.path.join(cfg.OUTPUT_DIR, 'log.txt')}")
    except Exception as e:
        print(f"[WARN] Could not initialize file logging in OUTPUT_DIR: {e}")

    # Monkey-patch annotations_to_instances to dump offending annotations on ValueError
    try:
        from detectron2.data import detection_utils as dutils
        _orig_annotations_to_instances = dutils.annotations_to_instances
        def _wrapped_annotations_to_instances(*args, **kwargs):
            # annotations may be positional or keyword arg; extract if available
            annotations = None
            if "annotations" in kwargs:
                annotations = kwargs.get("annotations")
            elif len(args) >= 1:
                annotations = args[0]
            try:
                return _orig_annotations_to_instances(*args, **kwargs)
            except ValueError as e:
                try:
                    os.makedirs("output_annotations", exist_ok=True)
                    dump_path = os.path.join("output_annotations", "bad_annotations_dump.json")
                    # Prepare a serializable dump: include types and reprs
                    dump = []
                    if isinstance(annotations, (list, tuple)):
                        for ann in annotations:
                            dump.append({
                                "id": ann.get("id") if isinstance(ann, dict) else None,
                                "seg_type": str(type(ann.get("segmentation"))) if isinstance(ann, dict) else str(type(ann)),
                                "seg_preview": (repr(ann.get("segmentation"))[:400] if isinstance(ann, dict) else repr(ann)[:400])
                            })
                    else:
                        dump.append({"args_repr": repr(args)[:400], "kwargs_repr": repr(kwargs)[:400]})

                    # Try to capture calling-frame context (dataset_dict, index) to identify the image/iteration
                    try:
                        import inspect, datetime
                        context = {"captured_at": datetime.datetime.utcnow().isoformat()}
                        # walk the stack to find useful locals: dataset_dict, index vars, and trainer self
                        for frame_info in inspect.stack():
                            loc = frame_info.frame.f_locals
                            if 'dataset_dict' in loc and isinstance(loc['dataset_dict'], dict):
                                dd = loc['dataset_dict']
                                # include a small snapshot of the dataset_dict and first annotation if present
                                snapshot = {'file_name': dd.get('file_name'), 'image_id': dd.get('image_id') or dd.get('id')}
                                anns = dd.get('annotations') or []
                                if isinstance(anns, (list, tuple)) and len(anns) > 0:
                                    a0 = anns[0]
                                    snapshot['first_annotation_preview'] = {k: a0.get(k) for k in ('id','category_id','bbox') if isinstance(a0, dict)}
                                context['dataset_dict_snapshot'] = snapshot
                                # capture index-like variables if available
                                for key in ('cur_idx', 'cur_index', 'idx', 'index', 'i'):
                                    if key in loc:
                                        try:
                                            context.setdefault('frame_index_vars', {})[key] = int(loc[key])
                                        except Exception:
                                            context.setdefault('frame_index_vars', {})[key] = repr(loc[key])
                                # also try to capture a trainer iteration if present nearby
                                if 'self' in loc:
                                    s = loc['self']
                                    try:
                                        if hasattr(s, 'iteration'):
                                            context['trainer_iteration'] = int(getattr(s, 'iteration'))
                                        elif hasattr(s, 'start_iter'):
                                            context['trainer_start_iter'] = int(getattr(s, 'start_iter'))
                                    except Exception:
                                        pass
                                break
                            # also attempt to capture trainer loop iteration from other frames
                            if 'self' in loc and isinstance(loc['self'], object):
                                s = loc['self']
                                try:
                                    if hasattr(s, 'iteration'):
                                        context.setdefault('trainer_iteration_candidates', []).append(int(getattr(s, 'iteration')))
                                except Exception:
                                    pass
                        # fallback minimal context
                        context.setdefault('notes','captured stack scan')
                    except Exception:
                        context = {"captured_at": None}

                    out_obj = {"error": str(e), "annotations_dump": dump, "context": context}
                    with open(dump_path, "w", encoding="utf-8") as df:
                        json.dump(out_obj, df, indent=2)
                    print(f"[ERROR] annotations_to_instances failed; dumped annotations+context to {dump_path}")
                except Exception as dump_e:
                    print("[ERROR] Failed to dump bad annotations:", dump_e)
                raise
        dutils.annotations_to_instances = _wrapped_annotations_to_instances
    except Exception as e:
        print("[WARN] Could not monkey-patch annotations_to_instances:", e)

    resume_ckpt = None
    def _find_local_checkpoint(output_dir):
        """Return best local checkpoint path from output dir, or None."""
        try:
            final_pth = os.path.join(output_dir, "model_final.pth")
            if os.path.isfile(final_pth):
                return final_pth
        except Exception:
            pass
        try:
            models = glob.glob(os.path.join(output_dir, "model_*.pth"))
            if models:
                models.sort(key=os.path.getmtime, reverse=True)
                return models[0]
        except Exception:
            pass
        return None
    try:
        last_path = os.path.join(cfg.OUTPUT_DIR, "last_checkpoint")
        if os.path.isfile(last_path):
            with open(last_path, "r", encoding="utf-8") as f:
                name = f.read().strip()
            if name:
                resume_ckpt = name if os.path.isabs(name) else os.path.join(cfg.OUTPUT_DIR, name)
    except Exception:
        resume_ckpt = None

    # Subclass DefaultTrainer to add COCO evaluation during training
    class DroneTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            from detectron2.evaluation import COCOEvaluator
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
            return COCOEvaluator(dataset_name, tasks=None, distributed=False, output_dir=output_folder)

        @classmethod
        def build_train_loader(cls, cfg):
            from detectron2.data import DatasetMapper, build_detection_train_loader
            from detectron2.utils import comm

            mapper = DatasetMapper(cfg, is_train=True)
            blur_prob = float(gaussian_blur_prob)
            sigma_min = float(os.environ.get("GAUSSIAN_BLUR_SIGMA_MIN", "0.5"))
            sigma_max = float(os.environ.get("GAUSSIAN_BLUR_SIGMA_MAX", "2.0"))

            if cv2 is not None and blur_prob > 0.0:
                mapper = RandomGaussianBlurMapper(
                    mapper,
                    prob=blur_prob,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                )
                print(
                    f"[AUG] Gaussian blur enabled: prob={blur_prob} "
                    f"sigma=[{sigma_min}, {sigma_max}]"
                )
            else:
                print("[AUG] Gaussian blur disabled (cv2 missing or --gaussian-blur-prob<=0)")

            loader = build_detection_train_loader(cfg, mapper=mapper)

            world_size = comm.get_world_size()
            if world_size > 1:
                _debug_log_train_loader_sampler(loader, prefix="[DEBUG]")

            return loader

    trainer = DroneTrainer(cfg)
    # resume=True will continue from last checkpoint if present
    try:
        if bool(resume):
            print(f"[INFO] resume=True, attempting to load checkpoint from OUTPUT_DIR: {resume_ckpt or '(none found)'}")
        trainer.resume_or_load(resume=bool(resume))
    except ValueError as e:
        msg = str(e)
        mismatch = "parameter group" in msg and "optimizer" in msg.lower()
        if bool(resume) and mismatch:
            print("[WARN] Resume checkpoint optimizer state is incompatible with current model/config.")
            local_fallback = None
            if resume_ckpt and os.path.isfile(resume_ckpt):
                local_fallback = resume_ckpt
            else:
                local_fallback = _find_local_checkpoint(cfg.OUTPUT_DIR)

            if local_fallback:
                cfg.MODEL.WEIGHTS = local_fallback
                print(f"[WARN] Falling back to weights-only load from local checkpoint: {cfg.MODEL.WEIGHTS}")
            else:
                # Avoid hard failure in offline environments when default weights is an URL.
                if str(cfg.MODEL.WEIGHTS).startswith(("http://", "https://")):
                    print("[WARN] No local checkpoint found and MODEL.WEIGHTS is a remote URL.")
                    print("[WARN] Falling back to random initialization (MODEL.WEIGHTS='') for offline run.")
                    cfg.MODEL.WEIGHTS = ""
                else:
                    print(f"[WARN] Falling back to weights-only load from MODEL.WEIGHTS={cfg.MODEL.WEIGHTS}")
            print("[WARN] To avoid this warning, clear OUTPUT_DIR/last_checkpoint or run with --no-resume.")
            # Recreate trainer/checkpointer to avoid internal partial-load state assertions.
            trainer = DroneTrainer(cfg)
            trainer.resume_or_load(resume=False)
        else:
            raise

    # Log explicit resume state to make optimizer/scheduler restoration obvious in logs.
    try:
        start_iter = int(getattr(trainer, "start_iter", 0))
        max_iter_now = int(getattr(trainer, "max_iter", cfg.SOLVER.MAX_ITER))
        print(f"[RESUME] trainer.start_iter={start_iter} max_iter={max_iter_now}")

        opt = getattr(trainer, "optimizer", None)
        if opt is not None and getattr(opt, "param_groups", None):
            lrs = []
            for pg in opt.param_groups:
                try:
                    lrs.append(float(pg.get("lr", float("nan"))))
                except Exception:
                    pass
            if lrs:
                lr_min = min(lrs)
                lr_max = max(lrs)
                print(f"[RESUME] optimizer lr range: min={lr_min:.8f}, max={lr_max:.8f}")

        if bool(resume) and start_iter <= 0:
            print("[RESUME][WARN] start_iter is 0 while resume=True (likely weights-only load)")
    except Exception as e:
        print(f"[RESUME][WARN] Could not inspect resume state: {e}")

    # --- DDP & data-loader sanity checks (help debug multi-node behavior) ---
    try:
        print("[DEBUG] torch.distributed available:", torch.distributed.is_available())
        print("[DEBUG] torch.distributed initialized:", torch.distributed.is_initialized())
        print(f"[DEBUG] Env RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
    except Exception:
        print("[DEBUG] Could not query torch.distributed state")

    # Inspect trainer.model for DDP wrapping
    try:
        is_ddp = isinstance(getattr(trainer, 'model', None), torch.nn.parallel.DistributedDataParallel)
        print(f"[DEBUG] trainer.model is DistributedDataParallel: {is_ddp}")
    except Exception:
        print("[DEBUG] Could not inspect trainer.model for DDP wrapper")

    # Build a temporary train loader to inspect its sampler (non-destructive)
    try:
        from detectron2.data import build_detection_train_loader
        tmp_loader = build_detection_train_loader(cfg)
        _debug_log_train_loader_sampler(tmp_loader, prefix="[DEBUG]")
    except Exception as e:
        print(f"[WARN] Could not build/inspect train loader: {e}")

    # synchronize all processes before starting training (if DDP active)
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception:
        pass

    # Register checkpoint cleanup hook (runs every CHECKPOINT_PERIOD steps, keeps last N checkpoints)
    try:
        _ckpt_period = getattr(cfg.SOLVER, 'CHECKPOINT_PERIOD', 100)
        trainer.register_hooks([CheckpointCleanupHook(cfg.OUTPUT_DIR, keep=4, period=_ckpt_period)])
    except Exception as e:
        print(f"[WARN] Could not register CheckpointCleanupHook: {e}")

    trainer.train()

    # ensure all processes reach this point before evaluation/plotting
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception:
        pass

    # determine rank/main process for multi-process setups; only rank 0 should do IO-heavy tasks
    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    try:
        rank = torch.distributed.get_rank() if is_distributed else 0
    except Exception:
        rank = 0
    is_main = (rank == 0)

    RUN_EVALUATION = True  # Set to True to run evaluation after training
    # Avoid a second distributed eval pass here: trainer already runs eval hooks.
    # Running this block on rank 0 only while world_size>1 can stall on collectives.
    if RUN_EVALUATION and is_distributed:
        if is_main:
            print("[INFO] Skipping custom post-train evaluation in distributed mode (avoids duplicate eval + rank mismatch timeout)")
    # Run optional extra evaluation only for single-process runs.
    elif RUN_EVALUATION and is_main:
        try:
            from detectron2.evaluation import COCOEvaluator, inference_on_dataset
            from detectron2.data import build_detection_test_loader
            eval_tasks = ("bbox",) if bool(eval_bbox_only) else None
            if eval_tasks is not None:
                print("[INFO] Evaluation tasks set to bbox-only for faster validation")
            _eval_out = os.path.join(cfg.OUTPUT_DIR, "inference")
            try:
                # detectron2>=0.6 signature: COCOEvaluator(dataset_name, tasks=None, ...)
                evaluator = COCOEvaluator(
                    val_dataset_name,
                    tasks=eval_tasks,
                    distributed=False,
                    output_dir=_eval_out,
                )
            except TypeError:
                # older signature: COCOEvaluator(dataset_name, cfg, ...)
                evaluator = COCOEvaluator(
                    val_dataset_name,
                    cfg,
                    distributed=False,
                    output_dir=_eval_out,
                    tasks=eval_tasks,
                )
            val_loader = build_detection_test_loader(cfg, val_dataset_name)
            results = inference_on_dataset(trainer.model, val_loader, evaluator)
            print(f"[INFO] Evaluation results: {results}")
        except Exception as e:
            print("[WARN] Evaluation step failed or unavailable:", e)

    # After training, prefer to use the trainer-produced weights for any predictor/visualization.
    try:
        import glob
        ckpt = None
        final_pth = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        if os.path.isfile(final_pth):
            ckpt = final_pth
        else:
            # look for model_*.pth and pick the most recent
            models = glob.glob(os.path.join(cfg.OUTPUT_DIR, "model_*.pth"))
            if models:
                models.sort(key=os.path.getmtime, reverse=True)
                ckpt = models[0]
            else:
                # try to read last_checkpoint file if present
                last_path = os.path.join(cfg.OUTPUT_DIR, "last_checkpoint")
                if os.path.isfile(last_path):
                    try:
                        with open(last_path, 'r', encoding='utf-8') as lf:
                            content = lf.read().strip()
                        if os.path.isfile(content):
                            ckpt = content
                    except Exception:
                        pass
        if ckpt:
            cfg.MODEL.WEIGHTS = ckpt
            print(f"[INFO] Using trainer checkpoint for visualization: {ckpt}")
        else:
            print("[WARN] No trainer checkpoint found; predictor will use cfg.MODEL.WEIGHTS (may be pretrained checkpoint)")
    except Exception as e:
        print("[WARN] Failed to locate trainer checkpoint:", e)
    # Only the main rank should perform plotting / prediction / heavy IO
    if is_main:
        try:
            import glob as _glob
            metrics_paths = []
            # common location: cfg.OUTPUT_DIR/metrics.json
            mpath = os.path.join(cfg.OUTPUT_DIR, 'metrics.json')
            if os.path.isfile(mpath):
                metrics_paths.append(mpath)
            # also search for metrics.json under output dir
            try:
                metrics_paths.extend([p for p in _glob.glob(os.path.join(cfg.OUTPUT_DIR, '**', 'metrics.json'), recursive=True) if os.path.isfile(p) and p not in metrics_paths])
            except Exception:
                pass
            if not metrics_paths:
                # try parent folder
                parent = os.path.dirname(cfg.OUTPUT_DIR)
                try:
                    metrics_paths.extend([p for p in _glob.glob(os.path.join(parent, '**', 'metrics.json'), recursive=True) if os.path.isfile(p)])
                except Exception:
                    pass
            if metrics_paths:
                mp = metrics_paths[0]
                try:
                    import json as _json
                    iters = []
                    total_loss = []
                    loss_cls = []
                    loss_box = []
                    loss_mask = []
                    lrs = []
                    times = []
                    # mask_rcnn metrics to collect if present
                    mask_acc = []
                    mask_fn = []
                    mask_fp = []
                    with open(mp, 'r', encoding='utf-8') as mf:
                        for line in mf:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entry = _json.loads(line)
                            except Exception:
                                continue
                            if 'iteration' in entry:
                                iters.append(int(entry.get('iteration', len(iters))))
                                total_loss.append(float(entry.get('total_loss', float('nan'))))
                                loss_cls.append(float(entry.get('loss_cls', float('nan'))))
                                loss_box.append(float(entry.get('loss_box_reg', float('nan'))))
                                loss_mask.append(float(entry.get('loss_mask', float('nan'))))
                                lrs.append(float(entry.get('lr', float('nan'))))
                                times.append(float(entry.get('time', float('nan'))))
                                # collect mask_rcnn metrics when present
                                mask_acc.append(float(entry.get('mask_rcnn/accuracy', float('nan'))))
                                mask_fn.append(float(entry.get('mask_rcnn/false_negative', float('nan'))))
                                mask_fp.append(float(entry.get('mask_rcnn/false_positive', float('nan'))))
                    if iters:
                        try:
                            import matplotlib.pyplot as _plt
                            import numpy as _np
                            fig, ax = _plt.subplots(figsize=(10, 6))
                            ax.plot(iters, total_loss, label='total_loss')
                            ax.plot(iters, loss_cls, label='loss_cls')
                            ax.plot(iters, loss_box, label='loss_box_reg')
                            ax.plot(iters, loss_mask, label='loss_mask')
                            ax.set_xlabel('iteration')
                            ax.set_ylabel('loss')
                            ax.legend(loc='upper right')
                            ax2 = ax.twinx()
                            ax2.plot(iters, lrs, color='tab:orange', linestyle='--', label='lr')
                            ax2.set_ylabel('lr')
                            ax2.legend(loc='upper left')
                            # Plot mask_rcnn metrics (accuracy / false_negative / false_positive) if available
                            try:
                                any_mask = any(not (_np.isnan(x)) for x in mask_acc + mask_fn + mask_fp)
                            except Exception:
                                any_mask = False
                            if any_mask:
                                fig2, axm = _plt.subplots(figsize=(10, 5))
                                plotted = False
                                try:
                                    axm.plot(iters, mask_acc, label='mask_rcnn/accuracy')
                                    plotted = True
                                except Exception:
                                    pass
                                try:
                                    axm.plot(iters, mask_fp, linestyle='--', label='mask_rcnn/false_positive')
                                    plotted = True
                                except Exception:
                                    pass
                                try:
                                    axm.plot(iters, mask_fn, linestyle=':', label='mask_rcnn/false_negative')
                                    plotted = True
                                except Exception:
                                    pass
                                if plotted:
                                    axm.set_title('Mask R-CNN: accuracy / false positive / false negative')
                                    axm.set_xlabel('iteration')
                                    axm.set_ylabel('metric')
                                    axm.legend(loc='best')
                                    acc_out = os.path.join(cfg.OUTPUT_DIR, 'metrics_mask_rcnn.png')
                                    fig2.tight_layout()
                                    fig2.savefig(acc_out, dpi=150)
                                    _plt.close(fig2)
                                    print(f"Saved mask_rcnn metrics plot to {acc_out}")

                            _plt.tight_layout()
                            metrics_png = os.path.join(cfg.OUTPUT_DIR, 'metrics_over_time.png')
                            fig.savefig(metrics_png, dpi=150)
                            _plt.close(fig)
                            # also write a CSV of selected metrics
                            csv_out = os.path.join(cfg.OUTPUT_DIR, 'metrics_over_time.csv')
                            try:
                                import csv as _csv
                                headers = ['iteration','total_loss','loss_cls','loss_box_reg','loss_mask','lr','time','mask_rcnn/accuracy','mask_rcnn/false_negative','mask_rcnn/false_positive']
                                with open(csv_out, 'w', newline='', encoding='utf-8') as cf:
                                    w = _csv.writer(cf)
                                    w.writerow(headers)
                                    for i in range(len(iters)):
                                        row = [iters[i], total_loss[i], loss_cls[i], loss_box[i], loss_mask[i], lrs[i], times[i], mask_acc[i], mask_fn[i], mask_fp[i]]
                                        w.writerow(row)
                                print(f"Wrote metrics plot to {metrics_png} and CSV to {csv_out}")
                            except Exception as e:
                                print('[WARN] Could not write metrics CSV:', e)
                        except Exception as e:
                            print('[WARN] Could not plot metrics:', e)
                except Exception as e:
                    print('[WARN] Failed to parse metrics file:', e)
            else:
                print('[INFO] No metrics.json found; skipping metrics plot')
        except Exception as e:
            print('[WARN] Metrics plotting failed:', e)

        print("\n--- TRAINING COMPLETE ---\n")
        print("Predicting validation grids...")
        predict_multiple_grids(cfg, val_dataset_name, grid_count=1, per_grid=6, out_prefix="val_grid", score_thresh=0.6, visualizer_scale=1.0, visualizer_min_distance=30, visualizer_y_offset=10)
        # ensure CUDA kernels have finished and synchronize with other ranks before teardown
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
        except Exception:
            pass
    else:
        # non-main ranks wait until main finishes IO to ensure files are written
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
        except Exception:
            pass

    # finalize distributed group if initialized (graceful shutdown)
    try:
        finalize_ddp(wait_seconds=1.0)
    except Exception:
        pass



def predict_multiple_grids(cfg=None, dataset_name="my_dataset_val", grid_count=5, per_grid=9, out_prefix=None, score_thresh=0.5, visualizer_scale=1.0, visualizer_min_distance=20, visualizer_y_offset=12):
    """Create `grid_count` separate grids, each containing `per_grid` images.

    - Prefer non-overlapping images across grids when dataset size allows.
    - Saves files named `<out_prefix>_1.png`, `_2.png`, ... or defaults to cfg.OUTPUT_DIR/val_predictions_grid_<i>.png
    """
    try:
        from detectron2.data import DatasetCatalog
    except Exception as e:
        print("[ERROR] detectron2 is required for prediction grids:", e)
        return

    if cfg is None:
        cfg = globals().get('TRAINER_CFG') or globals().get('cfg')
    if cfg is None:
        print("[ERROR] No cfg available for prediction.")
        return

    if dataset_name not in DatasetCatalog.list():
        print(f"[WARN] Dataset '{dataset_name}' not registered. Available: {DatasetCatalog.list()}")
        return

    dicts = list(DatasetCatalog.get(dataset_name))
    total_needed = grid_count * per_grid
    use_non_overlap = len(dicts) >= total_needed
    if use_non_overlap:
        random.shuffle(dicts)
    else:
        print(f"[WARN] Dataset has {len(dicts)} images; requested {total_needed}. Grids will sample with possible overlap.")

    # Use a cloned cfg for prediction to avoid mutating the training cfg
    try:
        cfg_pred = cfg.clone()
        # allow overriding score threshold for visualization without changing original cfg
        try:
            cfg_pred.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
        except Exception:
            pass
    except Exception:
        cfg_pred = cfg

    for gi in range(grid_count):
        out_file = None
        # Do not use name_offset in filenames; name_offset controls text position on images
        if out_prefix:
            out_file = os.path.join(cfg.OUTPUT_DIR, f"{out_prefix}_{gi+1}.png")
        else:
            out_file = os.path.join(cfg.OUTPUT_DIR, f"val_predictions_grid_{gi+1}.png")

        if use_non_overlap:
            start = gi * per_grid
            chunk = dicts[start:start + per_grid]
            # Build a temporary dataset view for sampling
            samples = chunk
            # call predict_val_grid but with preselected samples: create a small wrapper
            _predict_grid_from_samples(
                cfg_pred,
                samples,
                out_file,
                visualizer_scale=visualizer_scale,
                visualizer_min_distance=visualizer_min_distance,
                visualizer_y_offset=visualizer_y_offset,
            )
        else:
            # sample with replacement or random.sample if enough
            if len(dicts) >= per_grid:
                samples = random.sample(dicts, per_grid)
            else:
                samples = [random.choice(dicts) for _ in range(per_grid)]
            _predict_grid_from_samples(
                cfg_pred,
                samples,
                out_file,
                visualizer_scale=visualizer_scale,
                visualizer_min_distance=visualizer_min_distance,
                visualizer_y_offset=visualizer_y_offset,
            )


def _predict_grid_from_samples(cfg, samples, out_file, visualizer_scale=1.0, visualizer_min_distance=20, visualizer_y_offset=12):
    """Render a single grid given `samples` (list of dataset dicts)."""
    from PIL import Image
    try:
        from detectron2.utils.visualizer import Visualizer, ColorMode
        from detectron2.data import MetadataCatalog
        from detectron2.data.detection_utils import read_image
        from detectron2.engine import DefaultPredictor
    except Exception as e:
        print("[ERROR] detectron2 is required for prediction grid:", e)
        return

    # Subclass Visualizer to avoid label collisions and to avoid placing labels on top of boxes/masks
    class NonCollidingVisualizer(Visualizer):
        def __init__(self, *args, min_dist=20, y_offset=12, **kwargs):
            super().__init__(*args, **kwargs)
            self._used_positions = []
            self._occupied_rects = []  # list of (x1,y1,x2,y2) tuples where masks/boxes occupy
            self._min_dist = float(min_dist)
            self._y_offset = int(y_offset)

        def _distance(self, a, b):
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

        def _point_inside_rect(self, x, y, rect):
            x1, y1, x2, y2 = rect
            return x >= x1 and x <= x2 and y >= y1 and y <= y2

        def _point_occupied(self, x, y):
            # occupied if inside any occupied rect
            return any(self._point_inside_rect(x, y, r) for r in self._occupied_rects)

        def draw_box(self, box, *args, **kwargs):
            # record box bounds so labels avoid being placed on top
            try:
                import numpy as _np
                arr = _np.asarray(box)
                if arr.size >= 4:
                    x1, y1, x2, y2 = float(arr.flat[0]), float(arr.flat[1]), float(arr.flat[2]), float(arr.flat[3])
                else:
                    raise Exception()
            except Exception:
                try:
                    # box may be a sequence
                    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                except Exception:
                    x1 = y1 = x2 = y2 = 0.0

            # normalize
            lx, ty, rx, by = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
            self._occupied_rects.append((lx, ty, rx, by))
            return super().draw_box(box, *args, **kwargs)

        def draw_binary_mask(self, binary_mask, color, *, alpha=0.5, **kwargs):
            # record mask bbox then draw mask
            try:
                import numpy as _np
                arr = _np.asarray(binary_mask)
                if arr.ndim == 2:
                    ys, xs = _np.where(arr)
                    if xs.size and ys.size:
                        lx, rx = int(xs.min()), int(xs.max())
                        ty, by = int(ys.min()), int(ys.max())
                        self._occupied_rects.append((lx, ty, rx, by))
            except Exception:
                pass
            return super().draw_binary_mask(binary_mask, color, alpha=alpha, **kwargs)

        def draw_text(self, text, position, **kwargs):
            try:
                x = int(position[0])
                y = int(position[1])
            except Exception:
                return super().draw_text(text, position, **kwargs)

            # Estimate label size (approx) using text length and visualizer scale
            scale = getattr(self, 'scale', 1.0)
            label_h = max(10, int(12 * scale))
            label_w = max(20, int(len(str(text)) * 6 * scale))

            def rects_intersect(a, b):
                ax1, ay1, ax2, ay2 = a
                bx1, by1, bx2, by2 = b
                return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

            pad = max(2, int(self._min_dist // 2))

            attempt_y = y
            while True:
                # label bbox: assume bottom-left anchor at (x, attempt_y)
                lab_rect = (x, attempt_y - label_h, x + label_w, attempt_y)
                # pad occupied rects
                conflict = False
                for r in self._occupied_rects:
                    pr = (r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad)
                    if rects_intersect(lab_rect, pr):
                        conflict = True
                        break

                if conflict:
                    attempt_y += self._y_offset
                    continue

                # also ensure label point isn't too close to previous labels
                too_close = any(self._distance((x, attempt_y), p) < self._min_dist for p in self._used_positions)
                if too_close:
                    attempt_y += self._y_offset
                    continue

                break

            self._used_positions.append((x, attempt_y))
            return super().draw_text(text, (x, attempt_y), **kwargs)

    predictor = DefaultPredictor(cfg)
    # assume samples are valid dataset dicts
    metadata = MetadataCatalog.get(samples[0].get('dataset_name', 'my_dataset_val')) if samples and 'dataset_name' in samples[0] else MetadataCatalog.get('my_dataset_val')
    # Ensure metadata has readable `thing_classes` (list of strings). If not, try to use global CLASS_NAMES.
    try:
        tc = getattr(metadata, 'thing_classes', None)
        valid_tc = isinstance(tc, (list, tuple)) and all(isinstance(x, str) for x in tc) and len(tc) > 0
        if not valid_tc and 'CLASS_NAMES' in globals():
            try:
                cls_list = globals().get('CLASS_NAMES')
                if isinstance(cls_list, (list, tuple)) and all(isinstance(x, str) for x in cls_list):
                    setattr(metadata, 'thing_classes', list(cls_list))
            except Exception:
                pass
    except Exception:
        pass

    pil_images = []
    # prepare directory to save individual images used for this grid
    try:
        base = os.path.splitext(os.path.basename(out_file))[0] if out_file else "prediction_grid"
    except Exception:
        base = "prediction_grid"
    individuals_dir = os.path.join(cfg.OUTPUT_DIR, f"{base}_individuals")
    try:
        os.makedirs(individuals_dir, exist_ok=True)
    except Exception:
        individuals_dir = None

    for i, d in enumerate(samples):
        file_name = d.get("file_name")
        if not file_name:
            continue
        try:
            img_bgr = read_image(file_name, format="BGR")
        except Exception:
            img_bgr = read_image(os.path.join(os.getcwd(), file_name), format="BGR")
        outputs = predictor(img_bgr)
        img_rgb = img_bgr[:, :, ::-1].copy()
        vis = NonCollidingVisualizer(
            img_rgb,
            metadata=metadata,
            scale=visualizer_scale,
            instance_mode=ColorMode.IMAGE,
            min_dist=visualizer_min_distance,
            y_offset=visualizer_y_offset,
        )
        try:
            if "instances" in outputs:
                drawn = vis.draw_instance_predictions(outputs["instances"].to("cpu"))
            else:
                drawn = vis
        except Exception:
            drawn = vis

        # attempt to extract image (same logic as predict_val_grid)
        vis_img = None
        try:
            if hasattr(drawn, "get_image"):
                vis_img = drawn.get_image()
            elif hasattr(drawn, "get_output"):
                out = drawn.get_output()
                if hasattr(out, "get_image"):
                    vis_img = out.get_image()
                elif isinstance(out, np.ndarray):
                    vis_img = out
            elif isinstance(drawn, np.ndarray):
                vis_img = drawn
            else:
                if hasattr(vis, "get_output"):
                    out = vis.get_output()
                    if hasattr(out, "get_image"):
                        vis_img = out.get_image()
        except Exception:
            vis_img = None

        if vis_img is None:
            vis_img = img_rgb
        pil = Image.fromarray(vis_img)
        pil_images.append(pil)
        # save each individual visualized image to the individuals subfolder
        if individuals_dir:
            try:
                orig = os.path.basename(file_name)
                safe = os.path.splitext(orig)[0]
                save_name = f"{i+1:02d}_{safe}.png"
                save_path = os.path.join(individuals_dir, save_name)
                pil.save(save_path)
            except Exception as e:
                print(f"[WARN] Could not save individual image for {file_name}: {e}")

    if not pil_images:
        print("[WARN] No images rendered for this grid.")
        return

    cols = 3
    rows = math.ceil(len(pil_images) / cols)
    max_w = max(im.width for im in pil_images)
    max_h = max(im.height for im in pil_images)
    grid_img = Image.new("RGB", (cols * max_w, rows * max_h), (0, 0, 0))
    for idx, im in enumerate(pil_images):
        r = idx // cols
        c = idx % cols
        if im.width != max_w or im.height != max_h:
            im = im.resize((max_w, max_h))
        grid_img.paste(im, (c * max_w, r * max_h))

    os.makedirs(os.path.dirname(out_file) or cfg.OUTPUT_DIR, exist_ok=True)
    grid_img.save(out_file)
    print(f"Saved prediction grid to {out_file}")


if __name__ == "__main__":
    # When launched under torch.distributed.run (torchrun), each process
    # will execute this file. Guard the top-level call so importing the
    # module doesn't start training unintentionally.
    import argparse

    parser = argparse.ArgumentParser(description="Run default trainer with optional cfg overrides")
    parser.add_argument('--train-json', default="output_annotations/train_polygons.json", help='path to train COCO JSON')
    parser.add_argument('--val-json', default="output_annotations/val_polygons.json", help='path to val COCO JSON')
    parser.add_argument('--images-root', default="dataset/images", help='root folder for images')
    parser.add_argument('--val-images-root', default=None, help='root folder for val/test images (defaults to images-root)')
    parser.add_argument('--output-dir', default=None, help='output directory to write trainer outputs')
    parser.add_argument('--max-iter', type=int, default=None, help='override SOLVER.MAX_ITER')
    parser.add_argument('--ims-per-batch', type=int, default=None, help='override SOLVER.IMS_PER_BATCH')
    parser.add_argument('--base-lr', type=float, default=None, help='override SOLVER.BASE_LR')
    parser.add_argument('--num-workers', type=int, default=None, help='override DATALOADER.NUM_WORKERS')
    parser.add_argument('--batch-size-per-image', type=int, default=None, help='override MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE')
    parser.add_argument('--num-classes', type=int, default=None, help='override MODEL.ROI_HEADS.NUM_CLASSES')
    parser.add_argument('--config-file', default="maskrcnn_config.yaml", help='path to a detectron2 config file (or model_zoo key)')
    parser.add_argument('--weights', default=None, help='path or url to weights to set cfg.MODEL.WEIGHTS')
    parser.add_argument('-s', '--set', action='append', dest='set', help='extra cfg override in form KEY=VALUE (dotted path, can be repeated)')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='start training from scratch (do not resume from last checkpoint)')
    parser.add_argument('--epochs', type=int, default=None, help='number of epochs to train; if set, overrides --max-iter by computing iterations = epochs * ceil(num_images / IMS_PER_BATCH)')
    parser.add_argument('--fast-eval', action='store_true', help='speed up validation by increasing score threshold, capping detections, and focusing MaskDINO eval on boxes')
    parser.add_argument('--eval-bbox-only', action='store_true', help='run COCO bbox evaluation only (skip mask metrics) for much faster validation')
    parser.add_argument('--eval-score-thresh', type=float, default=None, help='override score threshold used during eval/inference (e.g., 0.5)')
    parser.add_argument('--eval-detections-per-image', type=int, default=None, help='override TEST.DETECTIONS_PER_IMAGE to cap per-image predictions during eval')
    parser.add_argument('--eval-focus-on-box', dest='eval_focus_on_box', action='store_true', help='for MaskDINO, focus eval outputs on boxes to reduce mask post-processing cost')
    parser.add_argument('--no-eval-focus-on-box', dest='eval_focus_on_box', action='store_false', help='disable MaskDINO TEST_FOUCUS_ON_BOX override')
    parser.add_argument('--gaussian-blur-prob', type=float, default=0.0, help='probability of applying random Gaussian blur augmentation during training (0 disables it)')
    parser.set_defaults(eval_focus_on_box=None)

    args = parser.parse_args()

    # initialize DDP if environment indicates a multi-process run
    is_torchrun = any(k in os.environ for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    if is_torchrun:
        setup_ddp_from_env()

    run_default_trainer(
        train_json_path=args.train_json,
        val_json_path=args.val_json,
        images_root=args.images_root,
        val_images_root=args.val_images_root,
        output_dir=args.output_dir,
        max_iter=args.max_iter,
        ims_per_batch=args.ims_per_batch,
        base_lr=args.base_lr,
        num_workers=args.num_workers,
        batch_size_per_image=args.batch_size_per_image,
        num_classes=args.num_classes,
        config_file=args.config_file,
        weights=args.weights,
        extra_cfg=args.set,
        resume=args.resume,
        epochs=args.epochs,
        eval_score_thresh=args.eval_score_thresh,
        eval_detections_per_image=args.eval_detections_per_image,
        eval_focus_on_box=args.eval_focus_on_box,
        fast_eval=args.fast_eval,
        eval_bbox_only=args.eval_bbox_only,
        gaussian_blur_prob=args.gaussian_blur_prob,
    )
