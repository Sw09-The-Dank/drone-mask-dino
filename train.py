import os
import random
import math
import json
import numpy as np
import torch
from detectron2.engine import HookBase

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
import torch.distributed as dist

dist.init_process_group(
    backend="nccl",
    timeout=datetime.timedelta(seconds=30)
)

# -----------------------------
# SANITY CHECK: CUDA
# -----------------------------

print("PyTorch version:", getattr(torch, '__version__', 'n/a'))
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", getattr(torch.version, 'cuda', 'n/a'))


def setup_ddp_from_env():
    """Initialize torch.distributed and detectron2 local PG from environment.

    Safe to call multiple times. Reads `WORLD_SIZE`, `RANK`, `LOCAL_RANK`,
    and `LOCAL_WORLD_SIZE` / `LOCAL_SIZE` to determine local process counts.
    """
    try:
        local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('LOCAL_RANK', '0')))
    except Exception:
        local_rank = 0
    try:
        world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('WORLD_SIZE', '1')))
    except Exception:
        world_size = 1

    # set CUDA device for this process
    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(local_rank)
            except Exception:
                pass
    except Exception:
        pass

    # init torch.distributed if needed
    try:
        if torch.distributed.is_available() and not torch.distributed.is_initialized() and world_size > 1:
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            try:
                torch.distributed.init_process_group(backend=backend, init_method='env://')
                rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
                print(f"DDP INIT: backend={backend} RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world_size}")
            except Exception as e:
                print(f"[WARN] torch.distributed.init_process_group failed: {e}")
    except Exception as e:
        print(f"[WARN] DDP setup problem: {e}")

    # Ensure detectron2 local process group exists
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                from detectron2.utils import comm as d2comm
                try:
                    n_local = int(os.environ.get('LOCAL_WORLD_SIZE', os.environ.get('LOCAL_SIZE', os.environ.get('NPROC_PER_NODE', '1'))))
                except Exception:
                    n_local = 1
                if n_local < 1:
                    n_local = 1
                try:
                    d2comm.create_local_process_group(n_local)
                    print(f"Created detectron2 local process group with {n_local} local workers")
                except Exception as e:
                    print(f"[WARN] Could not create detectron2 local process group: {e}")
            except Exception:
                pass
    except Exception:
        pass

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

    # init process group if needed
    try:
        if torch.distributed.is_available() and not torch.distributed.is_initialized() and world_size > 1:
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            try:
                torch.distributed.init_process_group(backend=backend, init_method='env://')
                rank = os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))
                print(f"DDP INIT: backend={backend} RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world_size}")
                # create detectron2 local process group so utilities like get_local_rank() work
                try:
                    try:
                        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', os.environ.get('LOCAL_SIZE', '1')))
                    except Exception:
                        local_world_size = 1
                    if local_world_size < 1:
                        local_world_size = 1
                    # import lazily to avoid hard dependency if detectron2 isn't used
                    try:
                        from detectron2.utils import comm as d2comm
                        d2comm.create_local_process_group(local_world_size)
                        print(f"Created detectron2 local process group with {local_world_size} local workers")
                    except Exception as e:
                        print(f"[WARN] Could not create detectron2 local process group: {e}")
                except Exception:
                    pass
            except Exception as e:
                print(f"[WARN] torch.distributed.init_process_group failed: {e}")
    except Exception as e:
        print(f"[WARN] DDP setup problem: {e}")
    # If the process group was initialized elsewhere (earlier), still ensure detectron2 local PG exists
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                from detectron2.utils import comm as d2comm
                # determine local workers per machine; prefer env vars, fallback to 1
                try:
                    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', os.environ.get('LOCAL_SIZE', '1')))
                except Exception:
                    local_world_size = 1
                if local_world_size < 1:
                    local_world_size = 1
                try:
                    d2comm.create_local_process_group(local_world_size)
                except Exception:
                    # may have been created already; ignore
                    pass
            except Exception:
                pass
    except Exception:
        pass



# (DDP initialization is handled by `setup_ddp_from_env()` where needed)


# -----------------------------
# DATASET REGISTRATION (register only the split needed at each phase)
# -----------------------------
CLASS_NAMES = ["rotor", "frame", "camera", "landinggear", "air2s", "neo", "mavic3m", "mini3pro"]


class CheckpointCleanupHook(HookBase):
    def __init__(self, output_dir, keep=4):
        self.output_dir = output_dir
        self.keep = keep
    def after_step(self):
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
                        output_dir=None,
                        max_iter=None,
                        ims_per_batch=None,
                        base_lr=None,
                        num_workers=None,
                        batch_size_per_image=None,
                        num_classes=None,
                        config_file=None,
                        weights=None,
                        extra_cfg=None,
                        resume=True,
                        epochs=None):
    try:
        from detectron2.data.datasets import register_coco_instances
        from detectron2.engine import DefaultTrainer
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
    except Exception as e:
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
    val_img_root = images_root
    train_candidate = os.path.join(images_root, "train")
    val_candidate = os.path.join(images_root, "val")
    if os.path.isdir(train_candidate):
        train_img_root = train_candidate
    if os.path.isdir(val_candidate):
        val_img_root = val_candidate

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
    val_img_root = _choose_existing_root(val_img_root, [os.path.join('images','val'), 'images', os.path.join('dataset','images','val')])

    # Attempt to locate train/val JSONs from several common locations and
    # register the first found path. This is more robust for different CWDs
    # (e.g. when running inside containers or orchestrators).
    def _find_json(candidates):
        # Also try paths relative to the script location (repo root when mounted)
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        for p in candidates:
            if not p:
                continue
            # Try several likely interpretations of the path
            try_paths = [p,
                         os.path.join(os.getcwd(), p),
                         os.path.join(script_dir, p)]
            # If p is just a basename, also try common annotation dirs under repo
            base = os.path.basename(p)
            if base and base != p:
                try_paths.extend([
                    os.path.join(script_dir, 'output_annotations', base),
                    os.path.join(os.getcwd(), 'output_annotations', base),
                ])
            for tp in try_paths:
                try:
                    if os.path.isfile(tp):
                        return os.path.abspath(tp)
                except Exception:
                    continue
        return None

    train_candidates = [train_json_path,
                        os.path.join('output_annotations', os.path.basename(train_json_path)),
                        os.path.join('dataset', 'annotations', os.path.basename(train_json_path)),
                        os.path.join('annotations', os.path.basename(train_json_path)),
                        os.path.join('dataset', 'annotations', 'train.json'),
                        os.path.join('output_annotations', 'train.json'),
                        os.path.join('annotations', 'train.json')]
    val_candidates = [val_json_path,
                      os.path.join('output_annotations', os.path.basename(val_json_path)),
                      os.path.join('dataset', 'annotations', os.path.basename(val_json_path)),
                      os.path.join('annotations', os.path.basename(val_json_path)),
                      os.path.join('dataset', 'annotations', 'val.json'),
                      os.path.join('output_annotations', 'val.json'),
                      os.path.join('annotations', 'val.json')]

    found_train = _find_json(train_candidates)
    if found_train:
        try:
            register_coco_instances(train_name, {}, found_train, train_img_root)
            print(f"Registered {train_name} -> {found_train} (images root: {train_img_root})")
            train_json_path = found_train
        except Exception as e:
            print(f"[WARN] Failed to register train JSON {found_train}: {e}")
    else:
        print(f"[WARN] Train JSON not found; searched: {train_candidates}")

    found_val = _find_json(val_candidates)
    if found_val:
        try:
            register_coco_instances(val_name, {}, found_val, val_img_root)
            print(f"Registered {val_name} -> {found_val} (images root: {val_img_root})")
            val_json_path = found_val
        except Exception as e:
            print(f"[WARN] Failed to register val JSON {found_val}: {e}")
    else:
        print(f"[WARN] Val JSON not found; searched: {val_candidates}")

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
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    except Exception:
        pass

   

    # Use the sanitized (clean) dataset names for training/testing if available
    cfg.DATASETS.TRAIN = (train_dataset_name,) if isinstance(train_dataset_name, str) else (train_name,)
    cfg.DATASETS.TEST = (val_dataset_name,) if isinstance(val_dataset_name, str) else (val_name,)
    cfg.DATALOADER.NUM_WORKERS = 12
    cfg.SOLVER.IMS_PER_BATCH = 12
    cfg.SOLVER.BASE_LR = 0.00005
    # cfg.SOLVER.STEPS = (3000,4000)
    cfg.SOLVER.MAX_ITER = 3000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256

     # Apply optional config/weight overrides provided by caller (CLI or function args)
    try:
        if config_file:
            try:
                cfg.merge_from_file(config_file)
                print(f"[INFO] Merged config file: {config_file}")
            except Exception:
                try:
                    # maybe a model_zoo short path
                    cfg.merge_from_file(model_zoo.get_config_file(config_file))
                    print(f"[INFO] Merged model_zoo config: {config_file}")
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
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(num_classes)
            print(f"[INFO] Set MODEL.ROI_HEADS.NUM_CLASSES = {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")

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
        
        
    # Infer number of classes from train JSON categories
    try:
        with open(train_json_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        cats = j.get("categories", [])
        if cats:
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cats)
            print(f"Set NUM_CLASSES = {len(cats)} from train JSON categories")
        else:
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            print("No categories found in train JSON; defaulting NUM_CLASSES=1")
    except Exception:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
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
    cfg.OUTPUT_DIR = "output_maskdino/trainer_output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.DATALOADER.NUM_WORKERS = 8

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

    trainer = DefaultTrainer(cfg)
    # resume=True will continue from last checkpoint if present
    trainer.resume_or_load(resume=bool(resume))
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
        sampler = getattr(tmp_loader, 'sampler', None)
        print(f"[DEBUG] Train loader sampler: {type(sampler).__name__ if sampler is not None else 'None'}")
        try:
            from torch.utils.data.distributed import DistributedSampler as _DS
            is_dist_sampler = isinstance(sampler, _DS)
            print(f"[DEBUG] Train loader uses DistributedSampler: {is_dist_sampler}")
        except Exception:
            print("[DEBUG] Could not determine if sampler is DistributedSampler")
    except Exception as e:
        print(f"[WARN] Could not build/inspect train loader: {e}")

    # synchronize all processes before starting training (if DDP active)
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception:
        pass

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
    # Run evaluation on the validation set using COCOEvaluator if available (only on main)
    if RUN_EVALUATION and is_main:
        try:
            from detectron2.evaluation import COCOEvaluator, inference_on_dataset
            from detectron2.data import build_detection_test_loader
            evaluator = COCOEvaluator(val_dataset_name, cfg, distributed=False, output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))
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
    parser.add_argument('--output-dir', default=None, help='output directory to write trainer outputs')
    parser.add_argument('--max-iter', type=int, default=None, help='override SOLVER.MAX_ITER')
    parser.add_argument('--ims-per-batch', type=int, default=None, help='override SOLVER.IMS_PER_BATCH')
    parser.add_argument('--base-lr', type=float, default=None, help='override SOLVER.BASE_LR')
    parser.add_argument('--num-workers', type=int, default=None, help='override DATALOADER.NUM_WORKERS')
    parser.add_argument('--batch-size-per-image', type=int, default=None, help='override MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE')
    parser.add_argument('--num-classes', type=int, default=None, help='override MODEL.ROI_HEADS.NUM_CLASSES')
    parser.add_argument('--config-file', default=None, help='path to a detectron2 config file (or model_zoo key)')
    parser.add_argument('--weights', default=None, help='path or url to weights to set cfg.MODEL.WEIGHTS')
    parser.add_argument('-s', '--set', action='append', dest='set', help='extra cfg override in form KEY=VALUE (dotted path, can be repeated)')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='start training from scratch (do not resume from last checkpoint)')
    parser.add_argument('--epochs', type=int, default=None, help='number of epochs to train; if set, overrides --max-iter by computing iterations = epochs * ceil(num_images / IMS_PER_BATCH)')

    args = parser.parse_args()

    # initialize DDP if environment indicates a multi-process run
    try:
        setup_ddp_from_env()
    except Exception:
        pass

    run_default_trainer(
        train_json_path=args.train_json,
        val_json_path=args.val_json,
        images_root=args.images_root,
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
    )
