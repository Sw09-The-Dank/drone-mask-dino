"""
Simple Gradio UI for running inference with either a Mask R-CNN or MaskDINO model.

Usage:
  - pip install gradio
  - Ensure detectron2 and maskdino (if using MaskDINO) are installed and available
  - Run: python scripts/inference_ui.py

The UI accepts a config file, weights file, a score threshold and an input image.
It loads a Detectron2 `DefaultPredictor` using the provided config and weights,
runs inference, and returns a visualized image plus an optional JSON summary.
"""
import os
import json
import torch
from typing import Optional

try:
    import gradio as gr
except Exception as e:
    raise RuntimeError("Gradio must be installed: pip install gradio") from e

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
try:
    from detectron2 import model_zoo
except Exception:
    model_zoo = None

try:
    # MaskDINO helper to add config options when using MaskDINO
    from maskdino import add_maskdino_config
except Exception:
    add_maskdino_config = None

# Fallback class list from `train.py` to populate MetadataCatalog when absent
DEFAULT_CLASS_NAMES = [
    "rotor", "frame", "camera", "landinggear",
    "air2s", "neo", "mavic3m", "mini3pro",
]

try:
    # Some configs depend on DeepLab project additions; mirror train_m.py
    from detectron2.projects.deeplab import add_deeplab_config
except Exception:
    add_deeplab_config = None

# Module-level predictor cache: avoids reloading the model on every inference
# click and ensures the old model is properly freed from GPU when inputs change.
_predictor_cache = {
    "key": None,       # (model_type, config_file, weights) tuple
    "predictor": None,
    "cfg": None,
}


def _release_predictor_cache():
    """Delete cached predictor and free GPU memory."""
    global _predictor_cache
    old = _predictor_cache["predictor"]
    if old is not None:
        try:
            del old
        except Exception:
            pass
        _predictor_cache["predictor"] = None
        _predictor_cache["cfg"] = None
        _predictor_cache["key"] = None
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def build_predictor(model_type: str, config_file: Optional[str], weights: Optional[str], score_thresh: float = 0.5):
    cfg = get_cfg()

    def _checkpoint_looks_compatible(wpath: str, mtype: str):
        """Best-effort compatibility check between checkpoint keys and model type."""
        try:
            if not wpath or not os.path.isfile(wpath):
                return True, ""
            ckpt = torch.load(wpath, map_location="cpu")
            sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            if not isinstance(sd, dict):
                return True, ""
            keys = list(sd.keys())
            if not keys:
                return True, ""

            has_maskrcnn_head = any("roi_heads.box_predictor.cls_score" in k for k in keys)
            has_maskdino_head = any("sem_seg_head.predictor.class_embed" in k or "query_feat" in k for k in keys)

            if mtype == "maskrcnn" and has_maskdino_head and not has_maskrcnn_head:
                return False, "Selected weights look like MaskDINO but model_type is maskrcnn."
            if mtype == "maskdino" and has_maskrcnn_head and not has_maskdino_head:
                return False, "Selected weights look like Mask R-CNN but model_type is maskdino."

            # Additional class-count compatibility checks by head tensor shape.
            if mtype == "maskrcnn":
                try:
                    key = "roi_heads.box_predictor.cls_score.weight"
                    if key in sd and hasattr(sd[key], "shape"):
                        head_classes = int(sd[key].shape[0])
                        cfg_classes = int(getattr(cfg.MODEL.ROI_HEADS, "NUM_CLASSES", 0)) + 1
                        if cfg_classes > 1 and head_classes != cfg_classes:
                            return False, (
                                f"Mask R-CNN class mismatch: checkpoint cls_score has {head_classes} outputs, "
                                f"but config expects {cfg_classes} (NUM_CLASSES={cfg_classes - 1})."
                            )
                except Exception:
                    pass
            else:
                try:
                    key = "sem_seg_head.predictor.class_embed.weight"
                    if key in sd and hasattr(sd[key], "shape"):
                        head_classes = int(sd[key].shape[0])
                        # MaskDINO uses SEM_SEG_HEAD.NUM_CLASSES, not ROI_HEADS.NUM_CLASSES
                        if hasattr(cfg.MODEL, "SEM_SEG_HEAD") and hasattr(cfg.MODEL.SEM_SEG_HEAD, "NUM_CLASSES"):
                            cfg_classes = int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
                        else:
                            cfg_classes = int(getattr(cfg.MODEL.ROI_HEADS, "NUM_CLASSES", 0))
                        # Some checkpoints include/no-include background class; accept +/-1.
                        if cfg_classes > 0 and head_classes not in (cfg_classes, cfg_classes + 1, max(1, cfg_classes - 1)):
                            return False, (
                                f"MaskDINO class mismatch: checkpoint class_embed has {head_classes} outputs, "
                                f"while config suggests about {cfg_classes} (MODEL.SEM_SEG_HEAD.NUM_CLASSES)."
                            )
                except Exception:
                    pass

            return True, ""
        except Exception:
            # Do not hard-fail on inspection issues; loading path below will report real errors.
            return True, ""

    # Add MaskDINO (+deeplab) config if requested and available
    if model_type == "maskdino":
        if add_maskdino_config is None:
            raise RuntimeError("MaskDINO config helper not available. Install MaskDINO or run MaskDINO inference externally.")
        # Mirror train_m.py: add DeepLab project options first so YAML keys
        # like MODEL.RESNETS.RES4_DILATION are present when merging.
        if add_deeplab_config is not None:
            try:
                add_deeplab_config(cfg)
            except Exception:
                pass
        add_maskdino_config(cfg)

    if config_file and os.path.isfile(config_file):
        try:
            cfg.merge_from_file(config_file)
        except Exception as e:
            raise RuntimeError(f"Failed to merge config file {config_file}: {e}") from e

    # Ensure TEST keys exist
    try:
        if not hasattr(cfg, "TEST"):
            from detectron2.config import CfgNode as CN
            cfg.TEST = CN()
    except Exception:
        pass

    if not hasattr(cfg.TEST, "IMS_PER_BATCH"):
        try:
            from detectron2.config import CfgNode as CN
            cfg.TEST.IMS_PER_BATCH = 1
        except Exception:
            pass

    if weights:
        # Resolve MODEL_ZOO:placeholder if present
        if isinstance(weights, str) and weights.startswith("MODEL_ZOO:"):
            mz_path = weights.split(":", 1)[1]
            try:
                from detectron2 import model_zoo as _mz
                try:
                    weights = _mz.get_checkpoint_url(mz_path)
                except Exception:
                    print(f"[WARN] failed to resolve model_zoo path {mz_path}; leaving weights empty")
                    weights = ""
            except Exception:
                print("[WARN] detectron2.model_zoo not available to resolve MODEL_ZOO: weights placeholder")
                weights = ""

        # Guard against config/weight mismatches that produce unusable outputs.
        try:
            ok, reason = _checkpoint_looks_compatible(weights, model_type)
            if not ok:
                raise RuntimeError(reason)
        except RuntimeError:
            raise
        except Exception:
            pass

        cfg.MODEL.WEIGHTS = weights

    # Device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Generic threshold setting
    try:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    except Exception:
        pass
    # MaskDINO may use a different field for instance threshold
    try:
        # best-effort set of possible MaskDINO test thresholds
        if hasattr(cfg.MODEL, "MaskDINO"):
            if hasattr(cfg.MODEL.MaskDINO, "TEST") and hasattr(cfg.MODEL.MaskDINO.TEST, "INST_SCORE_THRESH"):
                cfg.MODEL.MaskDINO.TEST.INST_SCORE_THRESH = score_thresh
    except Exception:
        pass

    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def get_cached_predictor(model_type: str, config_file: Optional[str], weights: Optional[str], score_thresh: float = 0.5):
    """Return a cached predictor, rebuilding only when model/config/weights change."""
    global _predictor_cache
    cache_key = (model_type, config_file, weights)
    if _predictor_cache["key"] != cache_key or _predictor_cache["predictor"] is None:
        _release_predictor_cache()
        predictor, cfg = build_predictor(model_type, config_file, weights, score_thresh)
        _predictor_cache["key"] = cache_key
        _predictor_cache["predictor"] = predictor
        _predictor_cache["cfg"] = cfg
    return _predictor_cache["predictor"], _predictor_cache["cfg"]


def prepare_output(prediction, image, cfg):
    out_json = {
        "instances": [],
    }
    instances = prediction.get("instances", None)
    if instances is None:
        return image, json.dumps(out_json, indent=2)

    cpu_instances = instances.to("cpu")
    boxes = cpu_instances.pred_boxes.tensor.numpy() if cpu_instances.has("pred_boxes") else None
    scores = cpu_instances.scores.numpy() if cpu_instances.has("scores") else None
    classes = cpu_instances.pred_classes.numpy() if cpu_instances.has("pred_classes") else None

    for i in range(len(cpu_instances)):
        entry = {}
        if boxes is not None:
            entry["bbox"] = boxes[i].tolist()
        if scores is not None:
            entry["score"] = float(scores[i])
        if classes is not None:
            entry["class"] = int(classes[i])
        out_json["instances"].append(entry)

    # Visualization
    try:
        meta = None
        try:
            test_ds = cfg.DATASETS.TEST[0]
            meta = MetadataCatalog.get(test_ds)
        except Exception:
            try:
                meta = MetadataCatalog.get("default")
            except Exception:
                meta = None

        # Ensure metadata contains class names for visualization; if the
        # dataset registration didn't set `thing_classes`, populate from
        # a sensible project-default list so labels render correctly.
        try:
            if meta is not None and (not hasattr(meta, 'thing_classes') or not getattr(meta, 'thing_classes')):
                try:
                    meta.thing_classes = DEFAULT_CLASS_NAMES
                except Exception:
                    try:
                        MetadataCatalog.get(test_ds).set(thing_classes=DEFAULT_CLASS_NAMES)
                    except Exception:
                        pass
        except Exception:
            pass

        # Use a NonCollidingVisualizer (based on train.py) so labels avoid
        # being placed on top of boxes/masks and remain readable.
        class NonCollidingVisualizer(Visualizer):
            def __init__(self, *args, min_dist=20, y_offset=12, **kwargs):
                super().__init__(*args, **kwargs)
                self._used_positions = []
                self._occupied_rects = []
                self._min_dist = float(min_dist)
                self._y_offset = int(y_offset)

            def _distance(self, a, b):
                return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

            def _point_inside_rect(self, x, y, rect):
                x1, y1, x2, y2 = rect
                return x >= x1 and x <= x2 and y >= y1 and y <= y2

            def _point_occupied(self, x, y):
                return any(self._point_inside_rect(x, y, r) for r in self._occupied_rects)

            def draw_box(self, box, *args, **kwargs):
                try:
                    import numpy as _np
                    arr = _np.asarray(box)
                    if arr.size >= 4:
                        x1, y1, x2, y2 = float(arr.flat[0]), float(arr.flat[1]), float(arr.flat[2]), float(arr.flat[3])
                    else:
                        raise Exception()
                except Exception:
                    try:
                        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    except Exception:
                        x1 = y1 = x2 = y2 = 0.0
                lx, ty, rx, by = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
                self._occupied_rects.append((lx, ty, rx, by))
                return super().draw_box(box, *args, **kwargs)

            def draw_binary_mask(self, binary_mask, color, *, alpha=0.5, **kwargs):
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
                    lab_rect = (x, attempt_y - label_h, x + label_w, attempt_y)
                    conflict = False
                    for r in self._occupied_rects:
                        pr = (r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad)
                        if rects_intersect(lab_rect, pr):
                            conflict = True
                            break
                    if conflict:
                        attempt_y += self._y_offset
                        continue
                    too_close = any(self._distance((x, attempt_y), p) < self._min_dist for p in self._used_positions)
                    if too_close:
                        attempt_y += self._y_offset
                        continue
                    break

                self._used_positions.append((x, attempt_y))
                return super().draw_text(text, (x, attempt_y), **kwargs)

        # Visualizer expects RGB input; keep original colors for output.
        nv = NonCollidingVisualizer(image, metadata=meta, scale=1.0, instance_mode=ColorMode.IMAGE)
        if cpu_instances is not None:
            try:
                nv = nv.draw_instance_predictions(cpu_instances)
            except Exception:
                pass
        vis = nv.get_image()
    except Exception:
        vis = image

    return vis, json.dumps(out_json, indent=2)


_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def _expand_folder(path: str, recursive: bool = False):
    """Return a sorted list of image file paths found under *path*."""
    import glob as _glob
    results = []
    if recursive:
        for root, _dirs, files in os.walk(path):
            for f in files:
                if os.path.splitext(f)[1].lower() in _IMAGE_EXTENSIONS:
                    results.append(os.path.join(root, f))
    else:
        for f in os.listdir(path):
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTENSIONS:
                results.append(os.path.join(path, f))
    return sorted(results)


def run_inference(images, model_type, config_file, weights, score_thresh, return_json, recursive: bool = False):
    """
    Accept either a single image (numpy array), a file path, a folder path,
    or a list of any mix of the above.  Folder paths are expanded to all
    image files they contain (optionally recursively).
    Returns a single visualized image (for one input) or a list of visualized
    images when multiple inputs were provided. JSON output is always
    available if requested, but visualization (boxes/masks) is applied
    regardless of the `return_json` flag.
    """
    if images is None:
        return None, "{}"

    # Normalize inputs to a list
    is_batch = isinstance(images, (list, tuple))
    imgs = list(images) if is_batch else [images]

    # If Gradio provided a list of uploaded file dicts (including folder drag),
    # extract the underlying temp file paths so we process the actual files.
    try:
        if isinstance(imgs, list) and imgs and all(isinstance(x, dict) for x in imgs):
            paths = []
            for item in imgs:
                # common keys: 'name' (temp path), 'orig_name' (original filename)
                p = item.get('name') or item.get('tmp_path') or item.get('orig_name')
                if p:
                    paths.append(p)
            if paths:
                # sort to keep deterministic order (use the pathname as given)
                paths = sorted(paths)
                imgs = paths
                is_batch = True
    except Exception:
        pass

    # Expand any directory paths in the list to image file paths.
    expanded = []
    for item in imgs:
        if isinstance(item, str) and os.path.isdir(item):
            folder_files = _expand_folder(item, recursive=recursive)
            if not folder_files:
                print(f"[WARN] No images found in folder: {item}")
            expanded.extend(folder_files)
            is_batch = True
        else:
            expanded.append(item)
    imgs = expanded

    if not imgs:
        return [], '{"error": "No images found."}'

    # Convert uploaded files/paths to numpy arrays if necessary. Try many
    # heuristics so Gradio folder-drag payloads are handled across versions.
    proc_imgs = []
    import numpy as _np
    from io import BytesIO
    from PIL import Image as _PILImage

    def _load_pil_from_path(p):
        try:
            return _np.asarray(_PILImage.open(p).convert('RGB'))
        except Exception:
            return None

    for img in imgs:
        loaded = None
        # Already an array
        try:
            if isinstance(img, _np.ndarray):
                proc_imgs.append(img)
                continue
        except Exception:
            pass

        # String path
        try:
            if isinstance(img, str):
                if os.path.exists(img):
                    loaded = _load_pil_from_path(img)
                else:
                    # try relative to cwd
                    alt = os.path.join(os.getcwd(), img)
                    if os.path.exists(alt):
                        loaded = _load_pil_from_path(alt)
                if loaded is not None:
                    proc_imgs.append(loaded)
                    continue
        except Exception:
            pass

        # File-like objects / Gradio FileData: prefer stable temp path reads first
        # so repeated clicks do not depend on file pointer state.
        try:
            for _attr in ('path', 'name', 'tmp_path'):
                _p = getattr(img, _attr, None)
                if isinstance(_p, str) and os.path.exists(_p):
                    loaded = _load_pil_from_path(_p)
                    if loaded is not None:
                        proc_imgs.append(loaded)
                        break
            if loaded is not None:
                continue

            if hasattr(img, 'read'):
                try:
                    img.seek(0)
                except Exception:
                    pass
                try:
                    pil = _PILImage.open(img).convert('RGB')
                    proc_imgs.append(_np.asarray(pil))
                    continue
                except Exception:
                    # try reading bytes
                    try:
                        b = img.read()
                        pil = _PILImage.open(BytesIO(b)).convert('RGB')
                        proc_imgs.append(_np.asarray(pil))
                        continue
                    except Exception:
                        pass
            # If stream-based load succeeded, append once.
            if loaded is not None:
                proc_imgs.append(loaded)
                continue
        except Exception:
            pass

        # Dict payloads from Gradio: try common keys and byte buffers
        try:
            if isinstance(img, dict):
                # keys may include: name, tmp_path, tempfile, file, data, orig_name
                for key in ('name', 'tmp_path', 'tempfile', 'file', 'path', 'orig_name'):
                    v = img.get(key)
                    if isinstance(v, str) and os.path.exists(v):
                        loaded = _load_pil_from_path(v)
                        if loaded is not None:
                            break
                    if hasattr(v, 'read'):
                        try:
                            pil = _PILImage.open(v).convert('RGB')
                            loaded = _np.asarray(pil)
                            break
                        except Exception:
                            try:
                                b = v.read()
                                pil = _PILImage.open(BytesIO(b)).convert('RGB')
                                loaded = _np.asarray(pil)
                                break
                            except Exception:
                                pass
                # raw bytes
                if loaded is None:
                    b = img.get('data') or img.get('bytes')
                    if isinstance(b, (bytes, bytearray)):
                        try:
                            pil = _PILImage.open(BytesIO(b)).convert('RGB')
                            loaded = _np.asarray(pil)
                        except Exception:
                            loaded = None
                if loaded is not None:
                    proc_imgs.append(loaded)
                    continue
        except Exception:
            pass

        # Raw bytes
        try:
            if isinstance(img, (bytes, bytearray)):
                pil = _PILImage.open(BytesIO(img)).convert('RGB')
                proc_imgs.append(_np.asarray(pil))
                continue
        except Exception:
            pass

        # Could not decode this item as an image — skip it.
        print(f"[WARN] Skipping unrecognised input item: {type(img)}")
        continue

    if not proc_imgs:
        return [], '{"error": "No decodable images found in current selection. Re-select files/folder and try again."}'

    predictor, cfg = get_cached_predictor(model_type, config_file.strip() if config_file else None, weights.strip() if weights else None, float(score_thresh))
    try:
        input_format = str(getattr(cfg.INPUT, "FORMAT", "BGR")).upper()
    except Exception:
        input_format = "BGR"

    vis_list = []
    json_list = []
    for i, img in enumerate(proc_imgs):
        try:
            # Prepare predictor input based on config input format.
            # Gradio/PIL-loaded images here are RGB.
            pred_img = img
            try:
                import numpy as _np
                if isinstance(img, _np.ndarray) and img.ndim == 3 and img.shape[2] >= 3:
                    if input_format == "RGB":
                        pred_img = img
                    else:
                        pred_img = img[:, :, ::-1]
            except Exception:
                pred_img = img

            outputs = predictor(pred_img)
            # Move instances to CPU immediately to free GPU mask tensors,
            # which are the main VRAM cost when DETECTIONS_PER_IMAGE is high.
            try:
                if "instances" in outputs and outputs["instances"] is not None:
                    outputs["instances"] = outputs["instances"].to("cpu")
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            # MaskDINO instance inference returns top-k predictions but does not
            # apply SCORE_THRESH_TEST. Filter instances here so the UI slider
            # consistently controls what is shown.
            try:
                instances = outputs.get("instances", None)
                if instances is not None and instances.has("scores"):
                    before_n = len(instances)
                    keep = instances.scores >= float(score_thresh)
                    outputs["instances"] = instances[keep]
                    after_n = int(keep.sum().item()) if hasattr(keep, "sum") else len(outputs["instances"])
                    if before_n > 0 and after_n == 0:
                        print(f"[INFO] image {i}: threshold {float(score_thresh):.2f} filtered all {before_n} detections")

                    # Degenerate output detector: many full-confidence detections in one class.
                    try:
                        inst2 = outputs.get("instances", None)
                        if inst2 is not None and len(inst2) >= 8 and inst2.has("scores") and inst2.has("pred_classes"):
                            s = inst2.scores
                            c = inst2.pred_classes
                            if len(c.unique()) == 1:
                                s_min = float(s.min().item())
                                s_max = float(s.max().item())
                                if s_min >= 0.999 and s_max <= 1.00001:
                                    print(
                                        f"[WARN] image {i}: degenerate predictions detected "
                                        f"(all class={int(c[0].item())}, all scores~1.0). "
                                        "This usually indicates config/weights class mismatch."
                                    )
                    except Exception:
                        pass
            except Exception:
                pass
            vis, out_json = prepare_output(outputs, img, cfg)
            # prepare_output returns RGB-arrays; Gradio accepts numpy arrays
            try:
                vis_list.append(vis)
            except Exception:
                vis_list.append(vis)
            try:
                json_list.append(json.loads(out_json))
            except Exception:
                json_list.append(out_json)
        except Exception as e:
            print(f"[WARN] Inference failed for image {i}: {e}")
            json_list.append({"error": str(e)})

    # If only a single input was provided, return a single image instead of a list
    if not is_batch:
        vis = vis_list[0] if vis_list else None
        out_json = json.dumps(json_list[0], indent=2) if (return_json and json_list) else ""
        return vis, out_json

    # Batch case: return gallery (list of images) and JSON array if requested
    out_json = json.dumps(json_list, indent=2) if return_json else ""
    return vis_list, out_json


def _release_inference_gpu_memory():
    """Release GPU memory held by the cached predictor after inference completes."""
    _release_predictor_cache()


def launch_ui():
    def _model_defaults(mtype: str):
        # Return (config_default, weights_default) for the chosen model type
        if mtype == "maskrcnn":
            cfg_default = "maskrcnn_config.yaml"
            # Prefer weights placed in output_maskdino/trainer_output (or any trainer* subfolder)
            trainer_candidate = "output_maskdino/trainer_output/model_final.pth"
            if os.path.isfile(trainer_candidate):
                return cfg_default, trainer_candidate
            try:
                import glob
                pths = glob.glob("output_maskdino/trainer*/*.pth")
                if pths:
                    return cfg_default, pths[0]
            except Exception:
                pass
            # fallback to root output_maskdino model_final.pth
            root_candidate = "output_maskdino/model_final.pth"
            if os.path.isfile(root_candidate):
                return cfg_default, root_candidate
            try:
                import glob
                pths = glob.glob("output_maskdino/*.pth")
                if pths:
                    return cfg_default, pths[0]
            except Exception:
                pass
            # default suggestion (may not exist) — points to the trainer_output path
            return cfg_default, trainer_candidate
        # maskdino defaults (repo-local)
        return "maskdino_drone_config.yaml", "output/model_final.pth"

    with gr.Blocks() as demo:
        gr.Markdown("# Inference UI — MaskDINO (default)\nUpload an image, choose config/weights, and run inference.")
        with gr.Row():
            model_type = gr.Dropdown(choices=["maskrcnn", "maskdino"], value="maskdino", label="Model Type")
            config_file = gr.Textbox(label="Config file (path)", value="maskdino_drone_config.yaml")
            weights = gr.Textbox(label="Weights file (path)", value="output/model_final.pth")

        def on_model_change(mtype):
            cfg_def, w_def = _model_defaults(mtype)
            return gr.update(value=cfg_def), gr.update(value=w_def)

        # Update config and weights textboxes when model type changes
        model_type.change(on_model_change, inputs=[model_type], outputs=[config_file, weights])
        with gr.Row():
            score = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Score threshold")
            return_json = gr.Checkbox(label="Return JSON summary", value=False)
        # file_count="multiple" + no file_types restriction lets users drag-drop
        # individual files OR an entire folder.
        img_in = gr.Files(
            label="Input images — drag & drop files or a folder here",
            file_count="multiple",
            elem_id="img_in_files",
        )
        gr.HTML("""
        <div id="folder-picker-wrapper" style="margin-top:4px">
            <input type="file" id="folder-picker-input" webkitdirectory multiple style="display:none">
            <button id="folder-picker-btn" type="button"
                style="padding:5px 14px;cursor:pointer;background:#f0f0f0;
                             border:1px solid #bbb;border-radius:4px;font-size:13px">
                &#128193; Select folder
            </button>
            <span id="folder-picker-status" style="margin-left:8px;font-size:12px;color:#555"></span>
        </div>
        """)
        img_out = gr.Gallery(label="Visualized output")
        json_out = gr.Textbox(label="JSON output", interactive=False)

        def _run(image, mtype, cfgf, wts, sc, rjson):
            try:
                vis, j = run_inference(image, mtype, cfgf, wts, sc, rjson)
                return vis, j
            except Exception as e:
                return [], f"Error: {e}"

        run_btn = gr.Button("Run Inference")
        run_btn.click(
            _run,
            inputs=[img_in, model_type, config_file, weights, score, return_json],
            outputs=[img_out, json_out],
        ).then(fn=_release_inference_gpu_memory, inputs=None, outputs=None)

        _folder_js = """
        async () => {
            const btn = document.getElementById('folder-picker-btn');
            const inp = document.getElementById('folder-picker-input');
            const status = document.getElementById('folder-picker-status');
            if (!btn || !inp) return;
            btn.onclick = () => inp.click();
            inp.addEventListener('change', async () => {
                const files = Array.from(inp.files);
                if (!files.length) return;
                const imgExts = /\\.(jpe?g|png|bmp|tiff?|webp)$/i;
                const imgFiles = files.filter(f => imgExts.test(f.name));
                if (!imgFiles.length) { status.textContent = 'No images found in folder.'; return; }
                status.textContent = `Selected ${imgFiles.length} image(s).`;
                try {
                    const fileInput = document.querySelector('#img_in_files input[type=file]');
                    if (!fileInput) throw new Error('Gradio file input not found');
                    const dt = new DataTransfer();
                    imgFiles.forEach(file => dt.items.add(file));
                    Object.defineProperty(fileInput, 'files', {value: dt.files, configurable: true});
                    fileInput.dispatchEvent(new Event('input', {bubbles:true}));
                    fileInput.dispatchEvent(new Event('change', {bubbles:true}));
                    status.textContent = `${imgFiles.length} image(s) ready.`;
                } catch(e) {
                    status.textContent = 'Folder selection error: ' + e.message;
                }
                inp.value = '';
            });
        }
        """
        demo.load(fn=None, js=_folder_js)

    print("* Open in browser: http://127.0.0.1:7860/")
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    launch_ui()
