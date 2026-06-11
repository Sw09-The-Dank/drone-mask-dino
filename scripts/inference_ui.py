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
import re
import tempfile
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
    "horstab", "verstab", "boom", "wing",
    "fuselage", "rotor-stump", "drone-body", "arm",
    "landing gear", "guard", "DNDN-concept", "Fixed-wing-concept",
    "Shahed", "DJI-Matrice-600-Pro", "DJI-S900", "DJI-Spark",
    "U842-Sport-Racing",
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

# Keep JSON payloads bounded so Gradio textbox rendering does not stall UI.
MAX_JSON_INSTANCES_PER_IMAGE = 100
MAX_TEXTBOX_JSON_CHARS = 250000
DEFAULT_CATEGORIES_JSON = """1 : horstab,
2 : verstab,
3 : boom,
4 : wing,
5 : fuselage,
6 : rotor-stump,
7 : drone-body,
8 : arm,
9 : landing gear,
10 : guard,
11 : DNDN-concept,
12 : Fixed-wing-concept,
13 : Shahed,
14 : DJI-Matrice-600-Pro,
15 : DJI-S900,
16 : DJI-Spark,
17 : U842-Sport-Racing,
"""


def _load_checkpoint_state_dict(weights_path: Optional[str]):
    """Return the model state_dict stored in a Detectron2-style checkpoint."""
    if not weights_path or not os.path.isfile(weights_path):
        return None

    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    return state_dict if isinstance(state_dict, dict) else None


def _get_cfg_num_classes(cfg, model_type: str) -> int:
    """Read the active class count from cfg for the requested model type."""
    try:
        if model_type == "maskdino" and hasattr(cfg.MODEL, "SEM_SEG_HEAD") and hasattr(cfg.MODEL.SEM_SEG_HEAD, "NUM_CLASSES"):
            return int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
    except Exception:
        pass

    try:
        return int(getattr(cfg.MODEL.ROI_HEADS, "NUM_CLASSES", 0))
    except Exception:
        return 0


def _set_cfg_num_classes(cfg, num_classes: int):
    """Force cfg NUM_CLASSES fields so model construction matches the checkpoint."""
    if num_classes is None or int(num_classes) <= 0:
        return False

    num_classes = int(num_classes)
    try:
        from detectron2.config import CfgNode as CN
    except Exception:
        CN = None

    try:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    except Exception:
        if CN is not None and not hasattr(cfg.MODEL, "ROI_HEADS"):
            cfg.MODEL.ROI_HEADS = CN()
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    try:
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    except Exception:
        if CN is not None and not hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
            cfg.MODEL.SEM_SEG_HEAD = CN()
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes

    return True


def _looks_generic_class_names(names) -> bool:
    if not isinstance(names, (list, tuple)) or not names:
        return True
    try:
        return all(str(name).startswith("class_") for name in names)
    except Exception:
        return False


def _parse_class_names_json(categories_json_text: Optional[str]):
    if not categories_json_text or not str(categories_json_text).strip():
        return None

    try:
        parsed_pairs = []
        text = str(categories_json_text).strip()
        simple_match_count = 0
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            match = re.match(r"^(\d+)\s*:\s*(.+?)\s*,?$", line)
            if not match:
                parsed_pairs = []
                break
            class_id = int(match.group(1))
            class_name = match.group(2).strip()
            if not class_name:
                raise ValueError(f"Invalid categories format on line {line_number}: missing class name")
            parsed_pairs.append((class_id, class_name))
            simple_match_count += 1

        if simple_match_count > 0:
            parsed_pairs.sort(key=lambda item: item[0])
            return [name for _class_id, name in parsed_pairs]

        payload = json.loads(text)
        categories = payload.get("categories", payload if isinstance(payload, list) else [])
        if not isinstance(categories, list) or not categories:
            return None

        if all(isinstance(category, dict) and isinstance(category.get("id"), int) for category in categories if isinstance(category, dict)):
            categories = sorted(categories, key=lambda category: int(category["id"]))

        names = [str(category.get("name", "")).strip() for category in categories if str(category.get("name", "")).strip()]
        return names or None
    except Exception as exc:
        raise ValueError(f"Invalid categories input: {exc}") from exc


def _resolve_class_names(cfg, categories_json_text: Optional[str] = None):
    explicit_names = _parse_class_names_json(categories_json_text)
    if explicit_names:
        return explicit_names

    expected_num_classes = _get_cfg_num_classes(cfg, "maskdino")
    dataset_names = []

    try:
        dataset_names.extend(list(getattr(cfg.DATASETS, "TEST", [])))
    except Exception:
        pass
    try:
        dataset_names.extend(list(getattr(cfg.DATASETS, "TRAIN", [])))
    except Exception:
        pass

    dataset_names = [name for name in dataset_names if name]
    fallback_names = None

    for dataset_name in dataset_names:
        try:
            meta = MetadataCatalog.get(dataset_name)
        except Exception:
            meta = None

        if meta is not None:
            existing_names = getattr(meta, "thing_classes", None)
            if isinstance(existing_names, (list, tuple)) and existing_names and not _looks_generic_class_names(existing_names):
                if expected_num_classes <= 0 or len(existing_names) == expected_num_classes:
                    return list(existing_names)
                if fallback_names is None:
                    fallback_names = list(existing_names)

    if expected_num_classes > 0 and len(DEFAULT_CLASS_NAMES) == expected_num_classes:
        return list(DEFAULT_CLASS_NAMES)
    if fallback_names:
        return fallback_names
    return list(DEFAULT_CLASS_NAMES) if DEFAULT_CLASS_NAMES else None


def _inspect_checkpoint(weights_path: Optional[str]):
    """Infer model family and class-count hints from checkpoint tensors."""
    info = {
        "state_dict": None,
        "detected_model_type": None,
        "num_classes": None,
        "num_classes_source": None,
        "maskdino_class_embed_outputs": None,
    }

    try:
        state_dict = _load_checkpoint_state_dict(weights_path)
        if not state_dict:
            return info

        info["state_dict"] = state_dict
        keys = list(state_dict.keys())
        has_maskrcnn_head = any("roi_heads.box_predictor.cls_score" in key for key in keys)
        has_maskdino_head = any("sem_seg_head.predictor.class_embed" in key or "query_feat" in key for key in keys)

        if has_maskrcnn_head and not has_maskdino_head:
            info["detected_model_type"] = "maskrcnn"
        elif has_maskdino_head and not has_maskrcnn_head:
            info["detected_model_type"] = "maskdino"

        maskrcnn_key = "roi_heads.box_predictor.cls_score.weight"
        if maskrcnn_key in state_dict and hasattr(state_dict[maskrcnn_key], "shape"):
            logits = int(state_dict[maskrcnn_key].shape[0])
            if logits > 1:
                info["num_classes"] = logits - 1
                info["num_classes_source"] = f"{maskrcnn_key} ({logits} logits including background)"

        label_key = "sem_seg_head.predictor.label_enc.weight"
        if label_key in state_dict and hasattr(state_dict[label_key], "shape"):
            label_classes = int(state_dict[label_key].shape[0])
            if label_classes > 0:
                info["num_classes"] = label_classes
                info["num_classes_source"] = label_key

        class_embed_key = "sem_seg_head.predictor.class_embed.weight"
        if class_embed_key in state_dict and hasattr(state_dict[class_embed_key], "shape"):
            info["maskdino_class_embed_outputs"] = int(state_dict[class_embed_key].shape[0])
            if info["num_classes"] is None and info["maskdino_class_embed_outputs"] > 0:
                info["num_classes_source"] = class_embed_key
    except Exception:
        return info

    return info


def _infer_checkpoint_num_classes(info, cfg, model_type: str):
    """Resolve an exact or best-effort class count from checkpoint metadata."""
    num_classes = info.get("num_classes")
    if num_classes is not None and int(num_classes) > 0:
        return int(num_classes), info.get("num_classes_source") or "checkpoint"

    if model_type != "maskdino":
        return None, None

    class_embed_outputs = info.get("maskdino_class_embed_outputs")
    if not class_embed_outputs or int(class_embed_outputs) <= 0:
        return None, None

    class_embed_outputs = int(class_embed_outputs)
    cfg_classes = _get_cfg_num_classes(cfg, model_type)
    candidates = [class_embed_outputs]
    if class_embed_outputs > 1:
        candidates.append(class_embed_outputs - 1)
    candidates = [candidate for candidate in dict.fromkeys(candidates) if candidate > 0]

    if cfg_classes in candidates:
        return cfg_classes, f"sem_seg_head.predictor.class_embed.weight matched config-compatible candidate from {class_embed_outputs} outputs"

    if len(candidates) == 1:
        return candidates[0], f"sem_seg_head.predictor.class_embed.weight ({class_embed_outputs} outputs)"

    return None, None


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

        checkpoint_info = _inspect_checkpoint(weights)
        detected_model_type = checkpoint_info.get("detected_model_type")
        if detected_model_type and detected_model_type != model_type:
            raise RuntimeError(
                f"Selected weights look like {detected_model_type} but model_type is {model_type}."
            )

        inferred_num_classes, infer_source = _infer_checkpoint_num_classes(checkpoint_info, cfg, model_type)
        if inferred_num_classes is not None:
            current_num_classes = _get_cfg_num_classes(cfg, model_type)
            if inferred_num_classes != current_num_classes:
                _set_cfg_num_classes(cfg, inferred_num_classes)
                print(
                    f"[INFO] Overrode cfg NUM_CLASSES from checkpoint: "
                    f"{current_num_classes} -> {inferred_num_classes} ({infer_source})"
                )
        elif checkpoint_info.get("maskdino_class_embed_outputs"):
            print(
                "[WARN] Could not infer an exact class count from checkpoint; "
                "keeping NUM_CLASSES from config."
            )

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


def _ensure_uint8_image(arr):
    """Convert arbitrary image array to uint8 HxWxC (0..255) for predictor input."""
    import numpy as _np

    a = _np.asarray(arr)
    if a.dtype == _np.uint8:
        return a

    # Common Gradio/PIL paths can produce float arrays in 0..1 or 0..255.
    if _np.issubdtype(a.dtype, _np.floating):
        maxv = float(_np.nanmax(a)) if a.size else 0.0
        if maxv <= 1.0:
            a = a * 255.0
    a = _np.nan_to_num(a, nan=0.0, posinf=255.0, neginf=0.0)
    a = _np.clip(a, 0, 255).astype(_np.uint8, copy=False)
    return a


def _rescale_normalized_instances_inplace(instances, image_shape):
    """If model returned 0..1-ish boxes, convert them to pixel coordinates in-place."""
    if instances is None or len(instances) == 0 or (not instances.has("pred_boxes")):
        return False

    try:
        import torch
        h, w = int(image_shape[0]), int(image_shape[1])
        boxes = instances.pred_boxes.tensor
        if boxes.numel() == 0 or h <= 0 or w <= 0:
            return False

        max_abs = float(boxes.abs().max().item())
        min_val = float(boxes.min().item())

        # Heuristic: normalized boxes stay close to 0..1 (occasionally slight negatives).
        if max_abs <= 2.5 and min_val >= -0.5:
            scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
            boxes.mul_(scale)
            boxes[:, 0::2].clamp_(0, max(0, w - 1))
            boxes[:, 1::2].clamp_(0, max(0, h - 1))
            return True
    except Exception:
        return False

    return False


def prepare_output(prediction, image, cfg, include_json: bool = True, categories_json_text: Optional[str] = None):
    out_json = {
        "instances": [],
    }
    instances = prediction.get("instances", None)
    if instances is None:
        return image, out_json if include_json else None

    cpu_instances = instances.to("cpu")
    boxes = cpu_instances.pred_boxes.tensor.numpy() if cpu_instances.has("pred_boxes") else None
    scores = cpu_instances.scores.numpy() if cpu_instances.has("scores") else None
    classes = cpu_instances.pred_classes.numpy() if cpu_instances.has("pred_classes") else None

    if include_json:
        total_instances = len(cpu_instances)
        limit = min(total_instances, MAX_JSON_INSTANCES_PER_IMAGE)
        for i in range(limit):
            entry = {}
            if boxes is not None:
                entry["bbox"] = boxes[i].tolist()
            if scores is not None:
                entry["score"] = float(scores[i])
            if classes is not None:
                entry["class"] = int(classes[i])
            out_json["instances"].append(entry)
        if total_instances > limit:
            out_json["truncated"] = True
            out_json["instances_returned"] = int(limit)
            out_json["instances_total"] = int(total_instances)

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
            current_names = getattr(meta, 'thing_classes', None) if meta is not None else None
            cfg_num_classes = _get_cfg_num_classes(cfg, "maskdino")
            names_missing = not current_names
            names_generic = _looks_generic_class_names(current_names)
            names_mismatch = bool(current_names) and cfg_num_classes > 0 and len(current_names) != cfg_num_classes

            if meta is not None and (names_missing or names_generic or names_mismatch):
                class_names = _resolve_class_names(cfg, categories_json_text)
                if cfg_num_classes > 0 and class_names and len(class_names) != cfg_num_classes:
                    print(
                        f"[WARN] Class-name count mismatch: resolved {len(class_names)} names "
                        f"for cfg NUM_CLASSES={cfg_num_classes}; labels may be incomplete."
                    )
                try:
                    meta.thing_classes = class_names
                except Exception:
                    try:
                        MetadataCatalog.get(test_ds).set(thing_classes=class_names)
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
                max_attempts = 200
                attempts = 0
                while attempts < max_attempts:
                    lab_rect = (x, attempt_y - label_h, x + label_w, attempt_y)
                    conflict = False
                    for r in self._occupied_rects:
                        pr = (r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad)
                        if rects_intersect(lab_rect, pr):
                            conflict = True
                            break
                    if conflict:
                        attempt_y += self._y_offset
                        attempts += 1
                        continue
                    too_close = any(self._distance((x, attempt_y), p) < self._min_dist for p in self._used_positions)
                    if too_close:
                        attempt_y += self._y_offset
                        attempts += 1
                        continue
                    break

                if attempts >= max_attempts:
                    # Fall back to the original location if no collision-free slot
                    # is found quickly to avoid UI hangs on dense predictions.
                    attempt_y = y

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

    return vis, out_json if include_json else None


def _build_detection_summary(instances, cfg, categories_json_text: Optional[str] = None):
    """Summarize detections by class name, count, and confidence statistics."""
    if instances is None:
        return "No detections."

    try:
        cpu_instances = instances.to("cpu")
    except Exception:
        cpu_instances = instances

    if len(cpu_instances) == 0:
        return "No detections."

    class_names = _resolve_class_names(cfg, categories_json_text) or []

    try:
        scores = cpu_instances.scores.tolist() if cpu_instances.has("scores") else []
    except Exception:
        scores = []
    try:
        class_ids = cpu_instances.pred_classes.tolist() if cpu_instances.has("pred_classes") else []
    except Exception:
        class_ids = []

    if not class_ids:
        total_count = len(cpu_instances)
        if scores:
            avg_conf = sum(float(score) for score in scores) / max(1, len(scores))
            max_conf = max(float(score) for score in scores)
            return f"Detections: {total_count}\nunknown: count={total_count}, avg_conf={avg_conf:.3f}, max_conf={max_conf:.3f}"
        return f"Detections: {total_count}"

    summary_by_class = {}
    for index, class_id in enumerate(class_ids):
        class_id = int(class_id)
        if 0 <= class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"class_{class_id}"

        item = summary_by_class.setdefault(class_name, {"count": 0, "scores": []})
        item["count"] += 1
        if index < len(scores):
            item["scores"].append(float(scores[index]))

    lines = [f"Detections: {len(cpu_instances)}"]
    for class_name in sorted(summary_by_class.keys()):
        item = summary_by_class[class_name]
        if item["scores"]:
            avg_conf = sum(item["scores"]) / len(item["scores"])
            max_conf = max(item["scores"])
            lines.append(
                f"{class_name}: count={item['count']}, avg_conf={avg_conf:.3f}, max_conf={max_conf:.3f}"
            )
        else:
            lines.append(f"{class_name}: count={item['count']}")

    return "\n".join(lines)


def _collect_detection_stats(instances, cfg, categories_json_text: Optional[str] = None):
    """Collect per-class count/confidence stats from instances."""
    if instances is None:
        return 0, {}

    try:
        cpu_instances = instances.to("cpu")
    except Exception:
        cpu_instances = instances

    if len(cpu_instances) == 0:
        return 0, {}

    class_names = _resolve_class_names(cfg, categories_json_text) or []

    try:
        scores = cpu_instances.scores.tolist() if cpu_instances.has("scores") else []
    except Exception:
        scores = []
    try:
        class_ids = cpu_instances.pred_classes.tolist() if cpu_instances.has("pred_classes") else []
    except Exception:
        class_ids = []

    class_stats = {}
    if not class_ids:
        class_stats["unknown"] = {
            "count": len(cpu_instances),
            "scores": [float(score) for score in scores],
        }
        return len(cpu_instances), class_stats

    for index, class_id in enumerate(class_ids):
        class_id = int(class_id)
        if 0 <= class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"class_{class_id}"

        item = class_stats.setdefault(class_name, {"count": 0, "scores": []})
        item["count"] += 1
        if index < len(scores):
            item["scores"].append(float(scores[index]))

    return len(cpu_instances), class_stats


def _format_detection_stats(total_count: int, class_stats: dict, title: Optional[str] = None):
    lines = []
    if title:
        lines.append(title)
    lines.append(f"Detections: {int(total_count)}")

    for class_name in sorted(class_stats.keys()):
        item = class_stats[class_name]
        count = int(item.get("count", 0))
        scores = item.get("scores", [])
        if scores:
            avg_conf = sum(float(score) for score in scores) / max(1, len(scores))
            max_conf = max(float(score) for score in scores)
            lines.append(f"{class_name}: count={count}, avg_conf={avg_conf:.3f}, max_conf={max_conf:.3f}")
        else:
            lines.append(f"{class_name}: count={count}")
    return "\n".join(lines)


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


def run_inference(images, model_type, config_file, weights, score_thresh, return_json, categories_json_text=None, recursive: bool = False):
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
        return None, "{}", "No detections."

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
        return [], '{"error": "No images found."}', "No detections."

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
        return [], '{"error": "No decodable images found in current selection. Re-select files/folder and try again."}', "No detections."

    try:
        parsed_class_names = _parse_class_names_json(categories_json_text)
    except ValueError as exc:
        return [], json.dumps({"error": str(exc)}, indent=2), str(exc)

    if parsed_class_names:
        print(f"[INFO] Loaded {len(parsed_class_names)} class names from UI categories JSON")

    predictor, cfg = get_cached_predictor(model_type, config_file.strip() if config_file else None, weights.strip() if weights else None, float(score_thresh))
    try:
        input_format = str(getattr(cfg.INPUT, "FORMAT", "BGR")).upper()
    except Exception:
        input_format = "BGR"

    # Diagnostic: Log what config/weights were actually loaded
    actual_cfg = config_file.strip() if config_file else "(none)"
    actual_wts = weights.strip() if weights else "(none)"
    print(f"[DIAG] Loaded config={actual_cfg}, weights={actual_wts}, input_format={input_format}")

    vis_list = []
    json_list = []
    summary_list = []
    batch_total_detections = 0
    batch_class_stats = {}
    
    for i, img in enumerate(proc_imgs):
        try:
            # Prepare predictor input based on config input format.
            # Gradio/PIL-loaded images here are RGB.
            pred_img = _ensure_uint8_image(img)
            try:
                import numpy as _np
                if isinstance(img, _np.ndarray) and img.ndim == 3 and img.shape[2] >= 3:
                    if input_format == "RGB":
                        pred_img = _ensure_uint8_image(img)
                    else:
                        pred_img = _ensure_uint8_image(img)[:, :, ::-1]
            except Exception:
                pred_img = img

            # Diagnostic: log input image shape
            if i == 0:
                try:
                    import numpy as _np
                    if isinstance(pred_img, _np.ndarray):
                        print(f"[DIAG] Input image shape: {pred_img.shape} (H x W x C or similar)")
                        pass
                except Exception:
                    pass

            outputs = predictor(pred_img)
            # Move instances to CPU immediately to free GPU mask tensors,
            # which are the main VRAM cost when DETECTIONS_PER_IMAGE is high.
            try:
                if "instances" in outputs and outputs["instances"] is not None:
                    outputs["instances"] = outputs["instances"].to("cpu")
                    if _rescale_normalized_instances_inplace(outputs["instances"], pred_img.shape):
                        print(f"[WARN] image {i}: detected normalized boxes; rescaled to pixel coordinates")
            except Exception:
                pass
            
            # Diagnostic: log raw model output before filtering
            try:
                raw_inst = outputs.get("instances", None)
                if raw_inst is not None and len(raw_inst) > 0 and i == 0:
                    print(f"[DIAG] image {i}: raw model output {len(raw_inst)} instances")
                    if raw_inst.has("scores"):
                        scores = raw_inst.scores
                        classes = raw_inst.pred_classes if raw_inst.has("pred_classes") else None
                        print(f"       scores: {float(scores.min()):.3f}-{float(scores.max()):.3f}, "
                              f"classes: {set(int(c.item()) for c in classes) if classes is not None else 'N/A'}")
                    # Check box coordinates
                    if raw_inst.has("pred_boxes"):
                        try:
                            boxes_list = raw_inst.pred_boxes.tensor.numpy()
                            if len(boxes_list) > 0:
                                print(f"       box[0]: {boxes_list[0]} (x1, y1, x2, y2)")
                        except Exception:
                            pass
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
                    elif after_n > 0:
                        try:
                            kept_scores = outputs["instances"].scores
                            top_score = float(kept_scores.max().item()) if len(kept_scores) else 0.0
                        except Exception:
                            top_score = -1.0
                        print(
                            f"[INFO] image {i}: threshold {float(score_thresh):.2f} kept {after_n}/{before_n} detections"
                            + (f" (top score {top_score:.3f})" if top_score >= 0.0 else "")
                        )

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
            vis, out_json = prepare_output(
                outputs,
                img,
                cfg,
                include_json=bool(return_json),
                categories_json_text=categories_json_text,
            )
            summary_text = _build_detection_summary(
                outputs.get("instances", None),
                cfg,
                categories_json_text=categories_json_text,
            )
            image_total, image_class_stats = _collect_detection_stats(
                outputs.get("instances", None),
                cfg,
                categories_json_text=categories_json_text,
            )
            batch_total_detections += int(image_total)
            for class_name, class_item in image_class_stats.items():
                merged = batch_class_stats.setdefault(class_name, {"count": 0, "scores": []})
                merged["count"] += int(class_item.get("count", 0))
                merged["scores"].extend(class_item.get("scores", []))
            # prepare_output returns RGB-arrays; Gradio accepts numpy arrays
            try:
                vis_list.append(vis)
            except Exception:
                vis_list.append(vis)
            summary_list.append(summary_text)
            if return_json:
                json_list.append(out_json if isinstance(out_json, dict) else {"error": "JSON generation failed"})
        except Exception as e:
            print(f"[WARN] Inference failed for image {i}: {e}")
            summary_list.append(f"Error: {e}")
            if return_json:
                json_list.append({"error": str(e)})

    # If only a single input was provided, return a single image instead of a list
    if not is_batch:
        vis = vis_list[0] if vis_list else None
        summary_text = summary_list[0] if summary_list else "No detections."
        if return_json and json_list:
            out_json = json.dumps(json_list[0], separators=(",", ":"))
            if len(out_json) > MAX_TEXTBOX_JSON_CHARS:
                out_json = json.dumps(
                    {
                        "warning": "JSON output truncated for UI responsiveness.",
                        "chars": len(out_json),
                    },
                    indent=2,
                )
        else:
            out_json = ""
        return vis, out_json, summary_text

    # Batch case: return gallery (list of images) and JSON array if requested
    if return_json:
        out_json = json.dumps(json_list, separators=(",", ":"))
        if len(out_json) > MAX_TEXTBOX_JSON_CHARS:
            out_json = json.dumps(
                {
                    "warning": "JSON output truncated for UI responsiveness.",
                    "images": len(json_list),
                    "chars": len(out_json),
                },
                indent=2,
            )
    else:
        out_json = ""
    per_image_summary_text = "\n\n".join(
        f"Image {index + 1}\n{summary}"
        for index, summary in enumerate(summary_list)
    ) if summary_list else "No detections."

    total_summary_text = _format_detection_stats(
        batch_total_detections,
        batch_class_stats,
        title="Total (all images)",
    )
    summary_text = f"{total_summary_text}\n\n{per_image_summary_text}" if summary_list else total_summary_text
    return vis_list, out_json, summary_text


def _release_inference_gpu_memory():
    """Release GPU memory held by the cached predictor on demand."""
    _release_predictor_cache()
    message = "Model cache cleared. GPU memory released."
    return message, "", None, None, gr.update(interactive=False)


def _write_download_file(prefix: str, suffix: str, content: str):
    """Write downloadable text content to a temp file and return its path."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=suffix, prefix=prefix) as tmp:
            tmp.write(content if content is not None else "")
            return tmp.name
    except Exception:
        return None


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

    with gr.Blocks(css="""
    .linked-input-card {
      border: 1px solid var(--border-color-primary, #b8bec7);
      border-radius: 10px;
      padding: 10px;
      background: var(--background-fill-secondary, #f7f9fc);
    }
    .section-card {
      border: 1px solid var(--border-color-primary, #b8bec7);
      border-radius: 12px;
      padding: 12px;
      background: var(--block-background-fill, var(--background-fill-primary, #ffffff));
      margin-bottom: 12px;
    }
    .compact-file .file-wrap,
    .compact-file .wrap,
    .compact-file [data-testid="file-upload"],
    .compact-file [data-testid="file-upload-dropzone"] {
            min-height: 112px !important;
            max-height: 112px !important;
      padding-top: 4px !important;
      padding-bottom: 4px !important;
    }
        .download-button-wrap {
            width: 80%;
            margin: 12px auto 0 auto;
        }
        .download-button-wrap button {
            width: 100%;
            border-radius: 12px !important;
            padding: 10px 16px !important;
        }
    """) as demo:
        gr.Markdown("# Inference UI — MaskDINO (default)\nUpload an image, choose config/weights, and run inference.\nUse the file pickers to browse config/checkpoint files outside the repo folder.")
        with gr.Group(elem_classes=["section-card"]):
            gr.Markdown("### Inputs and settings")
            with gr.Row():
                model_type = gr.Dropdown(choices=["maskrcnn", "maskdino"], value="maskdino", label="Model Type")
            with gr.Row():
                with gr.Group(elem_classes=["linked-input-card"]):
                    gr.Markdown("#### Config selector")
                    config_file = gr.Textbox(label="Config file (path)", value="maskdino_drone_config.yaml")
                    config_picker = gr.File(
                        label="Browse .yaml/.yml",
                        file_count="single",
                        file_types=[".yaml", ".yml"],
                        elem_classes=["compact-file"],
                    )
                with gr.Group(elem_classes=["linked-input-card"]):
                    gr.Markdown("#### Weights selector")
                    weights = gr.Textbox(label="Weights file (path)", value="output/model_final.pth")
                    weights_picker = gr.File(
                        label="Browse .pth/.pt",
                        file_count="single",
                        file_types=[".pth", ".pt"],
                        elem_classes=["compact-file"],
                    )
            categories_json = gr.Textbox(
                label="Categories",
                value=DEFAULT_CATEGORIES_JSON,
                lines=12,
                max_lines=12,
            )
            # file_count="multiple" + no file_types restriction lets users drag-drop
            # individual files OR an entire folder.
            with gr.Group(elem_classes=["linked-input-card"]):
                gr.Markdown("### Upload images for inference:")
                img_in = gr.Files(
                    label="Input images — drag & drop files or a folder here",
                    file_count="multiple",
                    elem_id="img_in_files",
                    elem_classes=["compact-file"],
                )
                gr.HTML("""
                    <div id="folder-picker-wrapper" style="margin-top:4px">
                        <input type="file" id="folder-picker-input" webkitdirectory multiple style="display:none">
                        <button id="folder-picker-btn" type="button"
                            style="padding:5px 14px;cursor:pointer;
                                   background:var(--button-secondary-background-fill,#f0f0f0);
                                   border:1px solid var(--border-color-primary,#bbb);
                                   border-radius:4px;font-size:13px;color:inherit">
                            &#128193; Select folder
                        </button>
                        <span id="folder-picker-status" style="margin-left:8px;font-size:12px;color:inherit"></span>
                    </div>
                    """)
            with gr.Row():
                score = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Score threshold")

        def on_model_change(mtype):
            cfg_def, w_def = _model_defaults(mtype)
            return gr.update(value=cfg_def), gr.update(value=w_def)

        def on_weights_pick(file_obj):
            if file_obj is None:
                return gr.update()
            # Gradio File returns an object with a temp path in `.name`.
            picked_path = getattr(file_obj, "name", None)
            if isinstance(picked_path, str) and picked_path.strip():
                return gr.update(value=picked_path)
            return gr.update()

        def on_config_pick(file_obj):
            if file_obj is None:
                return gr.update()
            picked_path = getattr(file_obj, "name", None)
            if isinstance(picked_path, str) and picked_path.strip():
                return gr.update(value=picked_path)
            return gr.update()

        # Update config and weights textboxes when model type changes
        model_type.change(on_model_change, inputs=[model_type], outputs=[config_file, weights])
        # Allow selecting config files from outside the repository.
        config_picker.change(on_config_pick, inputs=[config_picker], outputs=[config_file])
        # Allow selecting checkpoint files from outside the repository.
        weights_picker.change(on_weights_pick, inputs=[weights_picker], outputs=[weights])
        with gr.Row():
            run_btn = gr.Button("Run Inference")
            unload_btn = gr.Button("Unload Model", interactive=False)

        with gr.Group(elem_classes=["section-card"]):
            gr.Markdown("### Outputs")
            img_out = gr.Gallery(label="Visualized output")
            with gr.Row():
                with gr.Group(elem_classes=["linked-input-card"]):
                    gr.Markdown("#### JSON output")
                    json_out = gr.Textbox(label="JSON output", interactive=False, lines=14, max_lines=14)
                    json_download = gr.DownloadButton("Download JSON", value=None, elem_classes=["download-button-wrap"])
                with gr.Group(elem_classes=["linked-input-card"]):
                    gr.Markdown("#### Detection summary")
                    summary_out = gr.Textbox(label="Detection summary", interactive=False, lines=14, max_lines=14)
                    summary_download = gr.DownloadButton("Download detections (.txt)", value=None, elem_classes=["download-button-wrap"])

        def _run(image, mtype, cfgf, wts, sc, catjson):
            try:
                vis, j, summary = run_inference(image, mtype, cfgf, wts, sc, True, catjson)
                json_file = _write_download_file("inference_", ".json", j) if j else None
                txt_file = _write_download_file("detections_", ".txt", summary)
                return vis, j, summary, json_file, txt_file, gr.update(interactive=True)
            except Exception as e:
                err = f"Error: {e}"
                txt_file = _write_download_file("detections_", ".txt", err)
                return [], err, err, None, txt_file, gr.update(interactive=False)

        run_btn.click(
            _run,
            inputs=[img_in, model_type, config_file, weights, score, categories_json],
            outputs=[img_out, json_out, summary_out, json_download, summary_download, unload_btn],
        )
        unload_btn.click(
            _release_inference_gpu_memory,
            inputs=None,
            outputs=[json_out, summary_out, json_download, summary_download, unload_btn],
        )

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
