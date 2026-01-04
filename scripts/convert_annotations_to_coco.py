#!/usr/bin/env python3
"""Find/prepare COCO-format annotation JSONs and optionally register them with Detectron2.

Usage examples:
  python scripts/convert_annotations_to_coco.py --register
  python scripts/convert_annotations_to_coco.py --train-json path/to/train.json --val-json path/to/val.json --images-root dataset/images --register

The script will look for common paths if explicit JSONs are not provided:
  - annotations/train.json
  - annotations/val.json
  - dataset/annotations/train.json
  - dataset/annotations/val.json

If `--register` is passed the script attempts to import Detectron2 and call
`register_coco_instances` for dataset names `drone_train` and `drone_val`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Optional
import numpy as np

try:
    from pycocotools import mask as maskUtils
except Exception:
    maskUtils = None

# contour extraction: prefer OpenCV, fall back to skimage
try:
    import cv2
except Exception:
    cv2 = None
try:
    from skimage import measure
except Exception:
    measure = None


COMMON_TRAIN_PATHS = [
    "annotations/train.json",
    os.path.join("dataset", "annotations", "train.json"),
    os.path.join("dataset", "train.json"),
]

COMMON_VAL_PATHS = [
    "annotations/val.json",
    os.path.join("dataset", "annotations", "val.json"),
    os.path.join("dataset", "val.json"),
]


def find_first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def copy_json(src: str, dst_dir: str) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    return dst


def validate_json(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # basic COCO checks
        if not isinstance(data, dict):
            return False
        if "images" not in data or "annotations" not in data:
            return False
        return True
    except Exception:
        return False


def try_register(train_json: Optional[str], val_json: Optional[str], images_root: Optional[str]):
    try:
        from detectron2.data.datasets import register_coco_instances
        from detectron2.data import MetadataCatalog
    except Exception as e:
        print("[WARN] Could not import detectron2. Install detectron2 to register datasets:", e)
        print("You can still pass the JSON files to the trainer directly.")
        return False

    if train_json:
        print(f"Registering 'drone_train' -> json: {train_json}, images: {images_root or '(none)'}")
        register_coco_instances("drone_train", {}, train_json, images_root or "")
    if val_json:
        print(f"Registering 'drone_val' -> json: {val_json}, images: {images_root or '(none)'}")
        register_coco_instances("drone_val", {}, val_json, images_root or "")

    # Print brief metadata info
    try:
        if "drone_train" in MetadataCatalog.list():
            m = MetadataCatalog.get("drone_train")
            print("Registered metadata keys for drone_train:", list(m.keys()) if hasattr(m, 'keys') else 'ok')
    except Exception:
        pass
    return True


def rle_to_polygons(rle, height: int, width: int, approximate_eps: float = 1.0):
    """Convert COCO RLE mask to polygon segmentation list.

    Returns (polygons_list, area, bbox)
    """
    if maskUtils is None:
        raise RuntimeError("pycocotools is required for RLE decoding (install pycocotools)")

    # decodes to HxW array (or HxWxN if multiple)
    mask = maskUtils.decode(rle)
    if mask.ndim == 3:
        # merge multiple instance masks
        mask = np.any(mask, axis=2).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    # compute area and bbox
    area = float(mask.sum())
    ys, xs = np.where(mask)
    if ys.size == 0:
        bbox = [0.0, 0.0, 0.0, 0.0]
    else:
        x_min = float(xs.min())
        y_min = float(ys.min())
        x_max = float(xs.max())
        y_max = float(ys.max())
        bbox = [x_min, y_min, x_max - x_min + 1.0, y_max - y_min + 1.0]

    polygons = []
    # find contours
    if cv2 is not None:
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cnt.size < 6:
                continue
            if approximate_eps and approximate_eps > 0:
                eps = approximate_eps * cv2.arcLength(cnt, True) / 100.0
                cnt = cv2.approxPolyDP(cnt, eps, True)
            poly = cnt.reshape(-1, 2).flatten().tolist()
            if len(poly) >= 6:
                polygons.append([float(x) for x in poly])
    elif measure is not None:
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            # contour is (N, 2) in (row, col) => convert to (x, y)
            if contour.shape[0] < 3:
                continue
            poly = []
            for y, x in contour:
                poly.extend([float(x), float(y)])
            if len(poly) >= 6:
                polygons.append(poly)
    else:
        raise RuntimeError("Need either OpenCV (cv2) or scikit-image (skimage.measure) to extract polygons")

    return polygons, area, bbox


def convert_json_rle_to_polygons(json_path: str, out_path: str, approx_eps: float = 1.0):
    if maskUtils is None:
        raise RuntimeError("pycocotools is required for conversion; please install it")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_map = {img["id"]: img for img in data.get("images", [])}
    anns = data.get("annotations", [])
    converted = 0
    for ann in anns:
        seg = ann.get("segmentation")
        if isinstance(seg, dict) and ("counts" in seg or isinstance(seg.get("counts"), (list, str))):
            h, w = seg.get("size", (images_map.get(ann.get("image_id"), {}).get("height"), images_map.get(ann.get("image_id"), {}).get("width")))
            if h is None or w is None:
                # fallback: use image size if available
                img = images_map.get(ann.get("image_id"))
                if img:
                    h = img.get("height")
                    w = img.get("width")
            if h is None or w is None:
                print(f"[WARN] Could not determine size for image_id {ann.get('image_id')}; skipping ann {ann.get('id')}")
                continue
            try:
                polys, area, bbox = rle_to_polygons(seg, int(h), int(w), approximate_eps=approx_eps)
            except Exception as e:
                print(f"[WARN] Failed to convert RLE for ann {ann.get('id')}: {e}")
                continue
            if polys:
                # sanitize polygons: ensure plain Python lists of floats and drop tiny polygons
                clean_polys = []
                for poly in polys:
                    arr = np.asarray(poly).astype(float).flatten()
                    if arr.size < 6:
                        continue
                    # Drop degenerate polygons with zero area (very small bbox)
                    xs = arr[0::2]
                    ys = arr[1::2]
                    if (xs.max() - xs.min() < 1e-3) or (ys.max() - ys.min() < 1e-3):
                        continue
                    clean_polys.append(arr.tolist())
                if not clean_polys:
                    continue
                ann["segmentation"] = clean_polys
                ann["area"] = float(area)
                ann["bbox"] = [float(b) for b in bbox]
                ann.setdefault("iscrowd", 0)
                converted += 1
    # write out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Converted {converted} annotations to polygon segmentation and wrote: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Prepare/convert annotations and optionally register with Detectron2")
    p.add_argument("--train-json", help="Path to train COCO JSON")
    p.add_argument("--val-json", help="Path to val COCO JSON")
    p.add_argument("--images-root", help="Root dir for images (used when registering)")
    p.add_argument("--output-dir", default="output_annotations", help="Where to copy discovered JSON files")
    p.add_argument("--register", action="store_true", help="If set, try to register datasets with Detectron2")
    p.add_argument("--to-polygons", action="store_true", help="Convert RLE segmentations to polygon lists and write new JSON(s)")
    p.add_argument("--approx-eps", type=float, default=1.0, help="Approximation epsilon percent for polygon simplification (default=1.0)")
    args = p.parse_args()

    train_json = args.train_json or find_first_existing(COMMON_TRAIN_PATHS)
    val_json = args.val_json or find_first_existing(COMMON_VAL_PATHS)

    if not train_json and not val_json:
        print("[ERROR] No COCO JSON found. Provide --train-json/--val-json or place files at:")
        for x in COMMON_TRAIN_PATHS + COMMON_VAL_PATHS:
            print("  -", x)
        sys.exit(1)

    out_train = None
    out_val = None
    if train_json:
        if not validate_json(train_json):
            print(f"[ERROR] Train JSON appears invalid: {train_json}")
            sys.exit(1)
        out_train = copy_json(train_json, args.output_dir)
        print(f"Copied train JSON -> {out_train}")
    if val_json:
        if not validate_json(val_json):
            print(f"[ERROR] Val JSON appears invalid: {val_json}")
            sys.exit(1)
        out_val = copy_json(val_json, args.output_dir)
        print(f"Copied val JSON -> {out_val}")

    if args.register:
        ok = try_register(out_train, out_val, args.images_root)
        if not ok:
            print("[WARN] Registration failed. You can still pass the JSONs to your trainer.")

    if args.to_polygons:
        # convert copied JSON(s) in output dir
        if out_train:
            out_train_polys = os.path.join(args.output_dir, os.path.splitext(os.path.basename(out_train))[0] + "_polygons.json")
            convert_json_rle_to_polygons(out_train, out_train_polys, approx_eps=args.approx_eps)
        if out_val:
            out_val_polys = os.path.join(args.output_dir, os.path.splitext(os.path.basename(out_val))[0] + "_polygons.json")
            convert_json_rle_to_polygons(out_val, out_val_polys, approx_eps=args.approx_eps)

    print("Done. You can now point your trainer to the JSON(s) in:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
