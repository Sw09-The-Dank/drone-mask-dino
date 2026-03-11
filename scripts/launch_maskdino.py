#!/usr/bin/env python3
"""
Wrapper launcher that registers local COCO JSONs and then runs MaskDINO/train_net.py

This script accepts `--train-json`, `--val-json`, and `--images-root` to register
the provided JSONs under the standard COCO names (`coco_2017_train`, `coco_2017_val`)
so MaskDINO's config (which expects those dataset names) can load them.

All remaining args are forwarded to `MaskDINO.train_net`.
"""
import argparse
import runpy
import os
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train-json", default=None)
    parser.add_argument("--val-json", default=None)
    parser.add_argument("--images-root", default=None)
    # parse known so we keep the rest for MaskDINO.train_net
    args, rest = parser.parse_known_args()

    if args.train_json or args.val_json:
        try:
            from detectron2.data.datasets import register_coco_instances
        except Exception:
            print("detectron2 not available to register datasets", file=sys.stderr)
            raise

        # register under repository-unique names to avoid clobbering existing COCO metadata
        train_name = "drone_train_polygons"
        val_name = "drone_val_polygons"
        if args.train_json:
            register_coco_instances(train_name, {}, args.train_json, args.images_root or "")
            print(f"Registered {train_name} -> {args.train_json} (images root: {args.images_root})")
        if args.val_json:
            register_coco_instances(val_name, {}, args.val_json, args.images_root or "")
            print(f"Registered {val_name} -> {args.val_json} (images root: {args.images_root})")

        # ensure MaskDINO receives dataset overrides that point to our registered names
        # Detectron2/MaskDINO accepts tuple values via CLI like: DATASETS.TRAIN ('name',)
        dataset_overrides = [
            "DATASETS.TRAIN",
            "('" + train_name + "',)",
            "DATASETS.TEST",
            "('" + val_name + "',)",
        ]
    else:
        dataset_overrides = []

    # forward the rest of argv to MaskDINO.train_net; append dataset overrides if present
    sys.argv = ["MaskDINO/train_net.py"] + rest + dataset_overrides
    # ensure the MaskDINO package dir is on sys.path so `import maskdino` works
    maskdino_pkg_dir = os.path.join(os.getcwd(), "MaskDINO")
    if maskdino_pkg_dir not in sys.path:
        sys.path.insert(0, maskdino_pkg_dir)
    # Defensive monkeypatch: convert any numpy.ndarray segmentation fields
    # produced by upstream preprocessing into plain python lists before
    # Detectron2's `annotations_to_instances` consumes them. This avoids
    # modifying files inside the MaskDINO package while fixing runtime
    # errors where PolygonMasks expects lists, not numpy arrays.
    try:
        import numpy as _np
        from detectron2.data import detection_utils as _dutils

        _orig_annotations_to_instances = _dutils.annotations_to_instances

        def _patched_annotations_to_instances(annotations, image_size):
            new_annos = []
            for ann in annotations:
                if not isinstance(ann, dict):
                    new_annos.append(ann)
                    continue
                seg = ann.get("segmentation", None)
                if seg is None:
                    new_annos.append(ann)
                    continue
                # Make a shallow copy so we don't mutate caller objects
                ann_copy = ann.copy()
                # If segmentation is a numpy array -> convert
                if isinstance(seg, _np.ndarray):
                    if seg.ndim == 1:
                        ann_copy["segmentation"] = seg.tolist()
                    else:
                        # Try to convert array of polygons
                        try:
                            ann_copy["segmentation"] = [p.tolist() for p in seg]
                        except Exception:
                            ann_copy["segmentation"] = seg.flatten().tolist()
                elif isinstance(seg, list):
                    # Convert any numpy arrays inside the list
                    changed = False
                    new_seg = []
                    for poly in seg:
                        if isinstance(poly, _np.ndarray):
                            changed = True
                            new_seg.append(poly.tolist())
                        else:
                            new_seg.append(poly)
                    if changed:
                        ann_copy["segmentation"] = new_seg

                new_annos.append(ann_copy)

            return _orig_annotations_to_instances(new_annos, image_size)

        _dutils.annotations_to_instances = _patched_annotations_to_instances
        print("Applied runtime monkeypatch to Detectron2 annotations_to_instances()")
    except Exception:
        # If anything goes wrong, don't prevent training; fall back to default behavior
        pass
    # run the train_net.py file directly (some repos don't expose MaskDINO as an importable package)
    train_path = os.path.join(maskdino_pkg_dir, "train_net.py")
    runpy.run_path(train_path, run_name="__main__")


if __name__ == "__main__":
    main()
