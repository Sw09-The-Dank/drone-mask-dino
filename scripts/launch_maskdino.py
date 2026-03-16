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
import json
import tempfile


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train-json", default="/workspace/output_annotations/train_polygons_clean.json")
    parser.add_argument("--val-json", default="/workspace/output_annotations/val_polygons_clean.json")
    parser.add_argument("--images-root", default="/workspace/dataset/images/train",
                        help="Images root for train dataset (used if --val-images-root not provided)")
    parser.add_argument("--val-images-root", default="/workspace/dataset/images/val",
                        help="Images root for validation dataset (optional, defaults to --images-root)")
    parser.add_argument("--from-scratch", action="store_true",
                        help="If set, clear MODEL.WEIGHTS to train from random init and infer number of classes from --train-json")
    parser.add_argument("--config-file", default=None,
                        help="Path to MaskDINO config file. If omitted, uses maskdino_drone_config.yaml in repo root")
    parser.add_argument("--fix-json-root", action="store_true",
                        help="If set, rewrite train/val COCO JSONs so image file_name entries point to the provided images roots (uses basename + images-root)")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Override solver max iterations (SOLVER.MAX_ITER)")
    parser.add_argument("--base-lr", type=float, default=None,
                        help="Override base learning rate (SOLVER.BASE_LR)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override dataloader workers (DATALOADER.NUM_WORKERS)")
    parser.add_argument("--ims-per-batch", type=int, default=None,
                        help="Override images per batch (SOLVER.IMS_PER_BATCH)")
    parser.add_argument("--low-mem-eval", action='store_true',
                        help="Apply a set of conservative config overrides for low-memory evaluation (smaller images, fewer queries, fewer points, single-worker)")
    parser.add_argument("--low-mem-eval-aggressive", action='store_true',
                        help="Apply an aggressive low-memory preset (smaller test size, very few queries/points)"
                        )
    parser.add_argument("--resume", action='store_true', help="Resume from last checkpoint (forwarded to train_net.py)")
    parser.add_argument("--output", default="/workspace/output",
                        help="Optional output directory to set as OUTPUT_DIR for MaskDINO")
    # parse known so we keep the rest for MaskDINO.train_net
    args, rest = parser.parse_known_args()

    if args.train_json or args.val_json:
        # Optionally rewrite JSON file_name entries to point to provided image roots.
        # This fixes cases where the JSON references images in 'images/train' but
        # the validation images live in 'images/val'. Rewritten files are written
        # to temporary files and used for registration.
        if args.fix_json_root:
            if args.train_json and args.images_root:
                try:
                    with open(args.train_json, "r") as _f:
                        _j = json.load(_f)
                    for img in _j.get("images", []):
                        img["file_name"] = os.path.join(args.images_root, os.path.basename(img.get("file_name", "")))
                    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
                    with open(tf.name, "w") as _wf:
                        json.dump(_j, _wf)
                    args.train_json = tf.name
                    # When JSON contains absolute paths after rewrite, registration root can be empty
                    train_reg_root = ""
                except Exception as _e:
                    print(f"Failed to rewrite train json paths: {_e}", file=sys.stderr)
                    train_reg_root = args.images_root or ""
            else:
                train_reg_root = args.images_root or ""

            if args.val_json:
                val_root = args.val_images_root or args.images_root
                if val_root:
                    try:
                        with open(args.val_json, "r") as _f:
                            _jv = json.load(_f)
                        for img in _jv.get("images", []):
                            img["file_name"] = os.path.join(val_root, os.path.basename(img.get("file_name", "")))
                        tfv = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
                        with open(tfv.name, "w") as _wv:
                            json.dump(_jv, _wv)
                        args.val_json = tfv.name
                        val_reg_root = ""
                    except Exception as _e:
                        print(f"Failed to rewrite val json paths: {_e}", file=sys.stderr)
                        val_reg_root = args.val_images_root or args.images_root or ""
                else:
                    val_reg_root = args.val_images_root or args.images_root or ""
            else:
                val_reg_root = args.val_images_root or args.images_root or ""
        else:
            train_reg_root = args.images_root or ""
            val_reg_root = args.val_images_root or args.images_root or ""
        try:
            from detectron2.data.datasets import register_coco_instances
        except Exception:
            print("detectron2 not available to register datasets", file=sys.stderr)
            raise

        # register under repository-unique names to avoid clobbering existing COCO metadata
        train_name = "drone_train_polygons"
        val_name = "drone_val_polygons"
        if args.train_json:
            register_coco_instances(train_name, {}, args.train_json, train_reg_root)
            print(f"Registered {train_name} -> {args.train_json} (images root: {train_reg_root})")
        if args.val_json:
            register_coco_instances(val_name, {}, args.val_json, val_reg_root)
            print(f"Registered {val_name} -> {args.val_json} (images root: {val_reg_root})")

        # ensure MaskDINO receives dataset overrides that point to our registered names
        # Detectron2/MaskDINO accepts tuple values via CLI like: DATASETS.TRAIN ('name',)
        dataset_overrides = [
            "DATASETS.TRAIN",
            "('" + train_name + "',)",
            "DATASETS.TEST",
            "('" + val_name + "',)",
        ]

        # If the user provided an --output path to this launcher, translate it
        # into a config override for OUTPUT_DIR so MaskDINO will use it.
        if args.output:
            dataset_overrides.extend(["OUTPUT_DIR", args.output])

        # If user wants to train from scratch, try to infer number of classes
        # from the provided COCO train JSON and clear MODEL.WEIGHTS.
        if args.from_scratch:
            # Clear weights so MaskDINO will initialize from scratch
            # (many configs treat empty string as no pre-trained weights)
            if args.train_json:
                try:
                    with open(args.train_json, "r") as _f:
                        _j = json.load(_f)
                    # Prefer explicit categories list
                    if isinstance(_j.get("categories"), list) and len(_j["categories"]) > 0:
                        num_classes = len(_j["categories"])
                    else:
                        anns = _j.get("annotations", [])
                        cat_ids = {a.get("category_id") for a in anns if "category_id" in a}
                        num_classes = len(cat_ids)

                    if num_classes and num_classes > 0:
                        dataset_overrides.extend(["MODEL.WEIGHTS", ""])
                        dataset_overrides.extend(["MODEL.ROI_HEADS.NUM_CLASSES", str(num_classes)])
                        # Also set semantic head class count if present in the config
                        dataset_overrides.extend(["MODEL.SEM_SEG_HEAD.NUM_CLASSES", str(num_classes)])
                        print(f"Configured training from scratch with {num_classes} classes")
                    else:
                        print("Warning: could not infer number of classes from train JSON; clearing MODEL.WEIGHTS only", file=sys.stderr)
                        dataset_overrides.extend(["MODEL.WEIGHTS", ""])
                except Exception as _e:
                    print(f"Error reading train json to infer classes: {_e}", file=sys.stderr)
                    dataset_overrides.extend(["MODEL.WEIGHTS", ""])
            else:
                # No train json provided; just clear weights
                dataset_overrides.extend(["MODEL.WEIGHTS", ""])

        # Hyperparameter overrides from launcher flags
        if args.max_iter is not None:
            dataset_overrides.extend(["SOLVER.MAX_ITER", str(args.max_iter)])
        if args.base_lr is not None:
            dataset_overrides.extend(["SOLVER.BASE_LR", str(args.base_lr)])
        if args.num_workers is not None:
            dataset_overrides.extend(["DATALOADER.NUM_WORKERS", str(args.num_workers)])
        if args.ims_per_batch is not None:
            dataset_overrides.extend(["SOLVER.IMS_PER_BATCH", str(args.ims_per_batch)])
    else:
        dataset_overrides = []

    # If user requested low-memory evaluation presets, append conservative overrides
    if getattr(args, 'low_mem_eval', False):
        # These values are conservative and inspired by the project's `train.py` presets
        # to reduce GPU memory usage during inference/eval.
        low_mem_overrides = [
            "MODEL.MaskDINO.NUM_OBJECT_QUERIES", "32",
            "MODEL.MaskDINO.TRAIN_NUM_POINTS", "1024",
            "INPUT.MIN_SIZE_TEST", "512",
            "INPUT.MAX_SIZE_TEST", "512",
            "TEST.IMS_PER_BATCH", "1",
            "DATALOADER.NUM_WORKERS", "0",
            "SOLVER.IMS_PER_BATCH", "1",
        ]
        dataset_overrides.extend(low_mem_overrides)
    # Aggressive preset: even smaller and fewer queries/points
    if getattr(args, 'low_mem_eval_aggressive', False):
        aggressive = [
            "MODEL.MaskDINO.NUM_OBJECT_QUERIES", "16",
            "MODEL.MaskDINO.TRAIN_NUM_POINTS", "512",
            "INPUT.MIN_SIZE_TEST", "384",
            "INPUT.MAX_SIZE_TEST", "384",
            "MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE", "64",
            "TEST.IMS_PER_BATCH", "1",
            "DATALOADER.NUM_WORKERS", "0",
            "SOLVER.IMS_PER_BATCH", "1",
        ]
        dataset_overrides.extend(aggressive)

    # forward the rest of argv to MaskDINO.train_net; append dataset overrides if present
    # If the user didn't provide a config file, default to the project root drone config
    if not args.config_file:
        default_cfg = os.path.join(os.getcwd(), "maskdino_drone_config.yaml")
        # Prepend the default --config-file so MaskDINO uses it
        rest = ["--config-file", default_cfg] + rest
    else:
        # User provided a config-file to the launcher; forward it to MaskDINO
        rest = ["--config-file", args.config_file] + rest

    # Forward the resume flag to MaskDINO's train_net if requested
    if getattr(args, 'resume', False):
        # ensure '--resume' is included in CLI args passed to train_net
        rest = rest + ["--resume"]

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
    # Prefer a root-level `train_net.py` (so we don't have to modify files inside MaskDINO/)
    root_train = os.path.join(os.getcwd(), "train_net.py")
    if os.path.exists(root_train):
        train_path = root_train
    else:
        train_path = os.path.join(maskdino_pkg_dir, "train_net.py")
    runpy.run_path(train_path, run_name="__main__")


if __name__ == "__main__":
    main()
