#!/usr/bin/env python3
"""
Combined training entrypoint: registers local COCO JSONs, prepares overrides,
and launches MaskDINO's `train_net` in a way that's compatible with
`torch.distributed.run` (torchrun).

Usage (single-node or under torch.distributed.run):
  python train_maskdino.py [--train-json PATH] [--val-json PATH] [--images-root PATH]
      [--fix-json-root] [--from-scratch] [--config-file PATH] [other MaskDINO args]

This script merges the launcher + small dataset-cleaning logic so you can use
one file instead of separate `launch_maskdino.py` + `train_net.py` wrappers.
"""

import argparse
import json
import os
import runpy
import sys
import tempfile

import random
import logging

import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.utils.logger import setup_logger

# MaskDINO helpers: prefer the lowercase package name to avoid loading the
# same code under two different module names (which can cause double-import
# side-effects such as dataset registration). Fall back to the capitalized
# package if necessary.
import importlib.util
if importlib.util.find_spec("maskdino") is not None:
    from maskdino.config import add_maskdino_config
elif importlib.util.find_spec("MaskDINO") is not None:
    from MaskDINO.maskdino.config import add_maskdino_config
else:
    raise ImportError(
        "Could not import `add_maskdino_config` from MaskDINO; ensure the MaskDINO package is on PYTHONPATH or that you're running from the repository root."
    )
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
)
from detectron2.data import build_detection_train_loader


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        try:
            from detectron2.data import MetadataCatalog

            evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        except Exception:
            evaluator_type = "coco"

        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, distributed=True))
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder, distributed=True))
        if evaluator_type == "mapillary_vistas_panoptic_seg":
            try:
                from maskdino.evaluation import InstanceSegEvaluator as _ISE

                try:
                    evaluator_list.append(_ISE(dataset_name, output_dir=output_folder, distributed=True))
                except TypeError:
                    evaluator_list.append(_ISE(dataset_name, output_dir=output_folder))
            except Exception:
                # InstanceSegEvaluator not available; skip
                pass

        if len(evaluator_list) == 0:
            raise NotImplementedError(f"no Evaluator for dataset {dataset_name} (type: {evaluator_type})")
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg)



def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train-json", default="/workspace/output_annotations/train_polygons_clean.json")
    parser.add_argument("--val-json", default="/workspace/output_annotations/val_polygons_clean.json")
    parser.add_argument("--images-root", default="/workspace/dataset/images/train")
    parser.add_argument("--val-images-root", default="/workspace/dataset/images/val")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--fix-json-root", action="store_true")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--base-lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--ims-per-batch", type=int, default=None)
    parser.add_argument("--low-mem-eval", action="store_true")
    parser.add_argument("--low-mem-eval-aggressive", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", default="/workspace/output")
    # capture the rest to forward to MaskDINO.train_net as --opts
    args, rest = parser.parse_known_args()

    # Optionally rewrite JSON file_name entries to point to provided image roots.
    # If --fix-json-root is used, write temporary cleaned JSONs and point registration
    # to those temporary files so MaskDINO can load images from a uniform root.
    train_json_to_register = args.train_json
    val_json_to_register = args.val_json
    train_reg_root = args.images_root or ""
    val_reg_root = args.val_images_root or args.images_root or ""

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
                train_json_to_register = tf.name
                train_reg_root = ""
            except Exception as _e:
                print(f"Failed to rewrite train json paths: {_e}", file=sys.stderr)
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
                    val_json_to_register = tfv.name
                    val_reg_root = ""
                except Exception as _e:
                    print(f"Failed to rewrite val json paths: {_e}", file=sys.stderr)
                    val_reg_root = args.val_images_root or args.images_root or ""

    # Try to register the COCO instances using detectron2 helper
    try:
        from detectron2.data.datasets import register_coco_instances
    except Exception:
        print("detectron2 not available to register datasets", file=sys.stderr)
        raise

    train_name = "drone_train_polygons"
    val_name = "drone_val_polygons"
    if train_json_to_register and os.path.isfile(train_json_to_register):
        register_coco_instances(train_name, {}, train_json_to_register, train_reg_root)
        print(f"Registered {train_name} -> {train_json_to_register} (images root: {train_reg_root})")
    if val_json_to_register and os.path.isfile(val_json_to_register):
        register_coco_instances(val_name, {}, val_json_to_register, val_reg_root)
        print(f"Registered {val_name} -> {val_json_to_register} (images root: {val_reg_root})")

    # Build dataset overrides: MaskDINO expects DATASETS.TRAIN/TEST tuple overrides passed via --opts
    dataset_overrides = [
        "DATASETS.TRAIN",
        "('" + train_name + "',)",
        "DATASETS.TEST",
        "('" + val_name + "',)",
    ]

    if args.output:
        dataset_overrides.extend(["OUTPUT_DIR", args.output])

    if args.from_scratch:
        # Ensure model weights cleared and infer classes later inside MaskDINO if needed
        dataset_overrides.extend(["MODEL.WEIGHTS", ""])

    # Hyperparameter overrides
    if args.max_iter is not None:
        dataset_overrides.extend(["SOLVER.MAX_ITER", str(args.max_iter)])
    if args.base_lr is not None:
        dataset_overrides.extend(["SOLVER.BASE_LR", str(args.base_lr)])
    if args.num_workers is not None:
        dataset_overrides.extend(["DATALOADER.NUM_WORKERS", str(args.num_workers)])
    if args.ims_per_batch is not None:
        dataset_overrides.extend(["SOLVER.IMS_PER_BATCH", str(args.ims_per_batch)])

    if args.low_mem_eval:
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

    if args.low_mem_eval_aggressive:
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

    # If no config-file was provided, prefer the repo root config so users
    # can run the script from the workspace without explicitly passing it.
    if not args.config_file:
        candidates = [
            os.path.join(os.getcwd(), "maskdino_drone_config.yaml"),
            os.path.join(os.getcwd(), "config.yaml"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                args.config_file = c
                print(f"Defaulting --config-file to {c}")
                break

    # Build final argv to pass to MaskDINO.train_net.main via detectron2 parser
    try:
        from detectron2.engine import default_argument_parser
        parser2 = default_argument_parser()
        # Compose the argv used by train_net: start with config-file if provided
        argv = []
        if args.config_file:
            argv.extend(["--config-file", args.config_file])
        # Append any remaining raw args the user provided and then the dataset overrides
        argv.extend(rest)
        argv.extend(dataset_overrides)
        # Ensure resume flag forwarded
        if args.resume:
            argv.append("--resume")
        # Parse into args namespace for MaskDINO.train_net.main
        parsed = parser2.parse_args(argv)
    except Exception:
        parsed = None

    # Determine whether we're already in a distributed-launched process
    is_distributed_env = False
    try:
        ws = int(os.environ.get("WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
        if ws > 1 or os.environ.get("RANK") is not None or os.environ.get("LOCAL_RANK") is not None:
            is_distributed_env = True
    except Exception:
        is_distributed_env = False

    def train_worker(parsed_args):
        # Build config
        cfg = get_cfg()
        add_maskdino_config(cfg)
        if parsed_args.config_file:
                # Detect and pre-create any missing config nodes referenced in
                # the config file to avoid KeyError on merge (some configs may
                # reference keys added by external modules).
                try:
                    cfg.merge_from_file(parsed_args.config_file)
                except KeyError as e:
                    try:
                        import yaml

                        def _ensure_nodes(node, data):
                            from detectron2.config import CfgNode as CN

                            if not isinstance(data, dict):
                                return
                            for k, v in data.items():
                                if not hasattr(node, k):
                                    setattr(node, k, CN())
                                _ensure_nodes(getattr(node, k), v)

                        with open(parsed_args.config_file, "r") as _cf:
                            cfg_dict = yaml.safe_load(_cf)
                        _ensure_nodes(cfg, cfg_dict)
                        cfg.merge_from_file(parsed_args.config_file)
                    except Exception:
                        # Re-raise original error if we cannot recover
                        raise e
        # parsed_args comes from detectron2 default parser and contains `opts`
        if hasattr(parsed_args, "opts") and parsed_args.opts:
            cfg.merge_from_list(parsed_args.opts)
        cfg.freeze()

        # Setup logging and other default behaviors
        default_setup(cfg, parsed_args)
        setup_logger(output=cfg.OUTPUT_DIR, name="maskdino")

        # Distributed sync: ensure all processes have completed setup before
        # constructing loaders / starting training. This helps catch config
        # mismatches early and ensures workers are in lock-step.
        try:
            rank = comm.get_rank()
            world_size = comm.get_world_size()
        except Exception:
            rank = 0
            world_size = 1

        logging.getLogger("maskdino").info(f"[rank {rank}/{world_size}] setup complete, awaiting peers...")
        try:
            comm.synchronize()
        except Exception:
            # If comm isn't available (single-process), continue silently
            pass

        trainer = Trainer(cfg)

        # Barrier after Trainer construction to ensure dataset loaders and
        # samplers are created consistently across processes.
        logging.getLogger("maskdino").info(f"[rank {rank}/{world_size}] trainer constructed, syncing before resume/load")
        try:
            comm.synchronize()
        except Exception:
            pass

        trainer.resume_or_load(resume=parsed_args.resume)

        logging.getLogger("maskdino").info(f"[rank {rank}/{world_size}] starting training loop")
        try:
            result = trainer.train()
        finally:
            # Ensure all processes reach the end before exit/signal handling.
            try:
                comm.synchronize()
            except Exception:
                pass

        logging.getLogger("maskdino").info(f"[rank {rank}/{world_size}] training finished")
        return result

    if parsed is None:
        raise RuntimeError("Failed to construct detectron2 parser arguments; cannot continue")

    if is_distributed_env:
        # Use detectron2.launch to start the worker function in this process group
        ngpus_per_machine = getattr(parsed, "num_gpus", getattr(parsed, "num_gpus_per_machine", 1))
        num_machines = getattr(parsed, "num_machines", 1)
        machine_rank = getattr(parsed, "machine_rank", 0)
        dist_url = getattr(parsed, "dist_url", "tcp://127.0.0.1:29500")
        launch(
            train_worker,
            ngpus_per_machine,
            num_machines=num_machines,
            machine_rank=machine_rank,
            dist_url=dist_url,
            args=(parsed,),
        )
    else:
        # Single-process run
        train_worker(parsed)


if __name__ == '__main__':
    main()
