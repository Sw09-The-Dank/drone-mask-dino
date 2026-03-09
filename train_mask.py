#!/usr/bin/env python3
"""
Trainer wrapper for MaskDINO.
This script launches MaskDINO's training entrypoint (`MaskDINO/train_net.py`)
using `torch.distributed.run` (a.k.a. torchrun). It accepts common training
arguments and forwards remaining options to the training script.
"""
import argparse
import shlex
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Launcher for MaskDINO training")
    parser.add_argument("--config-file", "-c", required=True, help="Path to config yaml")
    parser.add_argument("--num-gpus", "-g", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--max-iter", type=int, help="Override MAX_ITER in config")
    parser.add_argument("--batch-size", type=int, help="Override SOLVER.IMS_PER_BATCH")
    parser.add_argument("--lr", type=float, help="Override SOLVER.BASE_LR")
    parser.add_argument("--dist-url", default="env://", help="URL used to set up distributed training")
    # dataset args matching train.py defaults
    parser.add_argument("--train-json", default="output_annotations/train_polygons.json", help="path to train COCO JSON")
    parser.add_argument("--val-json", default="output_annotations/val_polygons.json", help="path to val COCO JSON")
    parser.add_argument("--images-root", default="dataset/images", help="root folder for images")
    parser.add_argument("--num-workers", type=int, help="Override DATALOADER.NUM_WORKERS")
    parser.add_argument("--batch-size-per-image", type=int, help="Override MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE")
    parser.add_argument("--num-classes", type=int, help="Override MODEL.ROI_HEADS.NUM_CLASSES")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, help="Additional config options (KEY VALUE ...)")
    return parser.parse_args()


def build_command(args):
    # Build the torch distributed launcher command to run MaskDINO's train_net.py
    launcher = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(args.num_gpus)]
    # use wrapper that can register local COCO JSONs for MaskDINO
    launcher += ["scripts/launch_maskdino.py", "--train-json", args.train_json, "--val-json", args.val_json, "--images-root", args.images_root]
    launcher += ["--config-file", args.config_file, "--num-gpus", str(args.num_gpus)]
    # MaskDINO/train_net reads OUTPUT_DIR via cfg, so pass via opts later
    if args.resume:
        launcher.append("--resume")
    # Add overrides via opts or explicit shortcut args
    overrides = []
    if args.max_iter is not None:
        overrides += ["SOLVER.MAX_ITER", str(args.max_iter)]
    if args.batch_size is not None:
        overrides += ["SOLVER.IMS_PER_BATCH", str(args.batch_size)]
    if args.lr is not None:
        overrides += ["SOLVER.BASE_LR", str(args.lr)]
    if args.num_workers is not None:
        overrides += ["DATALOADER.NUM_WORKERS", str(args.num_workers)]
    if args.batch_size_per_image is not None:
        overrides += ["MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE", str(args.batch_size_per_image)]
    if args.num_classes is not None:
        overrides += ["MODEL.ROI_HEADS.NUM_CLASSES", str(args.num_classes)]
    # Append any user-provided opts
    if args.opts:
        # args.opts may start with '--'; strip that if present
        cleaned = [o for o in args.opts if o != "--"]
        overrides += cleaned
    if overrides:
        launcher += overrides
    # ensure OUTPUT_DIR is set to requested output dir
    launcher += ["OUTPUT_DIR", args.output_dir]
    return launcher


def main():
    args = parse_args()
    cmd = build_command(args)
    print("Running training command:")
    print(" ".join(shlex.quote(c) for c in cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training process exited with {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
