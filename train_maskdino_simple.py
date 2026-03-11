#!/usr/bin/env python3
"""
Minimal training launcher for MaskDINO.

Usage examples:
  # single-GPU
  python train_maskdino_simple.py --config-file MaskDINO/configs/...yaml --output-dir output

  # multi-GPU (uses torch.distributed.run)
  python train_maskdino_simple.py --config-file MaskDINO/configs/...yaml --num-gpus 4 --output-dir output

This script is intentionally small: it builds the appropriate command and
executes MaskDINO's training entrypoint (`MaskDINO/train_net.py`). Additional
config overrides can be passed after `--` as KEY VALUE pairs.
"""
import argparse
import shlex
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal MaskDINO training launcher")
    parser.add_argument("--config-file", "-c", required=True, help="Path to MaskDINO config YAML")
    parser.add_argument("--num-gpus", "-g", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, help="Additional config overrides (KEY VALUE ...)" )
    return parser.parse_args()


def build_cmd(args):
    train_entry = "MaskDINO/train_net.py"
    base = [sys.executable]
    if args.num_gpus and args.num_gpus > 1:
        # Use torch.distributed.run (torchrun)
        base = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(args.num_gpus)]
        base.append(train_entry)
    else:
        base.append(train_entry)

    cmd = base + ["--config-file", args.config_file]
    if args.resume:
        cmd.append("--resume")
    # Append output dir and any user overrides
    cmd += ["OUTPUT_DIR", args.output_dir]
    if args.opts:
        # strip leading '--' if present
        cleaned = [o for o in args.opts if o != "--"]
        cmd += cleaned
    return cmd


def main():
    args = parse_args()
    cmd = build_cmd(args)
    print("Running:")
    print(" ".join(shlex.quote(c) for c in cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training exited with {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
