#!/usr/bin/env python3
"""Simple inspector for PyTorch checkpoint files.

Usage:
  python scripts/inspect_checkpoint.py /path/to/checkpoint.pth

Prints top-level keys and useful metadata (optimizer lr, scheduler position,
AMP scaler presence, model param counts and sample shapes).
"""
import sys
import os
import traceback

def human(x):
    try:
        return tuple(x.shape)
    except Exception:
        return str(type(x))

def main(path):
    print('Inspecting checkpoint:', path)
    if not os.path.exists(path):
        print('ERROR: file not found')
        return 2
    try:
        import torch
    except Exception as e:
        print('ERROR: torch not available in this environment:', e)
        return 3

    try:
        ckpt = torch.load(path, map_location='cpu')
    except Exception as e:
        print('ERROR: failed to load checkpoint:', e)
        traceback.print_exc()
        return 4

    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())
    else:
        print('Loaded object type:', type(ckpt))
        keys = []
    print('Top-level keys:', keys)

    # Common entries
    for k in ('model','state_dict','model_state_dict'):
        if k in ckpt:
            model_state = ckpt[k]
            break
    else:
        model_state = ckpt if isinstance(ckpt, dict) else None

    if isinstance(model_state, dict):
        param_items = [(k, v) for k, v in model_state.items()]
        total = 0
        for _, v in param_items:
            try:
                total += int(v.numel())
            except Exception:
                pass
        print('Model parameter tensors found:', len(param_items))
        print('Total model parameters (approx):', total)
        print('Sample params:')
        for k, v in param_items[:6]:
            try:
                shape = tuple(v.shape)
                dtype = getattr(v, 'dtype', type(v))
            except Exception:
                shape = human(v)
                dtype = type(v)
            print(f'  {k:<60} shape={shape} dtype={dtype}')
    else:
        print('No model state dict found in checkpoint.')

    # Optimizer
    if 'optimizer' in ckpt:
        print('\nFound optimizer entry')
        opt = ckpt['optimizer']
        try:
            if isinstance(opt, dict) and 'param_groups' in opt:
                lrs = [g.get('lr') for g in opt['param_groups']]
                print('  optimizer.param_groups lr:', lrs)
            else:
                print('  optimizer type:', type(opt))
        except Exception as e:
            print('  failed to read optimizer info:', e)
    else:
        print('\nNo optimizer state found in checkpoint')

    # Scheduler
    if 'scheduler' in ckpt:
        print('\nFound scheduler entry')
        try:
            sched = ckpt['scheduler']
            if isinstance(sched, dict) and 'last_epoch' in sched:
                print('  scheduler.last_epoch:', sched.get('last_epoch'))
            else:
                print('  scheduler type:', type(sched))
        except Exception as e:
            print('  failed to read scheduler info:', e)
    else:
        print('\nNo scheduler state found in checkpoint')

    # AMP scaler
    if 'scaler' in ckpt or 'amp' in ckpt:
        print('\nFound AMP/GradScaler entry')
    else:
        print('\nNo AMP/GradScaler entry found')

    # Common metadata
    for m in ('epoch','iteration','iter','it'):
        if m in ckpt:
            print('Saved', m + ':', ckpt[m])

    print('\nDone')
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/inspect_checkpoint.py /path/to/checkpoint.pth')
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
