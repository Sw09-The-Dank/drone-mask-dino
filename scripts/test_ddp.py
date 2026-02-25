# test_ddp.py
import os
import socket
import time
import torch
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", 0))
rank_env = os.environ.get('RANK')
world_env = os.environ.get('WORLD_SIZE')

# Ensure correct CUDA device for this local rank
if torch.cuda.is_available():
    try:
        torch.cuda.set_device(local_rank)
    except Exception:
        pass

print('PROC START', rank_env, local_rank, world_env, 'HOST', socket.gethostname())

# Initialize process group using env:// (torch.distributed.run sets env vars)
backend = 'nccl' if torch.cuda.is_available() else 'gloo'
dist.init_process_group(backend=backend, init_method='env://')
try:
    r = dist.get_rank()
    ws = dist.get_world_size()
    print(f'DDP INIT: rank={r} world_size={ws} backend={backend}')
    # barrier to ensure all joined
    dist.barrier()
    if r == 0:
        print('ALL JOINED OK')
    # write a small heartbeat file so external tools can see activity
    try:
        hb_path = f"/workspace/ddp_heartbeat_rank_{r}.txt"
        with open(hb_path, 'a') as hf:
            hf.write(f"alive {time.time()}\n")
    except Exception:
        pass
finally:
    try:
        dist.destroy_process_group()
    except Exception:
        pass