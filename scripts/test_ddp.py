# test_ddp.py
import os, socket
import torch, torch.distributed as dist

print('PROC START', os.environ.get('RANK'), os.environ.get('LOCAL_RANK'), os.environ.get('WORLD_SIZE'), 'HOST', socket.gethostname())
dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
r = dist.get_rank(); ws = dist.get_world_size()
print(f'DDP INIT: rank={r} world_size={ws}')
# barrier to ensure all joined
dist.barrier()
if r == 0:
    print('ALL JOINED OK')
dist.destroy_process_group()