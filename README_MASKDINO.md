# MaskDINO — Quick Start & Examples

This file collects quick commands and examples to build, run, and train MaskDINO in this repository.

**Build Docker demo image**

Linux / macOS:

```bash
docker build -f Dockerfile.demo -t maskdino-demo:local .
```

Windows (PowerShell):

```powershell
docker build -f Dockerfile.demo -t maskdino-demo:local .
```

**Run container (GPU)**

Mount the repo into the container, enter a shell, then compile ops and run demo:

```bash
docker run --gpus all --rm -it -v "%CD%":/workspace maskdino-demo:local /bin/bash
# inside container
export TORCH_CUDA_ARCH_LIST="12.1+PTX;8.6;8.0;7.5"
/workspace/build_ops.sh
python MaskDINO/demo/demo.py --config-file MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml
```

**Compile CUDA ops at runtime (if needed)**

If the in-image compile fails, compile inside a running container or host with matching CUDA toolkit:

```bash
export TORCH_CUDA_ARCH_LIST="12.1+PTX;8.6;8.0;7.5"
bash /workspace/build_ops.sh
```

**Training examples**

- Single-GPU training using the launcher script:

```bash
python train_mask.py -c MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml -g 1 -o output
```

- Multi-GPU training (example: 4 GPUs):

```bash
python train_mask.py -c MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml -g 4 -o output --batch-size 16 --max-iter 90000
```

The launcher wraps `MaskDINO/train_net.py` with `torch.distributed.run` and forwards `--opts KEY VALUE ...` overrides.

Files you may use:

- Configs: [MaskDINO/configs](MaskDINO/configs#L1)
- Training launcher: [train_mask.py](train_mask.py#L1)
- Alternative launcher: [reain_mask.py](reain_mask.py#L1)
- Dockerfile demo: [Dockerfile.demo](Dockerfile.demo#L1)
- Runtime ops build script: [build_ops.sh](build_ops.sh#L1)

**Checkpoint key mismatch (warning)**

If you see a warning such as:

```
WARNING: The checkpoint state_dict contains keys that are not used by the model: stem.fc.{bias, weight}
```

You can either ignore it (if partial loading is expected) or strip those keys before loading. Example Python snippet to remove keys starting with `stem.fc.` and save a cleaned checkpoint:

```python
import torch
ckpt = torch.load('model.pth', map_location='cpu')
state = ckpt.get('model', ckpt)
filtered = {k: v for k, v in state.items() if not k.startswith('stem.fc.')}
if 'model' in ckpt:
    ckpt['model'] = filtered
torch.save(ckpt, 'model_stripped.pth')
```


python train_mask.py   -c MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml   -g 1   --train-json output_annotations/train_polygons.json   --val-json output_annotations/val_polygons.json   --images-root dataset/images   -o output_maskdino/trainer_output   --resume   --max-iter 3000   --batch-size 12   --lr 0.00005   --num-workers 12   --batch-size-per-image 256

**Troubleshooting**

- If CUDA arch errors occur during op compilation, prefer PTX-first arch list (`12.1+PTX`) in `TORCH_CUDA_ARCH_LIST` and fallback to older arches. currrently it will only run 8
- Ensure `git`, `cmake`, `ninja`, and a compatible `gcc` are available inside the build environment when compiling detectron2 or CUDA extensions.
- If detectron2 fails to install from source, run the install commands manually inside the container so you can inspect errors.

If you'd like, I can add an example config-to-opts mapping, a sample minimal dataset config, or a full reproducible CI-friendly Dockerfile. What should I add next?
