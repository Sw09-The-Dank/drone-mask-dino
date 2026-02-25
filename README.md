```markdown


copy detectron2 into folder

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install "numpy<2"

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

conda remove gcc gxx -y
sudo update-alternatives --set gcc /usr/bin/gcc-12
sudo update-alternatives --set g++ /usr/bin/g++-12


pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git

pip install scipy
pip install ninja

cd MaskDINO/maskdino/modeling/pixel_decoder/ops
rm -rf build *.so *.egg-info
bash make.sh
cd ../../../../../




Modify annotations:
# Create new category 'bird' and copy drone segmentations to it when 'neo' is on same image
python scripts/modify_segment_category.py --input output_annotations/train_polygons.json --output output_annotations/train_polygons_cat.json --from-category camera --to-category neo_camera --create-category neo_camera --copy --filter-category neo

```

## Use Cases

Below are common examples for running `train.py` with different options.

- Basic training with defaults:

```bash
python train.py
```

- Specify dataset JSONs and images root:

```bash
python train.py --train-json output_annotations/train_polygons.json \
    --val-json output_annotations/val_polygons.json \
    --images-root dataset/images
```

- Override common hyperparameters:

```bash
python train.py --max-iter 5000 --ims-per-batch 8 --base-lr 0.0001 --num-workers 4
```

- Set model/framing options:

```bash
python train.py --num-classes 8 --batch-size-per-image 512
```

- Use an alternate Detectron2 config or a model_zoo key and custom weights:

```bash
python train.py --config-file COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --weights detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```

- Advanced: apply arbitrary dotted-cfg overrides (can repeat `-s`):

```bash
python train.py -s SOLVER.BASE_LR=0.00005 -s SOLVER.MAX_ITER=2000 -s SOLVER.IMS_PER_BATCH=4
```

- Save outputs to a custom directory:

```bash
python train.py --output-dir output_maskdino/trainer_output_custom
```

- Train from scratch (do not resume from last checkpoint):

```bash
python train.py --no-resume
```

Notes:
- The `-s/--set` overrides accept booleans (`true`/`false`), integers and floats when parseable; otherwise the raw string is used.
- Run `python train.py --help` to see all available options.
