

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
