import os
from detectron2.data.datasets import register_coco_instances

def register_drone_coco(project_root=None):
    """
    Register the COCO-style drone dataset as 'drone_train'.
    project_root = path to the folder that contains 'input/'.
    """
    if project_root is None:
        # default: parent of this file (your repo root)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    image_root = os.path.join(project_root, "input", "images")
    ann_file   = os.path.join(project_root, "input", "annotations", "drone_train.json")

    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Image folder not found: {image_root}")

    register_coco_instances(
        "drone_train",  # dataset name
        {},             # extra metadata (optional)
        ann_file,
        image_root,
    )

    print(f"[dataset] Registered 'drone_train' with:")
    print(f"          ann_file = {ann_file}")
    print(f"          image_root = {image_root}")
