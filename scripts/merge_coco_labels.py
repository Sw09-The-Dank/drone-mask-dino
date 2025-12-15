import os
import glob
import json

LABEL_DIR = "input/labels"
OUT_ANN_PATH = "input/annotations/drone_train.json"
os.makedirs(os.path.dirname(OUT_ANN_PATH), exist_ok=True)

def main():
    json_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.json")))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {LABEL_DIR}")

    merged = {
        "images": [],
        "annotations": [],
        "categories": None,
    }

    ann_id = 1
    img_id = 1

    for i, json_path in enumerate(json_files):
        with open(json_path, "r") as f:
            data = json.load(f)

        # Use categories from the first file
        if merged["categories"] is None:
            merged["categories"] = data["categories"]
        else:
            # Optional: sanity check categories are identical
            pass

        # There is exactly one image in this json
        for img in data["images"]:
            img["id"] = img_id
            merged["images"].append(img)
            old_image_id = img["id"]

        # Fix annotation ids and image_ids to be unique
        for ann in data["annotations"]:
            ann["id"] = ann_id
            ann["image_id"] = img_id
            merged["annotations"].append(ann)
            ann_id += 1

        img_id += 1

    with open(OUT_ANN_PATH, "w") as f:
        json.dump(merged, f)

    print(f"Saved merged COCO annotations to {OUT_ANN_PATH}")
    print(f"Images: {len(merged['images'])}, Annotations: {len(merged['annotations'])}")

if __name__ == "__main__":
    main()
