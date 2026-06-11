#!/usr/bin/env python3
"""
Modify segmentation annotation category ids in a COCO JSON file.

Usage:
  python scripts/modify_segment_category.py \
    --input output_annotations/train.json \
    --output output_annotations/train_modified.json \
    --from-category drone --to-category bird \
    --filter-category person

If `--filter-category` is provided, the change will be applied only for annotations
whose `image_id` has at least one annotation with that filter category.

Categories may be specified by numeric id or by name.
"""

import argparse
import copy
import json
import sys
from collections import defaultdict


def resolve_category_id(categories, cat):
    # cat may be an integer id or a name
    try:
        cid = int(cat)
        # verify exists
        if any(c.get("id") == cid for c in categories):
            return cid
    except Exception:
        pass
    # fallback to name match
    for c in categories:
        if str(c.get("name")) == str(cat):
            return c.get("id")
    return None


def main():
    parser = argparse.ArgumentParser(description="Modify segmentation category ids in a COCO JSON.")
    parser.add_argument("--input", required=True, help="Input COCO JSON file path")
    parser.add_argument("--output", required=True, help="Output COCO JSON file path")
    parser.add_argument("--from-category", required=True, help="Category to change (id or name)")
    parser.add_argument("--to-category", required=True, help="Target category (id or name)")
    parser.add_argument("--filter-category", help="Only change when image contains this category (id or name)")
    parser.add_argument("--create-category", help="Create a new category with this name before applying changes")
    parser.add_argument("--create-category-id", type=int, help="Optional id to assign when creating a category")
    parser.add_argument("--copy", action="store_true", help="Copy matching segmentation annotations to new category instead of modifying in-place")

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", [])
    annotations = data.get("annotations", [])

    # Optionally create a new target category first
    if args.create_category:
        existing = next((c for c in categories if str(c.get("name")) == str(args.create_category)), None)
        if existing:
            print(f"Category '{args.create_category}' already exists with id {existing.get('id')}")
        else:
            max_cid = max((c.get("id", 0) for c in categories), default=0)
            new_id = args.create_category_id if args.create_category_id else max_cid + 1
            categories.append({"id": new_id, "name": args.create_category})
            print(f"Created category '{args.create_category}' with id {new_id}")

    from_id = resolve_category_id(categories, args.from_category)
    to_id = resolve_category_id(categories, args.to_category)
    filter_id = None
    if args.filter_category:
        filter_id = resolve_category_id(categories, args.filter_category)

    if from_id is None:
        print(f"Error: --from-category '{args.from_category}' not found in categories.")
        sys.exit(2)
    if to_id is None:
        print(f"Error: --to-category '{args.to_category}' not found in categories.")
        sys.exit(2)
    if args.filter_category and filter_id is None:
        print(f"Error: --filter-category '{args.filter_category}' not found in categories.")
        sys.exit(2)

    # Build image_id -> set(category_ids)
    image_cats = defaultdict(set)
    for ann in annotations:
        image_cats[ann.get("image_id")].add(ann.get("category_id"))

    changed = 0
    total_candidates = 0

    # prepare for creating new annotation ids if copying
    max_ann_id = max((a.get("id", 0) for a in annotations), default=0)

    new_annotations = []
    for ann in annotations:
        # consider only annotations that have segmentation (segments)
        if not ann.get("segmentation"):
            continue
        if ann.get("category_id") != from_id:
            continue
        total_candidates += 1
        if filter_id is not None and filter_id not in image_cats.get(ann.get("image_id"), set()):
            continue  # skip because filter not present on image

        if args.copy:
            # duplicate annotation and assign new category and id
            max_ann_id += 1
            new_ann = copy.deepcopy(ann)
            new_ann["id"] = max_ann_id
            new_ann["category_id"] = to_id
            new_annotations.append(new_ann)
            changed += 1
        else:
            # modify in-place
            ann["category_id"] = to_id
            changed += 1

    if args.copy and new_annotations:
        annotations.extend(new_annotations)

    # write output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    action = "Copied" if args.copy else "Modified"
    print(f"{action} {changed} annotations (candidates: {total_candidates}).")


if __name__ == "__main__":
    main()
