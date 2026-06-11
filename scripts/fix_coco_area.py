#!/usr/bin/env python3
"""
Add missing 'area' and 'iscrowd' fields to COCO-style annotation files.
Usage:
  python scripts/fix_coco_area.py path/to/annotations.json [more files...]
If no files provided, will try common paths under the repo.
"""
import json
import os
import sys

def poly_area(seg):
    def part_area(coords):
        xs = coords[0::2]
        ys = coords[1::2]
        s = 0.0
        for i in range(len(xs)):
            j = (i+1) % len(xs)
            s += xs[i]*ys[j] - xs[j]*ys[i]
        return abs(s) / 2.0
    if not seg:
        return 0.0
    if isinstance(seg[0], list):
        return sum(part_area(p) for p in seg if p)
    else:
        return part_area(seg)


def fix_file(path):
    print('Processing', path)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    anns = data.get('annotations')
    if not isinstance(anns, list):
        print('  no annotations list, skipping')
        return False
    changed = False
    for ann in anns:
        if 'area' not in ann:
            if 'bbox' in ann and ann.get('bbox'):
                _,_,w,h = ann['bbox'][:4]
                try:
                    ann['area'] = float(w*h)
                except Exception:
                    ann['area'] = 0.0
            elif 'segmentation' in ann and ann.get('segmentation'):
                try:
                    ann['area'] = float(poly_area(ann['segmentation']))
                except Exception:
                    ann['area'] = 0.0
            else:
                ann['area'] = 0.0
            changed = True
        if 'iscrowd' not in ann:
            ann['iscrowd'] = 0
            changed = True
    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print('  patched', path)
    else:
        print('  already OK')
    return changed


def main():
    paths = sys.argv[1:]
    if not paths:
        candidate = [
            'output_annotations/train_polygons_clean.json',
            'output_annotations/val_polygons_clean.json',
            'dataset/annotations/train.json',
            'dataset/annotations/val.json',
            'dataset/annotations/instances_train.json',
            'dataset/annotations/instances_val.json',
        ]
        paths = [p for p in candidate if os.path.exists(p)]
        if not paths:
            print('No files specified and no default annotation files found.')
            print('Usage: python scripts/fix_coco_area.py path/to/ann.json [more]')
            sys.exit(1)
    any_changed = False
    for p in paths:
        try:
            if fix_file(p):
                any_changed = True
        except Exception as e:
            print('ERROR processing', p, e)
    if any_changed:
        print('Done: files updated.')
    else:
        print('Done: nothing to change.')

if __name__ == '__main__':
    main()
