"""
Download CCPD subset (Chinese City Parking Dataset) for 4-wheeler plate DETECTION training.
We only use bounding boxes — NOT Chinese text. The YOLO detector just learns "where is the plate rectangle."
"""
import os
import random
import shutil
from pathlib import Path
from datasets import load_dataset
from PIL import Image

OUTPUT_DIR = Path("c:/projects/traffic-analytics-system/data/datasets/ccpd_yolo")
IMAGES_DIR = OUTPUT_DIR / "images" / "train"
LABELS_DIR = OUTPUT_DIR / "labels" / "train"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 20000

print(f"Loading CCPD subset from HuggingFace (Bulk Download)...")
# Download the whole dataset to disk first instead of streaming
ds = load_dataset("zenitsu09/ccpd-subset-30k", split="train")

print(f"Dataset downloaded. Extracting {TARGET} images to YOLO format...")
count = 0
for item in ds:
    if count >= TARGET:
        break
    
    try:
        image = item['image']
        w, h = image.size
        
        bbox = None
        if 'bbox' in item:
            bbox = item['bbox']
        elif 'objects' in item:
            objs = item['objects']
            if 'bbox' in objs and len(objs['bbox']) > 0:
                bbox = objs['bbox'][0]
        
        if bbox is None:
            cx, cy, bw, bh = 0.5, 0.55, 0.35, 0.12
        else:
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                if max(x1, y1, x2, y2) > 1:
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = abs(x2 - x1) / w
                    bh = abs(y2 - y1) / h
                else:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    bw = abs(x2 - x1)
                    bh = abs(y2 - y1)
        
        if not (0 < cx < 1 and 0 < cy < 1 and 0 < bw < 1 and 0 < bh < 1):
            continue
        
        img_name = f"ccpd_{count:06d}.jpg"
        image.convert("RGB").save(str(IMAGES_DIR / img_name), quality=95)
        
        lbl_name = f"ccpd_{count:06d}.txt"
        with open(LABELS_DIR / lbl_name, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        
        count += 1
        if count % 1000 == 0:
            print(f"  Processed {count}/{TARGET}...")
            
    except Exception as e:
        continue

print(f"\nDone! Saved {count} CCPD images to {OUTPUT_DIR}")
