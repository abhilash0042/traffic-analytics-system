"""
Prepare the OCR dataset for PARSeq training.
Removes unlabeled YOLO crops and creates train/val/test splits.
Creates a standard format that can be easily converted to LMDB.
"""
import os
import shutil
import random
from pathlib import Path

random.seed(42)

BASE = Path("c:/projects/traffic-analytics-system/data/datasets")
INPUT = BASE / "merged_ocr"
OUTPUT = BASE / "parseq_dataset"

def ensure_dirs():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    for split in ["train", "val", "test"]:
        (OUTPUT / split).mkdir(parents=True, exist_ok=True)

def process():
    ensure_dirs()
    
    files = list(INPUT.glob("*.jpg"))
    labeled_files = []
    
    for f in files:
        name = f.stem
        if name.startswith("perfect_") or name.startswith("lpindian_"):
            # Skip unlabeled YOLO crops
            continue
        
        # Extract plate text (first part of filename)
        text = name.split('_')[0].upper().replace(" ", "").replace("-", "")
        # Basic validation
        if len(text) < 6 or len(text) > 12:
            continue
            
        labeled_files.append((f, text))
        
    print(f"Total labeled files found: {len(labeled_files)}")
    
    random.shuffle(labeled_files)
    
    n = len(labeled_files)
    train_end = int(n * 0.85)
    val_end = int(n * 0.95)
    
    splits = {
        "train": labeled_files[:train_end],
        "val": labeled_files[train_end:val_end],
        "test": labeled_files[val_end:]
    }
    
    for split_name, split_pairs in splits.items():
        print(f"Copying {len(split_pairs)} pairs to {split_name} split...")
        gt_path = OUTPUT / split_name / "gt.txt"
        
        with open(gt_path, "w", encoding="utf-8") as f_gt:
            for i, (img_path, text) in enumerate(split_pairs):
                new_name = f"img_{i:06d}.jpg"
                out_img = OUTPUT / split_name / new_name
                shutil.copy(img_path, out_img)
                f_gt.write(f"{new_name}\t{text}\n")
                
    print("\nDataset split complete. Ready for LMDB conversion.")

if __name__ == "__main__":
    process()
