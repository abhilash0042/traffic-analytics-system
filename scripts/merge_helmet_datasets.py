"""Merge, Deduplicate and Rebalance Helmet Detection Datasets into 80/10/10 Split.

Sources merged:
- data/datasets/helmet_detection (41,140 images)
- data/datasets/helmet_detection_new (3,070 images)

Output:
- data/datasets/helmet_combined (train/valid/test with 80/10/10 balance)
"""

import os
import sys
import hashlib
import random
import shutil
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path("c:/projects/traffic-analytics-system")
DATASETS_DIR = ROOT_DIR / "data" / "datasets"
TARGET_DIR = DATASETS_DIR / "helmet_combined"
MANIFEST_FILE = DATASETS_DIR / "manifest.json"


def get_md5(file_path):
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def merge_and_rebalance_helmet():
    sources = [
        DATASETS_DIR / "helmet_detection",
        DATASETS_DIR / "helmet_detection_new"
    ]

    logger.info("Scanning helmet detection sources for image/label pairs...")
    seen_hashes = {}
    valid_pairs = []

    for src in sources:
        if not src.exists():
            continue
        for split in ["train", "valid", "test", "val"]:
            img_dir = src / split / "images"
            lbl_dir = src / split / "labels"
            if not img_dir.exists():
                continue

            for img_path in img_dir.glob("*.*"):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    continue

                file_hash = get_md5(img_path)
                if file_hash in seen_hashes:
                    continue  # Skip exact duplicate
                seen_hashes[file_hash] = img_path
                valid_pairs.append((img_path, lbl_path))

    logger.info(f"Total unique, valid helmet pairs found: {len(valid_pairs)}")

    # Shuffle with deterministic seed
    random.seed(42)
    random.shuffle(valid_pairs)

    n_total = len(valid_pairs)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.10)
    n_test = n_total - n_train - n_val

    splits = {
        "train": valid_pairs[:n_train],
        "valid": valid_pairs[n_train:n_train + n_val],
        "test": valid_pairs[n_train + n_val:]
    }

    logger.info(f"Target split sizes: Train={len(splits['train'])}, Valid={len(splits['valid'])}, Test={len(splits['test'])}")

    # Prepare target directories
    for split_name, pairs in splits.items():
        dst_img = TARGET_DIR / split_name / "images"
        dst_lbl = TARGET_DIR / split_name / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing {len(pairs)} pairs to {split_name} split...")
        for i, (img_src, lbl_src) in enumerate(pairs):
            dst_name = f"helmet_{split_name}_{i:06d}{img_src.suffix}"
            dst_lbl_name = f"helmet_{split_name}_{i:06d}.txt"
            shutil.copy2(img_src, dst_img / dst_name)
            shutil.copy2(lbl_src, dst_lbl / dst_lbl_name)

    # Write data.yaml
    yaml_content = f"""path: {TARGET_DIR.as_posix()}
train: train/images
val: valid/images
test: test/images
nc: 2
names:
  - helmet
  - no_helmet
"""
    with open(TARGET_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    logger.info(f"Wrote data.yaml to {TARGET_DIR / 'data.yaml'}")

    # Update manifest
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r") as f:
            manifest = json.load(f)
        manifest["merged_datasets"]["helmet_combined"] = {
            "path": str(TARGET_DIR),
            "total_images": n_total,
            "train": n_train,
            "valid": n_val,
            "test": n_test,
            "nc": 2,
            "names": ["helmet", "no_helmet"]
        }
        with open(MANIFEST_FILE, "w") as f:
            json.dump(manifest, f, indent=2)

    logger.info("Helmet dataset merge & rebalancing complete!")


if __name__ == "__main__":
    merge_and_rebalance_helmet()
