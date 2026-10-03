"""Merge, Clean and Deduplicate License Plate Detection Datasets into 80/10/10 Split.

Sources merged:
- data/datasets/perfect_indian_license_plates (2,298 images, YOLO)
- data/datasets/dashcop_yolo (3,034 images, YOLO)
- data/datasets/kaggle_indian_plates (47 images, Pascal VOC -> YOLO converted)
- data/datasets/kaggle_plates3_rescued (rescued labeled pairs, YOLO)

Output:
- data/datasets/plate_detection_combined (train/valid/test with 80/10/10 split)
"""

import os
import sys
import hashlib
import random
import shutil
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path("c:/projects/traffic-analytics-system")
DATASETS_DIR = ROOT_DIR / "data" / "datasets"
TARGET_DIR = DATASETS_DIR / "plate_detection_combined"
MANIFEST_FILE = DATASETS_DIR / "manifest.json"


def get_md5(file_path):
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def convert_voc_xml_to_yolo(xml_path, img_path):
    """Converts a Pascal VOC XML file to YOLO format lines: '0 x_center y_center width height'."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_node = root.find("size")
        if size_node is not None:
            w = float(size_node.find("width").text)
            h = float(size_node.find("height").text)
        else:
            with Image.open(img_path) as img:
                w, h = img.size

        if w <= 0 or h <= 0:
            return []

        yolo_lines = []
        for obj in root.findall("object"):
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            # Clamp
            xmin = max(0.0, min(w, xmin))
            xmax = max(0.0, min(w, xmax))
            ymin = max(0.0, min(h, ymin))
            ymax = max(0.0, min(h, ymax))

            bw = xmax - xmin
            bh = ymax - ymin
            if bw <= 1 or bh <= 1:
                continue

            x_center = (xmin + bw / 2.0) / w
            y_center = (ymin + bh / 2.0) / h
            norm_w = bw / w
            norm_h = bh / h

            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
        return yolo_lines
    except Exception as e:
        logger.warning(f"Error converting XML {xml_path}: {e}")
        return []


def merge_and_rebalance_plates():
    seen_hashes = {}
    valid_pairs = []  # list of tuples (img_path, label_content_lines or lbl_path)

    # 1. perfect_indian_license_plates
    p_src = DATASETS_DIR / "perfect_indian_license_plates"
    if p_src.exists():
        logger.info(f"Scanning {p_src}...")
        for img_path in (p_src / "images").rglob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            lbl_path = p_src / "labels" / img_path.parent.name / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                lbl_path = p_src / "labels" / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            h = get_md5(img_path)
            if h in seen_hashes:
                continue
            seen_hashes[h] = img_path
            valid_pairs.append(("yolo_file", img_path, lbl_path))

    # 2. dashcop_yolo
    d_src = DATASETS_DIR / "dashcop_yolo"
    if d_src.exists():
        logger.info(f"Scanning {d_src}...")
        for img_path in (d_src / "images").rglob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            rel_dir = img_path.parent.name
            lbl_path = d_src / "labels" / rel_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                lbl_path = d_src / "labels" / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            h = get_md5(img_path)
            if h in seen_hashes:
                continue
            seen_hashes[h] = img_path
            valid_pairs.append(("yolo_file", img_path, lbl_path))

    # 3. kaggle_indian_plates (VOC XMLs)
    k_src = DATASETS_DIR / "kaggle_indian_plates"
    if k_src.exists():
        logger.info(f"Scanning {k_src} for VOC XML conversions...")
        for img_path in k_src.rglob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            # Search for corresponding XML
            xml_candidates = list(k_src.rglob(f"{img_path.stem}.xml"))
            if not xml_candidates:
                continue
            xml_path = xml_candidates[0]
            h = get_md5(img_path)
            if h in seen_hashes:
                continue
            yolo_lines = convert_voc_xml_to_yolo(xml_path, img_path)
            if yolo_lines:
                seen_hashes[h] = img_path
                valid_pairs.append(("yolo_lines", img_path, yolo_lines))

    # 4. kaggle_plates3_rescued
    r_src = DATASETS_DIR / "kaggle_plates3_rescued"
    if r_src.exists():
        logger.info(f"Scanning {r_src}...")
        for img_path in (r_src / "images").glob("*.*"):
            lbl_path = r_src / "labels" / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            h = get_md5(img_path)
            if h in seen_hashes:
                continue
            seen_hashes[h] = img_path
            valid_pairs.append(("yolo_file", img_path, lbl_path))

    logger.info(f"Total unique plate detection samples collected: {len(valid_pairs)}")

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

    logger.info(f"Plate splits: Train={len(splits['train'])}, Valid={len(splits['valid'])}, Test={len(splits['test'])}")

    for split_name, items in splits.items():
        dst_img = TARGET_DIR / split_name / "images"
        dst_lbl = TARGET_DIR / split_name / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        for i, item in enumerate(items):
            mode = item[0]
            img_src = item[1]
            dst_name = f"plate_{split_name}_{i:06d}{img_src.suffix}"
            dst_lbl_name = f"plate_{split_name}_{i:06d}.txt"

            shutil.copy2(img_src, dst_img / dst_name)

            if mode == "yolo_file":
                lbl_src = item[2]
                shutil.copy2(lbl_src, dst_lbl / dst_lbl_name)
            else:
                lines = item[2]
                with open(dst_lbl / dst_lbl_name, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")

    # Write data.yaml
    yaml_content = f"""path: {TARGET_DIR.as_posix()}
train: train/images
val: valid/images
test: test/images
nc: 1
names:
  - license_plate
"""
    with open(TARGET_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    logger.info(f"Wrote data.yaml to {TARGET_DIR / 'data.yaml'}")

    # Update manifest
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r") as f:
            manifest = json.load(f)
        manifest["merged_datasets"]["plate_detection_combined"] = {
            "path": str(TARGET_DIR),
            "total_images": n_total,
            "train": n_train,
            "valid": n_val,
            "test": n_test,
            "nc": 1,
            "names": ["license_plate"]
        }
        with open(MANIFEST_FILE, "w") as f:
            json.dump(manifest, f, indent=2)

    logger.info("Plate detection dataset merge & rebalance complete!")


if __name__ == "__main__":
    merge_and_rebalance_plates()
