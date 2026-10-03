import os
import sys
import glob
import json
import yaml
from pathlib import Path
from collections import Counter, defaultdict

# Fix UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data" / "datasets"

print("================================================================================")
print("             IN-DEPTH LOCAL & HUGGING FACE DATASET AUDIT                       ")
print("================================================================================")
print(f"Data Root: {DATA_DIR}\n")

# Catalog definitions from publish_to_hf.py
from scripts.publish_to_hf import DATASET_CATALOG

local_audit = {}

def audit_yolo_dataset(name, path):
    info = {
        "type": "yolo_detection",
        "path": str(path),
        "exists": path.exists(),
        "splits": {},
        "yaml_config": None,
        "classes_in_yaml": [],
        "total_images": 0,
        "total_labels": 0,
        "total_boxes": 0,
        "class_distribution": Counter(),
        "issues": []
    }
    
    if not path.exists():
        info["issues"].append("Dataset directory does not exist locally.")
        return info
        
    yaml_path = path / "data.yaml"
    if yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                y_data = yaml.safe_load(f)
                info["yaml_config"] = y_data
                names = y_data.get("names", [])
                if isinstance(names, dict):
                    names = [names[k] for k in sorted(names.keys())]
                info["classes_in_yaml"] = names
        except Exception as e:
            info["issues"].append(f"Error reading data.yaml: {e}")
    else:
        info["issues"].append("Missing data.yaml")

    # Detect splits
    possible_splits = ["train", "val", "valid", "validation", "test"]
    found_splits = []
    
    # Check directory structure: either split/images + split/labels or images/split + labels/split or top-level images/labels
    for s in possible_splits:
        img_dir1 = path / s / "images"
        lbl_dir1 = path / s / "labels"
        img_dir2 = path / "images" / s
        lbl_dir2 = path / "labels" / s
        img_dir3 = path / s
        
        if img_dir1.exists():
            found_splits.append((s, img_dir1, lbl_dir1))
        elif img_dir2.exists():
            found_splits.append((s, img_dir2, lbl_dir2))
        elif img_dir3.exists() and any(img_dir3.glob("*.jpg")) or any(img_dir3.glob("*.png")):
            found_splits.append((s, img_dir3, path / "labels" / s if (path / "labels" / s).exists() else img_dir3))

    if not found_splits:
        # Check flat images/labels
        img_flat = path / "images"
        lbl_flat = path / "labels"
        if img_flat.exists():
            found_splits.append(("all", img_flat, lbl_flat))
        else:
            # Check if root has images
            imgs = list(path.glob("*.jpg")) + list(path.glob("*.png"))
            if imgs:
                found_splits.append(("root", path, path))

    num_classes = len(info["classes_in_yaml"]) if info["classes_in_yaml"] else None

    for split_name, img_dir, lbl_dir in found_splits:
        image_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        images = []
        for ext in image_exts:
            images.extend(list(img_dir.glob(ext)))
            images.extend(list(img_dir.glob(ext.upper())))
            
        labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
        
        split_boxes = 0
        split_class_dist = Counter()
        split_corrupt_lines = 0
        split_oob_boxes = 0
        split_invalid_classes = 0
        split_empty_labels = 0
        missing_label_count = 0
        
        img_stems = {img.stem: img for img in images}
        lbl_stems = {lbl.stem: lbl for lbl in labels}
        
        # Check images without labels
        for stem in img_stems:
            if stem not in lbl_stems:
                missing_label_count += 1

        # Check labels content
        for lbl_file in labels:
            try:
                content = lbl_file.read_text(encoding="utf-8").strip()
                if not content:
                    split_empty_labels += 1
                    continue
                for line_idx, line in enumerate(content.splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        split_corrupt_lines += 1
                        continue
                    cls_str, x_str, y_str, w_str, h_str = parts
                    try:
                        cls_id = int(cls_str)
                        x, y, w, h = float(x_str), float(y_str), float(w_str), float(h_str)
                    except ValueError:
                        split_corrupt_lines += 1
                        continue
                        
                    # Check class ID validity
                    if num_classes is not None and (cls_id < 0 or cls_id >= num_classes):
                        split_invalid_classes += 1
                        
                    # Check bounding box bounds [0, 1]
                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                        split_oob_boxes += 1
                    elif (x - w/2 < -0.1) or (y - h/2 < -0.1) or (x + w/2 > 1.1) or (y + h/2 > 1.1):
                        split_oob_boxes += 1
                        
                    split_boxes += 1
                    split_class_dist[cls_id] += 1
                    info["class_distribution"][cls_id] += 1
                    
            except Exception as e:
                split_corrupt_lines += 1

        info["splits"][split_name] = {
            "image_dir": str(img_dir),
            "label_dir": str(lbl_dir),
            "image_count": len(images),
            "label_count": len(labels),
            "box_count": split_boxes,
            "images_without_label_file": missing_label_count,
            "empty_label_files": split_empty_labels,
            "corrupt_lines": split_corrupt_lines,
            "out_of_bounds_boxes": split_oob_boxes,
            "invalid_class_ids": split_invalid_classes,
            "class_distribution": dict(split_class_dist)
        }
        
        info["total_images"] += len(images)
        info["total_labels"] += len(labels)
        info["total_boxes"] += split_boxes

        if split_corrupt_lines > 0:
            info["issues"].append(f"Split '{split_name}' has {split_corrupt_lines} corrupt/malformed label lines.")
        if split_invalid_classes > 0:
            info["issues"].append(f"Split '{split_name}' has {split_invalid_classes} bounding boxes with class IDs outside defined names ({num_classes}).")
        if split_oob_boxes > 0:
            info["issues"].append(f"Split '{split_name}' has {split_oob_boxes} bounding boxes with coordinates outside [0, 1].")

    return info


def audit_ocr_dataset(name, path):
    info = {
        "type": "ocr_text",
        "path": str(path),
        "exists": path.exists(),
        "gt_file": None,
        "total_samples": 0,
        "image_count": 0,
        "missing_images": 0,
        "char_distribution": Counter(),
        "char_set": set(),
        "empty_transcripts": 0,
        "issues": []
    }
    
    if not path.exists():
        info["issues"].append("Dataset directory does not exist locally.")
        return info

    gt_candidates = list(path.glob("gt.txt")) + list(path.glob("labels.txt")) + list(path.glob("train.txt")) + list(path.glob("*.csv")) + list(path.glob("*.json"))
    
    # Check for lmdb
    lmdb_data = list(path.glob("*.mdb")) + list(path.glob("data.mdb"))
    if lmdb_data:
        info["has_lmdb"] = True
        info["lmdb_files"] = [str(f.name) for f in lmdb_data]

    # Count images in folder
    image_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    images = []
    for ext in image_exts:
        images.extend(list(path.rglob(ext)))
    info["image_count"] = len(images)

    # If gt.txt exists
    gt_file = path / "gt.txt"
    if gt_file.exists():
        info["gt_file"] = str(gt_file)
        try:
            lines = gt_file.read_text(encoding="utf-8").splitlines()
            info["total_samples"] = len(lines)
            for idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t") if "\t" in line else line.split()
                if len(parts) < 2:
                    info["empty_transcripts"] += 1
                    continue
                img_name = parts[0]
                text = parts[1].strip().upper()
                if not text:
                    info["empty_transcripts"] += 1
                for c in text:
                    info["char_distribution"][c] += 1
                    info["char_set"].add(c)
        except Exception as e:
            info["issues"].append(f"Error parsing gt.txt: {e}")
    else:
        # Check sub-splits (train, val, test)
        for s in ["train", "val", "test"]:
            s_gt = path / s / "gt.txt"
            if s_gt.exists():
                lines = s_gt.read_text(encoding="utf-8").splitlines()
                info[f"{s}_samples"] = len(lines)
                info["total_samples"] += len(lines)
                for line in lines:
                    parts = line.strip().split("\t") if "\t" in line else line.strip().split()
                    if len(parts) >= 2:
                        text = parts[1].strip().upper()
                        for c in text:
                            info["char_distribution"][c] += 1
                            info["char_set"].add(c)

    info["char_set"] = sorted(list(info["char_set"]))
    info["char_distribution"] = dict(info["char_distribution"])
    return info


def audit_video_dataset(name, path):
    info = {
        "type": "video",
        "path": str(path),
        "exists": path.exists(),
        "video_count": 0,
        "total_video_bytes": 0,
        "xml_annotations": 0,
        "issues": []
    }
    if not path.exists():
        info["issues"].append("Directory does not exist.")
        return info
        
    videos = list(path.glob("*.mp4")) + list(path.glob("*.avi")) + list(path.glob("*.mkv"))
    info["video_count"] = len(videos)
    info["total_video_bytes"] = sum(v.stat().st_size for v in videos)
    
    xml_dir = PROJECT_ROOT / "data" / "datasets" / "videoset1_xml"
    if xml_dir.exists():
        xmls = list(xml_dir.glob("*.xml"))
        info["xml_annotations"] = len(xmls)
        info["xml_dir"] = str(xml_dir)
    else:
        info["xml_annotations"] = 0
        
    return info


print("\n>>> AUDITING ALL 10 LOCAL DATASETS <<<\n")

for cat_key, meta in DATASET_CATALOG.items():
    repo_name = meta["repo_name"]
    source_p = meta["source_path"]
    print(f"Auditing [{cat_key}] -> {source_p.name}...")
    
    task_cat = meta.get("task_categories", ["object-detection"])[0]
    if task_cat == "object-detection":
        audit_res = audit_yolo_dataset(cat_key, source_p)
    elif task_cat == "image-to-text":
        audit_res = audit_ocr_dataset(cat_key, source_p)
    elif task_cat == "video-classification":
        audit_res = audit_video_dataset(cat_key, source_p)
    else:
        audit_res = {"path": str(source_p), "exists": source_p.exists()}
        
    audit_res["repo_name"] = repo_name
    audit_res["title"] = meta["title"]
    local_audit[cat_key] = audit_res

# Save local audit to json
with open("local_dataset_audit.json", "w", encoding="utf-8") as f:
    json.dump(local_audit, f, indent=2)

print("\nSaved local dataset audit to local_dataset_audit.json")
