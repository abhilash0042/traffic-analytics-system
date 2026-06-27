"""
Validate and fix vehicle_detection dataset paths for local YOLO training.

Usage:
    python scripts/prepare_vehicle_dataset.py
    python download_datasets.py --vehicles   # fetch dataset first
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_utils import load_config, resolve_path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LOCAL_DATA_YAML = """path: {root}
train: train/images
val: valid/images
test: test/images
nc: 4
names:
  - bus
  - car
  - truck
  - van
"""


def count_split(root: Path, split: str) -> tuple[int, int]:
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    imgs = sum(1 for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if img_dir.is_dir() else 0
    lbls = sum(1 for p in lbl_dir.iterdir() if p.suffix == ".txt") if lbl_dir.is_dir() else 0
    return imgs, lbls


def main() -> None:
    config = load_config()
    root = resolve_path(config["training"]["vehicle"]["dataset_dir"])

    print("\n" + "=" * 60)
    print("PREPARE VEHICLE DATASET (LOCAL TRAINING)")
    print("=" * 60)

    if not (root / "train" / "images").is_dir():
        print(f"  [!!] Missing dataset at {root}")
        print("  Run: python download_datasets.py --vehicles")
        print("  Or:  https://universe.roboflow.com/model-rli8w/ua-detrac-10k-sample-znazr")
        raise SystemExit(1)

    totals = {}
    for split in ("train", "valid", "test"):
        totals[split] = count_split(root, split)
        print(f"  [OK] {split}: {totals[split][0]} images, {totals[split][1]} labels")

    yaml_path = root / "data.yaml"
    yaml_path.write_text(LOCAL_DATA_YAML.format(root=root.as_posix()), encoding="utf-8")
    print(f"  [OK] Wrote {yaml_path}")

    print("\n  Classes: bus(0), car(1), truck(2), van(3)")
    print("  Pipeline maps these to COCO-like IDs; motorcycles use yolo11s fallback.")
    print("\n  Train locally:")
    print("    python train_models.py --vehicles")
    print("\n  Or pack for Kaggle:")
    print("    python scripts/pack_vehicle_for_kaggle.py")


if __name__ == "__main__":
    main()
