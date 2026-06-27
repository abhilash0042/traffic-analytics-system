"""
Pack UA-DETRAC vehicle_detection dataset for Kaggle upload.

Dataset: ~20K train / ~2K valid / ~1K test (bus, car, truck, van).
Source: data/datasets/vehicle_detection (Roboflow UA-DETRAC 10K sample).

Usage:
    python scripts/pack_vehicle_for_kaggle.py
    python scripts/pack_vehicle_for_kaggle.py --output dist/vehicle_detection_kaggle.zip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_utils import load_config, resolve_path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KAGGLE_DATA_YAML = """path: /kaggle/input/traffic-vehicle-balanced/vehicle_detection
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


def ok(msg: str) -> None:
    print(f"  [OK] {msg}", flush=True)


def info(msg: str) -> None:
    print(f"  [>>] {msg}", flush=True)


def count_split(src_root: Path, split: str) -> tuple[int, int]:
    img_n = lbl_n = 0
    img_dir = src_root / split / "images"
    lbl_dir = src_root / split / "labels"
    if img_dir.is_dir():
        img_n = sum(1 for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if lbl_dir.is_dir():
        lbl_n = sum(1 for p in lbl_dir.iterdir() if p.is_file() and p.suffix == ".txt")
    return img_n, lbl_n


def add_split_to_zip(zf: zipfile.ZipFile, src_root: Path, split: str, prefix: str) -> tuple[int, int]:
    added_img = added_lbl = 0
    for sub, is_image in (("images", True), ("labels", False)):
        folder = src_root / split / sub
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if not path.is_file():
                continue
            if is_image and path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if not is_image and path.suffix != ".txt":
                continue
            arc = f"{prefix}/{split}/{sub}/{path.name}"
            zf.write(path, arc)
            if is_image:
                added_img += 1
            else:
                added_lbl += 1
    return added_img, added_lbl


def main() -> None:
    parser = argparse.ArgumentParser(description="Zip vehicle dataset for Kaggle")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "dist" / "vehicle_detection_kaggle.zip"),
        help="Output zip path",
    )
    args = parser.parse_args()

    config = load_config()
    src = resolve_path(config["training"]["vehicle"]["dataset_dir"])
    out_zip = Path(args.output)

    if not (src / "train" / "images").is_dir():
        raise SystemExit(
            f"Dataset not found: {src}\n"
            "Run: python download_datasets.py --vehicles\n"
            "  (Roboflow API key) or extract UA-DETRAC YOLO zip manually."
        )

    print("\n" + "=" * 60)
    print("PACK VEHICLE DATASET FOR KAGGLE")
    print("=" * 60)
    info(f"Source: {src}")
    info(f"Output: {out_zip}")
    info("Classes: bus, car, truck, van (UA-DETRAC — Indian/highway traffic)")
    info("Note: motorcycles use COCO fallback in pipeline until you add a 5th class.")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.is_file():
        out_zip.unlink()

    totals: dict[str, tuple[int, int]] = {}
    for split in ("train", "valid", "test"):
        totals[split] = count_split(src, split)
        ok(f"{split}: {totals[split][0]} images, {totals[split][1]} labels")

    prefix = "vehicle_detection"
    info("Creating zip (direct from source, ~10–25 min for ~23K images)...")

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(f"{prefix}/data.yaml", KAGGLE_DATA_YAML)
        for split in ("train", "valid", "test"):
            img_n, lbl_n = add_split_to_zip(zf, src, split, prefix)
            info(f"  zipped {split}: {img_n} images, {lbl_n} labels")

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    ok(f"Created {out_zip.name} ({size_mb:.1f} MB)")

    manifest = out_zip.parent / "KAGGLE_VEHICLE_UPLOAD.txt"
    manifest.write_text(
        f"""KAGGLE UPLOAD — Traffic Vehicle Detection (UA-DETRAC 10K)
============================================================

WHY TRAIN THIS FIRST:
  Better car / bus / truck / van boxes -> cleaner vehicle crops -> better plate & tracking.
  Motorcycles still detected via COCO fallback (yolo11s) until you extend the dataset.

UPLOAD THESE TO KAGGLE:
-----------------------

1) DATASET (required)
   File: {out_zip.resolve()}
   Kaggle slug: traffic-vehicle-balanced

   Create zip (if missing):
     python scripts/pack_vehicle_for_kaggle.py

2) NOTEBOOK
   File: {PROJECT_ROOT / "notebooks" / "kaggle_vehicle_training.ipynb"}

3) WEIGHTS (optional — continue from checkpoint)
   File: {PROJECT_ROOT / "models" / "vehicle_detector.pt"}
   Kaggle slug: traffic-vehicle-weights

ZIP CONTENTS:
  vehicle_detection/data.yaml
  vehicle_detection/train/images/  ({totals['train'][0]} images)
  vehicle_detection/train/labels/  ({totals['train'][1]} labels)
  vehicle_detection/valid/images/  ({totals['valid'][0]} images)
  vehicle_detection/valid/labels/  ({totals['valid'][1]} labels)
  vehicle_detection/test/images/   ({totals['test'][0]} images)
  vehicle_detection/test/labels/   ({totals['test'][1]} labels)

KAGGLE STEPS:
  1. Create -> Dataset -> upload {out_zip.name}
  2. Create -> Notebook -> import kaggle_vehicle_training.ipynb
  3. Settings -> GPU T4 x2 if available else T4, Internet ON
  4. Add Input -> traffic-vehicle-balanced
  5. Save Version -> Save & Run All (expect ~4–8 hours for 100 epochs)
  6. Download vehicle_detector.pt -> models/vehicle_detector.pt

LOCAL TRAINING (RTX 4050):
  python scripts/prepare_vehicle_dataset.py
  python train_models.py --vehicles

Full guide: docs/KAGGLE_VEHICLE_TRAINING.md
""",
        encoding="utf-8",
    )
    ok(f"Wrote upload manifest -> {manifest.name}")


if __name__ == "__main__":
    main()
