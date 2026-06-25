"""
Add blur / motion / low-light / noise variants to license plate training data.

Indian dashcam and CCTV footage often has motion blur, rain, night glare, and
compressed streams. This script duplicates a subset of train images with those
degradations so the detector learns to find plates in hard conditions.

Usage:
    python scripts/augment_plates_indian.py
    python scripts/augment_plates_indian.py --ratio 0.4
    python scripts/augment_plates_indian.py --dry-run

After augmenting, retrain:
    python train_models.py --plates --continue-best
    # or fresh: python train_models.py --plates --force
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_utils import load_config, resolve_path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def motion_blur(image: np.ndarray, size: int = 11) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0 / size
    return cv2.filter2D(image, -1, kernel)


def gaussian_blur(image: np.ndarray, k: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(image, (k, k), 0)


def low_light(image: np.ndarray, factor: float = 0.45) -> np.ndarray:
    dark = (image.astype(np.float32) * factor).astype(np.uint8)
    lab = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def add_noise(image: np.ndarray, sigma: float = 18.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def jpeg_artifacts(image: np.ndarray, quality: int = 35) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def rain_overlay(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    for _ in range(120):
        x1 = random.randint(0, w - 1)
        y1 = random.randint(0, h - 1)
        length = random.randint(8, 18)
        x2 = min(w - 1, x1 + random.randint(-2, 2))
        y2 = min(h - 1, y1 + length)
        cv2.line(out, (x1, y1), (x2, y2), (200, 200, 200), 1)
    return cv2.addWeighted(image, 0.85, out, 0.15, 0)


AUGMENTATIONS = {
    "blur": lambda img: gaussian_blur(img, 5),
    "motion": lambda img: motion_blur(img, 13),
    "night": lambda img: low_light(img, 0.42),
    "noise": lambda img: add_noise(img, 16.0),
    "jpeg": lambda img: jpeg_artifacts(img, 30),
    "rain": rain_overlay,
}


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def augment_dataset(dataset_dir: Path, ratio: float, dry_run: bool) -> int:
    train_images = dataset_dir / "images" / "train"
    train_labels = dataset_dir / "labels" / "train"
    if not train_images.is_dir():
        print(f"Missing train images: {train_images}")
        return 0

    images = sorted(
        path for path in train_images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print("No training images found.")
        return 0

    sample_count = max(1, int(len(images) * ratio))
    chosen = random.sample(images, min(sample_count, len(images)))
    created = 0

    print(f"Dataset : {dataset_dir}")
    print(f"Train   : {len(images)} images")
    print(f"Augment : {len(chosen)} images x {len(AUGMENTATIONS)} variants")

    for image_path in chosen:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        label_src = label_path_for(image_path)
        if not label_src.is_file():
            alt = train_labels / f"{image_path.stem}.txt"
            label_src = alt if alt.is_file() else None
        if label_src is None:
            continue

        stem = image_path.stem
        for suffix, fn in AUGMENTATIONS.items():
            out_name = f"{stem}_ind_{suffix}{image_path.suffix.lower()}"
            out_image = train_images / out_name
            out_label = train_labels / f"{stem}_ind_{suffix}.txt"

            if out_image.exists() and out_label.exists():
                continue

            augmented = fn(image)
            if dry_run:
                print(f"  would create {out_name}")
                created += 1
                continue

            cv2.imwrite(str(out_image), augmented)
            shutil.copy2(label_src, out_label)
            created += 1

    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment Indian plate train set for hard video conditions")
    parser.add_argument("--ratio", type=float, default=0.35, help="Fraction of train images to augment (default 0.35)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned files only")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    config = load_config()
    dataset_dir = resolve_path(config["training"]["plate"]["dataset_dir"])

    created = augment_dataset(dataset_dir, args.ratio, args.dry_run)
    if args.dry_run:
        print(f"\nDry run — would create {created} augmented image/label pairs.")
        return

    print(f"\nCreated {created} augmented image/label pairs.")
    print("Retrain with harder conditions:")
    print("  python train_models.py --plates --continue-best")


if __name__ == "__main__":
    main()
