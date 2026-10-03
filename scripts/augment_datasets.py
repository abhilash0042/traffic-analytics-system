"""Fast Domain-Specific Offline Dataset Augmentation Engine.

Applies realistic road-camera and dashcam physical augmentations:
- Motion blur (vehicle speed)
- JPEG compression artifacts (dashcam encoder artifacts)
- Rain & lens distortion
- Photometric / HSV color jitter
- Gaussian sensor noise (low-light CCTV)

All augmented datasets are saved to `data/datasets/augmented/` with clear
manifest tracking so they can be reviewed or deleted with a single command.
"""

import os
import sys
import math
import random
import json
import logging
from pathlib import Path
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path("c:/projects/traffic-analytics-system")
DATASETS_DIR = ROOT_DIR / "data" / "datasets"
AUG_BASE_DIR = DATASETS_DIR / "augmented"
MANIFEST_FILE = DATASETS_DIR / "manifest.json"


def apply_motion_blur(img: np.ndarray, max_kernel: int = 9) -> np.ndarray:
    """Applies fast directional motion blur."""
    size = random.choice([5, 7, 9])
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0 / size
    return cv2.filter2D(img, -1, kernel)


def apply_jpeg_compression(img: np.ndarray, min_q: int = 25, max_q: int = 60) -> np.ndarray:
    """Simulates severe video codec compression artifacts."""
    q = random.randint(min_q, max_q)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def apply_gaussian_noise(img: np.ndarray, sigma_range=(10, 25)) -> np.ndarray:
    """Simulates low-light high-ISO camera sensor noise."""
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_hsv_jitter(img: np.ndarray, h_gain=12, s_gain=25, v_gain=30) -> np.ndarray:
    """Alters lighting, hue, and saturation realistically."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-h_gain, h_gain)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(1.0 - s_gain / 100.0, 1.0 + s_gain / 100.0), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(1.0 - v_gain / 100.0, 1.0 + v_gain / 100.0), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_image(img: np.ndarray) -> np.ndarray:
    choice = random.random()
    if choice < 0.35:
        aug_img = apply_motion_blur(img)
    elif choice < 0.70:
        aug_img = apply_jpeg_compression(img)
    else:
        aug_img = apply_gaussian_noise(img)

    if random.random() < 0.5:
        aug_img = apply_hsv_jitter(aug_img)
    return aug_img


def augment_plate_detection(multiplier: int = 2, max_samples: int = 800):
    """Augments plate detection dataset and writes to isolated directory."""
    src_dir = DATASETS_DIR / "plate_detection_combined" / "train"

    target_dir = AUG_BASE_DIR / "plate_detection"
    target_img_dir = target_dir / "images"
    target_lbl_dir = target_dir / "labels"
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = list(src_dir.rglob("*.jpg")) + list(src_dir.rglob("*.png"))
    if not img_files:
        logger.warning(f"No source images found in {src_dir}")
        return

    random.seed(42)
    random.shuffle(img_files)
    img_files = img_files[:max_samples]

    logger.info(f"Augmenting {len(img_files)} plate detection images (multiplier x{multiplier})...")
    created_count = 0

    for img_path in img_files:
        lbl_path = DATASETS_DIR / "plate_detection_combined" / "train" / "labels" / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        lbl_content = lbl_path.read_text(encoding="utf-8")
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for m in range(multiplier):
            aug = augment_image(img)
            aug_name = f"aug_{img_path.stem}_m{m}{img_path.suffix}"
            aug_lbl_name = f"aug_{img_path.stem}_m{m}.txt"

            cv2.imwrite(str(target_img_dir / aug_name), aug, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            (target_lbl_dir / aug_lbl_name).write_text(lbl_content, encoding="utf-8")
            created_count += 1

    # Write data.yaml
    yaml_content = f"""path: {target_dir.as_posix()}
train: images
val: images
nc: 1
names:
  - license_plate
"""
    (target_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")
    logger.info(f"Generated {created_count} augmented plate detection images in {target_dir}")
    update_manifest("augmented_plate_detection", target_dir, created_count)


def augment_synthetic_ocr(multiplier: int = 2, max_samples: int = 2500):
    """Augments synthetic OCR plates with degradation artifacts."""
    src_dir = DATASETS_DIR / "kaggle_synthetic"
    if not src_dir.exists():
        logger.warning("kaggle_synthetic not found for OCR augmentation.")
        return

    target_dir = AUG_BASE_DIR / "ocr_synthetic"
    target_img_dir = target_dir / "images"
    target_img_dir.mkdir(parents=True, exist_ok=True)

    img_files = list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg"))
    if not img_files:
        return

    random.seed(42)
    random.shuffle(img_files)
    img_files = img_files[:max_samples]

    logger.info(f"Augmenting {len(img_files)} synthetic OCR crops (multiplier x{multiplier})...")
    gt_lines = []
    created_count = 0

    for img_path in img_files:
        text_label = img_path.stem.split("_")[0]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for m in range(multiplier):
            aug = augment_image(img)
            aug_name = f"ocr_aug_{img_path.stem}_m{m}.jpg"
            cv2.imwrite(str(target_img_dir / aug_name), aug, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            gt_lines.append(f"{aug_name}\t{text_label}")
            created_count += 1

    gt_file = target_dir / "gt.txt"
    gt_file.write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    logger.info(f"Generated {created_count} augmented OCR crops in {target_dir}")
    update_manifest("augmented_ocr_synthetic", target_dir, created_count)


def update_manifest(key: str, target_dir: Path, count: int):
    manifest = {}
    if MANIFEST_FILE.exists():
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            pass

    if "augmented_datasets" not in manifest:
        manifest["augmented_datasets"] = {}

    total_size_mb = sum(f.stat().st_size for f in target_dir.rglob('*') if f.is_file()) / (1024 * 1024)
    manifest["augmented_datasets"][key] = {
        "path": str(target_dir),
        "count": count,
        "size_mb": round(total_size_mb, 2),
        "removable": True,
        "rollback_command": "python scripts/manage_datasets.py --delete-augmented"
    }

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Updated manifest for {key} ({total_size_mb:.1f} MB)")


if __name__ == "__main__":
    augment_plate_detection(multiplier=2, max_samples=800)
    augment_synthetic_ocr(multiplier=2, max_samples=2500)
