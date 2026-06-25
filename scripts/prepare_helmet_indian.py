"""
Prepare helmet dataset for Indian traffic / dashcam training.

  1. Normalize duplicate class IDs (0/2 -> helmet, 1/3 -> no_helmet)
  2. Fix data.yaml to 2 classes
  3. Optionally merge extra Roboflow download (helmet_detection_extra/)
  4. Add blur / night / rain augmentations for realistic CCTV conditions

Usage:
    python scripts/prepare_helmet_indian.py
    python scripts/prepare_helmet_indian.py --merge-extra
    python scripts/prepare_helmet_indian.py --augment-only --ratio 0.4

Then train once:
    python train_models.py --helmets
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_utils import load_config, resolve_path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Roboflow export had 4 duplicate names; HF/PPE ids map to 2-class head model.
CLASS_REMAP = {0: 0, 2: 0, 1: 1, 3: 1}

HELMET_KEYWORDS = ("helmet", "with helmet", "hardhat", "hard hat")
NO_HELMET_KEYWORDS = ("no helmet", "no_helmet", "without", "head", "no-helmet")


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def info(msg: str) -> None:
    print(f"  [>>] {msg}")


def split_dirs(dataset_dir: Path) -> dict[str, tuple[Path, Path]]:
    layouts = [
        ("train", dataset_dir / "train" / "images", dataset_dir / "train" / "labels"),
        ("valid", dataset_dir / "valid" / "images", dataset_dir / "valid" / "labels"),
        ("test", dataset_dir / "test" / "images", dataset_dir / "test" / "labels"),
    ]
    found: dict[str, tuple[Path, Path]] = {}
    for name, img_dir, lbl_dir in layouts:
        if img_dir.is_dir():
            lbl_dir.mkdir(parents=True, exist_ok=True)
            found[name] = (img_dir, lbl_dir)
    return found


def normalize_label_line(class_id: int, parts: list[str]) -> str | None:
    mapped = CLASS_REMAP.get(class_id)
    if mapped is None:
        return None
    return " ".join([str(mapped), *parts[1:]])


def normalize_labels(label_dir: Path) -> int:
    changed = 0
    for label_path in label_dir.glob("*.txt"):
        lines = []
        file_changed = False
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            old_id = int(parts[0])
            new_line = normalize_label_line(old_id, parts)
            if new_line is None:
                continue
            if int(new_line.split()[0]) != old_id:
                file_changed = True
            lines.append(new_line)
        if file_changed:
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            changed += 1
    return changed


def write_data_yaml(dataset_dir: Path) -> None:
    yaml_path = dataset_dir / "data.yaml"
    content = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 2,
        "names": ["helmet", "no_helmet"],
    }
    yaml_path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    ok(f"Wrote {yaml_path.name} (2 classes: helmet, no_helmet)")


def class_map_from_names(names: list[str]) -> dict[int, int | None]:
    mapping: dict[int, int | None] = {}
    for idx, name in enumerate(names):
        lower = name.lower().strip()
        if any(key in lower for key in HELMET_KEYWORDS) and "no" not in lower:
            mapping[idx] = 0
        elif any(key in lower for key in NO_HELMET_KEYWORDS):
            mapping[idx] = 1
        elif lower in ("helmet",):
            mapping[idx] = 0
        elif lower in ("no_helmet", "no helmet"):
            mapping[idx] = 1
        else:
            mapping[idx] = None
    return mapping


def load_source_class_map(source_dir: Path) -> dict[int, int | None]:
    yaml_files = list(source_dir.rglob("data.yaml"))
    if not yaml_files:
        return CLASS_REMAP.copy()
    data = yaml.safe_load(yaml_files[0].read_text(encoding="utf-8"))
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x) if str(x).isdigit() else x)]
    return class_map_from_names([str(n) for n in names])


def convert_label_file(label_path: Path, class_map: dict[int, int | None]) -> list[str]:
    lines: list[str] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        src_id = int(parts[0])
        dst_id = class_map.get(src_id, CLASS_REMAP.get(src_id))
        if dst_id is None:
            continue
        lines.append(" ".join([str(dst_id), *parts[1:]]))
    return lines


def merge_extra_dataset(main_dir: Path, extra_dir: Path, prefix: str = "ind_") -> int:
    if not extra_dir.is_dir():
        info(f"No extra dataset at {extra_dir} — skip merge")
        return 0

    merged = 0
    sources = [extra_dir] if split_dirs(extra_dir) else sorted(extra_dir.iterdir())
    for source in sources:
        if not source.is_dir():
            continue
        class_map = load_source_class_map(source)
        info(f"Merging from {source.name} with class map {class_map}")

        extra_splits = split_dirs(source)
        if not extra_splits:
            continue

        main_train = main_dir / "train"
        img_out = main_train / "images"
        lbl_out = main_train / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        source_split = extra_splits.get("train") or next(iter(extra_splits.values()))
        src_img, src_lbl = source_split
        tag = re.sub(r"[^a-z0-9]+", "_", source.name.lower()).strip("_")

        for image_path in sorted(src_img.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = src_lbl / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            lines = convert_label_file(label_path, class_map)
            if not lines:
                continue

            out_stem = f"{prefix}{tag}_{image_path.stem}"
            out_img = img_out / f"{out_stem}{image_path.suffix.lower()}"
            out_lbl = lbl_out / f"{out_stem}.txt"
            if out_img.exists():
                continue

            shutil.copy2(image_path, out_img)
            out_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
            merged += 1

    ok(f"Merged {merged} extra images into train split")
    return merged


def motion_blur(image: np.ndarray, size: int = 11) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0 / size
    return cv2.filter2D(image, -1, kernel)


def low_light(image: np.ndarray, factor: float = 0.45) -> np.ndarray:
    dark = (image.astype(np.float32) * factor).astype(np.uint8)
    lab = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def rain_overlay(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    for _ in range(100):
        x1 = random.randint(0, w - 1)
        y1 = random.randint(0, h - 1)
        length = random.randint(8, 16)
        x2 = min(w - 1, x1 + random.randint(-2, 2))
        y2 = min(h - 1, y1 + length)
        cv2.line(out, (x1, y1), (x2, y2), (200, 200, 200), 1)
    return cv2.addWeighted(image, 0.85, out, 0.15, 0)


AUGMENTATIONS = {
    "blur": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
    "motion": lambda img: motion_blur(img, 13),
    "night": lambda img: low_light(img, 0.42),
    "rain": rain_overlay,
}


def augment_train(dataset_dir: Path, ratio: float, dry_run: bool) -> int:
    img_dir = dataset_dir / "train" / "images"
    lbl_dir = dataset_dir / "train" / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        return 0

    chosen = random.sample(images, max(1, int(len(images) * ratio)))
    created = 0
    info(f"Augmenting {len(chosen)} train images x {len(AUGMENTATIONS)} variants")

    for image_path in chosen:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        label_path = lbl_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue

        for suffix, fn in AUGMENTATIONS.items():
            out_stem = f"{image_path.stem}_ind_{suffix}"
            out_img = img_dir / f"{out_stem}{image_path.suffix.lower()}"
            out_lbl = lbl_dir / f"{out_stem}.txt"
            if out_img.exists():
                continue
            if dry_run:
                created += 1
                continue
            cv2.imwrite(str(out_img), fn(image))
            shutil.copy2(label_path, out_lbl)
            created += 1

    return created


def summarize(dataset_dir: Path) -> None:
    splits = split_dirs(dataset_dir)
    print("\nDataset summary:")
    for name, (img_dir, lbl_dir) in splits.items():
        count = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        print(f"  {name}: {count} images")

    label_dir = dataset_dir / "train" / "labels"
    counts: Counter[int] = Counter()
    for label_path in label_dir.glob("*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    print(f"  train class counts: {dict(counts)} (0=helmet, 1=no_helmet)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Indian helmet dataset for one-shot training")
    parser.add_argument("--merge-extra", action="store_true", help="Merge data/datasets/helmet_detection_extra")
    parser.add_argument("--augment-only", action="store_true", help="Skip normalize/merge, only augment")
    parser.add_argument("--ratio", type=float, default=0.35, help="Fraction of train images to augment")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Downsample helmet-only images to ~1:1 instance ratio after merge/augment",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=1.1,
        help="Helmet/no_helmet instance ratio for --balance (default 1.1)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    config = load_config()
    dataset_dir = resolve_path(config["training"]["helmet"]["dataset_dir"])
    extra_dir = dataset_dir.parent / "helmet_detection_extra"

    print("\n" + "=" * 60)
    print("PREPARE HELMET DATASET (Indian / dashcam)")
    print("=" * 60)

    if not args.augment_only:
        splits = split_dirs(dataset_dir)
        if not splits:
            print(f"Missing helmet dataset under {dataset_dir}")
            sys.exit(1)

        total_changed = 0
        for name, (_img, lbl) in splits.items():
            changed = normalize_labels(lbl)
            total_changed += changed
            ok(f"Normalized {changed} label files in {name}")

        write_data_yaml(dataset_dir)

        if args.merge_extra:
            merge_extra_dataset(dataset_dir, extra_dir)

    created = augment_train(dataset_dir, args.ratio, args.dry_run)
    if args.dry_run:
        info(f"Dry run — would create {created} augmented pairs")
    else:
        ok(f"Created {created} augmented train pairs")

    if args.balance:
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from balance_helmet_dataset import balance_train as balance_helmet_train

        print("\n" + "-" * 60)
        print("BALANCING TRAIN SPLIT")
        print("-" * 60)
        balance_helmet_train(dataset_dir, args.target_ratio, args.dry_run, args.seed)

    summarize(dataset_dir)
    print("\nNext:")
    print("  python scripts/balance_helmet_dataset.py   # re-balance only")
    print("  python train_models.py --helmets --continue-best")


if __name__ == "__main__":
    main()
