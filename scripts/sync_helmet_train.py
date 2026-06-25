"""Sync helmet train images/labels; quarantine unmatched files."""
from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    root = PROJECT_ROOT / "data" / "datasets" / "helmet_detection" / "train"
    img_dir = root / "images"
    lbl_dir = root / "labels"
    orphan_img = root / "_pruned" / "orphan_images"
    orphan_lbl = root / "_pruned" / "orphan_labels"
    orphan_img.mkdir(parents=True, exist_ok=True)
    orphan_lbl.mkdir(parents=True, exist_ok=True)

    for sub in ("_pruned/labels", "_pruned/orphan_labels"):
        src = root / sub
        if not src.exists():
            continue
        for path in list(src.iterdir()):
            if not path.is_file() or not path.name.endswith(".txt"):
                continue
            dest = lbl_dir / path.name
            if dest.exists():
                path.unlink(missing_ok=True)
            elif path.exists():
                try:
                    shutil.move(str(path), str(dest))
                except FileNotFoundError:
                    continue

    label_stems = {p.stem for p in lbl_dir.glob("*.txt")}
    moved_img = moved_lbl = 0
    for path in list(img_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem not in label_stems:
            shutil.move(str(path), str(orphan_img / path.name))
            moved_img += 1

    image_stems = {
        p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    }
    for path in list(lbl_dir.glob("*.txt")):
        if path.stem not in image_stems:
            shutil.move(str(path), str(orphan_lbl / path.name))
            moved_lbl += 1

    imgs = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    lbls = sum(1 for p in lbl_dir.glob("*.txt"))
    print(f"synced train pairs: {imgs} images, {lbls} labels")
    print(f"quarantined this run: {moved_img} images, {moved_lbl} labels")


if __name__ == "__main__":
    main()
