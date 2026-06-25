"""Repair helmet train split after interrupted balance; restore _pruned; re-balance."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def restore_all(root: Path) -> int:
    pr_img = root / "_pruned" / "images"
    pr_lbl = root / "_pruned" / "labels"
    img_dir = root / "images"
    lbl_dir = root / "labels"
    restored = 0
    for src_dir, dst_dir in ((pr_img, img_dir), (pr_lbl, lbl_dir)):
        if not src_dir.exists():
            continue
    for path in list(src_dir.iterdir()):
        if not path.is_file():
            continue
        dest = dst_dir / path.name
        if dest.exists():
            path.unlink(missing_ok=True)
            continue
        if not path.exists():
            continue
        try:
            shutil.move(str(path), str(dest))
        except FileNotFoundError:
            continue
        restored += 1
    return restored


def sync_orphans(root: Path) -> tuple[int, int]:
    img_dir = root / "images"
    lbl_dir = root / "labels"
    orphan_img = root / "_pruned" / "orphan_images"
    orphan_lbl = root / "_pruned" / "orphan_labels"
    orphan_img.mkdir(parents=True, exist_ok=True)
    orphan_lbl.mkdir(parents=True, exist_ok=True)

    label_stems = {p.stem for p in lbl_dir.glob("*.txt")}
    image_stems: dict[str, Path] = {}
    for path in img_dir.iterdir():
        if path.suffix.lower() in IMAGE_SUFFIXES:
            image_stems[path.stem] = path

    moved_lbl = moved_img = 0
    for label in list(lbl_dir.glob("*.txt")):
        if label.stem not in image_stems:
            shutil.move(str(label), str(orphan_lbl / label.name))
            moved_lbl += 1
    for stem, image in list(image_stems.items()):
        if stem not in label_stems:
            shutil.move(str(image), str(orphan_img / image.name))
            moved_img += 1
    return moved_img, moved_lbl


def main() -> None:
    root = PROJECT_ROOT / "data" / "datasets" / "helmet_detection" / "train"
    print("Restoring pruned files...")
    n = restore_all(root)
    print(f"  restored/moved {n} from _pruned")

    print("Syncing orphan labels/images...")
    moved_img, moved_lbl = sync_orphans(root)
    print(f"  orphan images: {moved_img}, orphan labels: {moved_lbl}")

    img_dir = root / "images"
    lbl_dir = root / "labels"
    imgs = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    lbls = sum(1 for p in lbl_dir.glob("*.txt"))
    print(f"train images: {imgs}, train labels: {lbls}")

    if imgs != lbls:
        print("WARNING: image/label count still mismatched")
        sys.exit(1)

    print("\nRunning balance...")
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from balance_helmet_dataset import balance_train, build_image_index, read_label_stats, summarize_rows
    from model_utils import load_config, resolve_path

    config = load_config()
    dataset_dir = resolve_path(config["training"]["helmet"]["dataset_dir"])
    lbl = dataset_dir / "train" / "labels"
    img = dataset_dir / "train" / "images"
    stats = read_label_stats(lbl, build_image_index(img))
    before = summarize_rows(stats)
    print(
        f"Before: helmet={before.get(0, 0)}, no_helmet={before.get(1, 0)}, "
        f"ratio={before.get(0, 0) / max(before.get(1, 0), 1):.2f}:1"
    )
    balance_train(dataset_dir, target_ratio=1.1, dry_run=False, seed=42, stats=stats)
    stats_after = read_label_stats(lbl, build_image_index(img))
    after = summarize_rows(stats_after)
    imgs_after = len(build_image_index(img))
    print(
        f"\nAfter: helmet={after.get(0, 0)}, no_helmet={after.get(1, 0)}, "
        f"ratio={after.get(0, 0) / max(after.get(1, 0), 1):.2f}:1"
    )
    print(f"Train images: {imgs_after}")
    print("\nRetrain: python train_models.py --helmets --continue-best")


if __name__ == "__main__":
    main()
