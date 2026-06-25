"""Move train/_pruned back into train (images + labels)."""
from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
root = PROJECT_ROOT / "data" / "datasets" / "helmet_detection" / "train"
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
        if not dest.exists():
            shutil.move(str(path), str(dest))
            restored += 1

print(f"restored {restored} files")
print(f"train images: {sum(1 for p in img_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})}")
print(f"train labels: {sum(1 for p in lbl_dir.glob('*.txt'))}")
