"""
Merge detection datasets into a unified YOLO dataset YAML manifest.

IMPORTANT: Only use datasets with FULL VEHICLE images for detection training.
  - dashcop_yolo: 2560x1440 full dashcam frames (Indian 2-wheelers) ✅
  - ccpd_yolo: 720x1160 full vehicle images (Chinese 4-wheelers) ✅
  - perfect_indian_license_plates: 272px PLATE CROPS — SKIP for detection ❌
  - license_plates_indian: 272px PLATE CROPS — SKIP for detection ❌

Instead of copying files (wastes disk), we create a dataset.yaml that points
YOLO to all source folders directly using YOLO's multi-path support.
"""
import random
import shutil
from pathlib import Path

random.seed(42)

BASE = Path("c:/projects/traffic-analytics-system/data/datasets")
OUTPUT = BASE / "unified_plate_detection"

# Only full-vehicle-image datasets (PLATE CROPS excluded)
DATASETS = [
    "dashcop_yolo",
    "ccpd_yolo"
]

def collect_all_pairs():
    all_pairs = []
    for ds_name in DATASETS:
        ds_path = BASE / ds_name
        if not ds_path.exists():
            print(f"Skipping {ds_name} (not found)")
            continue

        images = list(ds_path.rglob("*.jpg")) + list(ds_path.rglob("*.png"))
        valid = 0
        for img_path in images:
            img_str = str(img_path)
            if "\\images\\" in img_str or "/images/" in img_str:
                lbl_path = Path(
                    img_str.replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")
                ).with_suffix(".txt")
            else:
                lbl_path = img_path.with_suffix(".txt")

            if not lbl_path.exists():
                continue
            content = lbl_path.read_text().strip()
            if not content:
                continue
            all_pairs.append((img_path, lbl_path, ds_name))
            valid += 1

        print(f"  {ds_name}: {valid} valid pairs")
    return all_pairs

def process():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    # Create split dirs for symlink-style txt lists
    for split in ["train", "val", "test"]:
        (OUTPUT / split).mkdir(parents=True, exist_ok=True)

    print("Scanning datasets...")
    pairs = collect_all_pairs()
    print(f"Total: {len(pairs)} pairs")

    random.shuffle(pairs)
    n = len(pairs)
    train_end = int(n * 0.80)
    val_end = int(n * 0.95)

    splits = {
        "train": pairs[:train_end],
        "val":   pairs[train_end:val_end],
        "test":  pairs[val_end:]
    }

    # Write image-path lists for each split (YOLO accepts text-file lists of paths)
    for split_name, split_pairs in splits.items():
        list_file = OUTPUT / f"{split_name}.txt"
        with open(list_file, "w") as f:
            for img_path, _, _ in split_pairs:
                f.write(str(img_path.absolute().as_posix()) + "\n")
        print(f"  {split_name}: {len(split_pairs)} images -> {list_file.name}")

    # Write dataset.yaml using path lists (no file copying needed)
    yaml_content = (
        f"# Unified Indian + CCPD plate detection dataset\n"
        f"# Full vehicle images only — no plate crops\n\n"
        f"path: {OUTPUT.absolute().as_posix()}\n"
        f"train: train.txt\n"
        f"val:   val.txt\n"
        f"test:  test.txt\n\n"
        f"names:\n"
        f"  0: license_plate\n"
    )
    (OUTPUT / "dataset.yaml").write_text(yaml_content)
    print(f"\nDataset YAML written to {OUTPUT / 'dataset.yaml'}")
    print(f"\nSummary:")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val:   {len(splits['val'])} images")
    print(f"  Test:  {len(splits['test'])} images")
    print(f"\nNo files copied — YOLO reads from original locations.")

if __name__ == "__main__":
    process()
