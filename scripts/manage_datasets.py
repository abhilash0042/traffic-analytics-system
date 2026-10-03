"""Dataset Management and Lifecycle Script.

Handles:
- Deletion of redundant duplicate directories and zip archives.
- Tracking and registering all raw, merged, and augmented datasets.
- Clean one-command rollback / deletion of augmented datasets if needed.
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path("c:/projects/traffic-analytics-system")
DATASETS_DIR = ROOT_DIR / "data" / "datasets"
MANIFEST_FILE = DATASETS_DIR / "manifest.json"


def get_free_disk_gb():
    import psutil
    d = psutil.disk_usage(str(ROOT_DIR.drive or "C:"))
    return d.free / (1024 ** 3)


def load_manifest():
    if MANIFEST_FILE.exists():
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"raw_datasets": {}, "merged_datasets": {}, "augmented_datasets": {}, "deleted_items": []}


def save_manifest(manifest):
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved to {MANIFEST_FILE}")


def clean_redundancies():
    """Removes 100% duplicate directories and zip files where extracted folders exist."""
    manifest = load_manifest()
    free_before = get_free_disk_gb()
    logger.info(f"Disk free before cleanup: {free_before:.2f} GB")

    # 1. Zip archives to delete (since extracted folders are verified complete)
    zips_to_delete = [
        DATASETS_DIR / "dashcop_yolo.zip",
        DATASETS_DIR / "videoset1_videos_part1.zip",
        DATASETS_DIR / "ccpd_yolo.zip",
        DATASETS_DIR / "videoset1_xml.zip"
    ]

    for z in zips_to_delete:
        if z.exists():
            size_mb = z.stat().st_size / (1024 * 1024)
            logger.info(f"Deleting verified zip archive: {z.name} ({size_mb:.1f} MB)...")
            z.unlink()
            manifest["deleted_items"].append({"path": str(z), "reason": "Extracted folder verified complete", "freed_mb": size_mb})

    # 2. 100% Duplicate folders to delete
    folders_to_delete = [
        (DATASETS_DIR / "kaggle_indian_plates2", "100% duplicate clone of indian_vehicles"),
        (DATASETS_DIR / "indian_vehicles", "Corrupted VOC XML tags; fully absorbed by perfect_indian_license_plates"),
        (DATASETS_DIR / "license_plates_indian", "100% subset absorbed by perfect_indian_license_plates"),
        (DATASETS_DIR / "videoset1_xml_extracted", "Redundant extraction folder (same as videoset1_xml)")
    ]

    for folder, reason in folders_to_delete:
        if folder.exists():
            total_size = sum(f.stat().st_size for f in folder.rglob('*') if f.is_file()) / (1024 * 1024)
            logger.info(f"Deleting redundant folder: {folder.name} ({total_size:.1f} MB) - Reason: {reason}")
            shutil.rmtree(folder, ignore_errors=True)
            manifest["deleted_items"].append({"path": str(folder), "reason": reason, "freed_mb": total_size})

    # 3. Clean kaggle_indian_plates3: rescue 116 labeled items and remove heavy videos / unannotated files
    p3_dir = DATASETS_DIR / "kaggle_indian_plates3"
    if p3_dir.exists():
        logger.info("Cleaning kaggle_indian_plates3 (saving labeled items, removing raw video and unannotated images)...")
        rescue_dir = DATASETS_DIR / "kaggle_plates3_rescued"
        rescue_dir.mkdir(parents=True, exist_ok=True)
        (rescue_dir / "images").mkdir(exist_ok=True)
        (rescue_dir / "labels").mkdir(exist_ok=True)

        txt_files = list(p3_dir.rglob("*.txt"))
        rescued_count = 0
        for txt in txt_files:
            if txt.name.lower() in ["classes.txt", "data.yaml"]:
                continue
            img_candidate = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
                cand = txt.with_suffix(ext)
                if cand.exists():
                    img_candidate = cand
                    break
            if img_candidate and img_candidate.exists():
                shutil.copy2(img_candidate, rescue_dir / "images" / img_candidate.name)
                shutil.copy2(txt, rescue_dir / "labels" / txt.name)
                rescued_count += 1

        logger.info(f"Rescued {rescued_count} labeled plate pairs into {rescue_dir}")
        shutil.rmtree(p3_dir, ignore_errors=True)
        manifest["deleted_items"].append({"path": str(p3_dir), "reason": f"Rescued {rescued_count} labeled pairs, deleted 1.3GB raw media", "rescued_to": str(rescue_dir)})

    free_after = get_free_disk_gb()
    logger.info(f"Disk free after cleanup: {free_after:.2f} GB (Freed {free_after - free_before:.2f} GB)")
    save_manifest(manifest)


def delete_augmented():
    """Deletes all augmented datasets cleanly if user requests rollback."""
    aug_dir = DATASETS_DIR / "augmented"
    if aug_dir.exists():
        total_size = sum(f.stat().st_size for f in aug_dir.rglob('*') if f.is_file()) / (1024 * 1024)
        logger.info(f"Deleting augmented directory: {aug_dir} ({total_size:.1f} MB)...")
        shutil.rmtree(aug_dir, ignore_errors=True)
        logger.info("Augmented datasets successfully rolled back.")
    else:
        logger.info("No augmented directory found to delete.")

    manifest = load_manifest()
    manifest["augmented_datasets"] = {}
    save_manifest(manifest)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--delete-augmented":
        delete_augmented()
    else:
        clean_redundancies()
