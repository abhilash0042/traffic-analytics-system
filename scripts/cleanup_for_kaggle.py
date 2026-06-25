"""
Remove files not needed for training or Kaggle upload.

Usage:
    python scripts/cleanup_for_kaggle.py
    python scripts/cleanup_for_kaggle.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Temp logs from balance/repair runs (project root)
ROOT_LOGS = (
    "balance_final.log",
    "balance_run.log",
    "balance_dryrun.log",
    "balance_out.txt",
    "balance_done.txt",
    "repair_balance.log",
    "repair_balance.err",
    "finish_balance.log",
    "restore.log",
    "restore.err",
)

# Merged source copies — safe to delete after prepare_helmet_indian.py
EXTRA_DATASET_DIRS = (
    PROJECT_ROOT / "data" / "datasets" / "helmet_detection_extra",
    PROJECT_ROOT / "data" / "datasets" / "helmet_detection_new",
)

HELMET_TRAIN = PROJECT_ROOT / "data" / "datasets" / "helmet_detection" / "train"
PRUNED_DIR = HELMET_TRAIN / "_pruned"
CACHE_FILE = HELMET_TRAIN / "labels" / ".balance_stats.json"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _fmt_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean unwanted files before Kaggle upload")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("CLEANUP FOR KAGGLE UPLOAD")
    print("=" * 60)

    freed = 0

    def remove_path(path: Path, label: str) -> None:
        nonlocal freed
        if not path.exists():
            return
        size = _dir_size(path) if path.is_dir() else path.stat().st_size
        print(f"  [>>] {label}: {path.relative_to(PROJECT_ROOT)} ({_fmt_mb(size)})")
        if not args.dry_run:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        freed += size

    remove_path(PRUNED_DIR, "Pruned train split (not used for training)")
    remove_path(CACHE_FILE, "Balance stats cache")

    for name in ROOT_LOGS:
        remove_path(PROJECT_ROOT / name, "Temp log")

    for extra in EXTRA_DATASET_DIRS:
        if extra.is_dir():
            remove_path(extra, "Merged duplicate dataset")

    print(f"\n  {'Would free' if args.dry_run else 'Freed'}: {_fmt_mb(freed)}")
    if args.dry_run:
        print("\n  Run without --dry-run to delete.")
    else:
        print("\n  Next: python scripts/pack_helmet_for_kaggle.py")


if __name__ == "__main__":
    main()
