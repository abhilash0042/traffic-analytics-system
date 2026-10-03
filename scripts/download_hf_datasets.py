"""Automated Hugging Face Dataset Downloader for Teammates.

Pulls curated traffic analytics datasets from the Hugging Face organization / account
directly into the expected local project folder structure, and automatically unzips archives.

Usage:
    # 1. Download all datasets:
    python scripts/download_hf_datasets.py --all

    # 2. Download specific dataset:
    python scripts/download_hf_datasets.py --dataset vehicle
    python scripts/download_hf_datasets.py --dataset helmet
    python scripts/download_hf_datasets.py --dataset plate_detection
    python scripts/download_hf_datasets.py --dataset anpr_ocr
    python scripts/download_hf_datasets.py --dataset tracking_video

    # 3. Custom target organization / username:
    python scripts/download_hf_datasets.py --all --org thundarstrom
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict

# Ensure UTF-8 output encoding across Windows / Linux consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "datasets"

DEFAULT_ORG = "thundarstrom"

DATASET_MAP: Dict[str, Dict[str, str]] = {
    "vehicle": {
        "repo_name": "traffic-vehicle-detection",
        "target_dir": "vehicle_detection",
        "title": "Vehicle Detection (UA-DETRAC CCTV - 23.3k images)",
    },
    "helmet": {
        "repo_name": "traffic-helmet-violation",
        "target_dir": "helmet_combined",
        "title": "Helmet Violation Detection (42.5k images)",
    },
    "plate_detection": {
        "repo_name": "indian-license-plate-detection",
        "target_dir": "plate_detection_combined",
        "title": "Indian Plate Localization (3.7k images)",
    },
    "plate_ccpd": {
        "repo_name": "ccpd-license-plate-pretrain",
        "target_dir": "ccpd_yolo",
        "title": "CCPD Plate Backbone Pretraining (20k images)",
    },
    "anpr_ocr": {
        "repo_name": "indian-anpr-ocr-corpus",
        "target_dir": "parseq_dataset",
        "title": "Indian ANPR / PARSeq OCR Corpus (18.5k crops)",
    },
    "anpr_benchmark": {
        "repo_name": "indian-anpr-ocr-benchmark",
        "target_dir": "dashcop_ocr",
        "title": "DashCop Real-World OOD Benchmark (3k crops)",
    },
    "anpr_synthetic": {
        "repo_name": "synthetic-indian-anpr-ocr",
        "target_dir": "kaggle_synthetic",
        "title": "Synthetic Indian ANPR Corpus (18k crops)",
    },
    "tracking_video": {
        "repo_name": "traffic-surveillance-video-corpus",
        "target_dir": "videoset1_videos",
        "title": "2K Dashcam Video Corpus (50 streams + CVAT tracks)",
    },
    "sample_videos": {
        "repo_name": "traffic-pipeline-sample-videos",
        "target_dir": "sample_videos",
        "title": "Pipeline Integration & Demo Video Suite (6 clips)",
    },
    "augmented_plates": {
        "repo_name": "augmented-indian-plate-detection",
        "target_dir": "augmented/plate_detection",
        "title": "Augmented Indian Plate Perturbations (1.6k crops)",
    },
}


def extract_zips(target_path: Path) -> None:
    """Extracts any zip files found in the dataset directory."""
    zip_files = list(target_path.glob("*.zip"))
    for z in zip_files:
        print(f"   [EXTRACT] Unpacking {z.name}...", flush=True)
        try:
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(target_path)
            z.unlink()
            print(f"   [OK] Extracted {z.name}", flush=True)
        except Exception as e:
            print(f"   [WARN] Could not extract {z.name}: {e}", flush=True)


def download_single_dataset(key: str, org: str, token: str | None = None) -> bool:
    meta = DATASET_MAP.get(key)
    if not meta:
        print(f"[ERROR] Unknown dataset key: '{key}'")
        return False

    repo_id = f"{org}/{meta['repo_name']}"
    target_path = DATA_DIR / meta["target_dir"]
    target_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"[DOWNLOAD] Downloading: {meta['title']}")
    print(f"   Source Repo : https://huggingface.co/datasets/{repo_id}")
    print(f"   Local Target: {target_path}")
    print("=" * 70)

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(target_path),
            token=token,
            ignore_patterns=["*.msgpack", ".git*"],
        )
        extract_zips(target_path)
        print(f"   [SUCCESS] Successfully synced into {target_path}")
        return True
    except Exception as e:
        print(f"   [ERROR] Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Traffic Analytics Datasets from Hugging Face")
    parser.add_argument("--org", default=DEFAULT_ORG, help=f"Hugging Face organization or user slug (default: {DEFAULT_ORG})")
    parser.add_argument("--dataset", choices=list(DATASET_MAP.keys()), help="Download a specific dataset")
    parser.add_argument("--all", action="store_true", help="Download all available datasets")
    parser.add_argument("--token", help="Optional HF Token (if downloading private repositories)")
    args = parser.parse_args()

    if not HF_AVAILABLE:
        print("[ERROR] 'huggingface_hub' is required. Run: pip install huggingface_hub")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("       [HF] DATASET SYNCHRONIZATION TOOL       ")
    print("=" * 70)

    if args.dataset:
        keys = [args.dataset]
    elif args.all:
        keys = list(DATASET_MAP.keys())
    else:
        print("\n[CATALOG] Available datasets:")
        for k, v in DATASET_MAP.items():
            print(f"   * {k:<18} -> {v['title']}")
        print("\nPlease specify --all or --dataset <name> (e.g. --dataset vehicle).")
        return

    success = 0
    failed = 0
    for key in keys:
        if download_single_dataset(key, args.org, token=args.token):
            success += 1
        else:
            failed += 1

    print("\n" + "=" * 70)
    print(f"[SUMMARY] DONE: {success} downloaded, {failed} failed.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
