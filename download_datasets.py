"""
Dataset Download Helper Script
Traffic & Vehicle Analytics System

Downloads small substitute datasets (~900 MB total) instead of the massive
research-paper corpora (RideSafe-400, UA-DETRAC full, CCPD full, etc.).

Usage:
    python download_datasets.py                  # Status + quick-start guide
    python download_datasets.py --status         # Check what is already present
    python download_datasets.py --setup          # Create folder structure only
    python download_datasets.py --bootstrap      # Zero-API setup (model + demo video)
    python download_datasets.py --videos         # Download sample traffic videos
    python download_datasets.py --all            # Download everything possible
    python download_datasets.py --helmets        # Helmet dataset (Roboflow)
    python download_datasets.py --plates         # License plates (Kaggle or Roboflow)
    python download_datasets.py --vehicles       # Optional UA-DETRAC 10K (Roboflow)
    python download_datasets.py --manual         # Browser-based instructions

Credentials (free accounts):
    Roboflow: set ROBOFLOW_API_KEY or paste when prompted
              Sign up: https://app.roboflow.com
    Kaggle:   place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME + KAGGLE_KEY
              Sign up: https://www.kaggle.com
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "datasets")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
DEMO_VIDEO = os.path.join(PROJECT_ROOT, "assets", "videos", "sample_video.mp4")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# Small substitutes mapped to paper requirements (see docs/DATASET_STRATEGY.md)
ROBOFLOW_DATASETS = {
    "helmet_detection": {
        "workspace": "traffic-analysis-td0rl",
        "project": "motorcycle-helmet-q0wmd-qlt95",
        "version": 1,
        "name": "Motorcycle Helmet Detection (~1.3K images)",
        "url": "https://universe.roboflow.com/traffic-analysis-td0rl/motorcycle-helmet-q0wmd-qlt95",
    },
    "helmet_and_plates_combo": {
        "workspace": "helmet-and-number-plate-detection-project",
        "project": "helmet-and-number-plate-detection-for-motorbike-safety-iityz",
        "version": 1,
        "name": "Helmet + Number Plate combo (~20K images)",
        "url": "https://universe.roboflow.com/helmet-and-number-plate-detection-project/helmet-and-number-plate-detection-for-motorbike-safety-iityz",
    },
    "license_plates_roboflow": {
        "workspace": "object-detection-helmetslicense",
        "project": "motorcycle-helmet-and-license-plate-detection",
        "version": 1,
        "name": "License Plate + Helmet (~1.5K images)",
        "url": "https://universe.roboflow.com/object-detection-helmetslicense/motorcycle-helmet-and-license-plate-detection",
    },
    "vehicle_detection": {
        "workspace": "model-rli8w",
        "project": "ua-detrac-10k-sample-znazr",
        "version": 1,
        "name": "UA-DETRAC 10K Sample (~800 MB, optional fine-tuning)",
        "url": "https://universe.roboflow.com/model-rli8w/ua-detrac-10k-sample-znazr",
    },
}

KAGGLE_DATASETS = {
    "license_plates": {
        "slug": "kedarsai/indian-license-plates-with-labels",
        "name": "Indian License Plates with Labels (~300 MB)",
        "url": "https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels",
    },
}

SAMPLE_VIDEOS = [
    {
        "filename": "sample_traffic_cctv.mp4",
        "url": "https://raw.githubusercontent.com/anmspro/Traffic-Signal-Violation-Detection-System/master/Resources/CCTV%20Footage.mp4",
        "description": "CCTV traffic footage (GitHub, ~few MB)",
    },
    {
        "filename": "sample_traffic_highway.mp4",
        "url": "https://raw.githubusercontent.com/anmspro/Traffic-Signal-Violation-Detection-System/master/Resources/Highway-2.mp4",
        "description": "Highway traffic footage (GitHub)",
    },
]


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def missing(msg: str) -> None:
    print(f"  [--] {msg}")


def warn(msg: str) -> None:
    print(f"  [!!] {msg}")


def ensure_dirs() -> None:
    for name in (
        "license_plates",
        "helmet_detection",
        "vehicle_detection",
        "sample_videos",
    ):
        os.makedirs(os.path.join(DATA_DIR, name), exist_ok=True)
    ok(f"Created dataset folders under {DATA_DIR}")


def folder_stats(path: str) -> tuple[int, float]:
    if not os.path.isdir(path):
        return 0, 0.0
    file_count = 0
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file_count += 1
            total_size += os.path.getsize(os.path.join(dirpath, filename))
    return file_count, total_size / (1024 * 1024)


def verify_datasets() -> None:
    print("\n" + "=" * 60)
    print("DATASET STATUS CHECK")
    print("=" * 60)

    checks = {
        "License Plates": os.path.join(DATA_DIR, "license_plates"),
        "Helmet Detection": os.path.join(DATA_DIR, "helmet_detection"),
        "Vehicle Detection (optional)": os.path.join(DATA_DIR, "vehicle_detection"),
        "Sample Videos": os.path.join(DATA_DIR, "sample_videos"),
    }

    for name, path in checks.items():
        count, size_mb = folder_stats(path)
        if count > 0:
            ok(f"{name}: {count} files ({size_mb:.1f} MB)")
        elif os.path.isdir(path):
            missing(f"{name}: folder exists but empty")
        else:
            missing(f"{name}: not found")

    model_path = os.path.join(WEIGHTS_DIR, "yolov8n.pt")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        ok(f"YOLOv8n pre-trained (vehicle fallback): {size_mb:.1f} MB")
    else:
        missing("YOLOv8n pre-trained: run --bootstrap or first pipeline run")

    if os.path.exists(DEMO_VIDEO):
        size_mb = os.path.getsize(DEMO_VIDEO) / (1024 * 1024)
        ok(f"Demo video (assets/videos/sample_video.mp4): {size_mb:.1f} MB")
    else:
        missing("Demo video: run --bootstrap or --videos")

    print("\nPaper datasets NOT needed for this project:")
    print("  RideSafe-400 (50+ GB)  -> Roboflow helmet subset + logic-based violations")
    print("  UA-DETRAC full (20 GB) -> yolov8n.pt COCO weights + optional 10K sample")
    print("  CCPD full (48 GB)      -> Kaggle Indian plates (~300 MB)")
    print("  IDD (20 GB)            -> Skip (registration required)")


def print_quick_start() -> None:
    print("\n" + "=" * 60)
    print("QUICK START (no paper datasets, under 1 GB for essentials)")
    print("=" * 60)
    print("""
Step 1 - Already works without any training data:
  python download_datasets.py --bootstrap
  python ai_pipeline.py

Step 2 - Get training data for plates + helmets (~600-900 MB):
  a) Free Roboflow key -> https://app.roboflow.com (Settings -> API Keys)
  b) Free Kaggle account -> https://www.kaggle.com/settings -> Create API token

  set ROBOFLOW_API_KEY=your_key_here
  # Put kaggle.json in %USERPROFILE%\\.kaggle\\

  python download_datasets.py --all

Step 3 - Verify:
  python download_datasets.py --status

Minimum viable (zero custom downloads):
  Vehicle detection  -> yolov8n.pt (COCO, auto-downloads, 6 MB)
  Demo / testing     -> assets/videos/sample_video.mp4
  Plates + helmets   -> only needed when you train custom YOLO heads
""")


def print_manual_download_instructions() -> None:
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD (browser only, no API keys)")
    print("=" * 60)
    print("""
1) INDIAN LICENSE PLATES (~300 MB) [required for ANPR training]
   https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels
   Download ZIP -> extract to data/datasets/license_plates/

2) HELMET DETECTION (~150-500 MB) [required for helmet violations]
   https://universe.roboflow.com/traffic-analysis-td0rl/motorcycle-helmet-q0wmd-qlt95
   Download Dataset -> YOLOv8 -> extract to data/datasets/helmet_detection/

   OR one combined dataset (helmet + plate classes):
   https://universe.roboflow.com/helmet-and-number-plate-detection-project/helmet-and-number-plate-detection-for-motorbike-safety-iityz

3) VEHICLE FINE-TUNING [optional - COCO model already works]
   https://universe.roboflow.com/model-rli8w/ua-detrac-10k-sample-znazr
   Download Dataset -> YOLOv8 -> extract to data/datasets/vehicle_detection/

4) SAMPLE VIDEOS
   python download_datasets.py --videos
   OR use assets/videos/sample_video.mp4 in the project
""")


def load_env_credentials() -> None:
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.isfile(env_path):
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        pass


def check_hf_installed() -> bool:
    try:
        import datasets  # noqa: F401
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        warn("Hugging Face packages not installed. Run: pip install datasets huggingface_hub")
        return False


def download_plates_hf() -> bool:
    if not check_hf_installed():
        return False
    from hf_dataset_loader import download_license_plates_hf

    return download_license_plates_hf()


def download_helmets_hf() -> bool:
    if not check_hf_installed():
        return False
    from hf_dataset_loader import download_helmet_hf

    return download_helmet_hf()


def check_roboflow_installed() -> bool:
    try:
        import roboflow  # noqa: F401
        return True
    except ImportError:
        warn("Roboflow package not installed. Run: pip install roboflow")
        return False


def get_roboflow_api_key(interactive: bool = True) -> str | None:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if api_key:
        return api_key
    if not interactive or not sys.stdin.isatty():
        return None
    print("\nRoboflow API key required (free): https://app.roboflow.com -> Settings -> API Keys")
    api_key = input("Paste ROBOFLOW_API_KEY (or Enter to skip): ").strip()
    return api_key or None


def download_from_roboflow(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    save_dir: str,
    dataset_name: str,
) -> bool:
    from roboflow import Roboflow

    print(f"\nDownloading: {dataset_name}")
    print(f"  Source: {workspace}/{project} v{version}")
    print(f"  Target: {save_dir}")

    os.makedirs(save_dir, exist_ok=True)
    count_before, _ = folder_stats(save_dir)
    if count_before > 10:
        ok(f"Already present ({count_before} files) - skipping")
        return True

    try:
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        project_obj.version(version).download("yolov8", location=save_dir, overwrite=True)
        count_after, size_mb = folder_stats(save_dir)
        ok(f"Downloaded {count_after} files ({size_mb:.1f} MB)")
        return True
    except Exception as exc:
        warn(f"Roboflow download failed: {exc}")
        cfg = ROBOFLOW_DATASETS.get(
            next((k for k, v in ROBOFLOW_DATASETS.items() if v["project"] == project), ""),
            {},
        )
        if cfg.get("url"):
            print(f"  Manual fallback: {cfg['url']}")
        return False


def has_kaggle_credentials() -> bool:
    kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    access_token = os.path.join(os.path.expanduser("~"), ".kaggle", "access_token")
    has_token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
    return (
        os.path.isfile(kaggle_json)
        or os.path.isfile(access_token)
        or bool(os.environ.get("KAGGLE_USERNAME") and has_token)
    )


def check_kaggle_installed() -> bool:
    import importlib.util

    if importlib.util.find_spec("kaggle") is None:
        warn("Kaggle package not installed. Run: pip install kaggle")
        return False
    return has_kaggle_credentials()


def download_from_kaggle(slug: str, save_dir: str, dataset_name: str) -> bool:
    print(f"\nDownloading: {dataset_name}")
    print(f"  Source: kaggle datasets download -d {slug}")
    print(f"  Target: {save_dir}")

    os.makedirs(save_dir, exist_ok=True)
    count_before, _ = folder_stats(save_dir)
    if count_before > 10:
        ok(f"Already present ({count_before} files) - skipping")
        return True

    if not has_kaggle_credentials():
        warn("Kaggle credentials not found — skipping Kaggle download")
        return False

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(slug, path=save_dir, unzip=True, quiet=False)
        count_after, size_mb = folder_stats(save_dir)
        ok(f"Downloaded {count_after} files ({size_mb:.1f} MB)")
        return True
    except SystemExit:
        warn("Kaggle authentication failed — skipping")
        return False
    except Exception as exc:
        warn(f"Kaggle download failed: {exc}")
        print(f"  Manual fallback: https://www.kaggle.com/datasets/{slug}")
        return False


def download_roboflow_zip(url: str, save_dir: str, dataset_name: str) -> bool:
    """Download a public Roboflow Universe zip export (no account needed for some datasets)."""
    import zipfile

    print(f"\nDownloading: {dataset_name}")
    print(f"  Source: Roboflow public zip")
    print(f"  Target: {save_dir}")

    os.makedirs(save_dir, exist_ok=True)
    count_before, _ = folder_stats(save_dir)
    if count_before > 50:
        ok(f"Already present ({count_before} files) - skipping")
        return True

    zip_path = os.path.join(save_dir, "_download.zip")
    try:
        print("  Fetching zip archive...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(save_dir)
        os.remove(zip_path)
        count_after, size_mb = folder_stats(save_dir)
        ok(f"Downloaded {count_after} files ({size_mb:.1f} MB)")
        return count_after > 50
    except Exception as exc:
        warn(f"Roboflow zip download failed: {exc}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False


# Public Roboflow Universe zip links (YOLOv8 format) — add working links when available
ROBOFLOW_ZIP_URLS: dict[str, str] = {}


def download_file(url: str, dest_path: str, description: str) -> bool:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        ok(f"{description}: already exists ({size_mb:.1f} MB)")
        return True

    print(f"  Fetching {description}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        ok(f"{description}: {size_mb:.1f} MB")
        return True
    except Exception as exc:
        warn(f"Could not download {description}: {exc}")
        return False


def bootstrap() -> None:
    """Minimum setup with zero paper datasets and no API keys."""
    print("\n" + "=" * 60)
    print("BOOTSTRAP: minimum viable setup (no paper datasets)")
    print("=" * 60)
    ensure_dirs()
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    model_path = os.path.join(WEIGHTS_DIR, "yolov8n.pt")
    if not os.path.exists(model_path):
        print("\nFetching YOLOv8n (COCO pre-trained, detects cars/bikes/buses/trucks)...")
        try:
            from ultralytics import YOLO

            YOLO(model_path)
            ok("weights/yolov8n.pt ready")
        except Exception as exc:
            warn(f"Could not download yolov8n.pt: {exc}")
    else:
        ok("weights/yolov8n.pt already present")

    sample_dst = os.path.join(DATA_DIR, "sample_videos", "project_demo.mp4")
    if os.path.exists(DEMO_VIDEO):
        shutil.copy2(DEMO_VIDEO, sample_dst)
        ok("Copied demo video -> data/datasets/sample_videos/")
    else:
        download_sample_videos()

    print("\nYou can run the pipeline now:")
    print("  python ai_pipeline.py")
    print("\nFor license plate + helmet TRAINING, still run:")
    print("  python download_datasets.py --plates --helmets")


def download_sample_videos() -> None:
    print("\n" + "=" * 60)
    print("DOWNLOADING SAMPLE TRAFFIC VIDEOS")
    print("=" * 60)
    ensure_dirs()
    video_dir = os.path.join(DATA_DIR, "sample_videos")

    for item in SAMPLE_VIDEOS:
        dest = os.path.join(video_dir, item["filename"])
        download_file(item["url"], dest, item["description"])

    sample_dst = os.path.join(video_dir, "project_demo.mp4")
    if os.path.exists(DEMO_VIDEO):
        shutil.copy2(DEMO_VIDEO, sample_dst)
        ok("Included project demo video from assets/videos/")


def download_plates(use_kaggle: bool = True, api_key: str | None = None, use_hf: bool = True) -> bool:
    success = False
    if use_kaggle and check_kaggle_installed():
        cfg = KAGGLE_DATASETS["license_plates"]
        success = download_from_kaggle(
            cfg["slug"],
            os.path.join(DATA_DIR, "license_plates"),
            cfg["name"],
        )
    if success:
        return True

    if api_key and check_roboflow_installed():
        warn("Kaggle unavailable - trying Roboflow license plate dataset instead")
        cfg = ROBOFLOW_DATASETS["license_plates_roboflow"]
        if download_from_roboflow(
            api_key,
            cfg["workspace"],
            cfg["project"],
            cfg["version"],
            os.path.join(DATA_DIR, "license_plates"),
            cfg["name"],
        ):
            return True

    if use_hf:
        warn("Trying Hugging Face license plate dataset (no account needed)")
        if download_plates_hf():
            return True

    print_manual_download_instructions()
    return False


def download_helmets(api_key: str | None = None, use_hf: bool = True) -> bool:
    if api_key and check_roboflow_installed():
        cfg = ROBOFLOW_DATASETS["helmet_detection"]
        if download_from_roboflow(
            api_key,
            cfg["workspace"],
            cfg["project"],
            cfg["version"],
            os.path.join(DATA_DIR, "helmet_detection"),
            cfg["name"],
        ):
            return True

    if use_hf:
        warn("Trying Hugging Face helmet dataset (no account needed)")
        return download_helmets_hf()

    return False


def download_vehicles(api_key: str) -> bool:
    cfg = ROBOFLOW_DATASETS["vehicle_detection"]
    return download_from_roboflow(
        api_key,
        cfg["workspace"],
        cfg["project"],
        cfg["version"],
        os.path.join(DATA_DIR, "vehicle_detection"),
        cfg["name"],
    )


def download_all(use_hf: bool = True) -> None:
    load_env_credentials()
    ensure_dirs()
    download_sample_videos()

    api_key = get_roboflow_api_key(interactive=False)
    if not api_key and check_roboflow_installed():
        api_key = get_roboflow_api_key(interactive=True)

    if api_key:
        download_helmets(api_key, use_hf=use_hf)
        download_vehicles(api_key)
    else:
        warn("No Roboflow API key - using Hugging Face for plates and helmets")
        download_helmets(None, use_hf=use_hf)
        if use_hf:
            from hf_dataset_loader import download_vehicle_hf

            download_vehicle_hf()

    download_plates(use_kaggle=True, api_key=api_key, use_hf=use_hf)
    verify_datasets()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download manageable substitute datasets for Traffic Analytics System"
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--bootstrap", action="store_true", help="Minimum setup without API keys")
    parser.add_argument("--setup", action="store_true", help="Create folder structure only")
    parser.add_argument("--plates", action="store_true", help="License plate dataset")
    parser.add_argument("--helmets", action="store_true", help="Helmet detection dataset")
    parser.add_argument("--vehicles", action="store_true", help="Optional UA-DETRAC 10K sample")
    parser.add_argument("--videos", action="store_true", help="Sample traffic videos")
    parser.add_argument("--manual", action="store_true", help="Manual browser instructions")
    parser.add_argument("--status", action="store_true", help="Check download status")
    parser.add_argument("--hf", action="store_true", help="Use Hugging Face sources (no API key)")
    args = parser.parse_args()

    load_env_credentials()

    if args.setup:
        ensure_dirs()
        return

    if args.status:
        verify_datasets()
        return

    if args.manual:
        print_manual_download_instructions()
        return

    if args.bootstrap:
        bootstrap()
        verify_datasets()
        return

    if args.videos:
        download_sample_videos()
        verify_datasets()
        return

    if args.plates:
        ensure_dirs()
        api_key = get_roboflow_api_key(interactive=False)
        download_plates(use_kaggle=True, api_key=api_key, use_hf=True)
        verify_datasets()
        return

    if args.helmets:
        ensure_dirs()
        api_key = get_roboflow_api_key(interactive=False)
        download_helmets(api_key, use_hf=True)
        verify_datasets()
        return

    if args.vehicles:
        ensure_dirs()
        api_key = get_roboflow_api_key(interactive=False)
        if api_key and check_roboflow_installed():
            download_vehicles(api_key)
        else:
            warn("UA-DETRAC 10K requires Roboflow API key")
            print("  1. Sign up: https://app.roboflow.com")
            print("  2. set ROBOFLOW_API_KEY=your_key")
            print("  3. Re-run: python download_datasets.py --vehicles")
        verify_datasets()
        return

    if args.all or args.hf:
        download_all(use_hf=True)
        return

    verify_datasets()
    print_quick_start()


if __name__ == "__main__":
    main()
