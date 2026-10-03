"""Automated Hugging Face Dataset Publisher for Edge-AI Traffic Analytics System.

Uploads curated, deduplicated, and audited traffic datasets to Hugging Face
under the specified organization or user account with standardized dataset cards,
LFS attributes, split metadata, and automated archive packaging.

Usage:
    # 1. Publish all datasets:
    python scripts/publish_to_hf.py --all --public

    # 2. Publish a specific dataset only:
    python scripts/publish_to_hf.py --dataset vehicle --public
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output encoding across Windows / Linux consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from huggingface_hub import HfApi, create_repo, upload_file, whoami
    from huggingface_hub.utils import HfHubHTTPError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "datasets"

DEFAULT_LICENSE = "cc-by-4.0"

# Standard .gitattributes for Computer Vision / Video LFS handling
GITATTRIBUTES_CONTENT = """# Hugging Face Git LFS Configuration for Computer Vision Data
*.zip filter=lfs diff=lfs merge=lfs -text
*.tar.gz filter=lfs diff=lfs merge=lfs -text
*.tar filter=lfs diff=lfs merge=lfs -text
*.mp4 filter=lfs diff=lfs merge=lfs -text
*.avi filter=lfs diff=lfs merge=lfs -text
*.mkv filter=lfs diff=lfs merge=lfs -text
*.mdb filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
"""

# Catalog of distinct datasets to publish
DATASET_CATALOG: Dict[str, Dict[str, Any]] = {
    "vehicle": {
        "repo_name": "traffic-vehicle-detection",
        "title": "Edge-AI Traffic Vehicle Detection (UA-DETRAC CCTV)",
        "source_path": DATA_DIR / "vehicle_detection",
        "zip_name": "vehicle_detection.zip",
        "task_categories": ["object-detection"],
        "tags": ["traffic", "cctv", "yolo", "vehicle-detection", "ua-detrac", "smart-city"],
        "size_category": "10K<n<100K",
        "license": "cc-by-4.0",
        "description": """
## Dataset Summary
Curated and normalized **23,319 CCTV traffic images** from fixed intersection surveillance cameras (UA-DETRAC benchmark). Contains **215,109 annotated bounding boxes** in standard YOLO format across 4 vehicle classes: `car`, `bus`, `truck`, and `van`.

### Class Mapping
* **Class 0 (`car`)**: 177,403 bboxes (82.5%)
* **Class 1 (`bus`)**: 3,523 bboxes (1.6%)
* **Class 2 (`truck`)**: 16,051 bboxes (7.5%)
* **Class 3 (`van`)**: 18,132 bboxes (8.4%)

### Splits
* **Train**: 18,655 images (80%)
* **Validation**: 2,332 images (10%)
* **Test**: 2,332 images (10%)
* **Resolution**: 640x640 normalized (native 960x540)
""",
        "training_recipe": "python scripts/train_models.py --task vehicle --model yolov8s.pt --epochs 100 --imgsz 640 --batch 32",
    },
    "helmet": {
        "repo_name": "traffic-helmet-violation",
        "title": "Indian Traffic Helmet Violation Detection (Deduplicated & Rebalanced)",
        "source_path": DATA_DIR / "helmet_combined",
        "zip_name": "helmet_combined.zip",
        "task_categories": ["object-detection"],
        "tags": ["traffic", "helmet-detection", "yolo", "indian-roads", "two-wheeler", "traffic-safety"],
        "size_category": "10K<n<100K",
        "license": "cc-by-4.0",
        "description": """
## Dataset Summary
A unified, audited, and strictly deduplicated dataset of **42,559 images** and **~126,000 bounding boxes** for two-wheeler helmet compliance monitoring on Indian roads.

### Key Features
* **Zero Train/Test Leakage**: Fully verified perceptual hash partition (0 matching image hashes across splits).
* **Balanced Evaluation Split**: 80/10/10 split ensuring representative test benchmarking.
* **Classes**:
  - `0: helmet` (Full-face, open-face, construction/half helmets)
  - `1: no_helmet` (Bare heads, turbans, caps, scarves without safety helmets)

### Splits
* **Train**: 34,047 images (80%)
* **Validation**: 4,255 images (10%)
* **Test**: 4,257 images (10%)
""",
        "training_recipe": "python scripts/train_models.py --task helmet --model yolov8s.pt --epochs 120 --imgsz 640 --batch 32 --fl_gamma 1.5",
    },
    "plate_detection": {
        "repo_name": "indian-license-plate-detection",
        "title": "Indian Vehicle License Plate Localization (YOLO Format)",
        "source_path": DATA_DIR / "plate_detection_combined",
        "zip_name": "plate_detection_combined.zip",
        "task_categories": ["object-detection"],
        "tags": ["anpr", "alpr", "license-plate", "indian-plates", "yolo", "traffic"],
        "size_category": "1K<n<10K",
        "license": "cc-by-4.0",
        "description": """
## Dataset Summary
High-precision bounding box dataset containing **3,742 deduplicated real Indian vehicle images** for license plate detection and localization in crowded traffic.

### Composition
* Perfect Indian Plates (1,844 images)
* DashCop 2K Dashcam Crops (1,123 images)
* Cleaned VOC Rescued Subset (47 images)
* Merged and verified with zero train/test hash leakage.

### Splits
* **Train**: 2,993 images (80%)
* **Validation**: 374 images (10%)
* **Test**: 375 images (10%)
""",
        "training_recipe": "python scripts/train_models.py --task plate --model yolov8n.pt --epochs 150 --imgsz 640 --batch 16",
    },
    "plate_ccpd": {
        "repo_name": "ccpd-license-plate-pretrain",
        "title": "CCPD License Plate Localization Backbone Pretraining Corpus",
        "source_path": DATA_DIR / "ccpd_yolo",
        "zip_name": "ccpd_yolo.zip",
        "task_categories": ["object-detection"],
        "tags": ["anpr", "ccpd", "license-plate-detection", "yolo", "pretraining"],
        "size_category": "10K<n<100K",
        "license": "cc-by-4.0",
        "description": """
## Dataset Summary
**20,000 images** sampled from the Chinese City Parking Dataset (CCPD) converted to standard YOLO format.
Used exclusively for warm-up feature pretraining of the YOLOv8 visual backbone before fine-tuning on Indian license plates.

> **Important**: Backbone pretraining only. Do not use for final fine-tuning to prevent geographic plate shape bias.
""",
        "training_recipe": "python scripts/train_models.py --task plate --model yolov8n.pt --epochs 30 --data ccpd_yolo/data.yaml",
    },
    "anpr_ocr": {
        "repo_name": "indian-anpr-ocr-corpus",
        "title": "Indian License Plate Character Recognition (PARSeq & LMDB)",
        "source_path": DATA_DIR / "parseq_dataset",
        "zip_name": "parseq_dataset.zip",
        "task_categories": ["image-to-text"],
        "tags": ["anpr", "alpr", "ocr", "parseq", "indian-license-plate", "transformer"],
        "size_category": "10K<n<100K",
        "license": "cc-by-4.0",
        "description": """
## Dataset Summary
A curated dataset of **18,537 normalized license plate image crops** aligned strictly with Indian Motor Vehicle Act alphanumeric formats (`^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$`).

### Formats Provided
1. **Raw Crops & Ground Truth**: `parseq_dataset/` with `gt.txt` (Tab-delimited: `filename \\t text`).
2. **High-Throughput LMDB Store**: Ready for zero-bottleneck GPU memory-mapped I/O in PyTorch Lightning / PARSeq.

### Splits
* **Train**: 15,756 crops (85%)
* **Val**: 1,854 crops (10%)
* **Test**: 927 crops (5%)
* **Vocabulary**: 36 alphanumeric characters (0-9, A-Z)
""",
        "training_recipe": "python scripts/train_parseq.py --data_dir data/datasets/parseq_lmdb --batch_size 256 --max_epochs 50 --lr 7e-4",
    },
    "anpr_benchmark": {
        "repo_name": "indian-anpr-ocr-benchmark",
        "title": "DashCop Real-World Out-of-Distribution ANPR Benchmark",
        "source_path": DATA_DIR / "dashcop_ocr",
        "zip_name": "dashcop_ocr.zip",
        "task_categories": ["image-to-text"],
        "tags": ["anpr", "ocr", "benchmark", "dashcam", "out-of-distribution", "evaluation"],
        "size_category": "1K<n<10K",
        "license": "cc-by-4.0",
        "description": """
## Dataset Summary
A frozen evaluation benchmark of **3,034 real dashcam plate crops** captured in challenging conditions (low resolution median 56x34 px, motion blur, direct sunlight, rain).

### Policy
* **FROZEN TEST SET**: Never train, augment, or fine-tune models on this dataset.
* Use exclusively to evaluate zero-shot generalization of OCR models in real Indian road conditions.
""",
        "training_recipe": "# Evaluation only\\npython scripts/test_ocr.py --benchmark dashcop_ocr",
    },
    "anpr_synthetic": {
        "repo_name": "synthetic-indian-anpr-ocr",
        "title": "Synthetic Indian License Plate Character Generator Corpus",
        "source_path": DATA_DIR / "kaggle_synthetic",
        "zip_name": "kaggle_synthetic.zip",
        "task_categories": ["image-to-text"],
        "tags": ["anpr", "synthetic-data", "ocr", "indian-rto-codes", "data-augmentation"],
        "size_category": "10K<n<100K",
        "license": "cc-by-4.0",
        "description": """
## Dataset Summary
**18,000 synthetically generated Indian plate crops** covering all 36 Indian states and Union Territories with diverse fonts, spacing, distortion, and noise.
Ideal for pretraining sequence recognition models on rare RTO state codes before real data fine-tuning.
""",
        "training_recipe": "# Used for PARSeq vocabulary pre-training",
    },
    "tracking_video": {
        "repo_name": "traffic-surveillance-video-corpus",
        "title": "2K Dashcam Traffic Surveillance Video Corpus & CVAT Tracks",
        "source_path": DATA_DIR / "videoset1_videos",
        "task_categories": ["video-classification"],
        "tags": ["traffic", "tracking", "deepsort", "dashcam", "2k-video", "speed-estimation", "cvat"],
        "size_category": "10K<n<100K",
        "license": "cc-by-4.0",
        "description": """
## Dataset Summary
A continuous sequential video tracking dataset featuring **50 continuous 1-minute video drives in 2560x1440 (2K QHD)** at 25 FPS (75,000 frames total).

### Annotations Included
* Accompanied by **100 CVAT Video 1.1 XML annotation files** (`videoset1_xml`).
* **514,471 tracked bounding boxes** across continuous temporal IDs (`rider`, `motorcycle`, `helmet`, `no_helmet`, `license_plate`).
* Designed for training and evaluating DeepSORT / ByteTrack multi-object tracking, lane violation detection, and homography speed estimation.
""",
        "training_recipe": "python scripts/track_and_speed.py --video_dir data/datasets/videoset1_videos --xml_dir data/datasets/videoset1_xml",
    },
    "sample_videos": {
        "repo_name": "traffic-pipeline-sample-videos",
        "title": "End-to-End Pipeline Integration & Benchmark Video Suite",
        "source_path": DATA_DIR / "sample_videos",
        "zip_name": "sample_videos.zip",
        "task_categories": ["video-classification"],
        "tags": ["traffic", "pipeline-demo", "cctv", "benchmark-video", "edge-ai"],
        "size_category": "n<1K",
        "license": "cc-by-4.0",
        "description": """
## Dataset Summary
Curated suite of **6 high-definition video clips** (720p - 1080p) depicting diverse real-world traffic scenarios:
* Stationary CCTV highway feeds
* City intersection traffic light stops
* Nighttime driving with headlamp glare
* Congested two-wheeler corridors
Used for immediate pipeline smoke testing, RTSP stream simulation, and output visualization.
""",
        "training_recipe": "python -m src.pipeline --source data/datasets/sample_videos/sample_traffic_cctv.mp4 --view",
    },
    "augmented_plates": {
        "repo_name": "augmented-indian-plate-detection",
        "title": "Augmented Indian License Plate Perturbation Corpus",
        "source_path": DATA_DIR / "augmented" / "plate_detection",
        "zip_name": "augmented_plate_detection.zip",
        "task_categories": ["object-detection"],
        "tags": ["anpr", "data-augmentation", "robustness", "indian-plates", "yolo"],
        "size_category": "1K<n<10K",
        "license": "cc-by-4.0",
        "description": """
## Dataset Summary
**1,600 offline augmented images** with heavy weather perturbations (rain streaks, lens flare, night contrast reduction, camera perspective warps) designed to improve detector robustness in edge environments.
""",
        "training_recipe": "python scripts/train_models.py --task plate --augmented",
    },
}


def generate_dataset_card(key: str, namespace: str, metadata: Dict[str, Any]) -> str:
    """Generates standardized Hugging Face Dataset Card README.md."""
    repo_id = f"{namespace}/{metadata['repo_name']}"
    tags_yaml = "\n".join([f"  - {t}" for t in metadata.get("tags", [])])
    task_yaml = "\n".join([f"  - {tc}" for tc in metadata.get("task_categories", ["object-detection"])])

    content = f"""---
license: {metadata.get('license', DEFAULT_LICENSE)}
task_categories:
{task_yaml}
language:
  - en
tags:
{tags_yaml}
size_categories:
  - {metadata.get('size_category', '10K<n<100K')}
---

# {metadata['title']}

Part of the **Edge-AI Traffic & Vehicle Analytics System** repository by `{namespace}`.

{metadata['description'].strip()}

---

## How to Access and Download

### Using Automated Project Downloader (Extracts automatically)
```bash
# Clone / pull and auto-extract dataset
python scripts/download_hf_datasets.py --dataset {key} --org {namespace}
```

### Using `huggingface_hub` Python SDK
```python
from huggingface_hub import snapshot_download

# Download into local dataset directory
local_path = snapshot_download(
    repo_id="{repo_id}",
    repo_type="dataset",
    local_dir="data/datasets/{metadata['source_path'].name}"
)
print(f"Dataset downloaded to: {{local_path}}")
```

---

## Recommended Training / Evaluation Recipe

```bash
{metadata.get('training_recipe', '# See repository documentation for training scripts')}
```

---

## Citation & Maintainer
* **Maintained by**: `{namespace}`
* **Project**: Edge-AI Real-Time Traffic Violation Detection & ANPR
* **License**: {metadata.get('license', DEFAULT_LICENSE).upper()}
"""
    return content


def get_hf_token() -> Optional[str]:
    """Retrieves Hugging Face token from environment, .env file, or cache."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN=") or line.startswith("HUGGING_FACE_HUB_TOKEN="):
                val = line.split("=", 1)[1].strip().strip("\"'")
                if val:
                    return val
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        t = token_path.read_text(encoding="utf-8").strip()
        if t:
            return t
    return None


def zip_folder(source_dir: Path, target_zip: Path) -> None:
    """Packs directory contents into a zip archive with fast uncompressed store."""
    if target_zip.exists():
        target_zip.unlink()
    print(f"   [PACK] Compiling {target_zip.name}...", flush=True)
    with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(str(source_dir)):
            for file in files:
                fp = Path(root) / file
                if fp == target_zip or fp.name.endswith(".tmp"):
                    continue
                rel_path = fp.relative_to(source_dir)
                zf.write(fp, arcname=str(rel_path))


def publish_dataset(
    key: str,
    namespace: str,
    token: str,
    is_private: bool = False,
    dry_run: bool = False,
) -> bool:
    """Uploads a dataset to Hugging Face with auto-packaging and disk cleanup."""
    meta = DATASET_CATALOG.get(key)
    if not meta:
        print(f"[ERROR] Unknown dataset key: {key}", flush=True)
        return False

    repo_name = meta["repo_name"]
    repo_id = f"{namespace}/{repo_name}"
    source_path: Path = meta["source_path"]

    print("\n" + "=" * 70, flush=True)
    print(f"[DATASET] Processing: {meta['title']}", flush=True)
    print(f"   Target HF Repo: https://huggingface.co/datasets/{repo_id}", flush=True)
    print(f"   Source Folder : {source_path}", flush=True)
    print("=" * 70, flush=True)

    if not source_path.exists() or not source_path.is_dir():
        print(f"[WARN] Directory missing: {source_path}. Skipping.", flush=True)
        return False

    if dry_run:
        print("   [DRY-RUN OK] Ready for upload.", flush=True)
        return True

    api = HfApi(token=token)

    # 1. Create repository
    try:
        print(f"   Creating / verifying repository '{repo_id}'...", flush=True)
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=is_private,
            exist_ok=True,
            token=token,
        )
        print("   [OK] Repository verified.", flush=True)
    except Exception as e:
        print(f"   [ERROR] Error creating {repo_id}: {e}", flush=True)
        return False

    # 2. Upload metadata files (README, .gitattributes, data.yaml if exists)
    try:
        readme_content = generate_dataset_card(key, namespace, meta)
        readme_file = source_path / "README.md"
        readme_file.write_text(readme_content, encoding="utf-8")

        gitattr_file = source_path / ".gitattributes"
        gitattr_file.write_text(GITATTRIBUTES_CONTENT, encoding="utf-8")

        api.upload_file(
            path_or_fileobj=str(readme_file),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Update Dataset Card",
        )
        api.upload_file(
            path_or_fileobj=str(gitattr_file),
            path_in_repo=".gitattributes",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Configure Git LFS",
        )

        yaml_file = source_path / "data.yaml"
        if yaml_file.exists():
            api.upload_file(
                path_or_fileobj=str(yaml_file),
                path_in_repo="data.yaml",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                commit_message="Add YOLO data.yaml configuration",
            )
        print("   [OK] Metadata and configuration files uploaded.", flush=True)

        # 3. Handle data files upload
        if key == "tracking_video":
            # Upload MP4s and XMLs directly
            mp4_files = list(source_path.glob("*.mp4"))
            xml_dir = DATA_DIR / "videoset1_xml"
            xml_files = list(xml_dir.glob("*.xml")) if xml_dir.exists() else []
            print(f"   Uploading {len(mp4_files)} MP4 video streams and {len(xml_files)} track annotations...", flush=True)
            for idx, f in enumerate(mp4_files):
                print(f"   [{idx+1}/{len(mp4_files)}] Uploading {f.name} ({f.stat().st_size / (1024**2):.1f} MB)...", flush=True)
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=f"videos/{f.name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                )
            for idx, x in enumerate(xml_files):
                api.upload_file(
                    path_or_fileobj=str(x),
                    path_in_repo=f"annotations/{x.name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                )
        else:
            # Package into zip archive, upload, and delete zip to reclaim disk
            zip_name = meta.get("zip_name", f"{key}_dataset.zip")
            zip_path = DATA_DIR / zip_name
            if not zip_path.exists():
                zip_folder(source_path, zip_path)
            
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            print(f"   [UPLOADING] {zip_name} ({size_mb:,.1f} MB)...", flush=True)
            api.upload_file(
                path_or_fileobj=str(zip_path),
                path_in_repo=zip_name,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                commit_message=f"Upload {zip_name} data archive",
            )
            # Reclaim disk space
            if zip_path.exists():
                zip_path.unlink()
                print(f"   [CLEANUP] Freed {size_mb:,.1f} MB temporary archive.", flush=True)

        print(f"   [SUCCESS] Fully Published: https://huggingface.co/datasets/{repo_id}", flush=True)
        return True

    except Exception as e:
        print(f"   [ERROR] Upload failed for {repo_id}: {e}", flush=True)
        return False


def resolve_namespace(requested_namespace: Optional[str], token: Optional[str]) -> str:
    if not token:
        return requested_namespace or "thundarstrom"
    try:
        user_info = whoami(token=token)
        username = user_info.get("name", "thundarstrom")
        return requested_namespace or username
    except Exception:
        return requested_namespace or "thundarstrom"


def main():
    parser = argparse.ArgumentParser(description="Publish Traffic Analytics Datasets to Hugging Face")
    parser.add_argument("--org", help="Hugging Face organization or username (default: auto-detected)")
    parser.add_argument("--dataset", choices=list(DATASET_CATALOG.keys()), help="Publish a specific dataset")
    parser.add_argument("--all", action="store_true", help="Publish all registered datasets")
    parser.add_argument("--private", action="store_true", help="Set repository visibility to private")
    parser.add_argument("--public", action="store_true", default=True, help="Set repository visibility to public (default)")
    parser.add_argument("--dry-run", action="store_true", help="Preview dataset cards without uploading")
    parser.add_argument("--token", help="Hugging Face API token with write access (or set HF_TOKEN env var)")
    args = parser.parse_args()

    if not HF_AVAILABLE:
        print("[ERROR] 'huggingface_hub' is not installed.", flush=True)
        sys.exit(1)

    print("\n" + "=" * 70, flush=True)
    print("      [HF] SENTINELMESH - HUGGING FACE DATASET PUBLISHER       ", flush=True)
    print("=" * 70, flush=True)

    token = args.token or get_hf_token()
    if not token and not args.dry_run:
        print("\n[ERROR] Hugging Face write token not found!", flush=True)
        sys.exit(1)

    namespace = resolve_namespace(args.org, token)
    print(f"[TARGET] Publishing namespace: '{namespace}'", flush=True)

    is_private = args.private and not args.public

    if args.dataset:
        selected_keys = [args.dataset]
    elif args.all:
        selected_keys = list(DATASET_CATALOG.keys())
    else:
        print("\n[CATALOG] Available datasets:", flush=True)
        for k, v in DATASET_CATALOG.items():
            print(f"   * {k:<18} -> {v['repo_name']}")
        print("\nPlease specify --all or --dataset <name> (e.g. --dataset vehicle).")
        return

    success_count = 0
    fail_count = 0
    start_time = time.time()

    for key in selected_keys:
        ok = publish_dataset(key, namespace, token, is_private=is_private, dry_run=args.dry_run)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 70, flush=True)
    print(f"[SUMMARY] {success_count} succeeded, {fail_count} failed in {elapsed:.1f}s", flush=True)
    print(f"   Hub Profile / Namespace URL: https://huggingface.co/{namespace}", flush=True)
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    main()
