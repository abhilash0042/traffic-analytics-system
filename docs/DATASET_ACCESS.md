# Dataset Access & Teammate Handbook
### Edge-AI Traffic & Vehicle Analytics System (`sentielmesh`)

This guide explains how teammates can easily download, synchronize, and use the project's datasets from **Hugging Face Hub** (`sentielmesh`).

---

## 1. Quick Download via Project CLI (Recommended)

The project includes an automated synchronization script (`scripts/download_hf_datasets.py`) that places datasets directly into their expected local directory structure.

### Download All Datasets
```bash
python scripts/download_hf_datasets.py --all
```

### Download Individual Datasets
```bash
# 1. Vehicle Detection (UA-DETRAC CCTV)
python scripts/download_hf_datasets.py --dataset vehicle

# 2. Two-Wheeler Helmet Violation Detection
python scripts/download_hf_datasets.py --dataset helmet

# 3. Indian License Plate Localization
python scripts/download_hf_datasets.py --dataset plate_detection

# 4. License Plate Character Recognition (PARSeq / LMDB)
python scripts/download_hf_datasets.py --dataset anpr_ocr

# 5. Dashcam Tracking Video Corpus & CVAT Annotations
python scripts/download_hf_datasets.py --dataset tracking_video

# 6. Pipeline Smoke Test & Demo Videos
python scripts/download_hf_datasets.py --dataset sample_videos
```

---

## 2. Python Programmatic Access (`snapshot_download`)

If working in custom scripts or Jupyter notebooks:

```python
from huggingface_hub import snapshot_download

# Download Vehicle Detection Dataset
snapshot_download(
    repo_id="sentielmesh/traffic-vehicle-detection",
    repo_type="dataset",
    local_dir="data/datasets/vehicle_detection"
)

# Download Helmet Violation Dataset
snapshot_download(
    repo_id="sentielmesh/traffic-helmet-violation",
    repo_type="dataset",
    local_dir="data/datasets/helmet_combined"
)

# Download Indian License Plate Detection Dataset
snapshot_download(
    repo_id="sentielmesh/indian-license-plate-detection",
    repo_type="dataset",
    local_dir="data/datasets/plate_detection_combined"
)

# Download ANPR OCR Dataset & LMDB
snapshot_download(
    repo_id="sentielmesh/indian-anpr-ocr-corpus",
    repo_type="dataset",
    local_dir="data/datasets/parseq_dataset"
)
```

---

## 3. Dataset Mapping & Immediate Training Recipes

After downloading, you can immediately train models using the repository's training scripts:

| Sub-System | Download Command | Training Command |
| :--- | :--- | :--- |
| **Vehicle Detection** | `python scripts/download_hf_datasets.py --dataset vehicle` | `python scripts/train_models.py --task vehicle --model yolov8s.pt --epochs 100 --imgsz 640 --batch 32` |
| **Helmet Violation** | `python scripts/download_hf_datasets.py --dataset helmet` | `python scripts/train_models.py --task helmet --model yolov8s.pt --epochs 120 --imgsz 640 --batch 32` |
| **Plate Detection** | `python scripts/download_hf_datasets.py --dataset plate_detection` | `python scripts/train_models.py --task plate --model yolov8n.pt --epochs 150 --imgsz 640 --batch 16` |
| **License Plate OCR**| `python scripts/download_hf_datasets.py --dataset anpr_ocr` | `python scripts/train_parseq.py --data_dir data/datasets/parseq_lmdb --batch_size 256 --max_epochs 50` |
| **Tracking Pipeline**| `python scripts/download_hf_datasets.py --dataset sample_videos` | `python -m src.pipeline --source data/datasets/sample_videos/sample_traffic_cctv.mp4 --view` |

---

## 4. Verification

After syncing datasets, run the local verification check to ensure dataset integrity:
```bash
python scripts/verify_datasets.py
```
Expected result: **6/6 CHECKS PASSED**.
