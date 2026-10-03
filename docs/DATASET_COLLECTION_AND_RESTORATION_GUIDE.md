# Master Dataset Collection, Extraction & Restoration Guide
### Edge-AI Traffic & Vehicle Analytics System (`sentielmesh` / `thundarstrom`)
**Author:** Abhilash ([github.com/abhilash0042/traffic-analytics-system](https://github.com/abhilash0042/traffic-analytics-system))  
**Last Updated:** October 2026  
**Document Purpose:** Complete operational blueprint documenting all dataset collection sources, original origins, extraction paths, and automated restoration workflows. Use this whenever you set up a new environment or re-download datasets after clearing local disk storage.

---

## 1. Quick Restoration (One-Command Synced Downloader)

To automatically restore and extract all datasets to their exact required directory structures:

```bash
# Download and unpack all 10 curated datasets automatically
python scripts/download_hf_datasets.py --all

# Or download individual datasets on demand:
python scripts/download_hf_datasets.py --dataset vehicle
python scripts/download_hf_datasets.py --dataset helmet
python scripts/download_hf_datasets.py --dataset plate_detection
python scripts/download_hf_datasets.py --dataset anpr_ocr
python scripts/download_hf_datasets.py --dataset anpr_benchmark
python scripts/download_hf_datasets.py --dataset tracking_video
python scripts/download_hf_datasets.py --dataset sample_videos
```

---

## 2. Complete Dataset Inventory & Extraction Directory Map

| Identifier | Sub-System / ML Task | Original Source & Description | Hugging Face Repository | Local Extraction Path | File / Sample Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`vehicle`** | Vehicle Detection (Car, Bus, Truck, Van) | UA-DETRAC Traffic Benchmark (Fixed elevated CCTV intersection cameras). Normalized 640×640 px with clean class mappings. | [`thundarstrom/traffic-vehicle-detection`](https://huggingface.co/datasets/thundarstrom/traffic-vehicle-detection) | `data/datasets/vehicle_detection/` | 23,319 images (215,109 bboxes) |
| **`helmet`** | Two-Wheeler Helmet Violation Detection | Merged and deduplicated Indian traffic & CCTV footage. Zero train-test leakage. Rebalanced helmet/no_helmet classes. | [`thundarstrom/traffic-helmet-violation`](https://huggingface.co/datasets/thundarstrom/traffic-helmet-violation) | `data/datasets/helmet_combined/` | 42,559 images (~126k bboxes) |
| **`plate_detection`** | Indian License Plate Localization | Cleaned & deduplicated real Indian vehicles (Perfect Plates, DashCop 2K, Rescued VOC). Stratified 80/10/10 split. | [`thundarstrom/indian-license-plate-detection`](https://huggingface.co/datasets/thundarstrom/indian-license-plate-detection) | `data/datasets/plate_detection_combined/` | 3,742 images (3,742 bboxes) |
| **`plate_ccpd`** | Plate Backbone Pretraining | CCPD (Chinese City Parking Dataset). Used solely for early feature representation / warm-up. | [`thundarstrom/ccpd-license-plate-pretrain`](https://huggingface.co/datasets/thundarstrom/ccpd-license-plate-pretrain) | `data/datasets/ccpd_yolo/` | 20,000 images (20k bboxes) |
| **`anpr_ocr`** | ANPR / Character Recognition | High-resolution Indian syntax plate crops (`DL01AB1234`) with aligned labels and compiled LMDB binary store. | [`thundarstrom/indian-anpr-ocr-corpus`](https://huggingface.co/datasets/thundarstrom/indian-anpr-ocr-corpus) | `data/datasets/parseq_dataset/`<br/>`data/datasets/parseq_lmdb/` | 18,537 crops + LMDB |
| **`anpr_benchmark`** | Out-of-Distribution ANPR Benchmark | DashCop tightly cropped plates from 2K dashcam streams. Low-res, blurred, dirty plate benchmark (FROZEN test set). | [`thundarstrom/indian-anpr-ocr-benchmark`](https://huggingface.co/datasets/thundarstrom/indian-anpr-ocr-benchmark) | `data/datasets/dashcop_ocr/` | 3,034 crops |
| **`anpr_synthetic`** | ANPR Font Diversity | Algorithmic synthetic plate generator covering all 36 Indian states & union territory RTO font codes. | [`thundarstrom/synthetic-indian-anpr-ocr`](https://huggingface.co/datasets/thundarstrom/synthetic-indian-anpr-ocr) | `data/datasets/kaggle_synthetic/` | 18,000 crops |
| **`tracking_video`** | Multi-Object Tracking & Speed Estimation | 50 continuous 1-minute 2K QHD (2560×1440) video drives with 100 CVAT Video 1.1 XML track files. | [`thundarstrom/traffic-surveillance-video-corpus`](https://huggingface.co/datasets/thundarstrom/traffic-surveillance-video-corpus) | `data/datasets/videoset1_videos/`<br/>`data/datasets/videoset1_xml/` | 50 videos (75k frames, 514k tracks) |
| **`sample_videos`** | Pipeline Integration & Smoke Tests | CCTV intersection, highway night drive, and traffic light demo test clips. | [`thundarstrom/traffic-pipeline-sample-videos`](https://huggingface.co/datasets/thundarstrom/traffic-pipeline-sample-videos) | `data/datasets/sample_videos/`<br/>`assets/videos/` | 6 benchmark video clips |
| **`augmented_plates`**| Robustness & Adverse Weather Perturbations | Offline photometrically augmented plate crops (rain, fog, low-light, shadow, flare). Isolated from ground truth. | [`thundarstrom/augmented-indian-plate-detection`](https://huggingface.co/datasets/thundarstrom/augmented-indian-plate-detection) | `data/datasets/augmented/plate_detection/` | 1,600 augmented samples |

---

## 3. Original Upstream Data Sources

If you wish to access the raw upstream sources directly instead of the preprocessed Hugging Face hub:

1. **UA-DETRAC Benchmark (Vehicles):**
   * Official site: `http://detrac-db.rit.albany.edu/`
   * Roboflow subset: `https://universe.roboflow.com/model-rli8w/ua-detrac-10k-sample-znazr`
2. **Indian License Plates (Localization & OCR):**
   * Kaggle: `https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels`
   * Roboflow: `https://universe.roboflow.com/object-detection-helmetslicense/motorcycle-helmet-and-license-plate-detection`
3. **Helmet Violation & Rider Safety:**
   * Roboflow Motorcycle Helmet: `https://universe.roboflow.com/traffic-analysis-td0rl/motorcycle-helmet-q0wmd-qlt95`
   * Roboflow Helmet + Plate Combo: `https://universe.roboflow.com/helmet-and-number-plate-detection-project/helmet-and-number-plate-detection-for-motorbike-safety-iityz`
4. **CCPD (Chinese City Parking Dataset):**
   * GitHub: `https://github.com/detectRecog/CCPD`
5. **Dashcam 2K Traffic Tracking Corpus:**
   * High-resolution dashcam driving streams with CVAT annotations for vehicle trajectory tracking and speed estimation.

---

## 4. Python Programmatic Download (`snapshot_download`)

If you are running in Python, Google Colab, Kaggle, or a cloud VM, use this snippet:

```python
from huggingface_hub import snapshot_download

# 1. Download Vehicle Detection (UA-DETRAC)
snapshot_download(
    repo_id="thundarstrom/traffic-vehicle-detection",
    repo_type="dataset",
    local_dir="data/datasets/vehicle_detection"
)

# 2. Download Helmet Violation Dataset
snapshot_download(
    repo_id="thundarstrom/traffic-helmet-violation",
    repo_type="dataset",
    local_dir="data/datasets/helmet_combined"
)

# 3. Download Indian License Plate Detection
snapshot_download(
    repo_id="thundarstrom/indian-license-plate-detection",
    repo_type="dataset",
    local_dir="data/datasets/plate_detection_combined"
)

# 4. Download ANPR OCR Corpus & LMDB
snapshot_download(
    repo_id="thundarstrom/indian-anpr-ocr-corpus",
    repo_type="dataset",
    local_dir="data/datasets/parseq_dataset"
)

# 5. Download Sample Videos
snapshot_download(
    repo_id="thundarstrom/traffic-pipeline-sample-videos",
    repo_type="dataset",
    local_dir="data/datasets/sample_videos"
)
```

---

## 5. Dataset Verification & Health Check

After restoring datasets, run the automated dataset health verification script:

```bash
python scripts/verify_datasets.py
```

### Verification Checks Performed:
1. **Vehicle Detection YAML Structure:** Validates split counts, paths, and class labels (`car`, `bus`, `truck`, `van`).
2. **Helmet Detection Integrity:** Confirms zero data leakage between train/val/test splits and checks class balance.
3. **Plate Detection Bounding Boxes:** Validates coordinate normalization and bounding box limits.
4. **OCR Data Quality:** Checks character vocabulary adherence (alphanumeric Indian syntax).
5. **LMDB Storage:** Verifies transaction reading and entry counts for fast training.
6. **Video Suite Integrity:** Confirms availability and readable frame counts for test video streams.

Expected Output: **`6/6 CHECKS PASSED`**

---

## 6. Model Retraining Recipes

Once datasets are restored, you can immediately train models using the repository's native training scripts:

### A. Vehicle Detector (YOLOv8s)
```bash
python scripts/train_models.py --task vehicle --model yolov8s.pt --epochs 100 --imgsz 640 --batch 32
```

### B. Helmet Violation Detector (YOLOv8s)
```bash
python scripts/train_models.py --task helmet --model yolov8s.pt --epochs 120 --imgsz 640 --batch 32
```

### C. Indian License Plate Detector (YOLOv8n)
```bash
python scripts/train_models.py --task plate --model yolov8n.pt --epochs 150 --imgsz 640 --batch 16
```

### D. License Plate OCR (PARSeq Transformer)
```bash
python scripts/train_parseq.py --data_dir data/datasets/parseq_lmdb --batch_size 256 --max_epochs 50
```

---

## 7. Running the Full AI Analytics Pipeline

To run the complete system (vehicle detection, tracking, speed calculation, helmet violation, ANPR) on a video stream:

```bash
# Run on sample CCTV / dashcam video
python -m src.ai_pipeline --source assets/videos/output_video.mp4 --view

# Or run with custom configuration
python src/ai_pipeline.py --config configs/pipeline_config.yaml
```

Outputs are automatically saved into:
* Annotated Video: `output/output_video.mp4`
* Structured Event Log: `output/results_log.json`
* Violations Log: `output/violations.csv`

---

## 8. Git and Local Storage Notes

* **Git Tracking:** All source code, configs, pipeline scripts, documentation, and trained model weights (under `models/` tracked with Git LFS) are tracked in GitHub: [github.com/abhilash0042/traffic-analytics-system](https://github.com/abhilash0042/traffic-analytics-system).
* **Local Deletion:** You can safely delete the local `traffic-analytics-system` folder from your laptop. When you clone the repository on a fresh machine or server, simply run `python scripts/download_hf_datasets.py --all` to restore all data instantly.
