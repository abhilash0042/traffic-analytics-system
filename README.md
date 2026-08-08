# Traffic & Vehicle Analytics System

AI-powered traffic analytics from camera and dashcam footage: **vehicle detection**, **multi-object tracking**, **ANPR**, and **helmet violation** detection.

Major project built with **YOLO11** fine-tuning, **DeepSORT** tracking, and **EasyOCR**, informed by six research papers (see [`docs/`](docs/)).

**Repository:** [github.com/abhilash0042/traffic-analytics-system](https://github.com/abhilash0042/traffic-analytics-system)

---

## Features & Performance Results

| Module | Architecture / Model | Precision | Recall | mAP50 / F1 | R² Score / Accuracy | Status |
|---|---|---|---|---|---|---|
| **Vehicle Detection** | YOLO11m + UA-DETRAC | **97.59%** | **96.58%** | **98.25% mAP50** | **0.8839 mAP50-95** | Trained & Integrated |
| **License Plate Detection** | YOLO11s (Indian Plates) | **96.80%** | **95.40%** | **98.30% mAP50** | **0.8950 mAP50-95** | Trained & Integrated |
| **ANPR / OCR Engine** | EasyOCR + Layout Correction | **84.90%** | **82.10%** | **83.48% F1** | **84.90% Accuracy** | Pipeline Ready |
| **Helmet Violation Detection** | YOLO11s + Roboflow | **82.40%** | **78.60%** | **81.40% mAP50** | **76.20% F1-Score** | Trained & Integrated |
| **Speed Estimation** | Virtual-Line Tracking | — | — | **3.16 km/h MAE** | **R² = 0.942** | Integrated |

---

## Tech stack

- **Detection:** Ultralytics YOLO11 (YOLO11s / YOLO11m)
- **Tracking:** deep-sort-realtime / BoT-SORT
- **OCR:** EasyOCR with position-aware layout correction
- **Config:** YAML (`configs/pipeline_config.yaml`)
- **GPU:** CUDA PyTorch (tested on RTX 4050 Laptop)

---

## Project structure

```
traffic-analytics-system/
├── src/                        # Core AI pipelines & modules
│   ├── ai_pipeline.py          # Main pipeline (detect → track → ANPR → helmet)
│   ├── anpr_pipeline.py        # Plate detect + OCR + layout validation
│   ├── road_segmentation.py    # Drivable area mask & filtering
│   ├── speed_estimation.py    # Virtual line crossing & speed calculator
│   ├── zero_dce.py             # Low-light image enhancement
│   ├── model_utils.py          # Configuration & model loading utilities
│   └── training_utils.py       # Checkpoints & weight sync helpers
├── scripts/                    # Dataset download & model training scripts
│   ├── download_datasets.py    # Roboflow & HuggingFace dataset downloader
│   ├── train_models.py         # YOLO11 fine-tuning script
│   └── retrain_yolo_night.py   # Nighttime fine-tuning script
├── tests/                      # Diagnostic and test scripts
│   ├── test_ocr.py             # OCR engine unit tests
│   ├── test_indian_datasets.py # Validation tests on Indian traffic frames
│   └── compare_ocr_configs.py  # OCR configuration benchmark
├── configs/                    # Pipeline configuration files
├── assets/videos/              # Sample input videos
├── output/                     # Output video (output_video.mp4) & JSON logs
├── models/                     # Trained model weights
└── docs/                       # Research papers, plans, and evaluation report
```

---

## Quick start

### 1. Clone and set up environment

```powershell
git clone https://github.com/abhilash0042/traffic-analytics-system.git
cd traffic-analytics-system
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# GPU training/inference (RTX 4050/5060 — CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. Bootstrap demo

```powershell
python download_datasets.py --bootstrap
python ai_pipeline.py
```

Outputs: `output/output_video.mp4`, `output/results_log.json`

### 3. Download training datasets (optional)

```powershell
copy .env.example .env
# Add ROBOFLOW_API_KEY to .env

python download_datasets.py --plates --helmets --vehicles
```

---

## Model training

Full guide: [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md)

```powershell
python train_models.py --status
python train_models.py --plates      # done — 98.3% val mAP50
python train_models.py --helmets
python train_models.py --vehicles    # local RTX 4050 (~3–5 h)
# Or Kaggle GPU: docs/KAGGLE_VEHICLE_TRAINING.md
```

- Pause anytime: `Ctrl+C`
- Resume: `python train_models.py --plates --resume`
- Early stopping: `patience: 4` (stops when val mAP plateaus)

---

## Configuration

All paths and hyperparameters live in `configs/pipeline_config.yaml`:

| Setting | Default |
|---|---|
| Plate model | `weights/yolo11s.pt` → `models/plate_detector.pt` |
| Vehicle model | `weights/yolo11m.pt` → `models/vehicle_detector.pt` |
| Input video | `assets/videos/sample_video.mp4` |
| Batch (4050) | plates/helmets: 16, vehicles: 8 |

---

## Documentation

| Document | Description |
|---|---|
| [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md) | RTX 4050 training workflow, pause/resume, GPU setup |
| [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) | Full development phases |
| [`docs/DATASET_STRATEGY.md`](docs/DATASET_STRATEGY.md) | Lightweight datasets vs paper corpora |
| [`docs/RESEARCH_PAPERS_ANALYSIS.md`](docs/RESEARCH_PAPERS_ANALYSIS.md) | Six paper summaries and methodology |

---

## Environment variables

Copy `.env.example` to `.env`:

```env
ROBOFLOW_API_KEY=your_key_here
```

Kaggle credentials optional (HuggingFace fallback for license plate data).

---

## License

Academic / major project use. Research PDFs are kept locally in `docs/papers/` (not committed).
