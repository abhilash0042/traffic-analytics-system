# Traffic & Vehicle Analytics System

AI-powered traffic analytics from camera and dashcam footage: **vehicle detection**, **multi-object tracking**, **ANPR**, and **helmet violation** detection.

Major project built with **YOLO11** fine-tuning, **DeepSORT** tracking, and **EasyOCR**, informed by six research papers (see [`docs/`](docs/)).

**Repository:** [github.com/abhilash0042/traffic-analytics-system](https://github.com/abhilash0042/traffic-analytics-system)

---

## Features

| Module | Method | Status |
|---|---|---|
| Vehicle detection | YOLO11 + COCO transfer learning | Pipeline ready |
| Object tracking | DeepSORT (persistent IDs) | Pipeline ready |
| License plate detection | YOLO11 fine-tuned on Indian plates | **Trained — 98.3% mAP50** |
| Helmet detection | YOLO11 fine-tuned on Roboflow dataset | Training next |
| ANPR (OCR) | EasyOCR + regex validation + voting | Pipeline ready |

---

## Tech stack

- **Detection:** Ultralytics YOLO11 (YOLO11s / YOLO11m)
- **Tracking:** deep-sort-realtime
- **OCR:** EasyOCR
- **Config:** YAML (`configs/pipeline_config.yaml`)
- **GPU:** CUDA PyTorch (tested on RTX 4050 Laptop)

---

## Project structure

```
traffic-analytics-system/
├── ai_pipeline.py              # Main pipeline (detect → track → ANPR → helmet)
├── anpr_pipeline.py            # Plate detect + OCR + validation
├── train_models.py             # Fine-tune YOLO11 (pause/resume supported)
├── download_datasets.py        # Download Roboflow / HuggingFace datasets
├── model_utils.py              # Config + model path helpers
├── training_utils.py           # Checkpoints, GPU checks, weight sync
├── configs/pipeline_config.yaml
├── assets/videos/              # Demo input video
├── weights/                    # COCO pretrained weights (auto-downloaded)
├── models/                     # Fine-tuned detectors (after training)
├── data/datasets/              # Training data (download locally, gitignored)
├── runs/                       # Training logs & checkpoints (gitignored)
├── output/                     # Pipeline output video + JSON log
└── docs/                       # Research analysis, plans, training guide
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
python train_models.py --vehicles
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
