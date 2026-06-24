# Model Training Plan — RTX 4050 (Pause/Resume Ready)

This document is the step-by-step plan for fine-tuning all three YOLO11 detectors for the Traffic Analytics System. Training supports **pause anytime** and **resume later** on the same GPU or a stronger one (e.g. RTX 5060).

---

## Strategy Summary

| Principle | What we do |
|---|---|
| **Never from scratch** | Start from COCO pre-trained YOLO11 weights |
| **Transfer learning** | Fine-tune on domain datasets (plates, helmets, vehicles) |
| **Best checkpoint** | Always deploy `best.pt` (highest validation mAP), not `last.pt` |
| **Early stopping** | `patience` stops when mAP plateaus — saves time, avoids overfitting |
| **Mosaic schedule** | `close_mosaic=10` (Paper 4) — disable mosaic in last 10 epochs |
| **Pause/resume** | Ctrl+C anytime → resume with `--resume` from `last.pt` |
| **Upgrade path** | Later on RTX 5060: `--continue-best` with larger batch/model |

---

## Hardware Profile: RTX 4050 (~6 GB VRAM)

| Model | Base weights | Epochs | Batch | Dataset | Est. time |
|---|---|---|---|---|---|
| Plate detector | `yolo11s.pt` | 50 | 16 | License plates (~17K) | ~1–2 h |
| Helmet detector | `yolo11s.pt` | 50 | 16 | Helmet Roboflow (~6K) | ~45–90 min |
| Vehicle detector | `yolo11m.pt` | 100 | 8 | UA-DETRAC 10K | ~3–5 h |

**Total first run:** ~5–8 hours (run overnight or in stages).

If you hit CUDA OOM, halve `batch` in `configs/pipeline_config.yaml` and resume.

---

## Phase 0 — Pre-flight Checklist

```powershell
cd c:\projects\traffic-analytics-system
.\venv\Scripts\activate

# 0. GPU check — PyTorch must see your RTX 4050 (NOT torch+cpu)
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Expected output:

```
2.6.0+cu124
CUDA: True
```

If you see `+cpu` and `CUDA: False`, install CUDA PyTorch (~2.5 GB download):

```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Then verify again. Your driver reports CUDA 12.4 — `cu124` wheels are correct.

```powershell
# 1. Verify datasets
python download_datasets.py --status

# 2. Verify training readiness (datasets + GPU + checkpoints)
python train_models.py --status

# 3. Optional dry run (no GPU training)
python train_models.py --all --dry-run
```

Expected datasets:

| Task | Path | Required |
|---|---|---|
| Plates | `data/datasets/license_plates/data.yaml` | Yes |
| Helmets | `data/datasets/helmet_detection/data.yaml` | Yes |
| Vehicles | `data/datasets/vehicle_detection/data.yaml` | Optional but recommended |

---

## Phase 1 — Train on RTX 4050 (Recommended Order)

Train **plates → helmets → vehicles** (fastest models first, builds confidence).

### Step 1: License plate detector

```powershell
python train_models.py --plates
```

Output: `models/plate_detector.pt`

### Step 2: Helmet detector

```powershell
python train_models.py --helmets
```

Output: `models/helmet_detector.pt`

### Step 3: Vehicle detector

```powershell
python train_models.py --vehicles
```

Output: `models/vehicle_detector.pt`

### Or train all in one session

```powershell
python train_models.py --all
```

---

## Phase 2 — Pause & Resume (Any Time)

### How pause works

1. Press **Ctrl+C** in the training terminal.
2. Ultralytics saves `last.pt` in the run folder.
3. Best weights so far remain in `best.pt` and are copied to `models/` when a run completes.

### Checkpoint locations

```
runs/detect/traffic_analytics/
├── plate_detector/weights/
│   ├── best.pt    ← highest validation mAP
│   └── last.pt    ← resume from here
├── helmet_detector/weights/
└── vehicle_detector/weights/
```

### Resume a paused run

```powershell
# Resume one model
python train_models.py --plates --resume

# Resume all models that have checkpoints
python train_models.py --all --resume
```

### Check progress anytime

```powershell
python train_models.py --status
```

Shows: dataset ready, fine-tuned model exists, checkpoint epoch, best mAP50.

---

## Phase 3 — Validate & Run Pipeline

```powershell
# After all three models are trained
python train_models.py --status

# Run full analytics pipeline
python ai_pipeline.py
```

Expected outputs:

- `output_video.mp4` — annotated video
- `results_log.json` — detections, tracks, plates, helmet flags

---

## Phase 4 — Later Upgrade on RTX 5060 (Optional)

When you get a stronger GPU, **do not restart from COCO**. Continue from your best weights:

```powershell
# Continue fine-tuning with same settings (new run, starts from models/*.pt)
python train_models.py --vehicles --continue-best

# Or resume an interrupted run if you copied the whole project folder
python train_models.py --vehicles --resume
```

Suggested RTX 5060 config changes in `pipeline_config.yaml`:

```yaml
vehicle:
  base_weights: yolo11l.pt
  batch: 16
  epochs: 150
  patience: 30
  imgsz: 832   # if VRAM allows
```

---

## Training Techniques Used (Report-Ready)

| Technique | Config key | Paper / rationale |
|---|---|---|
| COCO transfer learning | `pretrained: true`, `base_weights: yolo11*.pt` | Standard best practice |
| Mosaic augmentation | Ultralytics default | Paper 4 (YOLOv8-FDD) |
| Close mosaic last N epochs | `close_mosaic: 10` | Paper 4 — stabilizes final epochs |
| Early stopping | `patience: 15–20` | Prevents overfitting |
| Mixed precision (AMP) | `amp: true` | Faster training, less VRAM on 4050 |
| Cosine LR schedule | `cos_lr: true` | Better convergence in later epochs |
| Periodic checkpoints | `save_period: 5` | Safety net if crash before epoch end |
| Image size 640 | `imgsz: 640` | Balance speed vs accuracy on 4050 |

**YOLOv8-FDD (Paper 4):** Custom architecture not in Ultralytics (~0.7% gain). We cite it in the report and use YOLO11 + the same training schedule instead.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| CUDA out of memory | Reduce `batch` by half in config, then `--resume` |
| Training seems stuck | Check GPU usage in Task Manager; first epoch compiles caches |
| `No dataset` error | Run `python download_datasets.py --plates --helmets --vehicles` |
| Resume says no checkpoint | Run without `--resume` to start fresh, or check `runs/detect/` exists |
| Want to retrain from scratch | Delete run folder under `runs/detect/traffic_analytics/<name>/` |
| Best model not in `models/` | Run completed? Copy manually: `copy runs\...\best.pt models\plate_detector.pt` |

---

## Success Criteria

| Model | Target mAP50 | Deploy path |
|---|---|---|
| Plate detector | ≥ 75% | `models/plate_detector.pt` |
| Helmet detector | ≥ 70% | `models/helmet_detector.pt` |
| Vehicle detector | ≥ 80% | `models/vehicle_detector.pt` |

View training curves:

```powershell
# TensorBoard (optional)
tensorboard --logdir runs/detect/traffic_analytics
```

Or open `runs/detect/traffic_analytics/<name>/results.csv` in Excel.

---

## Quick Command Reference

```powershell
python train_models.py --status              # Check everything
python train_models.py --plates              # Train plates
python train_models.py --helmets             # Train helmets
python train_models.py --vehicles            # Train vehicles
python train_models.py --all                 # Train all (fresh)
python train_models.py --all --resume        # Resume all paused runs
python train_models.py --vehicles --resume   # Resume one model
python train_models.py --vehicles --continue-best  # Fine-tune further from models/
python train_models.py --all --dry-run       # Validate config only
python ai_pipeline.py                        # Run pipeline after training
```
