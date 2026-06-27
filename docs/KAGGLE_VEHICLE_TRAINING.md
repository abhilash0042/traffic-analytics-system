# Train Vehicle Detector on Kaggle (Free GPU)

Fine-tune `vehicle_detector.pt` on **UA-DETRAC 10K** (~23K images, 4 classes) for Indian/highway traffic scenes.

**Expected result:** validation mAP50 **~92–97%** (Paper 4 reports >95% on UA-DETRAC with YOLOv8n; YOLO11m typically matches or beats this).

---

## Why train vehicles first?

| Downstream module | How better vehicle boxes help |
|---|---|
| **Plate OCR** | Plate detector runs on cleaner vehicle crops; fewer missed cars in jam |
| **Tracking** | Stable IDs in crowded Indian traffic |
| **ANPR** | Less overlap / wrong vehicle association |
| **Helmet** | Bikes still use **COCO motorcycle fallback** (this dataset has no motorcycle class) |

---

## Overview

| Item | Value |
|---|---|
| Notebook | `notebooks/kaggle_vehicle_training.ipynb` |
| Base model | YOLO11m (`yolo11m.pt`) |
| Epochs | 100 |
| Batch | 8 (use 4 if OOM) |
| Classes | bus, car, truck, van |
| Train images | ~20,409 |
| Est. time | **4–8 hours** on Kaggle T4 |
| Output | `vehicle_detector.pt` |

---

## Step 1 — Prepare dataset on your PC

Dataset should already exist at `data/datasets/vehicle_detection/` from:

```powershell
cd c:\projects\traffic-analytics-system
.\venv\Scripts\activate
python download_datasets.py --vehicles
```

If missing, download manually from [Roboflow UA-DETRAC 10K](https://universe.roboflow.com/model-rli8w/ua-detrac-10k-sample-znazr) (YOLO format) and extract to `data/datasets/vehicle_detection/`.

Fix local paths and verify counts:

```powershell
python scripts/prepare_vehicle_dataset.py
```

Pack for Kaggle (~2–4 GB zip, **10–25 min**):

```powershell
python scripts/pack_vehicle_for_kaggle.py
```

Creates **`dist/vehicle_detection_kaggle.zip`** and **`dist/KAGGLE_VEHICLE_UPLOAD.txt`**.

---

## Step 2 — Upload to Kaggle

1. [kaggle.com](https://www.kaggle.com) → **Create** → **Dataset** → **New Dataset**
2. Upload **`dist/vehicle_detection_kaggle.zip`**
3. Title: e.g. `Traffic Vehicle UA-DETRAC`
4. Slug: **`traffic-vehicle-balanced`** (or change `DATASET_SLUG` in notebook)

---

## Step 3 — Create Kaggle notebook

1. **Create** → **Notebook**
2. **Import** → upload `notebooks/kaggle_vehicle_training.ipynb`
3. **Settings:**
   - **Accelerator:** GPU T4 x2 (or T4 x1)
   - **Internet:** ON
4. **Add Input** → your `traffic-vehicle-balanced` dataset
5. **Save Version** → **Save & Run All**

---

## Step 4 — Download results

From **Output** panel:

| File | Use |
|---|---|
| `vehicle_detector.pt` | Copy to `models/vehicle_detector.pt` |
| `vehicle_training_output.zip` | Plots, CSV, weights |

```powershell
copy Downloads\vehicle_detector.pt c:\projects\traffic-analytics-system\models\vehicle_detector.pt
python ai_pipeline.py
```

Pipeline will show: `Loading vehicle detector: vehicle_detector.pt (fine-tuned)` and enable **motorcycle COCO fallback** for helmet checks.

---

## Step 5 — Local training (RTX 4050 alternative)

If you prefer training locally (~3–5 hours):

```powershell
python scripts/prepare_vehicle_dataset.py
python train_models.py --vehicles
```

Pause/resume:

```powershell
python train_models.py --vehicles --resume
```

---

## Hyperparameters (optimized for accuracy)

These match `configs/pipeline_config.yaml` and the Kaggle notebook:

| Param | Value | Why |
|---|---|---|
| `yolo11m.pt` | Medium model | Best accuracy/speed tradeoff on T4 / 4050 |
| `epochs` | 100 | UA-DETRAC needs longer than plates/helmets |
| `patience` | 15 | Stop when val mAP plateaus |
| `close_mosaic` | 10 | Paper 4 schedule — sharper final boxes |
| `hsv/degrees/scale` | Traffic CCTV aug | Glare, haze, angle variation |

**Expected metrics after full training:**

| Metric | Target |
|---|---|
| val mAP50 | **0.92 – 0.97** |
| test mAP50 | **0.90 – 0.96** |
| Pipeline vehicle recall (crowded jam) | **+15–25%** vs COCO fallback |

---

## Motorcycle note (important)

UA-DETRAC labels: **bus, car, truck, van** — no motorcycle.

The pipeline uses a **hybrid**:

1. Fine-tuned model → 4-wheel vehicles (much better in traffic jams)
2. `yolo11s` COCO fallback → **motorcycle only** (class 3) for helmet detection

Config (`configs/pipeline_config.yaml`):

```yaml
models:
  vehicle:
    motorcycle_fallback: true
    finetuned_classes: [0, 1, 2, 3]
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Dataset not found` | Check Kaggle slug; zip must contain `vehicle_detection/train/images/` |
| CUDA OOM | Set `BATCH = 4` in notebook |
| Session died | Download backup `vehicle_detector.pt`; re-run with `RESUME = True` |
| Pack zip slow | Normal for ~23K images; do not interrupt |
| mAP stuck below 0.85 | Verify `data.yaml` paths; ensure full 100 epochs |

---

## After vehicle training

Recommended order:

1. ✅ Plate detector (done ~98% mAP50)
2. ✅ Helmet detector (Kaggle — resume to 50 epochs if needed)
3. **→ Vehicle detector (this guide)**
4. Re-run `python ai_pipeline.py` on `helmant_demovedio.mp4` and CCTV sample
5. Compare vehicle box stability and plate read rate
