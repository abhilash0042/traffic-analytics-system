# Train Helmet Detector on Kaggle (Free GPU)

Step-by-step guide for fine-tuning `helmet_detector.pt` on Kaggle using your **balanced** dataset (~41K train images, 1.1:1 class ratio).

---

## Overview

| Item | Value |
|---|---|
| Notebook | `notebooks/kaggle_helmet_training.ipynb` |
| Base model | YOLO11s (`yolo11s.pt`) |
| Epochs | 50 |
| Batch | 16 (reduce to 8 if CUDA OOM) |
| Expected time | ~3–5 hours on T4 |
| Output | `helmet_detector.pt` + training plots |

---

## Step 1 — Pack dataset on your PC

From the project root:

```powershell
cd c:\projects\traffic-analytics-system
.\venv\Scripts\activate
python scripts/pack_helmet_for_kaggle.py
```

This creates **`dist/helmet_detection_kaggle.zip`** (~several GB). It includes:

- `train/` (40,764 balanced images)
- `valid/` and `test/`
- `data.yaml` with Kaggle paths

It **excludes** `train/_pruned/` (old majority-class images you don't need for training).

---

## Step 2 — Upload to Kaggle as a Dataset

1. Go to [kaggle.com](https://www.kaggle.com) → sign in.
2. **Create** → **Dataset** → **New Dataset**.
3. Drag **`dist/helmet_detection_kaggle.zip`** (or upload via browser).
4. Title: e.g. `Traffic Helmet Balanced`
5. Set visibility (Private is fine).
6. After upload, open the dataset → **Copy API command** and note the slug.
   - Default expected slug: **`traffic-helmet-balanced`**
   - Folder inside zip must be: **`helmet_detection/`**

If you use a different slug, edit the first code cell in the notebook:

```python
DATASET_SLUG = "your-dataset-slug-here"
```

---

## Step 3 — (Optional) Upload partial weights for `--continue-best`

If you already have a local checkpoint to refine:

1. Create a second small dataset on Kaggle with **`helmet_detector.pt`** (from `models/`).
2. Slug suggestion: **`traffic-helmet-weights`**
3. In the notebook, set:

```python
CONTINUE_FROM_INPUT = True
WEIGHTS_SLUG = "traffic-helmet-weights"
```

If starting fresh from COCO weights only, leave `CONTINUE_FROM_INPUT = False`.

---

## Step 4 — Create Kaggle Notebook

### Option A — Upload notebook file

1. **Create** → **Notebook**.
2. **File** → **Import Notebook** → upload `notebooks/kaggle_helmet_training.ipynb`.
3. **Settings** (right panel):
   - **Accelerator**: **GPU T4 x1**
   - **Internet**: **On** (downloads YOLO11s weights)
   - **Persistence**: **Files only** (default)

### Option B — Copy from GitHub

If the repo is public, paste the raw notebook URL in Kaggle import.

---

## Step 5 — Add dataset(s) to the notebook

In the notebook sidebar → **Add Input**:

1. **Required**: your `traffic-helmet-balanced` dataset.
2. **Optional**: `traffic-helmet-weights` if continuing from local `helmet_detector.pt`.

Click **Save Version** → **Save & Run All** (or **Run All**).

---

## Step 6 — While training

- Progress appears in the cell output (epoch, loss, mAP).
- Kaggle sessions last **~9–12 hours** — enough for this run.
- If disconnected, checkpoints are in `/kaggle/working/runs/.../weights/last.pt`.
- Re-run the notebook with `RESUME = True` in config cell to continue.

---

## Step 7 — Download results

After training completes, download from **Output** (right panel) or from `/kaggle/working/`:

| File | Use |
|---|---|
| `helmet_detector.pt` | Copy to `models/helmet_detector.pt` locally |
| `helmet_training_output.zip` | Full run folder (plots, CSV, weights) |

On your PC:

```powershell
copy Downloads\helmet_detector.pt c:\projects\traffic-analytics-system\models\helmet_detector.pt
python ai_pipeline.py
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Dataset not found` | Check slug matches; zip must contain `helmet_detection/data.yaml` |
| CUDA OOM | Set `BATCH = 8` in notebook |
| Session died mid-run | Download `last.pt` from Output, upload as weights dataset, set `RESUME = True` |
| Slow dataloader | Set `WORKERS = 2` (Kaggle has limited CPUs) |
| mAP lower than local plate model | Normal early in training; check `results.csv` by epoch 30+ |

---

## Kaggle limits (2025)

- **GPU**: ~30 hours/week per account (T4 or P100 depending on availability).
- **Dataset size**: up to ~20 GB per dataset (zip upload).
- **Output**: save important files to `/kaggle/working/` before session ends.

---

## After helmet — vehicle on Kaggle

Same pattern: pack `vehicle_detection`, upload, adapt notebook slug and task config. Vehicle uses `yolo11m`, batch 8, 100 epochs (~longer run).
