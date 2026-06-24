# Dataset Strategy & Download Guide
### Traffic & Vehicle Analytics System
### Solving the "Datasets Are Too Large" Problem

---

## The Problem

The research papers use massive datasets that are NOT practical for our laptops:

| Paper Dataset | Original Size | Why We Can't Use It |
|---|---|---|
| RideSafe-400 (Paper 3) | ~600K frames, 50+ GB | Too large, requires special access from IIIT Hyderabad |
| UA-DETRAC Full (Paper 4) | 140K+ frames, ~20 GB | Too large for local download |
| CCPD Full (Paper 2/5) | 250K+ images, ~48 GB | Way too large |
| IDD (Presentation) | ~20 GB | Requires registration |

## The Solution: Smart Subset Strategy

Instead of downloading full datasets, we use **3 strategies:**

1. **Use pre-made small subsets** from Roboflow (already in YOLO format!)
2. **Use small Kaggle datasets** (500 MB - 1 GB range)
3. **Leverage YOLOv8 pre-trained weights** (already trained on COCO — detects 80 classes including vehicles)

> **KEY INSIGHT:** YOLOv8 pre-trained on COCO already detects cars, motorcycles, buses, trucks, bicycles, and persons OUT OF THE BOX. We only need custom datasets for:
> - License plate detection (not in COCO)
> - Helmet vs no-helmet classification (not in COCO)
> - Fine-tuning for Indian traffic conditions (optional but improves accuracy)

---

## What We Actually Need to Download

### Dataset 1: Vehicle Detection (Fine-tuning)
**We DON'T need a separate dataset for basic vehicle detection — COCO pre-trained handles this!**

But for FINE-TUNING on traffic surveillance angles (overhead CCTV vs dashcam), use:

| Option | Source | Size | Images | Format | Link |
|---|---|---|---|---|---|
| **Option A (Recommended)** | UA-DETRAC 10K Sample on Roboflow | ~800 MB | 10,000 | YOLOv8 ready | https://universe.roboflow.com/model-rli8w/ua-detrac-10k-sample-znazr |
| Option B | Kaggle UA-DETRAC subset | ~2 GB | ~10,000 | Needs conversion | https://www.kaggle.com/datasets |
| Option C | Skip — use COCO pre-trained only | 0 MB | 0 | Already done | `yolov8n.pt` |

**Recommendation:** Start with **Option C** (pre-trained, no download needed). Move to **Option A** only if accuracy on surveillance footage is poor.

---

### Dataset 2: License Plate Detection & OCR (REQUIRED — not in COCO)
This is the most critical custom dataset we need.

| Option | Source | Size | Images | Format | Link |
|---|---|---|---|---|---|
| **Option A (Recommended)** | Kaggle: "Indian License Plates with Labels" | ~200-400 MB | ~4,000+ | YOLO format | Search "Indian license plate YOLO" on Kaggle |
| Option B | Kaggle: "Indian Vehicle Number Plate YOLO Annotation" | ~300 MB | ~2,500+ | YOLO format | Search on Kaggle |
| Option C | Roboflow: License plate datasets | ~200-500 MB | Varies | YOLOv8 ready | Search "license plate" on Roboflow Universe |

**Recommendation:** Use **Option A** — Indian-specific, YOLO-formatted, manageable size.

---

### Dataset 3: Helmet Detection (REQUIRED for violation detection)

| Option | Source | Size | Images | Format | Link |
|---|---|---|---|---|---|
| **Option A (Recommended)** | Roboflow: "Helmet and Number Plate Detection for Motorbike Safety" | ~500 MB | 8,000-20,000 | YOLOv8 ready | Search "motorcycle helmet" on Roboflow Universe |
| Option B | Roboflow: "Bike Helmet Detection" | ~200 MB | ~5,000 | YOLOv8 ready | Search "bike helmet detection" on Roboflow |
| Option C | Kaggle: Helmet detection datasets | ~300 MB | Varies | May need conversion | Search "helmet detection" on Kaggle |

**Recommendation:** Use **Option A** — includes both helmet AND number plate classes together!

---

### Dataset 4: Traffic Violation Detection (Optional — good for testing)

| Option | Source | Size | Images | Format | Link |
|---|---|---|---|---|---|
| **Option A** | Kaggle: "Traffic Violation Detection Dataset" | ~500 MB | Varies | YOLOv8/v9 | https://www.kaggle.com/datasets/guisahanes/traffic-violation-detection-dataset |
| Option B | Roboflow: "Traffic Violation Detection" | ~300 MB | Varies | YOLOv8 ready | Search on Roboflow Universe |
| Option C | Roboflow: "Wrong Way Driving Detection" | ~100 MB | Varies | YOLOv8 ready | Search on Roboflow Universe |

**Recommendation:** This is optional. Our violation detection is logic-based (tracking + geometry from Paper 5), not purely detection-based.

---

### Dataset 5: Sample Test Videos (REQUIRED for end-to-end testing)

| Option | Source | Size | Duration | Link |
|---|---|---|---|---|
| **Option A (Recommended)** | YouTube: Indian traffic CCTV footage | ~50-200 MB | 2-5 min clips | Download via yt-dlp |
| Option B | Intel OpenVINO sample videos | ~100 MB | Various | Intel's model zoo |
| Option C | Record your own (phone/dashcam) | ~50 MB | 1-2 min | Your camera |

**Recommendation:** Download 3-5 short clips from YouTube showing Indian traffic intersections.

---

## Total Download Summary

| Dataset | Size | Priority | Status |
|---|---|---|---|
| YOLOv8 pre-trained (COCO) | 6 MB | ✅ Must Have | ✅ Already done |
| Indian License Plates (Kaggle) | ~300 MB | 🔴 Must Have | ⬜ To download |
| Helmet Detection (Roboflow) | ~400 MB | 🔴 Must Have | ⬜ To download |
| Sample Traffic Videos | ~200 MB | 🔴 Must Have | ⬜ To download |
| UA-DETRAC 10K (Roboflow) | ~800 MB | 🟡 Nice to Have | ⬜ Optional |
| Traffic Violations (Kaggle) | ~500 MB | 🔵 Optional | ⬜ Optional |

### **Total MUST-HAVE download: ~900 MB** (under 1 GB!)
### **Total with optional: ~2.2 GB** (very manageable)

---

## Step-by-Step Download Instructions

### Step 1: Create the dataset folder structure
```bash
cd c:\projects\traffic-analytics-system
mkdir data
mkdir data\datasets
mkdir data\datasets\license_plates
mkdir data\datasets\helmet_detection
mkdir data\datasets\vehicle_detection
mkdir data\datasets\sample_videos
```

### Step 2: Download Indian License Plate Dataset from Kaggle

1. Go to https://www.kaggle.com
2. Search: **"Indian license plate YOLO"** or **"Indian License Plates with Labels"**
3. Download the ZIP file (~300 MB)
4. Extract to: `data/datasets/license_plates/`
5. Verify structure:
   ```
   license_plates/
   ├── images/
   │   ├── train/
   │   └── val/
   ├── labels/
   │   ├── train/
   │   └── val/
   └── data.yaml
   ```

### Step 3: Download Helmet Detection Dataset from Roboflow

1. Go to https://universe.roboflow.com
2. Search: **"motorcycle helmet detection"** or **"helmet and number plate"**
3. Click on a dataset with >5,000 images
4. Click **"Download Dataset"** → Select **"YOLOv8"** format
5. Download and extract to: `data/datasets/helmet_detection/`

### Step 4: Download Sample Traffic Videos

Option A - YouTube (use yt-dlp):
```bash
pip install yt-dlp
# Search for "Indian traffic CCTV footage" and download short clips
yt-dlp -f "best[height<=720]" --max-filesize 50M -o "data/datasets/sample_videos/%(title)s.%(ext)s" <VIDEO_URL>
```

Option B - Use the sample video we already have in the project.

### Step 5 (Optional): Download UA-DETRAC 10K from Roboflow

1. Go to https://universe.roboflow.com/model-rli8w/ua-detrac-10k-sample-znazr
2. Click **"Download Dataset"** → Select **"YOLOv8"** format
3. Download and extract to: `data/datasets/vehicle_detection/`

---

## How Each Dataset Maps to Our Pipeline

```
┌──────────────────────────────────────────────────┐
│            OUR AI PIPELINE                       │
├──────────────────────────────────────────────────┤
│                                                  │
│  1. Vehicle Detection ◄── COCO Pre-trained       │
│     (car, bike, bus, truck)  (Already have!)      │
│     Optional: + UA-DETRAC 10K fine-tuning        │
│                                                  │
│  2. Object Tracking ◄── No dataset needed        │
│     (DeepSORT)        (Algorithm-based)           │
│                                                  │
│  3. License Plate Detection ◄── Kaggle Indian LP │
│     (plate bounding box)       (~300 MB)          │
│                                                  │
│  4. License Plate OCR ◄── Same Kaggle dataset    │
│     (text recognition)   + EasyOCR pre-trained    │
│                                                  │
│  5. Helmet Detection ◄── Roboflow Helmet Dataset │
│     (helmet / no-helmet)    (~400 MB)             │
│                                                  │
│  6. Speed Estimation ◄── No dataset needed       │
│     (virtual line method)  (Algorithm-based)      │
│                                                  │
│  7. Violation Detection ◄── No dataset needed    │
│     (signal, wrong-way)    (Logic + tracking)     │
│                                                  │
│  8. Test/Demo ◄── Sample Traffic Videos           │
│                    (~200 MB)                      │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## Data Augmentation Strategy (from Papers)

Once we have the small datasets, we **augment** them to increase effective size (Paper 4 & 5 approach):

| Augmentation | Library | Purpose | Paper Reference |
|---|---|---|---|
| Mosaic | Ultralytics (built-in) | Combines 4 images into 1, diverse scenes | Paper 4 (disabled last 10 epochs) |
| Random brightness/contrast | OpenCV / Albumentations | Simulate day/night/weather | Paper 5 (augmented ×2.3) |
| Motion blur synthesis | OpenCV | Simulate camera shake / moving vehicles | Paper 5 |
| Random crop/flip | Ultralytics (built-in) | Spatial diversity | Paper 3, 4 |
| HSV jitter | Ultralytics (built-in) | Color robustness | Standard practice |

**Ultralytics YOLOv8 does most augmentations AUTOMATICALLY during training.** Just set:
```python
model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    augment=True,       # Enables auto-augmentation
    mosaic=1.0,         # Mosaic probability (1.0 = always)
    close_mosaic=10,    # Disable mosaic for last 10 epochs (Paper 4 approach)
)
```

---

## Comparison: Full Datasets vs Our Subsets

| Aspect | Research Papers (Full) | Our Strategy (Subsets) | Impact |
|---|---|---|---|
| Vehicle Detection | UA-DETRAC: 140K images | COCO pre-trained (0 download) + optional 10K | Minimal — pre-trained is excellent |
| License Plates | 11K-27K images | Kaggle: 4K images | ~2-5% accuracy drop, acceptable |
| Helmet | 37K helmet + 23K no-helmet | Roboflow: 5K-20K images | Similar or slightly lower F1 |
| Training Time | 4 days on 4× RTX 3080 Ti | 2-4 hours on single GPU/CPU | Much faster iteration |
| Total Storage | 50-250 GB | **~1-2 GB** | 100× smaller! |
| Accuracy (expected) | State-of-the-art | 80-90% of paper results | Good enough for project demo |

---

*Last Updated: 2026-06-24*
*Project: Traffic & Vehicle Analytics System*
