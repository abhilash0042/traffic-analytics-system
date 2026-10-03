# Dataset Manifest & Augmentation Tracker

This document tracks all active, merged, augmented, and deprecated datasets within the Traffic Analytics System.
It serves as an inventory and rollback log so that any generated or augmented datasets can be inspected, compared, or deleted cleanly at any time without impacting raw baseline datasets.

---

## 1. Dataset Status Summary

| Dataset Name | Type / Domain | Status | Format | Source / Generation Method | Removable / Rollback |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`vehicle_detection`** | Fixed CCTV (UA-DETRAC) | 🟢 **ACTIVE** (Fixed) | YOLO | Original (Class mapping corrected) | ❌ Raw Base Data |
| **`perfect_indian_license_plates`** | Scraped Indian Plates | 🟢 **ACTIVE** | YOLO | Consolidated clean base | ❌ Raw Base Data |
| **`dashcop_yolo`** | 1440p Indian Dashcam | 🟢 **ACTIVE** | YOLO | Original extracted | ❌ Raw Base Data |
| **`ccpd_yolo`** | Chinese License Plates | 🟢 **ACTIVE** (Pretrain) | YOLO | Original extracted | ❌ Pretrain Base |
| **`videoset1_videos`** | 50x Dashcam MP4s | 🟢 **ACTIVE** | Raw Video | DashCop 2K Video Corpus | ❌ Raw Base Data |
| **`videoset1_xml`** | Dashcam Track Annotations | 🟢 **ACTIVE** | CVAT XML | 514k Box Tracks | ❌ Raw Base Data |
| **`parseq_dataset`** | Real + Synthetic OCR Crops| 🟢 **ACTIVE** | Tab GT | Normalized OCR crops | ❌ Raw Base Data |
| **`dashcop_ocr`** | Real Dashcam OCR Crops | 🟢 **ACTIVE** (Benchmark)| Filename GT| Frozen Test Benchmark | ❌ Test Benchmark |
| **`helmet_combined`** | Unified Helmet Dataset | 🔷 **MERGED & BALANCED** | YOLO | Deduplicated 80/10/10 merge | ⚠️ Re-generable via script |
| **`plate_detection_combined`** | Unified Plate Detection | 🔷 **MERGED & BALANCED** | YOLO | Deduplicated 80/10/10 merge | ⚠️ Re-generable via script |
| **`augmented/plate_detection`** | Augmented Plate Crops | ⚡ **AUGMENTED** | YOLO | Albumentations (rain, blur, HSV) | ✅ **SAFE TO DELETE** |
| **`augmented/ocr_synthetic`** | Augmented Synthetic OCR | ⚡ **AUGMENTED** | Tab GT | Albumentations (JPEG, noise, motion)| ✅ **SAFE TO DELETE** |
| **`kaggle_indian_plates2`** | Duplicate clone | 🔴 **DELETED** | VOC XML | 100% duplicate of `indian_vehicles` | N/A (Redundant) |
| **`indian_vehicles`** | Corrupted XML tags | 🔴 **DELETED** | VOC XML | 100% absorbed into `perfect_indian` | N/A (Absorbed) |
| **`license_plates_indian`** | Duplicate subset | 🔴 **DELETED** | YOLO | 100% absorbed into `perfect_indian` | N/A (Absorbed) |
| **All `.zip` archives** | Compressed archives | 🔴 **DELETED** | ZIP | Extracted folders verified healthy | N/A (Freed ~9.5 GB) |

---

## 2. Augmented Datasets Registry

All augmented data is isolated in distinct folders or prefixed with `aug_` so they never overwrite or contaminate raw baseline ground truth.

```
data/datasets/
├── augmented/                          <-- ALL AUGMENTED DATA IS ISOLATED HERE
│   ├── plate_detection/                <-- Augmented Indian plate detection (rain, blur, perspective)
│   │   ├── images/
│   │   ├── labels/
│   │   └── data.yaml
│   └── ocr_synthetic/                  <-- Augmented synthetic crops for PARSeq pretraining
│       ├── images/
│       └── gt.txt
```

### How to Safely Delete / Roll Back Augmented Data
If augmented data is not yielding accuracy gains or if disk space needs to be reclaimed immediately, run:

```bash
# Delete all augmented datasets (reclaims disk space instantly)
python scripts/manage_datasets.py --action delete_augmented

# Or delete manually:
# Remove-Item -Recurse -Force "c:\projects\traffic-analytics-system\data\datasets\augmented"
```

---

## 3. Merged Datasets Registry

| Target Directory | Source Datasets Combined | Deduplication Rule | Train / Val / Test Split |
| :--- | :--- | :--- | :--- |
| `data/datasets/helmet_combined` | `helmet_detection` + `helmet_detection_new` + `helmet_detection_extra` | MD5 Image Hash Matching | 80% / 10% / 10% |
| `data/datasets/plate_detection_combined` | `perfect_indian_license_plates` + `dashcop_yolo` + `kaggle_indian_plates` + `kaggle_indian_plates3 (116 labeled)` | MD5 Image Hash Matching | 80% / 10% / 10% |

### How to Re-generate Merged Datasets
```bash
python scripts/merge_helmet_datasets.py
python scripts/merge_plate_detection.py
```
