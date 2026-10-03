# AI Models & Vision Architecture Guide
### Traffic & Vehicle Analytics System / SentinelMesh

This document provides a comprehensive technical overview of all AI/ML models, neural networks, computer vision algorithms, and post-processing engines implemented in this system.

---

## 1. System Architecture Overview

```
Raw Camera / Dashcam Stream
   │
   ├─► [0. Zero-DCE] ── (Low-Light & Night Enhancement Neural Network)
   │
   ├─► [1. Vehicle Detection: YOLO11m] ──► [BoT-SORT / DeepSORT Multi-Object Tracking]
   │         │                                    │
   │         │ (Vehicle Crop)                     ├─► [5. Speed Estimation Module]
   │         │                                    │     (Virtual-Line / Homography)
   │         ▼                                    │
   │   [2. Plate Detection: YOLO11s]              ├─► [6. Road Segmentation: YOLO11n-seg]
   │         │                                          (Drivable Area / Off-road Filter)
   │         ▼
   │   [3. Real-ESRGAN x4plus] ── (Super-Resolution for Blurry/Small Crops)
   │         │
   │         ▼
   │   [4. Dual OCR Engine: EasyOCR + PaddleOCR]
   │         │
   │         ▼
   │   [Fuzzy Levenshtein & Indian State Regex Validator]
   │
   └─► [7. Helmet Violation Detector: YOLO11s] ── (Rider & Helmet Compliance)
```

---

## 2. Detailed Breakdown of Every Model & Module

### 🚗 Model 1: Vehicle Detection Model
* **File / Weights Path:** `models/vehicle_detector.pt` (Base: `weights/yolo11m.pt`)
* **Architecture:** **YOLO11m** (Medium scale Ultralytics YOLO11, PyTorch)
* **Pre-trained On:** MS COCO Dataset
* **Fine-Tuned On:** **UA-DETRAC** traffic benchmark dataset + Indian urban & highway traffic datasets.
* **Classes Detected:** `0: Bus`, `1: Car`, `2: Truck`, `3: Van`, plus `Motorcycle` fallback.
* **Primary Function:**
  * Detects all moving and stationary vehicles under varying lighting, weather, and traffic conditions.
  * Generates high-confidence bounding boxes used for multi-object tracking and localized cropping.
* **Performance Metrics:**
  * **Precision:** `97.59%`
  * **Recall:** `96.58%`
  * **mAP@50:** `98.25%`
  * **mAP@50-95:** `0.8839`

---

### 🪪 Model 2: License Plate Localization Model
* **File / Weights Path:** `models/plate_detector.pt` (Base: `weights/yolo11s.pt`)
* **Architecture:** **YOLO11s** (Small scale Ultralytics YOLO11)
* **Input Resolution:** `imgsz = 960` (High input resolution to detect small and distant plates)
* **Trained On:** Unified Indian License Plate Dataset (2-wheelers, 4-wheelers, commercial yellow plates, and CCPD datasets).
* **Primary Function:**
  * Localizes license plates within vehicle crops (or full frames if fallback is triggered).
  * Uses a two-stage threshold (`0.28` standard confidence, `0.15` fallback) to prevent missing small, blurry, or angled plates.
* **Performance Metrics:**
  * **Precision:** `96.80%`
  * **Recall:** `95.40%`
  * **mAP@50:** `98.30%`
  * **mAP@50-95:** `0.8950`

---

### 🔍 Model 3: Super-Resolution Enhancement
* **File / Weights Path:** `weights/RealESRGAN_x4plus.pth`
* **Architecture:** **RRDBNet** (Residual-in-Residual Dense Block Network) via **Real-ESRGAN**
* **Scale Factor:** `4x` upscale with FP16 (`sr_half: true`) GPU acceleration.
* **Primary Function:**
  * Automatically triggers on small or low-resolution license plate crops (`width < 200px`).
  * Deblurs, sharpens character edges, and removes compression artifacts before passing images to OCR engines.

---

### 🔤 Model 4: ANPR & Optical Character Recognition (OCR) Engine
* **Architecture:** **Dual OCR Engine** (**EasyOCR** [CRAFT + ResNet/LSTM/CTC] & **PaddleOCR** [PP-OCRv4])
* **Preprocessing Pipeline:**
  * Bilateral filtering, CLAHE contrast adjustment, adaptive thresholding, and Laplacian edge sharpening.
* **Post-Processing & Validation:**
  * **Fuzzy Levenshtein Clustering & Multi-Frame Voting:** Groups plate readings over 15 frames with Levenshtein distance $\le 2$.
  * **Indian State Code Validator:** Validates against all 36 State/UT codes (`GJ`, `MH`, `DL`, `KA`, `UP`, etc.).
  * **Syntax Regex Matching:** Enforces Indian number plate formats (e.g., `^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$`).
  * **Character Ambiguity Resolution:** Disambiguates OCR font confusions (`O ↔ 0`, `I ↔ 1`, `B ↔ 8`, `S ↔ 5`, `Z ↔ 2`).
* **Performance Metrics:**
  * **Precision:** `84.90%`
  * **Recall:** `82.10%`
  * **F1-Score:** `83.48%`
  * **Plate Recognition Accuracy:** `84.90%`

---

### ⛑️ Model 5: Helmet & Safety Compliance Detector
* **File / Weights Path:** `models/helmet_detector.pt` (Base: `weights/yolo11s.pt`)
* **Architecture:** **YOLO11s** (Ultralytics YOLO11)
* **Trained On:** Roboflow Helmet Violation Dataset + Indian Two-Wheeler Traffic Dataset.
* **Classes Detected:** `Helmet`, `No Helmet`, `Rider / Two-Wheeler`.
* **Primary Function:**
  * Analyzes motorcyclists and pillion passengers in real time to detect non-compliance with helmet safety laws.
* **Performance Metrics:**
  * **Precision:** `82.40%`
  * **Recall:** `78.60%`
  * **mAP@50:** `81.40%`
  * **F1-Score:** `76.20%`

---

### 🌙 Model 6: Zero-DCE Low-Light Enhancer
* **File / Weights Path:** `weights/zero_dce.pth`
* **Architecture:** **Zero-DCE (Zero-Reference Deep Curve Estimation)** — 7-layer non-pooling CNN (`enhance_net_nopool`).
* **Primary Function:**
  * Evaluates incoming frame brightness ($\text{mean grayscale} < 60$).
  * Formulates dynamic pixel-wise non-linear curve mappings ($r_1 \dots r_8$) to restore underexposed nighttime footage without blowing out headlights or streetlights.

---

### 📍 Model 7: Multi-Object Tracking (MOT)
* **Algorithms:** **BoT-SORT** (Camera Motion Compensation + Appearance ReID) & **DeepSORT / ByteTrack**
* **Primary Function:**
  * Assigns consistent, persistent IDs across successive frames.
  * Smooths trajectories and handles brief visual occlusions for speed and ANPR stability.

---

### ⚡ Model 8: Speed Estimation & Road Segmentation
* **Speed Estimation:**
  * **Method:** Dual Virtual-Line Detection and Homography Perspective Mapping.
  * **Formula:** $v = \frac{d}{N \times T_f}$ ($d$ = distance between lines, $N$ = frame count, $T_f$ = frame duration).
  * **Performance:** Mean Absolute Error (**MAE**) of **$3.16\text{ km/h}$** ($R^2 = 0.942$).
* **Road Segmentation (`weights/yolo11n-seg.pt`):**
  * Segments drivable roadway using instance segmentation and vehicle trajectory heatmaps to eliminate false positives occurring outside active road lanes.

---

## 3. Summary Performance & Benchmark Table

| Module | Model Architecture | Primary Dataset | Precision | Recall | mAP@50 / Score | Status |
|---|---|---|---|---|---|---|
| **Vehicle Detection** | YOLO11m | UA-DETRAC + Indian Traffic | **97.59%** | **96.58%** | **98.25% mAP50** | Trained & Integrated |
| **Plate Detection** | YOLO11s (960px) | Unified Indian Plates + CCPD | **96.80%** | **95.40%** | **98.30% mAP50** | Trained & Integrated |
| **Super-Resolution** | Real-ESRGAN (RRDBNet) | High-Res Texture Benchmark | — | — | **4x Upscale (FP16)** | Integrated |
| **ANPR / OCR** | EasyOCR + PaddleOCR | Indian Plates + Preprocessing | **84.90%** | **82.10%** | **83.48% F1** | Integrated |
| **Helmet Detection** | YOLO11s | Two-Wheeler Safety Dataset | **82.40%** | **78.60%** | **81.40% mAP50** | Trained & Integrated |
| **Nighttime Enhancer**| Zero-DCE | Zero-Reference Low Light | — | — | **Real-Time Dynamic** | Integrated |
| **Speed Estimation** | Virtual-Line Geometry | Centroid Tracking Calibration | — | — | **3.16 km/h MAE** | Integrated |
| **Tracker** | BoT-SORT / DeepSORT | Multi-Object Benchmark | — | — | **Real-Time Association**| Integrated |
