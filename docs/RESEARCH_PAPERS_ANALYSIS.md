# Research Papers — Complete Analysis & Reference Guide
### Traffic & Vehicle Analytics System

> **Purpose:** This document is a permanent reference for all research papers used in this project.
> Use it to quickly look up any methodology, dataset, metric, or architecture detail during development.

---

## Table of Contents

1. [Paper 1: AI-Enhanced Real-Time Traffic Surveillance (Review)](#paper-1)
2. [Paper 2: Automatic Framework for Number Plate Detection (ANPR)](#paper-2)
3. [Paper 3: DashCop — Two-Wheeler Violations using Dashcam (WACV 2025)](#paper-3)
4. [Paper 4: YOLOv8-FDD — Improved YOLOv8 for Vehicle Detection (IEEE 2024)](#paper-4)
5. [Paper 5: Edge-AI Perception Node for Cooperative Road Safety (IISc/Bosch 2026)](#paper-5)
6. [Paper 6: Project Presentation (Internal)](#paper-6)
7. [Cross-Paper Comparison Table](#cross-paper-comparison)
8. [Combined Dataset Reference](#combined-dataset-reference)
9. [Technology Stack Derived from Papers](#technology-stack)

---

<a id="paper-1"></a>
## Paper 1: AI-Enhanced Real-Time Traffic Surveillance

| Field | Details |
|---|---|
| **Full Title** | AI-Enhanced Real-Time Traffic Surveillance: A Global Review of Vehicle Monitoring, Analytics, and Innovations Addressing Traditional Limitations |
| **Type** | Literature Review / Survey |
| **Pages** | 17 |
| **File** | `AI-Enhanced Real-Time Traffic Surveillance_.pdf` |
| **Our Use** | Problem Statement, Literature Review, Justification |

### What This Paper Covers
A comprehensive global review that identifies AI-based solutions addressing limitations of traditional traffic surveillance — limited coverage, processing delays, and lack of real-time analytics.

### Problem Statement (Use this in our report)
Traditional traffic surveillance systems face:
- **Limited Coverage:** Fixed cameras cover only localized point-specific areas
- **Processing Delays:** Centralized architecture → "minutes to hours" response time
- **No Real-Time Analytics:** Sensitive to illumination, weather, environmental noise
- **India-Specific:** 330M+ registered vehicles but only ~80,000 traffic officers (1 officer per 4,000 vehicles)

### Key Systems Reviewed

| System | Detection Model | Tracker | Application | Performance |
|---|---|---|---|---|
| YOLOv8 + DeepSORT | YOLOv8 | DeepSORT | General vehicle detection & tracking | 96.58% detection, 97.54% counting |
| DashCop (India) | Modified YOLOv8 | Cross-Association Tracker | Two-wheeler violations (helmet, triple riding) | F1: 72.18% (auto), 82.05% (HITL) |
| YOLO11 for ANPR | YOLO11 | Various | License plate recognition | 22% fewer params than YOLOv8m |
| YOLOv9 + DeepSORT | YOLOv9 | DeepSORT | Urban vehicle counting | Effective integration |
| ANPRSmaC | YOLOv8 + CNN | — | Number plate recognition | 0.994 accuracy |

### AI vs Traditional Systems (Use in report comparison)

| Dimension | Traditional | AI-Enhanced |
|---|---|---|
| Detection Accuracy | Inconsistent, environment-degraded | 95–98% under diverse conditions |
| Operating Hours | Limited by human factors | 24/7 continuous |
| Response Time | Minutes to hours | Real-time (seconds) |
| Scalability | Labor-intensive | Single system monitors multiple lanes |
| Environmental Performance | Severely limited in poor conditions | High accuracy in all weather |

### Key References from This Paper
- [DashCop GitHub](https://github.com/dash-cop/DashCop)
- [DashCop Project Page](https://dash-cop.github.io/)
- [NAYAN AI](https://www.nayan.co/services/traffic-monitoring/)
- [Ultralytics YOLO11 ANPR Blog](https://www.ultralytics.com/blog/using-ultralytics-yolo11-for-automatic-number-plate-recognition)

---

<a id="paper-2"></a>
## Paper 2: Automatic Framework for Number Plate Detection

| Field | Details |
|---|---|
| **Full Title** | An Automatic Framework for Number Plate Detection using OCR and Deep Learning Approach |
| **Authors** | Yash Shambharkar et al. (Symbiosis Institute of Technology, Pune) |
| **Published** | IJACSA, Vol. 14, No. 4, 2023 |
| **Pages** | 7 |
| **File** | `Paper_2-An_Automatic_Framework_for_Number_Plate_Detection - Copy.pdf` |
| **Our Use** | ANPR Pipeline Design, Indian License Plate Handling |

### Methodology (4-Stage Pipeline)

```
Stage 1: Image Acquisition
    ↓
Stage 2: Preprocessing
    → RGB to Grayscale conversion
    → Noise removal
    → Image binarization
    → Sobel edge detection
    ↓
Stage 3: License Plate Detection
    → CNN + YOLO for plate localization
    → Rectangular bounding box extraction
    → Thresholding for segmentation
    ↓
Stage 4: Character Recognition (OCR)
    → Feature extraction:
        - Distance to Wall (DTW)
        - Cross-Time Feature (CTF)
        - Active Region Ratio (ARR)
        - Height-to-Width Ratio (HWR)
    → CNN-based character classification
```

### Dataset Details

| Category | Count | Details |
|---|---|---|
| **Total Images** | **4,326** | From Kaggle |
| Two-Wheeler | 300 | Printed (375): SL(273), ML(102); Painted (125): SL(89), ML(36) |
| Three-Wheeler | 637 | Printed (157): SL(61), ML(96); Painted (480): SL(86), ML(394) |
| Car | 2,963 | Printed (2957): SL(2873), ML(84); Painted (6) |
| Truck/Pickup | 426 | Printed (82): SL(2), ML(80); Painted (344): SL(60), ML(284) |

> SL = Single Line, ML = Multi Line

### Key Results

| Metric | Value |
|---|---|
| Vehicle Detection Accuracy | **100%** |
| License Plate Detection Accuracy | **95%** (at detection threshold 0.5) |
| Character Recognition (OCR) Accuracy | **96.23%** |
| Small plate accuracy | 98% |
| Total loss | < 10% |
| Learning rate | 92% |
| Average accuracy (incl. multiline) | 94.9% |
| Character confidence > 90% | 93.7% of characters |

### Indian-Specific Challenges Documented
1. Registration plates vary: single-row vs double-row
2. Two-wheelers and three-wheelers often have **painted** (hand-written) plates
3. Auto-rickshaws use paint rather than printing
4. Different fonts, colors, sizes across vehicle types
5. Multi-line plates have different layouts per vehicle type

### Comparison with Other Methods

| Method | Detection % | Recognition % |
|---|---|---|
| YOLO + RSM | 97% | 90% |
| CNN + TensorFlow | 96.36% | 93% |
| Deep CNN + Fast RCNN | 99% | 91% |
| RCNN + SVM + ANN | 94.4% | 96% |
| **This Paper (Proposed)** | **100% (vehicle) / 95% (plate)** | **96.23%** |

### What We Take From This Paper
- ✅ The **4-stage ANPR pipeline** structure
- ✅ Preprocessing steps specific to Indian plates
- ✅ Kaggle dataset approach (4,326 images is feasible)
- ✅ Need to handle both **printed** and **painted** plates
- ✅ Feature extraction concepts for OCR (DTW, CTF, ARR, HWR)

---

<a id="paper-3"></a>
## Paper 3: DashCop — Two-Wheeler Traffic Violations (⭐ MOST IMPORTANT)

| Field | Details |
|---|---|
| **Full Title** | DashCop: Automated E-ticket Generation for Two-Wheeler Traffic Violations Using Dashcam Videos |
| **Authors** | Deepti Rawat, Keshav Gupta, Aryamaan Basu Roy, Ravi Kiran Sarvadevabhatla |
| **Institution** | IIIT Hyderabad, India |
| **Published** | WACV 2025 (IEEE/CVF) |
| **Pages** | 11 |
| **File** | `Two-Wheeler_Traffic_Violations_using_Dashcam (1).pdf` |
| **Our Use** | CORE ARCHITECTURE — Violation detection, rider-motorcycle association, E-ticket generation |

### Complete End-to-End Pipeline

```
Input: Dashcam Video (1920×1080, 25fps)
    │
    ├── Step (a): SAC Module (Segmentation & Cross-Association)
    │   ├── YOLOv8 backbone → multi-scale features
    │   ├── Detection Head → bounding boxes + class scores (rider, motorcycle)
    │   ├── Mask Coefficient Head → instance segmentation masks
    │   ├── Cross-Object Segmentation Head → rider↔motorcycle association masks
    │   └── Output: R-M instances (Rider-Motorcycle groups)
    │
    ├── Step (b): Cross-Association-based Tracking
    │   ├── Modified SORT with joint rider+motorcycle tracking
    │   ├── Kalman filter (motion) + ReID model SBS-R50 (appearance)
    │   ├── Gurobi solver for optimization
    │   └── Output: Tracked R-M instances with persistent IDs
    │
    ├── Step (c): Violation Detection
    │   ├── Helmet Detection: YOLOv8-x → 'helmet' / 'no-helmet' per rider
    │   │   └── Majority voting across frames per rider track
    │   ├── Triple Riding Classification: CNN on SAC features
    │   │   └── Classes: 'single', 'double', 'triple', 'none'
    │   │   └── Flagged if ANY frame in track = 'triple'
    │   └── Output: Per-track violation labels
    │
    ├── Step (d): ANPR Module (for violated tracks only)
    │   ├── License Plate Detection: YOLOv8-x on R-M instance crops
    │   ├── Multi-line → single-line plate conversion
    │   ├── OCR: CRNN trained on real + synthetic + augmented plates
    │   └── Majority vote across frames for final plate number
    │
    └── Step (e): E-Ticket Generation
        └── (License plate + Violation type + R-M track frames) → Authorities
```

### Novel Technical Contributions

#### 1. SAC Module (Segmentation and Cross-Association)
- **What it does:** Learns to detect which rider belongs to which motorcycle directly in image space
- **How it works:**
  - Standard YOLO instance segmentation → rider masks + motorcycle masks
  - Novel **cross-object segmentation head**: rider query attends to motorcycle locations (and vice versa)
  - Association score computed using IoU of cross-masks:
    ```
    a(k,l) = 0.5 × (IoU(A_r^k, S_m^l) + IoU(S_r^k, A_m^l · mask(B_r^k)))
    ```
- **Why it's better:** IOU-based methods fail in dense traffic; SAC operates in image space and learns robust associations
- **Result:** 84.04% association score (vs 58.50% for prior methods)

#### 2. Cross-AssociationSORT Tracker
- **What it does:** Tracks rider-motorcycle pairs as a SINGLE entity
- **How it's different:** Standard trackers (DeepSORT, ByteTrack, BotSORT) track riders and motorcycles independently
- **Optimization:** Joint maximization with 3 terms:
  ```
  maximize: λ₁ × Σ(rider_scores) + λ₂ × Σ(motorcycle_scores) + λ₃ × Σ(association_scores)
  ```
- **Solver:** Gurobi (linear programming)
- **Result:** HOTA 60.41, MOTA 51.96, IDF1 64.58 (best among all trackers tested)

#### 3. Helmet Detection
- **Model:** YOLOv8-x
- **Training data:** 37K helmet + 23K no-helmet bounding boxes
- **Training:** 70 epochs, Adam, lr=0.01, 4× RTX 3080 Ti, 1 day
- **Method:** Frame-level detection → match to rider tracks → majority vote

#### 4. Triple Riding Classification
- **Input:** R-M instance ROI crops from tracker
- **Architecture:** Learnable conv layer on frozen SAC mask coefficient head + linear layer
- **Classes:** single, double, triple, none
- **Decision:** Track flagged if ≥1 frame classified as 'triple'

#### 5. License Plate OCR
- **Detection:** YOLOv8-x trained for license plate regions
- **OCR:** CRNN architecture trained on:
  - Real-world plates
  - Augmented plates
  - Synthetic plates (best performance when all 3 combined)
- **Consolidation:** Majority vote across frames for final plate number

### Dataset: RideSafe-400

| Statistic | Value |
|---|---|
| Total videos | **400** |
| Split | 200 train / 100 validation / 100 test |
| Resolution | 1920 × 1080 pixels |
| Frame rate | 25 fps |
| Duration per video | 60–72 seconds |
| Total frames | ~600,000 |
| Camera | DDPAI X2S Pro dashcam |
| R-M Instance annotations | 354K (23K tracks) |
| Motorcycle annotations | 356K (24K tracks) |
| Rider annotations | 311K (19K tracks) |
| Helmet annotations | 194K (10K tracks) |
| No-helmet annotations | 149K (8K tracks) |
| License plate annotations | 27K (with plate numbers) |
| 6 annotated classes | R-M instance, rider, motorcycle, helmet, no-helmet, license plate |

### Complete Performance Results

| Module | Metric | Value |
|---|---|---|
| **E-Ticket (Automated)** | Precision / Recall / F1 | 84.21 / 63.16 / **72.18%** |
| **E-Ticket (Human-in-Loop)** | Precision / Recall / F1 | 100.0 / 69.57 / **82.05%** |
| **Helmet Detection (Ours, frame)** | Precision / Recall / F1 | 68.75 / 62.78 / **65.63%** |
| **Triple Riding (TR Classifier)** | Precision / Recall / F1 | 69.22 / 66.29 / **67.72%** |
| **R-M Association (SAC)** | Association Score | **84.04%** |
| **R-M Association (IOU baseline)** | Association Score | 80.93% |
| **R-M Association (Prior work)** | Association Score | 58.50% |
| **Tracking (CrossAssocSORT)** | HOTA / MOTA / IDF1 | **60.41 / 51.96 / 64.58** |
| **Tracking (DeepOCSORT)** | HOTA / MOTA / IDF1 | 58.21 / 50.78 / 62.57 |
| **Tracking (ByteTrack)** | HOTA / MOTA / IDF1 | 56.86 / 48.60 / 60.04 |
| **LP OCR (CRNN, all data)** | CER / Accuracy | 0.0400 / **85.71%** |
| **LP OCR (GoogleOCR)** | CER / Accuracy | 0.2698 / 33.34% |
| **LP OCR (TrOCR)** | CER / Accuracy | 0.1691 / 16.41% |

### Training Details

| Component | Hardware | Duration | Optimizer | Epochs |
|---|---|---|---|---|
| SAC Module | 4× RTX 3080 Ti | 4 days | Adam, lr=0.01 | 200 |
| Helmet Detector | 4× RTX 3080 Ti | 1 day | Adam, lr=0.01 | 70 |
| Triple Riding Classifier | — | — | — | — |
| License Plate OCR (CRNN) | — | — | — | — |

### Loss Function (SAC)
```
L_SAC = λ₁·L_CIOU + λ₂·L_DFL + λ₃·L_cls + λ₄·L_seg + λ₅·L_crossseg
Where: λ₁=7.5, λ₂=1.5, λ₃=0.5, λ₄=7, λ₅=7
```

### Failure Cases Documented
- Glare and motion blur → misses triple riders
- Occlusions → helmet detection false negatives
- Distant occluded R-M instances → triple riding false positives
- Opposite-lane vehicles → low OCR recognition rate

---

<a id="paper-4"></a>
## Paper 4: YOLOv8-FDD — Improved YOLOv8 for Vehicle Detection

| Field | Details |
|---|---|
| **Full Title** | YOLOv8-FDD: A Real-Time Vehicle Detection Method Based on Improved YOLOv8 |
| **Authors** | Xiaojia Liu, Yipeng Wang, Dexin Yu, Zimin Yuan |
| **Institution** | Jimei University, Xiamen, China |
| **Published** | IEEE Access, Volume 12, 2024 |
| **Pages** | 17 |
| **File** | `YOLOv8-FDD_A_Real-Time_Vehicle_Detection_Method_Based_on_Improved_YOLOv8.pdf` |
| **Our Use** | Detection model optimization, UA-DETRAC dataset reference |

### 4 Key Improvements

#### 1. Feature Sharing Detection Head
- **Problem:** YOLOv8 decoupled head uses separate convolutions for P3, P4, P5 → 25% of total parameters in head alone
- **Solution:** Share convolutional weights across all 3 detection layers + GroupNorm (GN)
- **Effect:** Dramatic parameter reduction with minimal accuracy loss

#### 2. Feature Dynamic Interaction Detection Head
- **Problem:** Classification and regression branches don't interact → inconsistent predictions
- **Solution:**
  - Interactable Features Extractor (residual connection of 3×3 convs)
  - Regression Feature Dynamic Interactor (Deformable Convolution DCN)
  - Classification Feature Dynamic Interactor (Feature Dynamic Filter)
  - Layer Attention Mechanism for task-specific feature segmentation
- **Effect:** Better alignment between box confidence and position accuracy

#### 3. Dilation-wise Residual (DWR) Module
- **Where:** Replaces Bottleneck in C2f at backbone layers P4 and P5
- **How:** Two-step: Region Residualization (3×3 conv + BN + ReLU) → Semantic Residualization (3 branches of dilated depth convolutions)
- **Effect:** Enhanced multi-scale feature extraction from expandable receptive fields

#### 4. DySample (Dynamic Upsampling)
- **What:** Replaces nearest-neighbor interpolation upsampling in YOLOv8 neck
- **How:** Uses point sampling with learned positional biases + bilinear interpolation initialization + range-factor offset constraints + grouped upsampling
- **Effect:** Better feature information after upsampling, smoother transitions

### Dataset: UA-DETRAC

| Statistic | Value |
|---|---|
| **Source** | University at Albany |
| **Total video** | 10 hours |
| **Locations** | 24 locations in Beijing and Tianjin, China |
| **Camera** | Canon EOS 550D |
| **Frame rate** | 25 fps |
| **Angle** | Close to surveillance camera view |
| **Vehicles labeled** | 8,250 |
| **Bounding boxes** | 1.21 million |
| **Vehicle classes** | Cars, Buses, Vans, Other vehicles |
| **Weather conditions** | Cloudy, Nighttime, Sunny, Rainy |
| **Used in paper** | 10,870 images (extracted at intervals) |
| **Split** | 8,564 training / 2,306 validation (80:20) |
| **Access** | http://detrac-db.rit.albany.edu/ |

### Private Traffic Surveillance Dataset (also used)
- Real roadside monitoring from urban and rural China
- Classes: Cars, Buses, Trucks
- Weather: Sunny, Rainy, Snowy
- Time: Day and Night
- 9,846 training + 3,282 validation images

### Experimental Setup

| Setting | Value |
|---|---|
| OS | Windows 11 64-bit |
| CPU | Intel Core i5-12400F |
| GPU | NVIDIA GeForce RTX 3060 |
| Python | 3.8 |
| PyTorch | 1.13.1 |
| CUDA | 11.6 |
| Input size | 640 × 640 pixels |
| Epochs | 100 |
| Batch size | 16 |
| Learning rate | 0.01 |
| Optimizer | SGD |
| Data augmentation | Mosaic (off for last 10 epochs) |
| Pre-trained weights | **None** (trained from scratch) |

### Key Results (UA-DETRAC)

| Metric | YOLOv8n (Original) | YOLOv8-FDD (Proposed) | Change |
|---|---|---|---|
| Parameters | 3.01M (100%) | 2.19M (**72.89%**) | -27.11% |
| GFLOPs | 8.1 | 8.5 | +0.4 |
| mAP50 | baseline | baseline + **0.7%** | ↑ |
| mAP50-95 | baseline | baseline + **1.3%** | ↑ |
| FPS | >300 | >300 | Same |

### What We Take From This Paper
- ✅ **UA-DETRAC dataset** — downloadable, 4-class vehicle detection, real traffic scenes
- ✅ The subset approach: extract 10K images at frame intervals from the full dataset
- ✅ YOLOv8n is already excellent; lightweight improvements are possible
- ✅ Experimental setup reference (GPU, training params, etc.)
- ✅ Error analysis approach: false negative + false positive statistics

---

<a id="paper-5"></a>
## Paper 5: Edge-AI Perception Node (⭐ SYSTEM ARCHITECTURE REFERENCE)

| Field | Details |
|---|---|
| **Full Title** | Edge-AI Perception Node for Cooperative Road-Safety Enforcement and Connected-Vehicle Integration |
| **Authors** | Shree Charran R and Rahul Kumar Dubey |
| **Institutions** | Indian Institute of Science (IISc) Bengaluru + Bosch Global Software Technologies |
| **Published** | arXiv:2601.07845, January 2026 |
| **Pages** | 10 |
| **File** | `edge ai.pdf` |
| **Our Use** | PRODUCTION SYSTEM DESIGN — architecture, violation logic, speed estimation, auto-ROI, deployment |

### Complete System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    VIDEO INPUT (RTSP/USB)                │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  OBJECT DETECTION — YOLOv8-Nano (TensorRT FP16)        │
│  Classes: vehicles, zebra crossings, lanes, dividers,   │
│           license plates                                │
│  Output: bounding boxes + class scores + confidence     │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  MULTI-OBJECT TRACKING — DeepSORT                       │
│  - Kalman filter (motion model)                         │
│  - CNN appearance embeddings (re-identification)        │
│  - Persistent ID assignment                             │
│  - Occlusion recovery and re-association                │
│  Output: tracked objects with IDs + trajectories        │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  AUTOMATIC ROI GENERATION                               │
│  1. Detect static classes: zebra_crossing, lane, divider│
│  2. Compute convex-hull polygons enclosing detections   │
│  3. Temporal-average hull vertices over K frames        │
│  4. Rasterize binary masks aligned to averaged polygons │
│  Output: violation zone masks (no manual calibration!)  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  VIOLATION REASONING (5 Types)                          │
│                                                         │
│  1. SIGNAL JUMP:                                        │
│     zebra upper edge → virtual stop line                │
│     trigger: vehicle crosses stop line during red phase  │
│                                                         │
│  2. ZEBRA-CROSSING BREACH:                              │
│     trigger: vehicle enters zebra polygon during         │
│     pedestrian phase                                    │
│                                                         │
│  3. WRONG-WAY DRIVING:                                  │
│     estimate nominal lane vector v̂ₙ from lane centroids │
│     trigger: vehicle motion v̂ᵢ · v̂ₙ < 0 (dot product)  │
│                                                         │
│  4. ILLEGAL U-TURN:                                     │
│     divider openings form 3 zones (A,B,C at ±90°)      │
│     trigger: rapid traversal of all 3 zones             │
│                                                         │
│  5. SPEEDING:                                           │
│     two virtual lines (Start, Stop) spaced by distance d│
│     v = d / (N × Tᶠ)                                   │
│     where N = frame count between crossings             │
│           Tᶠ = frame interval in seconds                │
│     trigger: v > speed_limit                            │
│                                                         │
│  Output: violation events with evidence                 │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  LICENSE PLATE RECOGNITION                              │
│  - Detection: YOLOv8 sub-model (11,271 Indian plates)   │
│  - Preprocessing: grayscale, bilateral filter, adaptive  │
│    thresholding, CLAHE                                  │
│  - OCR: Tesseract 5 + regex validation (AA00AA0000)     │
│  - Multi-frame confidence voting over T frames           │
│  Output: plate text + confidence                        │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  LOGGING & ANALYTICS                                    │
│  - SQL database (timestamp, track_id, class, plate_hash)│
│  - Spatiotemporal heat maps                             │
│  - Peak-hour profiling                                  │
│  - Policy analytics                                     │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  V2X / COOPERATIVE OUTPUT (Optional)                    │
│  - MQTT over TLS 1.2 to ITS message broker              │
│  - CAM/DENM-style JSON payloads                         │
│  - Rate-limited: max 10 Hz, 3-5s dedup windows          │
│  - Privacy: salted SHA-256 plate hashes only             │
└─────────────────────────────────────────────────────────┘
```

### Speed Estimation Formula
```
v = d / (N × Tᶠ)

Where:
  v  = estimated vehicle speed (km/h) — convert from m/s
  d  = distance between virtual Start and Stop lines (meters)
  N  = number of frames between when vehicle crosses Start and Stop
  Tᶠ = time per frame = 1 / FPS (seconds)

Example at 30 fps, 100m calibrated distance, 60 frame gap:
  v = 100 / (60 × 0.0333) = 100 / 2.0 = 50 m/s = 180 km/h
```

### Hardware Performance

| Metric | Value |
|---|---|
| Hardware | NVIDIA Jetson Nano (128-core Maxwell, 4GB LPDDR4) |
| Power consumption | **9.6 W** |
| Inference speed | **28-30 fps** |
| Violation detection accuracy | **97.7%** |
| OCR precision | **84.9%** |
| Model size (TensorRT FP16) | **36 MB** |
| GPU utilization | 84% |
| Temperature (35-40°C ambient) | < 67°C with passive cooling |
| mAP@0.5 (YOLOv8n) | **48.9%** |
| mAP improvement vs YOLOv4-Tiny | **+10.7%** |
| Accuracy-per-watt improvement | **1.4×** |

### Speed Estimation MAE by Frame Rate

| Frame Rate | MAE (km/h) | Notes |
|---|---|---|
| 10 fps | Higher | Undersamples fast vehicles |
| 20 fps | Medium | Better but still some discretization |
| **30 fps** | **3.16 km/h** | **Optimal — matches camera acquisition rate** |
| 40 fps | Slightly higher | I/O overhead introduces latency |

### Datasets Used

| Dataset | Size | Purpose |
|---|---|---|
| COCO + CityPersons | — | Pre-training/initialization |
| Speed-detection videos | 5 videos (3 min each, 1080p, 30fps) | Speed estimation validation |
| Violation-detection videos | 6 videos (day/dusk/night, glare) | Multi-violation testing |
| Indian license plates (Kaggle) | **11,271 images** (augmented ×2.3) | ANPR training |
| Final training set | 18,420 frames | Combined |
| Final validation set | 4,604 frames | Combined |
| Training | SGD, lr=0.01, momentum 0.9, 150 epochs | TensorRT FP16 export |

### Software Stack

| Component | Version |
|---|---|
| PyTorch | 2.0 |
| TensorRT | 8.5 (FP16, layer fusion) |
| OpenCV | 4.5 (GStreamer-enabled) |
| Tracker | DeepSORT (Kalman + ReID) |
| OCR | Tesseract 5 (CPU) + regex |
| Message Broker | Mosquitto (MQTT/TLS 1.2) |
| Time Sync | NTP/PTP; GPS fallback |

### Detector Comparison on Jetson Nano (640×640 input)

| Model | mAP@0.5 | FPS | Comments |
|---|---|---|---|
| YOLOv4-Tiny (2020) | 38.2 | 21 | Early edge baseline |
| PP-YOLOE-S (2022) | 46.7 | 24 | Balanced accuracy-speed |
| **YOLOv8n (2023)** | **48.9** | **28** | **C2f + SPPF; TensorRT-optimized** |
| YOLOv11n (2024) | 51.3 | 30+ | Transformer-neck; SoA efficiency |
| NanoDet-Plus (2022) | 43.5 | 32 | Ultra-lightweight; lower precision |

### What We Take From This Paper
- ✅ **Complete production system architecture** (our primary reference)
- ✅ **Automatic ROI generation** algorithm (convex hull → no manual calibration)
- ✅ **Speed estimation** formula and validation
- ✅ **All 5 violation detection algorithms** with exact logic
- ✅ **Kaggle Indian LP dataset** (11,271 images)
- ✅ **TensorRT FP16 optimization** for deployment
- ✅ **MQTT/V2X event format** for backend API design
- ✅ **OCR preprocessing pipeline:** grayscale → bilateral filter → adaptive threshold → CLAHE

---

<a id="paper-6"></a>
## Paper 6: Project Presentation (Internal)

| Field | Details |
|---|---|
| **Title** | Traffic & Vehicle Analytics System |
| **Type** | Project overview presentation |
| **Pages** | 6 |
| **File** | `traffic and vehicle Project Presentation.pdf` |

### Hardware Architecture Defined
- CCTV / IP Cameras
- Dashcam units
- Edge GPU Device
- CPU System + RAM & Storage
- Network Connectivity

### System Constraints Identified
1. Real-time processing requirement
2. High GPU computation load
3. Memory & storage limitations
4. Network latency issues
5. Weather & low-light conditions
6. Vehicle occlusion in traffic
7. Scalability for multiple cameras

### Datasets Listed

| Dataset | Purpose | Access |
|---|---|---|
| BrnoCompSpeed | Speed Estimation | Research Access (GitHub) |
| UA-DETRAC | Vehicle Detection & Tracking | Public Benchmark |
| UFPR-ALPR | License Plate Recognition | Academic (email request) |
| CCPD | License Plate Edge Cases | Open Source (GitHub) |
| IDD (Indian Driving Dataset) | Indian Traffic Detection | Registration (IIIT Hyderabad) |

### Dataset Strategy
- Volume: **50K–100K annotated object instances**
- Storage: **~250GB curated corpus**
- Distribution: **80% Training / 10% Validation / 10% Testing**
- Augmentation: OpenCV-based (weather degradation, etc.)

---

<a id="cross-paper-comparison"></a>
## Cross-Paper Comparison Table

| Feature | Paper 1 (Review) | Paper 2 (ANPR) | Paper 3 (DashCop) | Paper 4 (YOLOv8-FDD) | Paper 5 (Edge-AI) |
|---|---|---|---|---|---|
| **Detection Model** | YOLOv8 (reviewed) | CNN + YOLO | YOLOv8-x | YOLOv8n (improved) | YOLOv8-Nano |
| **Tracker** | DeepSORT (reviewed) | — | Cross-AssociationSORT | — | DeepSORT |
| **OCR** | Various (reviewed) | Custom CNN | CRNN | — | Tesseract 5 |
| **Violations** | Multi (reviewed) | None (plate only) | Helmet, Triple Riding | None (detection only) | 5 types |
| **Dataset** | — | 4,326 (Kaggle) | RideSafe-400 (600K frames) | UA-DETRAC (10K images) | COCO + 11K plates |
| **Hardware** | — | Jetson TX2 | 4× RTX 3080 Ti | RTX 3060 | Jetson Nano |
| **FPS** | — | — | — | >300 | 28-30 |
| **Power** | — | — | — | — | 9.6 W |
| **Key Metric** | 96.58% det acc | 96.23% OCR acc | 72.18% E-ticket F1 | mAP+1.3% | 97.7% violation acc |

---

<a id="combined-dataset-reference"></a>
## Combined Dataset Reference

### Datasets We Can Actually Download & Use

| # | Dataset | Download Size | Images/Frames | Classes | Source | Priority |
|---|---|---|---|---|---|---|
| 1 | **COCO Pre-trained (YOLOv8)** | ~6 MB | — | 80 (incl. car, motorcycle, bus, truck, person) | Already downloaded (`yolov8n.pt`) | ✅ Done |
| 2 | **UA-DETRAC (subset)** | ~2-3 GB | 10,870 images | Car, Bus, Van, Other | http://detrac-db.rit.albany.edu/ | 🔴 High |
| 3 | **CCPD (Chinese City Parking)** | ~1.5 GB | 250K+ images | License plates | https://github.com/detectRecog/CCPD | 🔴 High |
| 4 | **Kaggle Indian LP** | ~500 MB | 11,271 images | Indian license plates | Kaggle search: "Indian license plate" | 🟡 Medium |
| 5 | **Sample traffic videos** | ~200 MB | Various | — | Intel/YouTube samples | ✅ Done |
| 6 | **RideSafe-400** | Very Large | 600K frames | 6 classes (R-M, rider, motorcycle, helmet, etc.) | https://dash-cop.github.io/ (request) | 🟡 Medium |
| 7 | **BrnoCompSpeed** | ~5 GB | Videos | Speed estimation | GitHub | 🔵 Low |
| 8 | **IDD** | ~20 GB | — | Indian traffic | IIIT Hyderabad (registration) | 🔵 Low |

### Practical Minimum Dataset (What we need to start)
1. **YOLOv8 pre-trained weights** (COCO) — ✅ already have
2. **UA-DETRAC subset** (10K images) — for vehicle detection fine-tuning
3. **CCPD or Kaggle LP dataset** — for license plate detection + OCR
4. **2-3 sample traffic videos** (1-3 min each) — for end-to-end testing

**Total estimated download: ~4-5 GB**

---

<a id="technology-stack"></a>
## Technology Stack (Derived from Papers)

### AI / Computer Vision
| Component | Technology | Paper Source |
|---|---|---|
| Object Detection | **YOLOv8** (Nano or Medium) | Papers 1, 3, 4, 5 |
| Object Tracking | **DeepSORT** | Papers 1, 3, 5 |
| License Plate Detection | **YOLOv8** (fine-tuned) | Papers 2, 3, 5 |
| License Plate OCR | **EasyOCR / CRNN / Tesseract 5** | Papers 2, 3, 5 |
| Rider-Motorcycle Association | **SAC Module** (simplified) | Paper 3 |
| Helmet Detection | **YOLOv8 classifier** | Paper 3 |
| Triple Riding Detection | **CNN classifier** | Paper 3 |

### Frameworks & Libraries
| Library | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Core language |
| PyTorch | 2.0+ | Deep learning framework |
| Ultralytics | Latest | YOLOv8 implementation |
| OpenCV | 4.5+ | Video processing, preprocessing |
| deep-sort-realtime | Latest | Multi-object tracking |
| EasyOCR | Latest | Optical character recognition |
| NumPy / Pandas / SciPy | Latest | Data processing |
| Matplotlib | Latest | Visualization |

### Backend (Phase 3)
| Component | Technology | Purpose |
|---|---|---|
| API Framework | FastAPI | REST APIs |
| Database | PostgreSQL | Structured data storage |
| Caching | Redis | Real-time data caching |
| Evidence Storage | Local filesystem / S3 | Violation images/videos |

### Deployment (Future)
| Component | Technology | Paper Source |
|---|---|---|
| Edge Device | NVIDIA Jetson Nano | Paper 5 |
| Model Optimization | TensorRT FP16 | Paper 5 |
| Message Protocol | MQTT / TLS 1.2 | Paper 5 |
| Time Sync | NTP / PTP | Paper 5 |

---

## Quick Lookup: "How Does Paper X Do Y?"

### How to detect vehicles?
→ **Paper 4**: YOLOv8n on UA-DETRAC, 4 classes (car/bus/van/other), mAP50 >95%

### How to track vehicles across frames?
→ **Paper 5**: DeepSORT with Kalman filter + CNN appearance embeddings

### How to detect license plates?
→ **Paper 2**: CNN/YOLO → rectangular bounding box → thresholding
→ **Paper 5**: YOLOv8 fine-tuned on 11,271 Indian plates

### How to read license plate text?
→ **Paper 2**: OCR with feature extraction (DTW, CTF, ARR, HWR) → 96.23% accuracy
→ **Paper 3**: CRNN trained on real+synthetic+augmented data → 85.71% accuracy
→ **Paper 5**: Tesseract 5 + regex validation (AA00AA0000) → 84.9% accuracy

### How to detect helmet violations?
→ **Paper 3**: YOLOv8-x trained on 37K helmet + 23K no-helmet boxes → majority voting per track

### How to detect triple riding?
→ **Paper 3**: CNN classifier on R-M instance crops → single/double/triple/none

### How to estimate vehicle speed?
→ **Paper 5**: `v = d / (N × Tᶠ)`, best at 30fps, MAE = 3.16 km/h

### How to detect signal jumping?
→ **Paper 5**: Auto-detect zebra crossing → upper edge = virtual stop line → check if vehicle crosses during red

### How to detect wrong-way driving?
→ **Paper 5**: Compute lane direction vector → if vehicle_vector · lane_vector < 0, it's wrong-way

### How to auto-generate ROI without manual calibration?
→ **Paper 5**: Detect static features (zebra, lane, divider) → compute convex hull → temporal average → rasterize masks

### How to associate riders with motorcycles?
→ **Paper 3**: SAC module — cross-object segmentation masks → IoU-based association score = 84.04%

---

*Last Updated: 2026-06-24*
*Project: Traffic & Vehicle Analytics System*
