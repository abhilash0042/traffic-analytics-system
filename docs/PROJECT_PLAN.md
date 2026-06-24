# Project Development Plan
### Traffic & Vehicle Analytics System
### Based on Research Papers Analysis

> This document outlines the complete step-by-step development plan derived from our 6 research papers.
> Each phase maps directly to the methodologies validated in those papers.

---

## Project Summary

**What we are building:**
An AI-powered traffic analytics platform that processes traffic camera/dashcam footage using YOLOv8 and DeepSORT to generate:
- Real-time vehicle detection, counting, and classification
- Vehicle speed estimation
- Traffic violation detection (signal jump, wrong-way, speeding, helmet, triple riding)
- Automatic license plate recognition (ANPR)
- Congestion monitoring and trend analytics
- Automated E-ticket evidence generation

**Our architecture is a combination of:**
- **Paper 5** (Edge-AI Node) → System architecture, violation reasoning, auto-ROI, speed estimation
- **Paper 3** (DashCop) → Rider-motorcycle association, helmet/triple-riding detection, ANPR pipeline
- **Paper 4** (YOLOv8-FDD) → Optimized vehicle detection backbone
- **Paper 2** (ANPR Framework) → License plate preprocessing and OCR methodology

---

## Phase 1: Core Detection & Tracking Pipeline
**Timeline: Week 1-2**
**Paper Reference: Papers 1, 4, 5**

### Objective
Build and validate the foundational YOLOv8 + DeepSORT pipeline that detects and tracks vehicles in real-time.

### Tasks

#### 1.1 Environment Setup ✅ DONE
- [x] Python virtual environment created
- [x] Installed: PyTorch, Ultralytics, OpenCV, EasyOCR, deep-sort-realtime
- [x] Downloaded YOLOv8n pre-trained weights (COCO)
- [x] Downloaded sample traffic video

#### 1.2 Vehicle Detection Module
- [x] Load YOLOv8n pre-trained model
- [x] Run inference on sample video
- [x] Filter to relevant COCO classes: car(2), motorcycle(3), bus(5), truck(7)
- [ ] Evaluate detection accuracy on sample frames
- [ ] Test with YOLOv8s/m if YOLOv8n accuracy is insufficient

> **Paper 4 Reference:** YOLOv8n achieves >95% mAP50 on UA-DETRAC with just 3.01M parameters

#### 1.3 Object Tracking Module
- [x] Integrate DeepSORT with YOLOv8 detections
- [x] Assign and maintain persistent IDs across frames
- [ ] Test tracking consistency (ID switches, lost tracks)
- [ ] Tune DeepSORT parameters: max_age=30, n_init=3, IoU threshold=0.45

> **Paper 5 Reference:** DeepSORT with Kalman filter + CNN embeddings → >98% event-aggregation accuracy

#### 1.4 Output Generation
- [x] Save annotated output video with bounding boxes and track IDs
- [x] Generate JSON log with per-frame tracking data
- [ ] Add vehicle class labels to annotations (car/bike/bus/truck)
- [ ] Add vehicle count summary per class

### Deliverables
- Working `ai_pipeline.py` with detection + tracking
- `output_video.mp4` with annotated bounding boxes and IDs
- `results_log.json` with per-frame tracking data

---

## Phase 2: License Plate Detection & Recognition (ANPR)
**Timeline: Week 2-3**
**Paper Reference: Papers 2, 3, 5**

### Objective
Build a complete ANPR pipeline: detect license plates → preprocess → OCR → validate text.

### Tasks

#### 2.1 License Plate Detection
- [ ] Download CCPD dataset (~1.5 GB) or Kaggle Indian LP dataset
- [ ] Fine-tune YOLOv8 specifically for license plate detection
- [ ] Train on CCPD/Kaggle data (80/20 split)
- [ ] Evaluate plate detection mAP

> **Paper 5 Reference:** YOLOv8 sub-model trained on 11,271 Indian plates

#### 2.2 Plate Preprocessing (Paper 2 & 5 methodology)
- [ ] Crop detected plate region from frame
- [ ] Apply preprocessing pipeline:
  ```
  Original → Grayscale → Bilateral Filtering → Adaptive Thresholding → CLAHE
  ```
- [ ] Handle multi-line plates: convert to single-line format

> **Paper 2 Reference:** Grayscale → Noise removal → Binarization → Sobel edge detection
> **Paper 5 Reference:** Grayscale → Bilateral filter → Adaptive threshold → CLAHE

#### 2.3 Character Recognition (OCR)
- [ ] Test EasyOCR on preprocessed plate crops
- [ ] Test Tesseract 5 on preprocessed plate crops
- [ ] Compare accuracy between the two
- [ ] Add regex-based syntactic validation: `AA00AA0000` or `AA00A0000` format
- [ ] Implement multi-frame confidence voting (majority vote across frames for same track)

> **Paper 3 Reference:** CRNN (real+synthetic+augmented data) → 85.71% accuracy
> **Paper 5 Reference:** Tesseract 5 + regex validation → 84.9% accuracy
> **Paper 2 Reference:** Custom CNN OCR → 96.23% accuracy

#### 2.4 ANPR Integration
- [ ] Connect plate detection → preprocessing → OCR → validation into single pipeline
- [ ] Only run ANPR on tracked vehicles (not every frame — performance optimization)
- [ ] Store best plate reading per track (majority vote)

### Deliverables
- `anpr_pipeline.py` — standalone ANPR module
- Plate detection model (fine-tuned YOLOv8)
- OCR accuracy report on test set

---

## Phase 3: Speed Estimation & Traffic Analytics
**Timeline: Week 3-4**
**Paper Reference: Paper 5**

### Objective
Implement speed estimation using the virtual-line method and basic traffic analytics.

### Tasks

#### 3.1 Speed Estimation
- [ ] Implement virtual Start/Stop line definition (configurable)
- [ ] Track when each vehicle ID crosses Start and Stop lines
- [ ] Calculate speed using Paper 5's formula:
  ```python
  speed_mps = distance_meters / (frame_count * time_per_frame)
  speed_kmph = speed_mps * 3.6
  ```
- [ ] Validate at 30 fps (optimal per Paper 5, MAE = 3.16 km/h)
- [ ] Flag vehicles exceeding configured speed limit

#### 3.2 Vehicle Counting & Classification
- [ ] Count vehicles crossing a defined line per class (car/bike/bus/truck)
- [ ] Calculate count per unit time (vehicles/minute, vehicles/hour)
- [ ] Generate class distribution statistics

#### 3.3 Congestion Analytics
- [ ] Calculate vehicle density per frame (vehicles per unit area)
- [ ] Estimate congestion level: Free Flow / Moderate / Heavy / Gridlock
- [ ] Track congestion trends over time
- [ ] Identify peak hours from temporal patterns

### Deliverables
- `speed_estimator.py` — speed estimation module
- `traffic_analytics.py` — counting, classification, congestion analysis
- Analytics output: JSON with speed data, counts, congestion levels

---

## Phase 4: Violation Detection
**Timeline: Week 4-6**
**Paper Reference: Papers 3, 5**

### Objective
Implement automated violation detection for 5+ violation types using the methodologies from Papers 3 and 5.

### Tasks

#### 4.1 Automatic ROI Generation (Paper 5)
- [ ] Detect static road features: zebra crossings, lane markings, dividers
- [ ] Compute convex-hull polygons over detected static features
- [ ] Temporal-average hull vertices over K=10 frames to reduce jitter
- [ ] Rasterize binary masks for violation zone definitions
- [ ] Support manual ROI override (fallback)

> **Paper 5 Algorithm:**
> ```
> For each frame:
>   1. Select static classes: zebra_crossing, lane, divider
>   2. Compute convex-hull polygons enclosing detections
>   3. Temporal-average hull vertices over K frames
>   4. Rasterize binary masks aligned to averaged polygons
> Persist masks for geometric/temporal checks
> ```

#### 4.2 Geometric Violations (Paper 5)

**Signal Jump Detection:**
- [ ] Identify virtual stop line from zebra crossing upper edge
- [ ] Check if tracked vehicle crosses stop line during red phase
- [ ] Log violation with evidence frame

**Wrong-Way Detection:**
- [ ] Estimate nominal lane direction vector from lane centroids
- [ ] Compute dot product of vehicle motion vector with lane vector
- [ ] Flag if dot product < 0 (vehicle moving against traffic flow)

**Speeding Detection:**
- [ ] Use speed estimation from Phase 3
- [ ] Compare against configured speed limit per zone
- [ ] Flag overspeeding vehicles with evidence

**Illegal U-Turn Detection:**
- [ ] Define 3 zones (A, B, C) at divider openings at ±90°
- [ ] Track vehicle trajectory through zones
- [ ] Flag if vehicle rapidly traverses all 3 zones

**Zebra-Crossing Breach:**
- [ ] Monitor vehicles entering zebra polygon during pedestrian phase
- [ ] Flag as violation

#### 4.3 Two-Wheeler Violations (Paper 3)

**Rider-Motorcycle Association (Simplified SAC):**
- [ ] Detect motorcycles and persons/riders using YOLOv8
- [ ] Spatially associate riders to nearby motorcycles using bounding box overlap
- [ ] Group riders belonging to same motorcycle as R-M instance
- [ ] Track R-M instances jointly

**Helmet Detection:**
- [ ] Fine-tune YOLOv8 for helmet/no-helmet classification
- [ ] Dataset: Look for helmet detection dataset on Kaggle/Roboflow
- [ ] Apply per-frame detection → match to rider tracks → majority vote
- [ ] Flag tracks where rider has no-helmet as violation

> **Paper 3 Reference:** 37K helmet + 23K no-helmet bboxes, YOLOv8-x, 70 epochs

**Triple Riding Detection:**
- [ ] Crop R-M instance bounding box from frame
- [ ] Train simple classifier: single / double / triple / none
- [ ] Flag track if any frame classified as 'triple'

> **Paper 3 Reference:** CNN on frozen SAC features + linear layer

#### 4.4 Violation Evidence Package
- [ ] For each violation: capture evidence frames
- [ ] Run ANPR on violating vehicle (extract plate number)
- [ ] Generate violation record: { type, timestamp, plate, evidence_frames, track_id, speed }

### Deliverables
- `violation_detector.py` — all violation detection modules
- `roi_generator.py` — automatic ROI generation
- Violation log with evidence (JSON + images)

---

## Phase 5: Backend & Database
**Timeline: Week 6-7**
**Paper Reference: Paper 5 (logging & analytics design)**

### Objective
Build the FastAPI backend with PostgreSQL for data storage, querying, and API access.

### Tasks

#### 5.1 Database Schema
- [ ] Design tables: vehicles, violations, analytics_snapshots, cameras
- [ ] Violation record: timestamp, camera_id, track_id, violation_type, plate_number, plate_confidence, evidence_path, speed (if applicable)
- [ ] Analytics record: timestamp, camera_id, vehicle_counts_by_class, congestion_level, avg_speed

#### 5.2 FastAPI Backend
- [ ] REST endpoints:
  - `POST /api/process-video` — submit video for analysis
  - `GET /api/violations` — list violations (filtered, paginated)
  - `GET /api/analytics/summary` — traffic summary stats
  - `GET /api/analytics/trends` — temporal trends
  - `GET /api/violations/{id}/evidence` — get violation evidence
- [ ] JWT authentication
- [ ] WebSocket endpoint for real-time alerts

#### 5.3 Integration
- [ ] Connect AI pipeline output to database ingestion
- [ ] Evidence image storage (local filesystem or S3)
- [ ] Real-time event publishing (WebSocket or MQTT)

### Deliverables
- FastAPI application with REST endpoints
- PostgreSQL database with schema
- API documentation (Swagger/OpenAPI)

---

## Phase 6: Frontend Dashboard
**Timeline: Week 7-8**
**Paper Reference: None (custom design)**

### Objective
Build a modern, interactive dashboard for viewing analytics and violations.

### Tasks
- [ ] Real-time vehicle count display
- [ ] Congestion level indicator
- [ ] Violation alerts feed
- [ ] Traffic trend charts (hourly, daily)
- [ ] Speed distribution graphs
- [ ] Vehicle class distribution pie chart
- [ ] Violation evidence viewer (images + plate text)
- [ ] Camera feed viewer (if RTSP available)

### Deliverables
- Complete web dashboard
- Connected to FastAPI backend

---

## Phase 7: Optimization & Documentation
**Timeline: Week 8-9**

### Tasks
- [ ] Model optimization (TensorRT FP16 if deploying to edge)
- [ ] Pipeline performance profiling (target: ≥25 fps)
- [ ] Final accuracy evaluation on test datasets
- [ ] Project report (LaTeX/Word):
  - Introduction & Problem Statement (from Paper 1)
  - Literature Review (from all papers)
  - Methodology (our implementation details)
  - Results & Evaluation (our metrics vs paper benchmarks)
  - Conclusion & Future Work

---

## Performance Targets (from Papers)

| Metric | Paper Benchmark | Our Target |
|---|---|---|
| Vehicle Detection Accuracy | 96.58% (Paper 1) | ≥90% |
| Vehicle Counting Accuracy | 97.54% (Paper 1) | ≥90% |
| Speed Estimation MAE | 3.16 km/h (Paper 5) | ≤5 km/h |
| License Plate OCR Accuracy | 84.9-96.23% (Papers 2,3,5) | ≥80% |
| Violation Detection Accuracy | 97.7% (Paper 5) | ≥85% |
| Helmet Detection F1 | 65.63% (Paper 3) | ≥60% |
| Triple Riding F1 | 67.72% (Paper 3) | ≥60% |
| Processing Speed | 28-30 fps (Paper 5) | ≥15 fps (local machine) |

---

## File Structure (Planned)

```
traffic-analytics-system/
├── docs/
│   ├── RESEARCH_PAPERS_ANALYSIS.md    ← Paper analysis reference (this project)
│   └── PROJECT_PLAN.md                ← This development plan
│
├── src/
│   ├── detection/
│   │   ├── vehicle_detector.py        ← YOLOv8 vehicle detection
│   │   ├── plate_detector.py          ← YOLOv8 license plate detection
│   │   └── helmet_detector.py         ← YOLOv8 helmet classification
│   │
│   ├── tracking/
│   │   └── tracker.py                 ← DeepSORT integration
│   │
│   ├── anpr/
│   │   ├── preprocessor.py            ← Plate image preprocessing
│   │   ├── ocr_engine.py              ← EasyOCR / Tesseract wrapper
│   │   └── plate_validator.py         ← Regex validation for plate format
│   │
│   ├── violations/
│   │   ├── roi_generator.py           ← Automatic ROI generation (convex hull)
│   │   ├── speed_estimator.py         ← Virtual-line speed estimation
│   │   ├── signal_jump.py             ← Signal jump detection
│   │   ├── wrong_way.py               ← Wrong-way driving detection
│   │   ├── helmet_violation.py        ← Helmet rule violation
│   │   └── triple_riding.py           ← Triple riding detection
│   │
│   ├── analytics/
│   │   ├── vehicle_counter.py         ← Counting & classification
│   │   ├── congestion_analyzer.py     ← Density & congestion estimation
│   │   └── trend_analyzer.py          ← Temporal trend analysis
│   │
│   └── pipeline.py                    ← Unified processing pipeline
│
├── backend/
│   ├── app.py                         ← FastAPI application
│   ├── models.py                      ← Database models
│   ├── routes/                        ← API routes
│   └── services/                      ← Business logic
│
├── frontend/                          ← Dashboard (Phase 6)
│
├── models/                            ← Trained model weights
│   ├── yolov8n.pt                     ← Pre-trained vehicle detection
│   ├── plate_detector.pt              ← Fine-tuned plate detection
│   └── helmet_detector.pt             ← Fine-tuned helmet detection
│
├── data/
│   ├── datasets/                      ← Downloaded datasets
│   ├── sample_videos/                 ← Test videos
│   └── outputs/                       ← Pipeline outputs
│
├── configs/
│   └── pipeline_config.yaml           ← Configuration file
│
├── ai_pipeline.py                     ← Current prototype (Phase 1)
├── requirements.txt                   ← Python dependencies
└── README.md                          ← Project overview
```

---

*Last Updated: 2026-06-24*
*Project: Traffic & Vehicle Analytics System*
