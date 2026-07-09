"""
Traffic Analytics Pipeline v2
Detection + BoT-SORT Tracking + Speed Estimation + Road Segmentation
+ ANPR + Helmet violation hints

Changes from v1:
  - Replaced deep-sort-realtime with Ultralytics built-in BoT-SORT tracker
  - Added speed estimation (virtual-line method from Paper 5)
  - Added road segmentation for drivable area inference
  - Added action zone gating for ANPR/helmet (GPU savings)
  - Improved object localization with NMS IoU tuning
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from anpr_pipeline import ANPREngine, LegacyOCREngine
from model_utils import (
    has_finetuned_model,
    inference_device_label,
    load_config,
    resolve_inference_device,
    resolve_model_path,
    resolve_path,
)
from road_segmentation import RoadSegmenter
from speed_estimation import SpeedEstimator

PROJECT_ROOT = Path(__file__).resolve().parent


def load_yolo_model(model_cfg: dict, label: str) -> tuple[YOLO, str, Path]:
    path, source = resolve_model_path(model_cfg, label)
    print(f"Loading {label}: {path.name} ({source})")
    return YOLO(str(path)), source, path


def expand_bbox(x1: int, y1: int, x2: int, y2: int, frame_shape, padding: float = 0.0):
    h, w = frame_shape[:2]
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    )


# UA-DETRAC fine-tuned IDs -> pipeline IDs (COCO-compatible for tracking / helmet)
FINETUNED_VEHICLE_TO_PIPELINE = {
    0: 5,  # bus
    1: 2,  # car
    2: 7,  # truck
    3: 8,  # van (not COCO; avoids clashing with motorcycle=3)
}


def collect_vehicle_detections(
    frame,
    vehicle_model: YOLO,
    vehicle_source: str,
    vehicle_cfg: dict,
    moto_model: YOLO | None,
    device: str | int,
) -> list[list]:
    """Return DeepSORT-format detections: [[x,y,w,h], conf, cls_id]."""
    detections: list[list] = []
    conf = float(vehicle_cfg.get("confidence", 0.35))
    nms_iou = float(vehicle_cfg.get("nms_iou", 0.45))

    if vehicle_source == "fine-tuned":
        ft_classes = vehicle_cfg.get("finetuned_classes", [0, 1, 2, 3])
        results = vehicle_model.predict(
            frame,
            classes=ft_classes,
            conf=conf,
            iou=nms_iou,
            device=device,
            verbose=False,
        )
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                pipeline_cls = FINETUNED_VEHICLE_TO_PIPELINE.get(cls_id, cls_id)
                detections.append(
                    [[x1, y1, x2 - x1, y2 - y1], float(box.conf[0]), pipeline_cls]
                )

        if vehicle_cfg.get("motorcycle_fallback", True) and moto_model is not None:
            moto_conf = float(vehicle_cfg.get("motorcycle_confidence", 0.32))
            moto_results = moto_model.predict(
                frame,
                classes=[3],
                conf=moto_conf,
                device=device,
                verbose=False,
            )
            for result in moto_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(
                        [[x1, y1, x2 - x1, y2 - y1], float(box.conf[0]), 3]
                    )
    else:
        coco_classes = vehicle_cfg.get("classes", [2, 3, 5, 7])
        results = vehicle_model.predict(
            frame,
            classes=coco_classes,
            conf=conf,
            iou=nms_iou,
            device=device,
            verbose=False,
        )
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    [[x1, y1, x2 - x1, y2 - y1], float(box.conf[0]), int(box.cls[0])]
                )

    return detections


def detect_helmet_violation(
    helmet_model, frame, vehicle_bbox, confidence: float, device: str | int = 0
) -> tuple[bool, str]:
    x1, y1, x2, y2 = vehicle_bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False, ""

    results = helmet_model.predict(crop, conf=confidence, device=device, verbose=False)
    saw_rider = False
    saw_helmet = False
    saw_no_helmet = False

    for result in results:
        if result.boxes is None:
            continue
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = str(names.get(cls_id, cls_id)).lower()
            # 2-class model: 0=helmet, 1=no_helmet
            if cls_id == 1 or any(token in cls_name for token in ("no_helmet", "no helmet", "no-helmet", "without")):
                saw_no_helmet = True
            elif cls_id == 0 or "helmet" in cls_name:
                saw_helmet = True
            elif any(token in cls_name for token in ("rider", "person", "bike", "motor", "head")):
                saw_rider = True

    if saw_no_helmet:
        return True, "NO HELMET"
    if saw_rider and not saw_helmet:
        return True, "NO HELMET?"
    if saw_helmet:
        return False, "HELMET OK"
    return False, ""


def draw_speed_overlay(
    frame: np.ndarray,
    speed_kmh: float | None,
    bbox: tuple[int, int, int, int],
    is_speeding: bool = False,
) -> None:
    """Draw speed label above the vehicle bounding box."""
    if speed_kmh is None:
        return
    x1, y1, x2, y2 = bbox
    label = f"{speed_kmh:.0f} km/h"
    color = (0, 0, 255) if is_speeding else (255, 200, 0)  # red if speeding, yellow otherwise

    # Background rectangle for readability
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - 35)), (x1 + tw + 6, max(0, y1 - 35) + th + 6), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (x1 + 3, max(0, y1 - 35) + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def draw_virtual_lines(
    frame: np.ndarray,
    start_y: int,
    stop_y: int,
) -> None:
    """Draw the virtual speed estimation lines on the frame."""
    h, w = frame.shape[:2]
    cv2.line(frame, (0, start_y), (w, start_y), (0, 255, 255), 2)
    cv2.putText(frame, "START", (10, start_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.line(frame, (0, stop_y), (w, stop_y), (0, 255, 255), 2)
    cv2.putText(frame, "STOP", (10, stop_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Traffic Analytics Pipeline — YOLO detection + tracking + ANPR + speed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        metavar="YAML",
        help="Path to pipeline YAML config (default: configs/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--video", "-v",
        default=None,
        metavar="PATH",
        help="Input video file path (overrides config paths.video_input)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="PATH",
        help="Output video file path (overrides config paths.video_output)",
    )
    parser.add_argument(
        "--results", "-r",
        default=None,
        metavar="PATH",
        help="Results JSON log path (overrides config paths.results_log)",
    )
    parser.add_argument(
        "--device", "-d",
        default=None,
        metavar="DEV",
        help="Inference device: 0 for GPU, 'cpu' for CPU (overrides config pipeline.device)",
    )
    parser.add_argument(
        "--no-anpr",
        action="store_true",
        default=False,
        help="Disable ANPR (plate detection + OCR)",
    )
    parser.add_argument(
        "--no-helmet",
        action="store_true",
        default=False,
        help="Disable helmet violation detection",
    )
    parser.add_argument(
        "--no-speed",
        action="store_true",
        default=False,
        help="Disable speed estimation",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Vehicle detection confidence threshold (overrides config models.vehicle.confidence)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # --- Apply CLI overrides ---
    if args.video:
        config["paths"]["video_input"] = args.video
    if args.output:
        config["paths"]["video_output"] = args.output
    if args.results:
        config["paths"]["results_log"] = args.results
    if args.device is not None:
        config["pipeline"]["device"] = int(args.device) if args.device.isdigit() else args.device
    if args.no_anpr:
        config["pipeline"]["enable_anpr"] = False
    if args.no_helmet:
        config["pipeline"]["enable_helmet"] = False
    if args.no_speed:
        config.setdefault("speed_estimation", {})["enabled"] = False
    if args.conf is not None:
        config["models"]["vehicle"]["confidence"] = args.conf
    paths = config["paths"]
    pipeline_cfg = config["pipeline"]
    tracking_cfg = config.get("tracking", {})
    vehicle_cfg = config["models"]["vehicle"]
    device = resolve_inference_device(config)
    use_cuda = str(device) != "cpu"

    print(f"Inference device: {inference_device_label(device)}")

    # --- Load vehicle detection model ---
    vehicle_model, vehicle_source, _ = load_yolo_model(vehicle_cfg, "vehicle detector")

    moto_model = None
    if vehicle_source == "fine-tuned" and vehicle_cfg.get("motorcycle_fallback", True):
        moto_path = resolve_path(vehicle_cfg.get("fallback", "weights/yolo11s.pt"))
        moto_model = YOLO(str(moto_path))
        print(f"  Motorcycle detect: {moto_path.name} (COCO fallback for helmet pipeline)")

    # --- Load ANPR ---
    plate_model = None
    anpr = None
    if pipeline_cfg.get("enable_anpr", True):
        if has_finetuned_model(config["models"]["plate"]):
            plate_model, plate_source, _ = load_yolo_model(config["models"]["plate"], "plate detector")
            anpr = ANPREngine(plate_model, config)
            print(f"  ANPR mode: fine-tuned plate detector ({plate_source})")
        else:
            anpr = LegacyOCREngine(config)
            print("  ANPR mode: legacy OCR fallback (train plate model for better results)")

    # --- Load helmet model ---
    helmet_model = None
    if pipeline_cfg.get("enable_helmet", True) and has_finetuned_model(config["models"]["helmet"]):
        helmet_model, helmet_source, _ = load_yolo_model(config["models"]["helmet"], "helmet detector")
        print(f"  Helmet model source: {helmet_source}")
    elif pipeline_cfg.get("enable_helmet", True):
        print("  Helmet detection: waiting for fine-tuned model")

    # --- Initialize DeepSORT tracker (accepts combined detections) ---
    from deep_sort_realtime.deepsort_tracker import DeepSort
    deepsort = DeepSort(
        max_age=int(tracking_cfg.get("max_age", 30)),
        n_init=int(tracking_cfg.get("n_init", 3)),
        nms_max_overlap=1.0,  # Handled by YOLO
    )
    tracker_type = "deepsort"
    print("  Tracker: DeepSORT (initialized with custom combined detections)")

    # --- Initialize Speed Estimator ---
    speed_estimator = SpeedEstimator(config)
    speed_enabled = speed_estimator.enabled
    if speed_enabled:
        start_y, stop_y = speed_estimator.virtual_lines
        print(f"  Speed estimation: enabled (virtual lines at y={start_y}, y={stop_y})")
    else:
        print("  Speed estimation: disabled")

    # --- Initialize Road Segmenter ---
    road_segmenter = RoadSegmenter(config)
    if road_segmenter.enabled:
        print(f"  Road segmentation: enabled (warmup={road_segmenter.warmup_frames} frames)")
    else:
        print("  Road segmentation: disabled")

    # --- Initialize Zero-DCE Enhancer ---
    try:
        from zero_dce import ZeroDCEEnhancer, is_dark_frame
        zero_dce_enhancer = ZeroDCEEnhancer(device=device)
        print("  Zero-DCE: Enabled for low-light enhancement")
    except Exception as e:
        zero_dce_enhancer = None
        print(f"  Zero-DCE: Disabled ({e})")

    # --- Open video ---
    video_path = resolve_path(paths["video_input"])
    output_path = resolve_path(paths["video_output"])
    results_path = resolve_path(paths["results_log"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    helmet_conf = float(config["models"]["helmet"].get("confidence", 0.45))
    log_every = int(pipeline_cfg.get("log_every_n_frames", 30))
    draw_speed_lines = bool(pipeline_cfg.get("draw_speed_lines", True))
    draw_road_mask = bool(pipeline_cfg.get("draw_road_mask", False))

    frame_count = 0
    results_log = []
    total_tracks_seen = set()

    print(f"Processing: {video_path}")
    print(f"Vehicle model source: {vehicle_source}")
    print(f"Output: {output_path}")
    print(f"FPS: {fps}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # --- Step 1: Vehicle detection ---
        # Run vehicle detection on the ORIGINAL unenhanced frame (as the vehicle model was trained on it)
        detections = collect_vehicle_detections(
            frame,
            vehicle_model,
            vehicle_source,
            vehicle_cfg,
            moto_model,
            device,
        )

        # Enhance the frame for ANPR and downstream tasks if it's too dark
        if zero_dce_enhancer is not None and is_dark_frame(frame, threshold=60):
            frame = zero_dce_enhancer.enhance(frame)

        # --- Step 2: Road segmentation update & filtering ---
        if road_segmenter.enabled:
            road_segmenter.update_from_detections(frame, detections, frame_count)
            if road_segmenter.is_ready:
                detections = road_segmenter.filter_detections(detections)

        # --- Step 3: DeepSORT tracking ---
        # deepsort.update_tracks takes detections in [[x, y, w, h], confidence, class_id]
        tracks = deepsort.update_tracks(detections, frame=frame)

        frame_logs = []

        if anpr is not None and hasattr(anpr, "begin_frame"):
            anpr.begin_frame(frame_count, frame)

        # Draw virtual speed lines
        if speed_enabled and draw_speed_lines:
            draw_virtual_lines(frame, *speed_estimator.virtual_lines)

        # Draw road mask overlay
        if draw_road_mask and road_segmenter.is_ready:
            frame = road_segmenter.draw_road_overlay(frame, alpha=0.15)

        # --- Step 4: Process tracked objects ---
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cls_id = int(track.det_class)
            conf_val = float(track.det_conf or 0.0)

            # Map fine-tuned class IDs to pipeline IDs
            if vehicle_source == "fine-tuned":
                pipeline_cls = FINETUNED_VEHICLE_TO_PIPELINE.get(cls_id, cls_id)
            else:
                pipeline_cls = cls_id

            total_tracks_seen.add(track_id)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID:{track_id}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # --- Speed estimation ---
            speed_kmh = None
            is_over_speed = False
            if speed_enabled:
                speed_kmh = speed_estimator.update(
                    track_id, (x1, y1, x2, y2), frame_count, fps
                )
                is_over_speed = speed_estimator.is_speeding(track_id)
                draw_speed_overlay(frame, speed_kmh, (x1, y1, x2, y2), is_over_speed)

            # --- ANPR (with action zone gating) ---
            plate_text = ""
            if anpr is not None:
                run_anpr = True
                if road_segmenter.enabled and road_segmenter.is_ready:
                    run_anpr = road_segmenter.is_in_action_zone(
                        (x1, y1, x2, y2), zone_type="anpr"
                    )
                if run_anpr:
                    plate_text = anpr.update_track(track_id, frame, (x1, y1, x2, y2), frame_count)
                    if pipeline_cfg.get("draw_plate_boxes", True) and hasattr(anpr, "get_track_boxes"):
                        for plate_box in anpr.get_track_boxes(track_id):
                            anpr.draw_plate_box(frame, plate_box)
                    anpr.draw_plate(frame, plate_text, (x1, y2))

            # --- Helmet detection (with action zone gating) ---
            helmet_violation = False
            helmet_label = ""
            if helmet_model is not None and pipeline_cls == 3:
                run_helmet = True
                if road_segmenter.enabled and road_segmenter.is_ready:
                    run_helmet = road_segmenter.is_in_action_zone(
                        (x1, y1, x2, y2), zone_type="helmet"
                    )
                if run_helmet:
                    helmet_violation, helmet_label = detect_helmet_violation(
                        helmet_model,
                        frame,
                        (x1, y1, x2, y2),
                        helmet_conf,
                        device=device,
                    )
                    if helmet_label:
                        color = (0, 0, 255) if helmet_violation else (255, 165, 0)
                        cv2.putText(
                            frame,
                            helmet_label,
                            (x1, y2 + 45 if plate_text else y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                        )

            frame_logs.append(
                {
                    "track_id": track_id,
                    "class_id": pipeline_cls,
                    "bbox": [x1, y1, x2, y2],
                    "plate_text": plate_text,
                    "helmet_violation": helmet_violation,
                    "helmet_label": helmet_label,
                    "speed_kmh": round(speed_kmh, 1) if speed_kmh else None,
                    "speeding": is_over_speed,
                }
            )

        results_log.append({"frame": frame_count, "tracks": frame_logs})
        out.write(frame)

        if frame_count % log_every == 0:
            active_tracks = len(frame_logs)
            road_status = "ready" if road_segmenter.is_ready else "warmup"
            print(
                f"Processed {frame_count} frames... "
                f"(tracks: {active_tracks}, total IDs: {len(total_tracks_seen)}, "
                f"road: {road_status})"
            )

    cap.release()
    out.release()

    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results_log, handle, indent=4)

    print(f"\nVideo processing complete. Saved to '{output_path}'.")
    print(f"Saved tracking logs to '{results_path}'.")
    print(f"Total unique tracks: {len(total_tracks_seen)}")
    print(f"Tracker: {tracker_type} (Ultralytics built-in)")
    if speed_enabled:
        speeding_count = sum(
            1 for tid in total_tracks_seen if speed_estimator.is_speeding(tid)
        )
        print(f"Speed estimation: {speeding_count} vehicles flagged for speeding")

    # --- Excel Export ---
    print("\nExporting summary to Excel...")
    try:
        import pandas as pd
        summary_data = {}
        for frame_data in results_log:
            for track in frame_data["tracks"]:
                tid = track["track_id"]
                if tid not in summary_data:
                    summary_data[tid] = {
                        "Track ID": tid,
                        "Class ID": track["class_id"],
                        "Max Speed (km/h)": 0,
                        "Speeding Violation": False,
                        "Plate Text": "",
                        "Helmet Status": "",
                    }
                
                # Update with latest/max values
                if track["speed_kmh"] is not None:
                    summary_data[tid]["Max Speed (km/h)"] = max(
                        summary_data[tid]["Max Speed (km/h)"], track["speed_kmh"]
                    )
                if track["speeding"]:
                    summary_data[tid]["Speeding Violation"] = True
                if track["plate_text"]:
                    summary_data[tid]["Plate Text"] = track["plate_text"]
                if track["helmet_label"]:
                    summary_data[tid]["Helmet Status"] = track["helmet_label"]

        # Map class IDs to names for readability
        CLASS_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck", 8: "Van"}
        for tid in summary_data:
            cls_id = summary_data[tid]["Class ID"]
            summary_data[tid]["Vehicle Type"] = CLASS_NAMES.get(cls_id, f"Unknown({cls_id})")

        df = pd.DataFrame(list(summary_data.values()))
        # Reorder columns
        df = df[["Track ID", "Vehicle Type", "Max Speed (km/h)", "Speeding Violation", "Plate Text", "Helmet Status"]]
        
        excel_path = output_path.parent / "vehicle_summary.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"Successfully saved vehicle summary to '{excel_path}'")
    except ImportError:
        print("Could not export to Excel: pandas or openpyxl is not installed.")
    except Exception as e:
        print(f"Error exporting to Excel: {e}")



if __name__ == "__main__":
    main()
