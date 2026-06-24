"""
Traffic Analytics Pipeline
Detection + Tracking + ANPR + Helmet violation hints

Uses fine-tuned models when available in models/, otherwise falls back to
COCO pre-trained YOLOv8 weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from anpr_pipeline import ANPREngine, LegacyOCREngine
from model_utils import has_finetuned_model, load_config, resolve_model_path, resolve_path

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


def detect_helmet_violation(helmet_model, frame, vehicle_bbox, confidence: float) -> tuple[bool, str]:
    x1, y1, x2, y2 = vehicle_bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False, ""

    results = helmet_model.predict(crop, conf=confidence, verbose=False)
    saw_rider = False
    saw_helmet = False
    saw_no_helmet = False

    for result in results:
        if result.boxes is None:
            continue
        names = result.names
        for box in result.boxes:
            cls_name = str(names.get(int(box.cls[0]), "")).lower()
            if any(token in cls_name for token in ("no_helmet", "without", "no helmet", "no-helmet")):
                saw_no_helmet = True
            elif "helmet" in cls_name:
                saw_helmet = True
            elif any(token in cls_name for token in ("rider", "person", "bike", "motor")):
                saw_rider = True

    if saw_no_helmet:
        return True, "NO HELMET"
    if saw_rider and not saw_helmet:
        return True, "NO HELMET?"
    if saw_helmet:
        return False, "HELMET OK"
    return False, ""


def main():
    config = load_config()
    paths = config["paths"]
    pipeline_cfg = config["pipeline"]
    tracking_cfg = config["tracking"]
    vehicle_cfg = config["models"]["vehicle"]

    vehicle_model, vehicle_source, _ = load_yolo_model(vehicle_cfg, "vehicle detector")

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
            print("           Run: python download_datasets.py --plates && python train_models.py --plates")

    helmet_model = None
    if pipeline_cfg.get("enable_helmet", True) and has_finetuned_model(config["models"]["helmet"]):
        helmet_model, helmet_source, _ = load_yolo_model(config["models"]["helmet"], "helmet detector")
        print(f"  Helmet model source: {helmet_source}")
    elif pipeline_cfg.get("enable_helmet", True):
        print("  Helmet detection: waiting for fine-tuned model")
        print("           Run: python download_datasets.py --helmets && python train_models.py --helmets")

    tracker = DeepSort(
        max_age=int(tracking_cfg["max_age"]),
        n_init=int(tracking_cfg["n_init"]),
    )

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

    vehicle_classes = vehicle_cfg.get("classes", [2, 3, 5, 7])
    vehicle_conf = float(vehicle_cfg.get("confidence", 0.35))
    helmet_conf = float(config["models"]["helmet"].get("confidence", 0.45))
    log_every = int(pipeline_cfg.get("log_every_n_frames", 30))

    frame_count = 0
    results_log = []

    print(f"Processing: {video_path}")
    print(f"Vehicle model source: {vehicle_source}")
    print(f"Output: {output_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        detections = []

        results = vehicle_model.predict(
            frame,
            classes=vehicle_classes,
            conf=vehicle_conf,
            verbose=False,
        )

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append([[x1, y1, x2 - x1, y2 - y1], conf, cls_id])

        tracks = tracker.update_tracks(detections, frame=frame)
        frame_logs = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cls_id = track.det_class

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

            plate_text = ""
            if anpr is not None:
                plate_text = anpr.update_track(track_id, frame, (x1, y1, x2, y2), frame_count)
                anpr.draw_plate(frame, plate_text, (x1, y2))

            helmet_violation = False
            helmet_label = ""
            if helmet_model is not None and cls_id == 3:
                helmet_violation, helmet_label = detect_helmet_violation(
                    helmet_model,
                    frame,
                    (x1, y1, x2, y2),
                    helmet_conf,
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
                    "class_id": cls_id,
                    "bbox": [x1, y1, x2, y2],
                    "plate_text": plate_text,
                    "helmet_violation": helmet_violation,
                    "helmet_label": helmet_label,
                }
            )

        results_log.append({"frame": frame_count, "tracks": frame_logs})
        out.write(frame)

        if frame_count % log_every == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()

    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results_log, handle, indent=4)

    print(f"Video processing complete. Saved to '{output_path}'.")
    print(f"Saved tracking logs to '{results_path}'.")


if __name__ == "__main__":
    main()
