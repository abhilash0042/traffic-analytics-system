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
from model_utils import (
    has_finetuned_model,
    inference_device_label,
    load_config,
    resolve_inference_device,
    resolve_model_path,
    resolve_path,
)

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
    """Return DeepSORT detections: [[x,y,w,h], conf, cls_id]."""
    detections: list[list] = []
    conf = float(vehicle_cfg.get("confidence", 0.35))

    if vehicle_source == "fine-tuned":
        ft_classes = vehicle_cfg.get("finetuned_classes", [0, 1, 2, 3])
        results = vehicle_model.predict(
            frame,
            classes=ft_classes,
            conf=conf,
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


def main():
    config = load_config()
    paths = config["paths"]
    pipeline_cfg = config["pipeline"]
    tracking_cfg = config["tracking"]
    vehicle_cfg = config["models"]["vehicle"]
    device = resolve_inference_device(config)
    use_cuda = str(device) != "cpu"

    print(f"Inference device: {inference_device_label(device)}")

    vehicle_model, vehicle_source, _ = load_yolo_model(vehicle_cfg, "vehicle detector")

    moto_model = None
    if vehicle_source == "fine-tuned" and vehicle_cfg.get("motorcycle_fallback", True):
        moto_path = resolve_path(vehicle_cfg.get("fallback", "weights/yolo11s.pt"))
        moto_model = YOLO(str(moto_path))
        print(f"  Motorcycle detect: {moto_path.name} (COCO fallback for helmet pipeline)")

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

        detections = collect_vehicle_detections(
            frame,
            vehicle_model,
            vehicle_source,
            vehicle_cfg,
            moto_model,
            device,
        )

        tracks = tracker.update_tracks(detections, frame=frame)
        frame_logs = []

        if anpr is not None and hasattr(anpr, "begin_frame"):
            anpr.begin_frame(frame_count, frame)

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
                if pipeline_cfg.get("draw_plate_boxes", True) and hasattr(anpr, "get_track_boxes"):
                    for plate_box in anpr.get_track_boxes(track_id):
                        anpr.draw_plate_box(frame, plate_box)
                anpr.draw_plate(frame, plate_text, (x1, y2))

            helmet_violation = False
            helmet_label = ""
            if helmet_model is not None and cls_id == 3:
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
