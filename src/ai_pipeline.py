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
  - Added per-track class stabilization
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.trackers import BOTSORT

from src.anpr_pipeline import ANPREngine, LegacyOCREngine
from src.model_utils import (
    has_finetuned_model,
    inference_device_label,
    load_config,
    resolve_inference_device,
    resolve_model_path,
    resolve_path,
)
from src.road_segmentation import RoadSegmenter
from src.speed_estimation import SpeedEstimator


PROJECT_ROOT = Path(__file__).resolve().parent


# ==========================================================
# BoT-SORT detection adapter
# ==========================================================

class TrackerResults:
    """
    Minimal Results-like adapter for the installed
    Ultralytics trackers.

    BoT-SORT expects:
        .xyxy
        .xywh
        .conf
        .cls
        __len__()
        __getitem__()
    """

    def __init__(
        self,
        boxes_xyxy,
        conf,
        cls,
    ):
        self.xyxy = boxes_xyxy
        self.conf = conf
        self.cls = cls

    @property
    def xywh(self):
        xyxy = self.xyxy

        if xyxy.numel() == 0:
            return xyxy.clone()

        if xyxy.ndim == 1:
            xyxy = xyxy.unsqueeze(0)

        result = xyxy.clone()

        result[:, 0] = (
            xyxy[:, 0] + xyxy[:, 2]
        ) / 2

        result[:, 1] = (
            xyxy[:, 1] + xyxy[:, 3]
        ) / 2

        result[:, 2] = (
            xyxy[:, 2] - xyxy[:, 0]
        )

        result[:, 3] = (
            xyxy[:, 3] - xyxy[:, 1]
        )

        return result

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, index):
        xyxy = self.xyxy[index]
        conf = self.conf[index]
        cls = self.cls[index]

        if xyxy.ndim == 1:
            xyxy = xyxy.unsqueeze(0)

        if conf.ndim == 0:
            conf = conf.unsqueeze(0)

        if cls.ndim == 0:
            cls = cls.unsqueeze(0)

        return TrackerResults(
            xyxy,
            conf,
            cls,
        )


# ==========================================================
# Model loading
# ==========================================================

def load_yolo_model(
    model_cfg: dict,
    label: str,
) -> tuple[YOLO, str, Path]:

    path, source = resolve_model_path(
        model_cfg,
        label,
    )

    print(
        f"Loading {label}: "
        f"{path.name} ({source})"
    )

    return (
        YOLO(str(path)),
        source,
        path,
    )


# ==========================================================
# Bounding box utility
# ==========================================================

def expand_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_shape,
    padding: float = 0.0,
):
    h, w = frame_shape[:2]

    pad_x = int(
        (x2 - x1) * padding
    )

    pad_y = int(
        (y2 - y1) * padding
    )

    return (
        max(
            0,
            x1 - pad_x,
        ),
        max(
            0,
            y1 - pad_y,
        ),
        min(
            w,
            x2 + pad_x,
        ),
        min(
            h,
            y2 + pad_y,
        ),
    )


# ==========================================================
# Fine-tuned vehicle IDs -> pipeline IDs
# ==========================================================

# UA-DETRAC fine-tuned IDs -> pipeline IDs
# COCO-compatible for tracking / helmet.
FINETUNED_VEHICLE_TO_PIPELINE = {
    0: 5,  # bus
    1: 2,  # car
    2: 7,  # truck
    3: 8,  # van
}


# ==========================================================
# Vehicle detection
# ==========================================================

def collect_vehicle_detections(
    frame,
    vehicle_model: YOLO,
    vehicle_source: str,
    vehicle_cfg: dict,
    moto_model: YOLO | None,
    device: str | int,
) -> list[list]:
    """
    Return detections in:

        [[x, y, w, h], confidence, class_id]

    This format is converted into TrackerResults
    before being passed to BoT-SORT.
    """

    detections: list[list] = []

    conf = float(
        vehicle_cfg.get(
            "confidence",
            0.35,
        )
    )

    nms_iou = float(
        vehicle_cfg.get(
            "nms_iou",
            0.45,
        )
    )

    # ======================================================
    # Fine-tuned detector
    # ======================================================

    if vehicle_source == "fine-tuned":

        ft_classes = vehicle_cfg.get(
            "finetuned_classes",
            [0, 1, 2, 3],
        )

        results = vehicle_model.predict(
            frame,
            classes=ft_classes,
            conf=conf,
            iou=nms_iou,
            imgsz=int(
                vehicle_cfg.get(
                    "imgsz",
                    640,
                )
            ),
            device=device,
            verbose=False,
        )

        # Targeted second pass for difficult CCTV frames.
        total_first_pass = sum(
            len(r.boxes)
            for r in results
        )

        if total_first_pass <= 1:

            results = vehicle_model.predict(
                frame,
                classes=ft_classes,
                conf=float(
                    vehicle_cfg.get(
                        "fallback_confidence",
                        0.15,
                    )
                ),
                iou=nms_iou,
                imgsz=int(
                    vehicle_cfg.get(
                        "fallback_imgsz",
                        960,
                    )
                ),
                device=device,
                verbose=False,
            )

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                cls_id = int(
                    box.cls[0]
                )

                pipeline_cls = (
                    FINETUNED_VEHICLE_TO_PIPELINE.get(
                        cls_id,
                        cls_id,
                    )
                )

                detections.append(
                    [
                        [
                            x1,
                            y1,
                            x2 - x1,
                            y2 - y1,
                        ],
                        float(
                            box.conf[0]
                        ),
                        pipeline_cls,
                    ]
                )

        # ==================================================
        # Motorcycle fallback
        # ==================================================

        if (
            vehicle_cfg.get(
                "motorcycle_fallback",
                True,
            )
            and moto_model is not None
        ):

            moto_conf = float(
                vehicle_cfg.get(
                    "motorcycle_confidence",
                    0.32,
                )
            )

            moto_results = moto_model.predict(
                frame,
                classes=[3],
                conf=moto_conf,
                device=device,
                verbose=False,
            )

            for result in moto_results:

                for box in result.boxes:

                    x1, y1, x2, y2 = (
                        box.xyxy[0].tolist()
                    )

                    detections.append(
                        [
                            [
                                x1,
                                y1,
                                x2 - x1,
                                y2 - y1,
                            ],
                            float(
                                box.conf[0]
                            ),
                            3,
                        ]
                    )

    # ======================================================
    # COCO detector
    # ======================================================

    else:

        coco_classes = vehicle_cfg.get(
            "classes",
            [2, 3, 5, 7],
        )

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

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                detections.append(
                    [
                        [
                            x1,
                            y1,
                            x2 - x1,
                            y2 - y1,
                        ],
                        float(
                            box.conf[0]
                        ),
                        int(
                            box.cls[0]
                        ),
                    ]
                )

    return detections


# ==========================================================
# Helmet detection
# ==========================================================

def detect_helmet_violation(
    helmet_model,
    frame,
    vehicle_bbox,
    confidence: float,
    device: str | int = 0,
) -> tuple[bool, str]:

    x1, y1, x2, y2 = vehicle_bbox

    crop = frame[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return False, ""

    results = helmet_model.predict(
        crop,
        conf=confidence,
        device=device,
        verbose=False,
    )

    saw_rider = False
    saw_helmet = False
    saw_no_helmet = False

    for result in results:

        if result.boxes is None:
            continue

        names = result.names

        for box in result.boxes:

            cls_id = int(
                box.cls[0]
            )

            cls_name = str(
                names.get(
                    cls_id,
                    cls_id,
                )
            ).lower()

            # 2-class model:
            # 0 = helmet
            # 1 = no_helmet
            if (
                cls_id == 1
                or any(
                    token in cls_name
                    for token in (
                        "no_helmet",
                        "no helmet",
                        "no-helmet",
                        "without",
                    )
                )
            ):

                saw_no_helmet = True

            elif (
                cls_id == 0
                or "helmet" in cls_name
            ):

                saw_helmet = True

            elif any(
                token in cls_name
                for token in (
                    "rider",
                    "person",
                    "bike",
                    "motor",
                    "head",
                )
            ):

                saw_rider = True

    if saw_no_helmet:
        return True, "NO HELMET"

    if saw_rider and not saw_helmet:
        return True, "NO HELMET?"

    if saw_helmet:
        return False, "HELMET OK"

    return False, ""


# ==========================================================
# Speed overlay
# ==========================================================

def draw_speed_overlay(
    frame: np.ndarray,
    speed_kmh: float | None,
    bbox: tuple[int, int, int, int],
    is_speeding: bool = False,
) -> None:

    if speed_kmh is None:
        return

    x1, y1, x2, y2 = bbox

    label = (
        f"{speed_kmh:.0f} km/h"
    )

    color = (
        (0, 0, 255)
        if is_speeding
        else (255, 200, 0)
    )

    (tw, th), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2,
    )

    cv2.rectangle(
        frame,
        (
            x1,
            max(
                0,
                y1 - 35,
            ),
        ),
        (
            x1 + tw + 6,
            max(
                0,
                y1 - 35,
            ) + th + 6,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        label,
        (
            x1 + 3,
            max(
                0,
                y1 - 35,
            ) + th + 2,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


# ==========================================================
# Virtual speed lines
# ==========================================================

def draw_virtual_lines(
    frame: np.ndarray,
    start_y: int,
    stop_y: int,
) -> None:

    h, w = frame.shape[:2]

    cv2.line(
        frame,
        (
            0,
            start_y,
        ),
        (
            w,
            start_y,
        ),
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "START",
        (
            10,
            start_y - 8,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )

    cv2.line(
        frame,
        (
            0,
            stop_y,
        ),
        (
            w,
            stop_y,
        ),
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "STOP",
        (
            10,
            stop_y - 8,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )


# ==========================================================
# CLI
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Traffic Analytics Pipeline — "
            "YOLO detection + tracking + "
            "ANPR + speed"
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--config",
        "-c",
        default=None,
        metavar="YAML",
        help=(
            "Path to pipeline YAML config "
            "(default: configs/pipeline_config.yaml)"
        ),
    )

    parser.add_argument(
        "--video",
        "-v",
        default=None,
        metavar="PATH",
        help=(
            "Input video file path "
            "(overrides config paths.video_input)"
        ),
    )

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="PATH",
        help=(
            "Output video file path "
            "(overrides config paths.video_output)"
        ),
    )

    parser.add_argument(
        "--results",
        "-r",
        default=None,
        metavar="PATH",
        help=(
            "Results JSON log path "
            "(overrides config paths.results_log)"
        ),
    )

    parser.add_argument(
        "--device",
        "-d",
        default=None,
        metavar="DEV",
        help=(
            "Inference device: 0 for GPU, "
            "'cpu' for CPU "
            "(overrides config pipeline.device)"
        ),
    )

    parser.add_argument(
        "--no-anpr",
        action="store_true",
        default=False,
        help=(
            "Disable ANPR "
            "(plate detection + OCR)"
        ),
    )

    parser.add_argument(
        "--no-helmet",
        action="store_true",
        default=False,
        help=(
            "Disable helmet violation detection"
        ),
    )

    parser.add_argument(
        "--no-speed",
        action="store_true",
        default=False,
        help=(
            "Disable speed estimation"
        ),
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Vehicle detection confidence threshold "
            "(overrides config models.vehicle.confidence)"
        ),
    )

    return parser.parse_args()



# ==========================================================
# Persistent vehicle state
# ==========================================================

def update_track_state(
    track_state: dict,
    track_id: int,
    frame_number: int,
    fps: float,
    class_id: int,
    bbox: tuple[int, int, int, int],
    confidence: float,
) -> dict:
    """
    Maintain persistent vehicle-level state across frames.

    The state is intentionally lightweight:
      - identity
      - stable vehicle class
      - lifetime
      - position history
      - confidence statistics
      - bounding-box history
    """

    x1, y1, x2, y2 = bbox

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    if track_id not in track_state:
        track_state[track_id] = {
            "track_id": track_id,
            "class_id": class_id,
            "first_frame": frame_number,
            "last_frame": frame_number,
            "frames_seen": 0,
            "positions": [],
            "bboxes": [],
            "confidence_sum": 0.0,
            "confidence_min": confidence,
            "confidence_max": confidence,
            "max_speed_kmh": 0.0,
            "speeding": False,
        }

    state = track_state[track_id]

    # Use the stabilized class.
    state["class_id"] = class_id

    state["last_frame"] = frame_number
    state["frames_seen"] += 1

    state["confidence_sum"] += confidence
    state["confidence_min"] = min(
        state["confidence_min"],
        confidence,
    )
    state["confidence_max"] = max(
        state["confidence_max"],
        confidence,
    )

    # Keep position history bounded.
    state["positions"].append(
        {
            "frame": frame_number,
            "x": center_x,
            "y": center_y,
        }
    )

    state["bboxes"].append(
        {
            "frame": frame_number,
            "bbox": [
                x1,
                y1,
                x2,
                y2,
            ],
        }
    )

    # Keep memory under control.
    # We do NOT need thousands of positions per vehicle.
    max_history = 150

    if len(state["positions"]) > max_history:
        state["positions"] = state["positions"][
            -max_history:
        ]

    if len(state["bboxes"]) > max_history:
        state["bboxes"] = state["bboxes"][
            -max_history:
        ]

    state["average_confidence"] = (
        state["confidence_sum"]
        / state["frames_seen"]
    )

    state["duration_seconds"] = (
        state["last_frame"]
        - state["first_frame"]
    ) / max(fps, 1.0)

    return state


def calculate_traffic_metrics(
    track_state: dict,
    frame_width: int,
    frame_height: int,
) -> dict:
    """
    Calculate lightweight scene-level traffic metrics.

    These metrics are based on unique tracked vehicles,
    not raw detections.
    """

    vehicle_count = len(track_state)

    if vehicle_count == 0:
        return {
            "vehicle_count": 0,
            "density": 0.0,
            "average_track_duration": 0.0,
            "vehicle_classes": {},
        }

    vehicle_classes = {}

    durations = []

    for state in track_state.values():

        class_id = str(
            state["class_id"]
        )

        vehicle_classes[class_id] = (
            vehicle_classes.get(
                class_id,
                0,
            ) + 1
        )

        durations.append(
            state["duration_seconds"]
        )

    # Simple normalized scene density.
    #
    # This is NOT vehicles/km² because we don't yet
    # have a calibrated physical road area.
    scene_area = (
        frame_width * frame_height
    )

    density = (
        vehicle_count
        / max(
            scene_area / 100000.0,
            1.0,
        )
    )

    return {
        "vehicle_count": vehicle_count,
        "density": round(
            density,
            4,
        ),
        "average_track_duration": round(
            sum(durations)
            / len(durations),
            2,
        ),
        "vehicle_classes": vehicle_classes,
    }

# ==========================================================
# Live processing state
# ==========================================================

def write_live_state(
    frame_number,
    active_tracks,
    total_ids,
    road_status,
    fps,
    total_frames,
    live_state_path,
):
    """Atomically write the current processing state for monitoring."""

    state = {
        "status": "PROCESSING",
        "frame": frame_number,
        "total_frames": total_frames,
        "progress_percent": (
            round((frame_number / total_frames) * 100, 1)
            if total_frames
            else 0
        ),
        "fps": fps,
        "active_vehicles": active_tracks,
        "unique_vehicles_seen": total_ids,
        "road_status": road_status,
    }

    tmp_path = live_state_path.with_suffix(".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f)

    tmp_path.replace(live_state_path)


# ==========================================================
# Main
# ==========================================================

def main():

    args = parse_args()

    config = load_config(
        args.config
    )

    # ==========================================================
    # Apply CLI overrides
    # ==========================================================

    if args.video:

        config["paths"][
            "video_input"
        ] = args.video

    if args.output:

        config["paths"][
            "video_output"
        ] = args.output

    if args.results:

        config["paths"][
            "results_log"
        ] = args.results

    if args.device is not None:

        config["pipeline"][
            "device"
        ] = (
            int(args.device)
            if args.device.isdigit()
            else args.device
        )

    if args.no_anpr:

        config["pipeline"][
            "enable_anpr"
        ] = False

    if args.no_helmet:

        config["pipeline"][
            "enable_helmet"
        ] = False

    if args.no_speed:

        config.setdefault(
            "speed_estimation",
            {},
        )["enabled"] = False

    if args.conf is not None:

        config["models"][
            "vehicle"
        ]["confidence"] = args.conf

    paths = config["paths"]

    pipeline_cfg = config["pipeline"]

    tracking_cfg = config.get(
        "tracking",
        {},
    )

    vehicle_cfg = config["models"][
        "vehicle"
    ]

    # ==========================================================
    # Device
    # ==========================================================

    device = resolve_inference_device(
        config
    )

    print(
        f"Inference device: "
        f"{inference_device_label(device)}"
    )

    # ==========================================================
    # Vehicle detector
    # ==========================================================

    (
        vehicle_model,
        vehicle_source,
        _,
    ) = load_yolo_model(
        vehicle_cfg,
        "vehicle detector",
    )

    moto_model = None

    if (
        vehicle_source == "fine-tuned"
        and vehicle_cfg.get(
            "motorcycle_fallback",
            True,
        )
    ):

        moto_path = resolve_path(
            vehicle_cfg.get(
                "fallback",
                "weights/yolo11s.pt",
            )
        )

        moto_model = YOLO(
            str(moto_path)
        )

        print(
            f"  Motorcycle detect: "
            f"{moto_path.name} "
            "(COCO fallback for helmet pipeline)"
        )

    # ==========================================================
    # ANPR
    # ==========================================================

    plate_model = None
    anpr = None

    if pipeline_cfg.get(
        "enable_anpr",
        True,
    ):

        if has_finetuned_model(
            config["models"]["plate"]
        ):

            (
                plate_model,
                plate_source,
                _,
            ) = load_yolo_model(
                config["models"]["plate"],
                "plate detector",
            )

            anpr = ANPREngine(
                plate_model,
                config,
            )

            print(
                "  ANPR mode: "
                "fine-tuned plate detector "
                f"({plate_source})"
            )

        else:

            anpr = LegacyOCREngine(
                config
            )

            print(
                "  ANPR mode: legacy OCR fallback "
                "(train plate model for better results)"
            )

    # ==========================================================
    # Helmet detector
    # ==========================================================

    helmet_model = None

    if (
        pipeline_cfg.get(
            "enable_helmet",
            True,
        )
        and has_finetuned_model(
            config["models"]["helmet"]
        )
    ):

        (
            helmet_model,
            helmet_source,
            _,
        ) = load_yolo_model(
            config["models"]["helmet"],
            "helmet detector",
        )

        print(
            f"  Helmet model source: "
            f"{helmet_source}"
        )

    elif pipeline_cfg.get(
        "enable_helmet",
        True,
    ):

        print(
            "  Helmet detection: "
            "waiting for fine-tuned model"
        )

    # ==========================================================
    # Open video
    # ==========================================================

    video_path = resolve_path(
        paths["video_input"]
    )

    output_path = resolve_path(
        paths["video_output"]
    )

    results_path = resolve_path(
        paths["results_log"]
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"Error: Could not open video "
            f"{video_path}"
        )

        return

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    ) or 0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = int(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    ) or int(
        tracking_cfg.get(
            "frame_rate",
            30,
        )
    )

    print(
        f"Input video FPS: {fps}"
    )

    # ==========================================================
    # Initialize tracker
    # ==========================================================

    tracker_type = str(
        tracking_cfg.get(
            "tracker",
            "botsort",
        )
    ).lower()

    if tracker_type == "botsort":

        from ultralytics.cfg import get_cfg

        botsort_cfg_path = (
            Path(
                __import__(
                    "ultralytics"
                ).__file__
            ).resolve().parent
            / "cfg"
            / "trackers"
            / "botsort.yaml"
        )

        tracker_args = get_cfg(
            str(
                botsort_cfg_path
            )
        )

        tracker_args.track_high_thresh = float(
            tracking_cfg.get(
                "track_high_thresh",
                tracker_args.track_high_thresh,
            )
        )

        tracker_args.track_low_thresh = float(
            tracking_cfg.get(
                "track_low_thresh",
                tracker_args.track_low_thresh,
            )
        )

        tracker_args.new_track_thresh = float(
            tracking_cfg.get(
                "new_track_thresh",
                tracker_args.new_track_thresh,
            )
        )

        tracker_args.track_buffer = int(
            tracking_cfg.get(
                "track_buffer",
                tracker_args.track_buffer,
            )
        )

        tracker_args.match_thresh = float(
            tracking_cfg.get(
                "match_thresh",
                tracker_args.match_thresh,
            )
        )

        tracker = BOTSORT(
            args=tracker_args,
        )

        print(
            "  Tracker: BoT-SORT "
            "(Ultralytics)"
        )

    else:

        from deep_sort_realtime.deepsort_tracker import (
            DeepSort,
        )

        tracker = DeepSort(
            max_age=int(
                tracking_cfg.get(
                    "max_age",
                    30,
                )
            ),
            n_init=int(
                tracking_cfg.get(
                    "n_init",
                    3,
                )
            ),
            nms_max_overlap=1.0,
        )

        tracker_type = "deepsort"

        print(
            "  Tracker: DeepSORT "
            "(legacy fallback)"
        )

    # ==========================================================
    # Speed estimator
    # ==========================================================

    speed_estimator = SpeedEstimator(
        config
    )

    speed_enabled = (
        speed_estimator.enabled
    )

    if speed_enabled:

        (
            start_y,
            stop_y,
        ) = speed_estimator.virtual_lines

        print(
            f"  Speed estimation: enabled "
            f"(virtual lines at "
            f"y={start_y}, y={stop_y})"
        )

    else:

        print(
            "  Speed estimation: disabled"
        )

    # ==========================================================
    # Road segmenter
    # ==========================================================

    road_segmenter = RoadSegmenter(
        config
    )

    if road_segmenter.enabled:

        print(
            f"  Road segmentation: enabled "
            f"(warmup="
            f"{road_segmenter.warmup_frames} frames)"
        )

    else:

        print(
            "  Road segmentation: disabled"
        )

    # ==========================================================
    # Zero-DCE enhancer
    # ==========================================================

    try:

        from src.zero_dce import (
            ZeroDCEEnhancer,
            is_dark_frame,
        )

        zero_dce_enhancer = (
            ZeroDCEEnhancer(
                device=device
            )
        )

        print(
            "  Zero-DCE: Enabled "
            "for low-light enhancement"
        )

    except Exception as e:

        zero_dce_enhancer = None
        is_dark_frame = None

        print(
            f"  Zero-DCE: Disabled ({e})"
        )

    # ==========================================================
    # Video writer
    # ==========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (
            width,
            height,
        ),
    )

    helmet_conf = float(
        config["models"][
            "helmet"
        ].get(
            "confidence",
            0.45,
        )
    )

    log_every = int(
        pipeline_cfg.get(
            "log_every_n_frames",
            30,
        )
    )

    draw_speed_lines = bool(
        pipeline_cfg.get(
            "draw_speed_lines",
            True,
        )
    )

    draw_road_mask = bool(
        pipeline_cfg.get(
            "draw_road_mask",
            False,
        )
    )

    frame_count = 0

    results_log = []

    total_tracks_seen = set()

    # Live processing state for external monitoring.
    live_state_path = Path("/tmp/traffic_live_state.json")

    # ==========================================================
    # Core traffic intelligence state
    # ==========================================================

    track_history = defaultdict(
        lambda: deque(maxlen=50)
    )

    track_first_frame = {}
    track_last_frame = {}

    track_direction = {}

    track_stationary_frames = defaultdict(int)

    direction_counts = defaultdict(int)

    peak_active_vehicles = 0

    total_vehicle_observations = 0

    # ==========================================================
    # Per-track class history.
    #
    # Detector classes can fluctuate across frames, so
    # downstream traffic intelligence uses the dominant
    # class for each track.
    # ==========================================================

    track_class_history = {}

    # Persistent state for every vehicle track.
    # This is the foundation for traffic intelligence.
    track_state = {}

    print(
        f"Processing: {video_path}"
    )

    print(
        f"Vehicle model source: "
        f"{vehicle_source}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"FPS: {fps}"
    )

    # ==========================================================
    # Main processing loop
    # ==========================================================

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # ======================================================
        # Step 1: Vehicle detection
        # ======================================================

        detections = (
            collect_vehicle_detections(
                frame,
                vehicle_model,
                vehicle_source,
                vehicle_cfg,
                moto_model,
                device,
            )
        )

        # ======================================================
        # Zero-DCE enhancement
        # ======================================================

        if (
            zero_dce_enhancer is not None
            and is_dark_frame is not None
            and is_dark_frame(
                frame,
                threshold=60,
            )
        ):

            frame = (
                zero_dce_enhancer.enhance(
                    frame
                )
            )

        # ======================================================
        # Step 2: Road segmentation update & filtering
        # ======================================================

        if road_segmenter.enabled:

            road_segmenter.update_from_detections(
                frame,
                detections,
                frame_count,
            )

            if road_segmenter.is_ready:

                for i, det in enumerate(
                    detections
                ):

                    if (
                        len(det) != 3
                        or len(det[0]) != 4
                    ):

                        print(
                            f"BAD DETECTION "
                            f"#{i}: {det!r}"
                        )

                detections = (
                    road_segmenter.filter_detections(
                        detections
                    )
                )

        # ======================================================
        # Step 3: Tracking
        # ======================================================

        if tracker_type == "botsort":

            if detections:

                boxes_tensor = torch.tensor(
                    [
                        [
                            det[0][0],
                            det[0][1],
                            det[0][0]
                            + det[0][2],
                            det[0][1]
                            + det[0][3],
                            det[1],
                            det[2],
                        ]
                        for det in detections
                    ],
                    dtype=torch.float32,
                )

            else:

                boxes_tensor = torch.empty(
                    (
                        0,
                        6,
                    ),
                    dtype=torch.float32,
                )

            # --------------------------------------------------
            # Adapter for installed Ultralytics tracker.
            #
            # columns:
            #   0: x1
            #   1: y1
            #   2: x2
            #   3: y2
            #   4: confidence
            #   5: class_id
            # --------------------------------------------------

            tracking_results = TrackerResults(
                boxes_tensor[:, :4],
                boxes_tensor[:, 4],
                boxes_tensor[:, 5],
            )

            tracks = tracker.update(
                tracking_results,
                frame,
            )

        else:

            tracks = tracker.update_tracks(
                detections,
                frame=frame,
            )

        # ======================================================
        # Step 4: Process tracked objects
        # ======================================================

        frame_logs = []
        active_vehicle_count = 0

        # Optional ANPR frame hook.
        if (
            anpr is not None
            and hasattr(
                anpr,
                "begin_frame",
            )
        ):

            anpr.begin_frame(
                frame_count,
                frame,
            )

        # ======================================================
        # Draw virtual speed lines
        # ======================================================

        if (
            speed_enabled
            and draw_speed_lines
        ):

            draw_virtual_lines(
                frame,
                *speed_estimator.virtual_lines,
            )

        # ======================================================
        # Draw road mask
        # ======================================================

        if (
            draw_road_mask
            and road_segmenter.is_ready
        ):

            frame = (
                road_segmenter.draw_road_overlay(
                    frame,
                    alpha=0.15,
                )
            )

        # ======================================================
        # Process each tracked object
        # ======================================================

        for track in tracks:

            # ==================================================
            # BoT-SORT
            # ==================================================

            if tracker_type == "botsort":

                # BoT-SORT output:
                #
                # [x1, y1, x2, y2,
                #  track_id, confidence,
                #  class_id, detection_index]

                if len(track) < 7:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    track[:4],
                )

                track_id = int(
                    track[4]
                )

                conf_val = float(
                    track[5]
                )

                cls_id = int(
                    track[6]
                )

            # ==================================================
            # DeepSORT fallback
            # ==================================================

            else:

                if not track.is_confirmed():
                    continue

                track_id = int(
                    track.track_id
                )

                x1, y1, x2, y2 = map(
                    int,
                    track.to_ltrb(),
                )

                cls_id = int(
                    track.det_class
                )

                conf_val = float(
                    track.det_conf or 0.0
                )

            # ==================================================
            # Clamp bounding box to frame
            # ==================================================

            x1 = max(
                0,
                min(
                    x1,
                    width - 1,
                ),
            )

            y1 = max(
                0,
                min(
                    y1,
                    height - 1,
                ),
            )

            x2 = max(
                0,
                min(
                    x2,
                    width - 1,
                ),
            )

            y2 = max(
                0,
                min(
                    y2,
                    height - 1,
                ),
            )

            if (
                x2 <= x1
                or y2 <= y1
            ):
                continue

            # ==================================================
            # Stabilize vehicle class across the lifetime
            # of the track.
            # ==================================================

            raw_class_id = cls_id

            class_votes = (
                track_class_history.setdefault(
                    track_id,
                    {},
                )
            )

            class_votes[raw_class_id] = (
                class_votes.get(
                    raw_class_id,
                    0,
                ) + 1
            )

            # Dominant class = class observed most often.
            #
            # In a tie, keep the current detector class so
            # a genuinely changed classification can recover.
            dominant_class = max(
                class_votes.items(),
                key=lambda item: (
                    item[1],
                    (
                        1
                        if item[0]
                        == raw_class_id
                        else 0
                    ),
                ),
            )[0]

            pipeline_cls = dominant_class

            total_tracks_seen.add(
                track_id
            )

            # ==================================================
            # Core trajectory intelligence
            # ==================================================

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            track_history[track_id].append(
                (
                    frame_count,
                    cx,
                    cy,
                )
            )

            if track_id not in track_first_frame:
                track_first_frame[track_id] = frame_count

            track_last_frame[track_id] = frame_count

            total_vehicle_observations += 1

            history = track_history[track_id]

            # --------------------------------------------------
            # Direction estimation
            # --------------------------------------------------

            direction = "unknown"

            if len(history) >= 5:

                _, old_x, old_y = history[0]
                _, new_x, new_y = history[-1]

                dx = new_x - old_x
                dy = new_y - old_y

                displacement = (
                    (dx * dx + dy * dy) ** 0.5
                )

                if displacement < 8:
                    direction = "stationary"

                elif abs(dy) > abs(dx):

                    if dy > 0:
                        direction = "down"
                    else:
                        direction = "up"

                else:

                    if dx > 0:
                        direction = "right"
                    else:
                        direction = "left"

            track_direction[track_id] = direction

            # --------------------------------------------------
            # Stationary vehicle detection
            # --------------------------------------------------

            if direction == "stationary":

                track_stationary_frames[
                    track_id
                ] += 1

            else:

                track_stationary_frames[
                    track_id
                ] = 0

            active_vehicle_count += 1

            # --------------------------------------------------
            # Persistent vehicle state
            # --------------------------------------------------

            current_track_state = update_track_state(
                track_state,
                track_id,
                frame_count,
                fps,
                pipeline_cls,
                (
                    x1,
                    y1,
                    x2,
                    y2,
                ),
                conf_val,
            )

            # ==================================================
            # Draw bounding box
            # ==================================================

            cv2.rectangle(
                frame,
                (
                    x1,
                    y1,
                ),
                (
                    x2,
                    y2,
                ),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"ID:{track_id}",
                (
                    x1,
                    max(
                        20,
                        y1 - 10,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # ==================================================
            # Speed estimation
            # ==================================================

            speed_kmh = None
            is_over_speed = False

            if speed_enabled:

                speed_kmh = (
                    speed_estimator.update(
                        track_id,
                        (
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                        frame_count,
                        fps,
                    )
                )

                is_over_speed = (
                    speed_estimator.is_speeding(
                        track_id
                    )
                )

                draw_speed_overlay(
                    frame,
                    speed_kmh,
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    is_over_speed,
                )

                if speed_kmh is not None:
                    current_track_state["max_speed_kmh"] = max(
                        current_track_state["max_speed_kmh"],
                        speed_kmh,
                    )

                if is_over_speed:
                    current_track_state["speeding"] = True

            # ==================================================
            # ANPR
            # ==================================================

            plate_text = ""

            if anpr is not None:

                run_anpr = True

                if (
                    road_segmenter.enabled
                    and road_segmenter.is_ready
                ):

                    run_anpr = (
                        road_segmenter.is_in_action_zone(
                            (
                                x1,
                                y1,
                                x2,
                                y2,
                            ),
                            zone_type="anpr",
                        )
                    )

                if run_anpr:

                    plate_text = (
                        anpr.update_track(
                            track_id,
                            frame,
                            (
                                x1,
                                y1,
                                x2,
                                y2,
                            ),
                            frame_count,
                        )
                    )

                    if (
                        pipeline_cfg.get(
                            "draw_plate_boxes",
                            True,
                        )
                        and hasattr(
                            anpr,
                            "get_track_boxes",
                        )
                    ):

                        for plate_box in (
                            anpr.get_track_boxes(
                                track_id
                            )
                        ):

                            anpr.draw_plate_box(
                                frame,
                                plate_box,
                            )

                    anpr.draw_plate(
                        frame,
                        plate_text,
                        (
                            x1,
                            y2,
                        ),
                    )

            # ==================================================
            # Helmet detection
            # ==================================================

            helmet_violation = False
            helmet_label = ""

            # IMPORTANT:
            # Use stabilized pipeline_cls here rather than
            # raw_class_id so helmet processing doesn't switch
            # on/off because of one bad detector classification.
            if (
                helmet_model is not None
                and pipeline_cls == 3
            ):

                run_helmet = True

                if (
                    road_segmenter.enabled
                    and road_segmenter.is_ready
                ):

                    run_helmet = (
                        road_segmenter.is_in_action_zone(
                            (
                                x1,
                                y1,
                                x2,
                                y2,
                            ),
                            zone_type="helmet",
                        )
                    )

                if run_helmet:

                    (
                        helmet_violation,
                        helmet_label,
                    ) = detect_helmet_violation(
                        helmet_model,
                        frame,
                        (
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                        helmet_conf,
                        device=device,
                    )

                    if helmet_label:

                        color = (
                            (0, 0, 255)
                            if helmet_violation
                            else (255, 165, 0)
                        )

                        cv2.putText(
                            frame,
                            helmet_label,
                            (
                                x1,
                                (
                                    y2 + 45
                                    if plate_text
                                    else y2 + 20
                                ),
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                        )

            # ==================================================
            # Frame log
            # ==================================================

            frame_logs.append(
                {
                    "track_id": track_id,

                    # Stabilized class.
                    "class_id": pipeline_cls,

                    # Raw detector class for this frame.
                    "raw_class_id": raw_class_id,

                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],

                    "center": [
                        round(cx, 1),
                        round(cy, 1),
                    ],

                    "direction": track_direction.get(
                        track_id,
                        "unknown",
                    ),

                    "stationary_frames": track_stationary_frames.get(
                        track_id,
                        0,
                    ),

                    "plate_text": plate_text,

                    "helmet_violation": (
                        helmet_violation
                    ),

                    "helmet_label": (
                        helmet_label
                    ),

                    "speed_kmh": (
                        round(
                            speed_kmh,
                            1,
                        )
                        if speed_kmh is not None
                        else None
                    ),

                    "speeding": (
                        is_over_speed
                    ),

                    "confidence": conf_val,
                }
            )

        peak_active_vehicles = max(
            peak_active_vehicles,
            active_vehicle_count,
        )

        # ======================================================
        # Save frame results
        # ======================================================

        results_log.append(
            {
                "frame": frame_count,
                "tracks": frame_logs,
            }
        )

        out.write(frame)

        # ======================================================
        # Progress logging
        # ======================================================

        if (
            frame_count % log_every
            == 0
        ):

            active_tracks = len(
                frame_logs
            )

            road_status = (
                "ready"
                if road_segmenter.is_ready
                else "warmup"
            )

            total_ids = len(total_tracks_seen)

            write_live_state(
                frame_count,
                active_tracks,
                total_ids,
                road_status,
                fps,
                total_frames,
                live_state_path,
            )

            print(
                f"Processed "
                f"{frame_count} frames... "
                f"(tracks: "
                f"{active_tracks}, "
                f"total IDs: "
                f"{total_ids}, "
                f"road: "
                f"{road_status})"
            )

    # ==========================================================
    # Cleanup
    # ==========================================================

    cap.release()
    out.release()

    write_live_state(
        frame_count,
        0,
        len(total_tracks_seen),
        "complete",
        fps,
        total_frames,
        live_state_path,
    )

    with open(
        live_state_path,
        "r+",
        encoding="utf-8",
    ) as f:
        state = json.load(f)
        state["status"] = "COMPLETE"

        f.seek(0)
        json.dump(state, f)
        f.truncate()

    # ==========================================================
    # Traffic metrics
    # ==========================================================

    traffic_metrics = calculate_traffic_metrics(
        track_state,
        width,
        height,
    )

    print("\nTraffic metrics:")
    print(
        f"  Unique vehicles: "
        f"{traffic_metrics['vehicle_count']}"
    )
    print(
        f"  Average track duration: "
        f"{traffic_metrics['average_track_duration']:.2f}s"
    )
    print(
        f"  Vehicle classes: "
        f"{traffic_metrics['vehicle_classes']}"
    )

    # ==========================================================
    # Save JSON
    # ==========================================================

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            results_log,
            handle,
            indent=4,
        )

    print(
        "\nVideo processing complete. "
        f"Saved to '{output_path}'."
    )

    print(
        "Saved tracking logs to "
        f"'{results_path}'."
    )

    print(
        "Total unique tracks: "
        f"{len(total_tracks_seen)}"
    )

    print(
        f"Tracker: {tracker_type}"
    )

    # ==========================================================
    # Core Traffic Intelligence
    # ==========================================================

    print(
        "\n" + "=" * 58
    )

    print(
        "TRAJECTORY INTELLIGENCE"
    )

    print(
        "=" * 58
    )

    direction_counts = defaultdict(int)

    for tid, direction in track_direction.items():

        if direction != "unknown":

            direction_counts[
                direction
            ] += 1

    # ----------------------------------------------------------
    # Track duration statistics
    # ----------------------------------------------------------

    durations = []

    for tid in total_tracks_seen:

        if (
            tid in track_first_frame
            and tid in track_last_frame
        ):

            duration = (
                track_last_frame[tid]
                - track_first_frame[tid]
                + 1
            ) / max(fps, 1)

            durations.append(
                duration
            )

    average_duration = (
        sum(durations) / len(durations)
        if durations
        else 0.0
    )

    # ----------------------------------------------------------
    # Stopped vehicles
    # ----------------------------------------------------------

    stopped_vehicles = []

    for tid in total_tracks_seen:

        if track_stationary_frames.get(
            tid,
            0,
        ) >= 15:

            stopped_vehicles.append(
                tid
            )

    print(
        f"Unique vehicles: "
        f"{len(total_tracks_seen)}"
    )

    print(
        f"Peak active vehicles: "
        f"{peak_active_vehicles}"
    )

    print(
        f"Average track duration: "
        f"{average_duration:.2f}s"
    )

    print(
        f"Direction distribution: "
        f"{dict(direction_counts)}"
    )

    print(
        f"Potential stopped vehicles: "
        f"{len(stopped_vehicles)}"
    )

    # ----------------------------------------------------------
    # Save trajectory intelligence
    # ----------------------------------------------------------

    trajectory_summary = {

        "unique_vehicles": len(
            total_tracks_seen
        ),

        "peak_active_vehicles":
            peak_active_vehicles,

        "average_track_duration_seconds":
            round(
                average_duration,
                2,
            ),

        "direction_distribution":
            dict(direction_counts),

        "potential_stopped_vehicles":
            stopped_vehicles,

        "vehicle_directions": {
            str(tid): direction
            for tid, direction
            in track_direction.items()
        },

        "track_durations": {
            str(tid): round(
                (
                    track_last_frame[tid]
                    - track_first_frame[tid]
                    + 1
                ) / max(fps, 1),
                2,
            )
            for tid in total_tracks_seen
            if tid in track_first_frame
            and tid in track_last_frame
        },
    }

    trajectory_path = (
        output_path.parent
        / "trajectory_intelligence.json"
    )

    with open(
        trajectory_path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            trajectory_summary,
            handle,
            indent=4,
        )

    print(
        "Saved trajectory intelligence to "
        f"'{trajectory_path}'."
    )

    # ==========================================================
    # Vehicle track summary
    # ==========================================================

    print("\nVehicle track summary:")

    for track_id in sorted(track_state):

        state = track_state[track_id]

        print(
            f"  ID {track_id}: "
            f"class={state['class_id']}, "
            f"frames={state['frames_seen']}, "
            f"duration={state['duration_seconds']:.1f}s, "
            f"avg_conf={state['average_confidence']:.2f}"
        )

    # ==========================================================
    # Save traffic summary
    # ==========================================================

    traffic_summary = {
        "video": str(video_path),
        "frames": frame_count,
        "fps": fps,
        "resolution": [
            width,
            height,
        ],
        "vehicles": traffic_metrics,
        "trajectory_intelligence": trajectory_summary,
        "tracker": tracker_type,
    }

    summary_path = (
        results_path.parent
        / "traffic_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            traffic_summary,
            handle,
            indent=4,
        )

    print(
        f"Saved traffic summary to "
        f"'{summary_path}'."
    )

    # ==========================================================
    # Class stability summary
    # ==========================================================

    print(
        "\nClass stability by track:"
    )

    for track_id in sorted(
        track_class_history
    ):

        votes = (
            track_class_history[
                track_id
            ]
        )

        dominant = max(
            votes.items(),
            key=lambda item: item[1],
        )[0]

        print(
            f"  ID {track_id}: "
            f"{votes} | "
            f"dominant={dominant}"
        )

    # ==========================================================
    # Speed summary
    # ==========================================================

    if speed_enabled:

        speeding_count = sum(
            1
            for tid in total_tracks_seen
            if speed_estimator.is_speeding(
                tid
            )
        )

        print(
            "Speed estimation: "
            f"{speeding_count} vehicles "
            "flagged for speeding"
        )

    # ==========================================================
    # Excel Export
    # ==========================================================

    print(
        "\nExporting summary to Excel..."
    )

    try:

        import pandas as pd

        summary_data = {}

        for frame_data in results_log:

            for track in frame_data[
                "tracks"
            ]:

                tid = track[
                    "track_id"
                ]

                if tid not in summary_data:

                    summary_data[tid] = {
                        "Track ID": tid,
                        "Class ID": track[
                            "class_id"
                        ],
                        "Max Speed (km/h)": 0,
                        "Speeding Violation": False,
                        "Plate Text": "",
                        "Helmet Status": "",
                    }

                # ==================================================
                # Max speed
                # ==================================================

                if (
                    track["speed_kmh"]
                    is not None
                ):

                    summary_data[tid][
                        "Max Speed (km/h)"
                    ] = max(
                        summary_data[tid][
                            "Max Speed (km/h)"
                        ],
                        track[
                            "speed_kmh"
                        ],
                    )

                # ==================================================
                # Speeding
                # ==================================================

                if track[
                    "speeding"
                ]:

                    summary_data[tid][
                        "Speeding Violation"
                    ] = True

                # ==================================================
                # Plate
                # ==================================================

                if track[
                    "plate_text"
                ]:

                    summary_data[tid][
                        "Plate Text"
                    ] = track[
                        "plate_text"
                    ]

                # ==================================================
                # Helmet
                # ==================================================

                if track[
                    "helmet_label"
                ]:

                    summary_data[tid][
                        "Helmet Status"
                    ] = track[
                        "helmet_label"
                    ]

        # ======================================================
        # Map class IDs to names.
        # ======================================================

        CLASS_NAMES = {
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck",
            8: "Van",
        }

        for tid in summary_data:

            cls_id = summary_data[tid][
                "Class ID"
            ]

            summary_data[tid][
                "Vehicle Type"
            ] = CLASS_NAMES.get(
                cls_id,
                f"Unknown({cls_id})",
            )

        # ======================================================
        # DataFrame
        # ======================================================

        df = pd.DataFrame(
            list(
                summary_data.values()
            )
        )

        df = df[
            [
                "Track ID",
                "Vehicle Type",
                "Max Speed (km/h)",
                "Speeding Violation",
                "Plate Text",
                "Helmet Status",
            ]
        ]

        excel_path = (
            output_path.parent
            / "vehicle_summary.xlsx"
        )

        df.to_excel(
            excel_path,
            index=False,
        )

        print(
            "Successfully saved vehicle "
            "summary to "
            f"'{excel_path}'"
        )

    except ImportError:

        print(
            "Could not export to Excel: "
            "pandas or openpyxl is not installed."
        )

    except Exception as e:

        print(
            f"Error exporting to Excel: {e}"
        )


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":
    main()