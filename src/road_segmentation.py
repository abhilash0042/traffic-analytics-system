"""
Road Segmentation Module for Traffic Analytics Pipeline.

Uses YOLO11-seg for instance segmentation. For fixed CCTV cameras, runs
segmentation once on the first frame to generate a static road mask (ROI)
that is reused for all subsequent frames.

Since COCO-pretrained YOLO11-seg does not have explicit 'road' classes,
this module uses a two-pronged approach:
  1. Accumulate vehicle detection masks over initial frames to infer
     the drivable area (where vehicles travel = road).
  2. Optionally load a fine-tuned road segmentation model (BDD100K).

The road mask is used to:
  - Filter out off-road false positive detections
  - Define action zones for targeted ANPR/helmet detection
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.model_utils import resolve_path


class RoadSegmenter:
    """
    Generate and cache road/drivable-area masks for the pipeline.

    For fixed CCTV: runs once, caches forever.
    For dashcam: refreshes every N frames.
    """

    def __init__(self, config: dict[str, Any]):
        seg_cfg = config.get("segmentation", {})

        self.enabled = bool(seg_cfg.get("enabled", False))
        self.static_roi = bool(seg_cfg.get("static_roi", True))
        self.roi_refresh_frames = int(seg_cfg.get("roi_refresh_frames", 0))
        self.filter_off_road = bool(seg_cfg.get("filter_off_road", True))
        self.warmup_frames = int(seg_cfg.get("warmup_frames", 10))

        # Action zone config
        zones_cfg = seg_cfg.get("action_zones", {})
        self.anpr_zone_lane = zones_cfg.get("anpr_zone_lane", None)
        self.helmet_zone_lanes = zones_cfg.get("helmet_zone_lanes", None)
        self.anpr_zone_thresh = float(zones_cfg.get("anpr_zone_threshold", 0.4))
        self._seg_model = None
        model_path = seg_cfg.get("model")
        if model_path and self.enabled:
            resolved = resolve_path(model_path)
            if resolved.is_file():
                try:
                    from ultralytics import YOLO

                    self._seg_model = YOLO(str(resolved))
                    print(f"  Road segmentation model: {resolved.name}")
                except Exception as exc:
                    print(f"  Warning: Could not load segmentation model: {exc}")

        # Cached masks
        self._road_mask: np.ndarray | None = None
        self._road_mask_frame: int = -1
        self._vehicle_accumulator: np.ndarray | None = None
        self._warmup_count: int = 0

    def update_from_detections(
        self,
        frame: np.ndarray,
        detections: list[list],
        frame_number: int,
    ) -> None:
        """
        Accumulate vehicle detection positions to infer drivable area.

        During the warmup period, builds up a heat map of where vehicles
        appear. After warmup, thresholds this into a binary road mask.
        """
        if not self.enabled:
            return

        # If we already have a static mask and don't need to refresh, skip
        if (
            self._road_mask is not None
            and self.static_roi
            and self.roi_refresh_frames <= 0
        ):
            return

        # Check if periodic refresh is needed
        if (
            self._road_mask is not None
            and self.roi_refresh_frames > 0
            and frame_number % self.roi_refresh_frames != 0
        ):
            return

        h, w = frame.shape[:2]

        # Initialize accumulator
        if self._vehicle_accumulator is None:
            self._vehicle_accumulator = np.zeros((h, w), dtype=np.float32)

        # Accumulate bounding box positions
        for det in detections:
            bbox, conf, cls_id = det
            x, y, bw, bh = bbox
            x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 > x1 and y2 > y1:
                self._vehicle_accumulator[y1:y2, x1:x2] += 1.0

        self._warmup_count += 1

        # After warmup, generate the road mask
        if self._warmup_count >= self.warmup_frames and self._road_mask is None:
            self._generate_road_mask_from_accumulator(h, w)

    def _generate_road_mask_from_accumulator(self, h: int, w: int) -> None:
        """Convert accumulated vehicle positions into a binary road mask."""
        if self._vehicle_accumulator is None:
            return

        # Normalize
        acc = self._vehicle_accumulator.copy()
        max_val = acc.max()
        if max_val > 0:
            acc = acc / max_val

        # Threshold: areas where vehicles appeared in >10% of warmup frames
        binary = (acc > 0.1).astype(np.uint8) * 255

        # Dilate to fill gaps between vehicle paths
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        # Close small holes
        closed = cv2.morphologyEx(
            dilated, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30)),
        )

        # Fill from convex hull of the road region
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros((h, w), dtype=np.uint8)
        if contours:
            # Use the largest contour(s)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours[:3]:  # top 3 largest road regions
                hull = cv2.convexHull(cnt)
                cv2.fillConvexPoly(mask, hull, 255)

        self._road_mask = mask
        print(f"  Road mask generated: {np.count_nonzero(mask)} / {h * w} pixels "
              f"({100 * np.count_nonzero(mask) / (h * w):.1f}% coverage)")

    def update_from_segmentation(self, frame: np.ndarray, frame_number: int) -> None:
        """Run the segmentation model to generate road mask (if model is loaded)."""
        if not self.enabled or self._seg_model is None:
            return

        if self._road_mask is not None and self.static_roi:
            return

        results = self._seg_model.predict(frame, device=0, verbose=False, retina_masks=True)
        result = results[0]

        h, w = frame.shape[:2]
        road_mask = np.zeros((h, w), dtype=np.uint8)

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes
            # Look for vehicle classes as road proxy
            vehicle_classes = {2, 5, 7}  # car, bus, truck in COCO
            for mask, box in zip(masks, boxes):
                cls_id = int(box.cls[0])
                if cls_id in vehicle_classes:
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                    road_mask[mask > 0.5] = 255

            # Dilate vehicle masks to approximate road surface
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
            road_mask = cv2.dilate(road_mask, kernel, iterations=3)

        self._road_mask = road_mask
        self._road_mask_frame = frame_number

    def filter_detections(
        self, detections: list[list], min_overlap: float = 0.3
    ) -> list[list]:
        """
        Filter out detections that are mostly outside the road mask.

        Args:
            detections: List of [[x, y, w, h], conf, cls_id].
            min_overlap: Minimum fraction of bbox area that must be on road.

        Returns:
            Filtered detections.
        """
        if not self.enabled or not self.filter_off_road or self._road_mask is None:
            return detections

        h, w = self._road_mask.shape
        filtered = []

        for det in detections:
            bbox, conf, cls_id = det
            x, y, bw, bh = bbox
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))

            if x2 <= x1 or y2 <= y1:
                continue

            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area <= 0:
                continue

            road_pixels = np.count_nonzero(self._road_mask[y1:y2, x1:x2])
            overlap = road_pixels / bbox_area

            if overlap >= min_overlap:
                filtered.append(det)

        return filtered

    def is_in_action_zone(
        self,
        bbox: tuple[int, int, int, int],
        zone_type: str = "anpr",
    ) -> bool:
        """
        Check if a tracked vehicle is in an action zone.

        For now, uses simple position-based zones (bottom half of frame
        = capture zone for ANPR where plates are larger/clearer).
        """
        if not self.enabled:
            return True  # If segmentation disabled, always allow

        x1, y1, x2, y2 = bbox
        cy = (y1 + y2) / 2

        if self._road_mask is not None:
            h = self._road_mask.shape[0]
            if zone_type == "anpr":
                # ANPR zone: customizable threshold (closer = larger plates)
                return cy > h * self.anpr_zone_thresh
            elif zone_type == "helmet":
                # Helmet zone: bottom 50% of frame (closer = clearer rider)
                return cy > h * 0.5
        return True

    def get_road_mask(self) -> np.ndarray | None:
        """Return the current road mask (or None if not ready)."""
        return self._road_mask

    def draw_road_overlay(
        self,
        frame: np.ndarray,
        alpha: float = 0.25,
        color: tuple[int, int, int] = (0, 200, 0),
    ) -> np.ndarray:
        """Draw semi-transparent road mask overlay on frame."""
        if self._road_mask is None:
            return frame

        overlay = frame.copy()
        mask_bool = self._road_mask > 0
        overlay[mask_bool] = (
            overlay[mask_bool] * (1 - alpha)
            + np.array(color, dtype=np.uint8) * alpha
        ).astype(np.uint8)
        return overlay

    @property
    def is_ready(self) -> bool:
        """Whether the road mask has been generated."""
        return self._road_mask is not None
