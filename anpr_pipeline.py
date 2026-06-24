"""
ANPR pipeline: plate detection -> preprocessing -> OCR -> validation -> voting.

Methodology from Papers 2, 3, and 5:
  - YOLO plate detector (fine-tuned, not COCO)
  - Grayscale -> bilateral filter -> adaptive threshold -> CLAHE
  - EasyOCR (pre-trained)
  - Regex validation for Indian plates
  - Multi-frame majority vote per track
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import easyocr
import numpy as np


@dataclass
class PlateReading:
    text: str
    confidence: float
    raw_text: str


@dataclass
class TrackPlateState:
    votes: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    best_text: str = ""
    best_confidence: float = 0.0


class LegacyOCREngine:
    """Fallback ANPR when plate detector is not fine-tuned yet (OCR on vehicle lower ROI)."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        anpr_cfg = config.get("anpr", {})
        self.ocr_every_n = int(anpr_cfg.get("ocr_every_n_frames", 3))
        self.patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in anpr_cfg.get("plate_patterns", [])
        ]
        languages = anpr_cfg.get("ocr_languages", ["en"])
        self.reader = easyocr.Reader(languages, gpu=False, verbose=False)
        self.track_states: dict[int, TrackPlateState] = defaultdict(TrackPlateState)

    def normalize_text(self, text: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", text.upper())

    def validate_plate(self, text: str) -> bool:
        if not text or len(text) < 6:
            return False
        return any(pattern.match(text) for pattern in self.patterns)

    def update_track(
        self,
        track_id: int,
        frame: np.ndarray,
        vehicle_bbox: tuple[int, int, int, int] | None,
        frame_number: int,
    ) -> str:
        if vehicle_bbox is None or frame_number % self.ocr_every_n != 0:
            return self.track_states[track_id].best_text

        x1, y1, x2, y2 = vehicle_bbox
        h, w = frame.shape[:2]
        if (x2 - x1) < 80 or (y2 - y1) < 80:
            return self.track_states[track_id].best_text

        roi_y1 = max(0, int(y1 + 0.65 * (y2 - y1)))
        roi = frame[roi_y1:min(h, y2), max(0, x1):min(w, x2)]
        if roi.size == 0:
            return self.track_states[track_id].best_text

        results = self.reader.readtext(roi, detail=1, paragraph=False)
        if not results:
            return self.track_states[track_id].best_text

        best = max(results, key=lambda item: float(item[2]))
        normalized = self.normalize_text(str(best[1]))
        if not normalized:
            return self.track_states[track_id].best_text

        state = self.track_states[track_id]
        state.votes.append(normalized)
        confidence = float(best[2])
        if confidence > state.best_confidence:
            state.best_confidence = confidence
            state.best_text = normalized

        if state.votes:
            voted_text, _ = Counter(state.votes).most_common(1)[0]
            if self.validate_plate(voted_text):
                state.best_text = voted_text
        return state.best_text

    def draw_plate(self, frame: np.ndarray, plate_text: str, anchor: tuple[int, int]) -> None:
        if not plate_text:
            return
        x, y = anchor
        cv2.putText(
            frame,
            f"Plate: {plate_text}",
            (x, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 128, 255),
            2,
        )


class ANPREngine:
    def __init__(self, plate_model, config: dict[str, Any]):
        self.plate_model = plate_model
        self.config = config
        anpr_cfg = config.get("anpr", {})
        prep_cfg = config.get("preprocessing", {})

        self.confidence = float(config["models"]["plate"].get("confidence", 0.4))
        self.min_width = int(anpr_cfg.get("min_plate_width", 60))
        self.min_height = int(anpr_cfg.get("min_plate_height", 20))
        self.ocr_every_n = int(anpr_cfg.get("ocr_every_n_frames", 3))
        self.vote_window = int(anpr_cfg.get("vote_window", 10))
        self.patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in anpr_cfg.get("plate_patterns", [])
        ]
        self.use_vehicle_crop = bool(config.get("pipeline", {}).get("anpr_on_vehicle_crop", True))
        self.crop_padding = float(config.get("pipeline", {}).get("vehicle_crop_padding", 0.10))

        self.prep = prep_cfg
        languages = anpr_cfg.get("ocr_languages", ["en"])
        self.reader = easyocr.Reader(languages, gpu=False, verbose=False)
        self.track_states: dict[int, TrackPlateState] = defaultdict(TrackPlateState)

    def preprocess_plate(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return crop

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(
            gray,
            int(self.prep.get("bilateral_d", 11)),
            float(self.prep.get("bilateral_sigma_color", 17)),
            float(self.prep.get("bilateral_sigma_space", 17)),
        )
        thresh = cv2.adaptiveThreshold(
            filtered,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            int(self.prep.get("adaptive_block_size", 11)),
            int(self.prep.get("adaptive_c", 2)),
        )
        clahe = cv2.createCLAHE(
            clipLimit=float(self.prep.get("clahe_clip_limit", 2.0)),
            tileGridSize=(
                int(self.prep.get("clahe_tile_size", 8)),
                int(self.prep.get("clahe_tile_size", 8)),
            ),
        )
        return clahe.apply(thresh)

    def normalize_text(self, text: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text.upper())
        return cleaned

    def validate_plate(self, text: str) -> bool:
        if not text or len(text) < 6:
            return False
        return any(pattern.match(text) for pattern in self.patterns)

    def run_ocr(self, crop: np.ndarray) -> PlateReading | None:
        if crop.shape[0] < self.min_height or crop.shape[1] < self.min_width:
            return None

        processed = self.preprocess_plate(crop)
        results = self.reader.readtext(processed, detail=1, paragraph=False)
        if not results:
            results = self.reader.readtext(crop, detail=1, paragraph=False)
        if not results:
            return None

        best = max(results, key=lambda item: float(item[2]))
        raw_text = str(best[1])
        normalized = self.normalize_text(raw_text)
        if not normalized:
            return None

        confidence = float(best[2])
        if self.validate_plate(normalized):
            return PlateReading(text=normalized, confidence=confidence, raw_text=raw_text)

        # Keep partial reads if they look plate-like (for voting across frames)
        if len(normalized) >= 8 and any(char.isdigit() for char in normalized):
            return PlateReading(text=normalized, confidence=confidence * 0.7, raw_text=raw_text)
        return None

    def detect_plates_in_region(self, frame: np.ndarray, region: tuple[int, int, int, int] | None = None):
        search = frame
        offset_x, offset_y = 0, 0

        if region is not None:
            x1, y1, x2, y2 = region
            h, w = frame.shape[:2]
            pad_x = int((x2 - x1) * self.crop_padding)
            pad_y = int((y2 - y1) * self.crop_padding)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            search = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1

        if search.size == 0:
            return []

        results = self.plate_model.predict(search, conf=self.confidence, verbose=False)
        plates = []
        for result in results:
            for box in result.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                plates.append(
                    (
                        int(offset_x + bx1),
                        int(offset_y + by1),
                        int(offset_x + bx2),
                        int(offset_y + by2),
                        float(box.conf[0]),
                    )
                )
        return plates

    def update_track(
        self,
        track_id: int,
        frame: np.ndarray,
        vehicle_bbox: tuple[int, int, int, int] | None,
        frame_number: int,
    ) -> str:
        if frame_number % self.ocr_every_n != 0:
            state = self.track_states[track_id]
            return state.best_text

        region = vehicle_bbox if self.use_vehicle_crop else None
        plate_boxes = self.detect_plates_in_region(frame, region)

        if not plate_boxes and vehicle_bbox is not None and self.use_vehicle_crop:
            plate_boxes = self.detect_plates_in_region(frame, region=None)

        state = self.track_states[track_id]
        for x1, y1, x2, y2, det_conf in plate_boxes:
            crop = frame[y1:y2, x1:x2]
            reading = self.run_ocr(crop)
            if reading is None:
                continue

            state.votes.append(reading.text)
            combined_conf = reading.confidence * det_conf
            if combined_conf > state.best_confidence:
                state.best_confidence = combined_conf
                state.best_text = reading.text

        if state.votes:
            counts = Counter(state.votes)
            voted_text, _ = counts.most_common(1)[0]
            if self.validate_plate(voted_text):
                state.best_text = voted_text

        return state.best_text

    def draw_plate(self, frame: np.ndarray, plate_text: str, anchor: tuple[int, int]) -> None:
        if not plate_text:
            return
        x, y = anchor
        cv2.putText(
            frame,
            f"Plate: {plate_text}",
            (x, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
