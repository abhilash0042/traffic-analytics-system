"""
ANPR pipeline: plate detection -> preprocessing -> OCR -> validation -> voting.

Tuned for Indian traffic / dashcam footage:
  - Low-confidence fallback detection for small or blurry plates
  - Multi-variant OCR (threshold, CLAHE, sharpen, inverted)
  - Upscale tiny plate crops before OCR
  - Frame denoise + CLAHE for moving / low-light video
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

# Common EasyOCR confusions on Indian plates (apply in digit vs letter zones).
_OCR_DIGIT_FIX = str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"})
_OCR_LETTER_FIX = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "6": "G"})


@dataclass
class PlateReading:
    text: str
    confidence: float
    raw_text: str


@dataclass
class TrackPlateState:
    votes: deque[str] = field(default_factory=deque)
    best_text: str = ""
    best_confidence: float = 0.0
    last_boxes: list[tuple[int, int, int, int, float]] = field(default_factory=list)


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
        roi = frame[roi_y1 : min(h, y2), max(0, x1) : min(w, x2)]
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

    def get_track_boxes(self, track_id: int) -> list[tuple[int, int, int, int, float]]:
        return self.track_states[track_id].last_boxes

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
        plate_cfg = config["models"]["plate"]
        pipeline_cfg = config.get("pipeline", {})

        self.confidence = float(plate_cfg.get("confidence", 0.28))
        self.confidence_fallback = float(plate_cfg.get("detect_conf_fallback", 0.15))
        self.detect_imgsz = int(plate_cfg.get("detect_imgsz", 960))
        self.min_width = int(anpr_cfg.get("min_plate_width", 32))
        self.min_height = int(anpr_cfg.get("min_plate_height", 12))
        self.upscale_min_width = int(anpr_cfg.get("upscale_min_width", 160))
        self.ocr_every_n = int(anpr_cfg.get("ocr_every_n_frames", 1))
        self.vote_window = int(anpr_cfg.get("vote_window", 15))
        self.multi_variant_ocr = bool(anpr_cfg.get("multi_variant_ocr", True))
        self.fuzzy_validation = bool(anpr_cfg.get("fuzzy_validation", True))
        self.detect_full_frame_fallback = bool(anpr_cfg.get("detect_full_frame_fallback", True))
        self.patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in anpr_cfg.get("plate_patterns", [])
        ]
        if self.fuzzy_validation:
            self.patterns.append(re.compile(r"^[A-Z]{2}[0-9OIZSB]{1,2}[A-Z]{1,3}[0-9OIZSB]{1,4}$"))

        self.use_vehicle_crop = bool(pipeline_cfg.get("anpr_on_vehicle_crop", True))
        self.crop_padding = float(pipeline_cfg.get("vehicle_crop_padding", 0.15))
        self.video_enhance = bool(pipeline_cfg.get("video_enhance_for_anpr", True))

        self.prep = prep_cfg
        languages = anpr_cfg.get("ocr_languages", ["en"])
        self.reader = easyocr.Reader(languages, gpu=False, verbose=False)
        self.track_states: dict[int, TrackPlateState] = defaultdict(self._new_track_state)

    def _new_track_state(self) -> TrackPlateState:
        return TrackPlateState(votes=deque(maxlen=self.vote_window))

    def enhance_frame_for_detection(self, frame: np.ndarray) -> np.ndarray:
        if not self.video_enhance or not self.prep.get("enable_video_enhance", True):
            return frame

        strength = int(self.prep.get("denoise_strength", 7))
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, 7, 21)

        if not self.prep.get("low_light_clahe", True):
            return denoised

        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=float(self.prep.get("video_clahe_clip", 3.0)),
            tileGridSize=(
                int(self.prep.get("clahe_tile_size", 8)),
                int(self.prep.get("clahe_tile_size", 8)),
            ),
        )
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def upscale_crop(self, crop: np.ndarray) -> np.ndarray:
        height, width = crop.shape[:2]
        if width >= self.upscale_min_width:
            return crop
        scale = self.upscale_min_width / max(width, 1)
        new_size = (max(self.upscale_min_width, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(crop, new_size, interpolation=cv2.INTER_CUBIC)

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        amount = float(self.prep.get("sharpen_amount", 1.2))
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

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
            clipLimit=float(self.prep.get("clahe_clip_limit", 3.0)),
            tileGridSize=(
                int(self.prep.get("clahe_tile_size", 8)),
                int(self.prep.get("clahe_tile_size", 8)),
            ),
        )
        return clahe.apply(thresh)

    def preprocess_variants(self, crop: np.ndarray) -> list[np.ndarray]:
        if crop.size == 0:
            return []

        upscaled = self.upscale_crop(crop)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=float(self.prep.get("clahe_clip_limit", 3.0)),
            tileGridSize=(
                int(self.prep.get("clahe_tile_size", 8)),
                int(self.prep.get("clahe_tile_size", 8)),
            ),
        )
        clahe_gray = clahe.apply(gray)
        sharpened = self.sharpen(upscaled)
        variants = [
            upscaled,
            cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(self.preprocess_plate(upscaled), cv2.COLOR_GRAY2BGR),
            sharpened,
            cv2.cvtColor(cv2.bitwise_not(clahe_gray), cv2.COLOR_GRAY2BGR),
        ]
        if not self.multi_variant_ocr:
            return [variants[0]]
        return variants

    def normalize_text(self, text: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", text.upper())

    def correct_indian_plate(self, text: str) -> str:
        """Fix common OCR confusions using Indian plate layout (LL DD LLL DDDD style)."""
        if len(text) < 6:
            return text

        chars = list(text)
        # State code + series letters
        for idx in range(min(4, len(chars))):
            if idx < 2 or (len(chars) > 4 and idx in (3, 4)):
                chars[idx] = chars[idx].translate(_OCR_LETTER_FIX)
        # Remaining positions are usually digits
        for idx in range(2, len(chars)):
            if idx in (2, 3) and len(chars) >= 8:
                chars[idx] = chars[idx].translate(_OCR_DIGIT_FIX)
            elif idx >= 4:
                if idx < len(chars) - 4 and idx <= 5:
                    chars[idx] = chars[idx].translate(_OCR_LETTER_FIX)
                else:
                    chars[idx] = chars[idx].translate(_OCR_DIGIT_FIX)
        return "".join(chars)

    def validate_plate(self, text: str) -> bool:
        if not text or len(text) < 6:
            return False
        if any(pattern.match(text) for pattern in self.patterns):
            return True
        if self.fuzzy_validation and len(text) >= 8 and text[:2].isalpha():
            digit_count = sum(char.isdigit() for char in text[2:])
            return digit_count >= 3
        return False

    def run_ocr(self, crop: np.ndarray) -> PlateReading | None:
        if crop.shape[0] < self.min_height or crop.shape[1] < self.min_width:
            if crop.shape[0] < 8 or crop.shape[1] < 20:
                return None
            crop = self.upscale_crop(crop)

        best_reading: PlateReading | None = None
        for variant in self.preprocess_variants(crop):
            results = self.reader.readtext(variant, detail=1, paragraph=False)
            for _box, raw_text, conf in results:
                normalized = self.normalize_text(str(raw_text))
                if not normalized:
                    continue
                corrected = self.correct_indian_plate(normalized)
                confidence = float(conf)
                candidate = corrected if self.validate_plate(corrected) else corrected
                reading = PlateReading(text=candidate, confidence=confidence, raw_text=str(raw_text))
                if best_reading is None or reading.confidence > best_reading.confidence:
                    best_reading = reading

        if best_reading is None:
            return None

        if self.validate_plate(best_reading.text):
            return best_reading

        if len(best_reading.text) >= 7 and any(char.isdigit() for char in best_reading.text):
            return PlateReading(
                text=best_reading.text,
                confidence=best_reading.confidence * 0.75,
                raw_text=best_reading.raw_text,
            )
        return None

    def _predict_plates(self, search: np.ndarray, confidence: float):
        return self.plate_model.predict(
            search,
            conf=confidence,
            imgsz=self.detect_imgsz,
            verbose=False,
        )

    def detect_plates_in_region(
        self,
        frame: np.ndarray,
        region: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[int, int, int, int, float]]:
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

        plates: list[tuple[int, int, int, int, float]] = []
        for confidence in (self.confidence, self.confidence_fallback):
            results = self._predict_plates(search, confidence)
            for result in results:
                if result.boxes is None:
                    continue
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
            if plates:
                break

        # Deduplicate overlapping boxes — keep highest confidence
        plates.sort(key=lambda item: item[4], reverse=True)
        filtered: list[tuple[int, int, int, int, float]] = []
        for plate in plates:
            x1, y1, x2, y2, _conf = plate
            duplicate = False
            for fx1, fy1, fx2, fy2, _ in filtered:
                ix1, iy1 = max(x1, fx1), max(y1, fy1)
                ix2, iy2 = min(x2, fx2), min(y2, fy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area = max(1, (x2 - x1) * (y2 - y1))
                    if inter / area > 0.5:
                        duplicate = True
                        break
            if not duplicate:
                filtered.append(plate)
        return filtered

    def update_track(
        self,
        track_id: int,
        frame: np.ndarray,
        vehicle_bbox: tuple[int, int, int, int] | None,
        frame_number: int,
    ) -> str:
        state = self.track_states[track_id]
        if frame_number % self.ocr_every_n != 0:
            return state.best_text

        detect_frame = self.enhance_frame_for_detection(frame)
        region = vehicle_bbox if self.use_vehicle_crop else None
        plate_boxes = self.detect_plates_in_region(detect_frame, region)

        if not plate_boxes and vehicle_bbox is not None and self.detect_full_frame_fallback:
            plate_boxes = self.detect_plates_in_region(detect_frame, region=None)

        state.last_boxes = plate_boxes
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
            voted_text, _ = Counter(state.votes).most_common(1)[0]
            corrected = self.correct_indian_plate(voted_text)
            if self.validate_plate(corrected):
                state.best_text = corrected

        return state.best_text

    def get_track_boxes(self, track_id: int) -> list[tuple[int, int, int, int, float]]:
        return self.track_states[track_id].last_boxes

    def draw_plate_box(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int, float],
        color: tuple[int, int, int] = (0, 165, 255),
    ) -> None:
        x1, y1, x2, y2, conf = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"plate {conf:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

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
