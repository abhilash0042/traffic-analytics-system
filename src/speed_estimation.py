"""
Speed Estimation Module for Traffic Analytics Pipeline.

Supports two methods:
  1. Virtual-Line: Two horizontal lines with known real-world distance (Paper 5)
  2. Homography: 4-point perspective transform for pixel→meter conversion

Paper 5 formula: v = d / (N × Tᶠ)
  - d  = distance between virtual lines (meters)
  - N  = frame count between crossings
  - Tᶠ = 1 / FPS (seconds per frame)

Validated: MAE = 3.16 km/h at 30 fps (Paper 5).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TrackSpeedState:
    """Per-track state for speed calculation."""

    centroids: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=120)
    )
    frame_numbers: deque[int] = field(default_factory=lambda: deque(maxlen=120))
    crossed_start: int | None = None  # frame number when crossed start line
    crossed_stop: int | None = None  # frame number when crossed stop line
    speed_mps: float | None = None  # meters per second
    speed_kmh: float | None = None  # km/h
    smoothed_speeds: deque[float] = field(
        default_factory=lambda: deque(maxlen=15)
    )
    direction: str = ""  # "approaching" or "receding"


class SpeedEstimator:
    """
    Estimate vehicle speed from tracked bounding box centroids.

    Usage:
        estimator = SpeedEstimator(config)
        # Per frame, per track:
        speed = estimator.update(track_id, bbox, frame_number, fps)
    """

    def __init__(self, config: dict[str, Any]):
        speed_cfg = config.get("speed_estimation", {})

        self.enabled = bool(speed_cfg.get("enabled", True))
        self.method = str(speed_cfg.get("method", "virtual_line"))

        # Virtual-line config
        vl_cfg = speed_cfg.get("virtual_line", {})
        self.start_y = int(vl_cfg.get("start_y", 200))
        self.stop_y = int(vl_cfg.get("stop_y", 500))
        self.distance_meters = float(vl_cfg.get("distance_meters", 30.0))

        # Smoothing
        self.smoothing_window = int(speed_cfg.get("smoothing_window", 5))
        self.speed_limit_kmh = float(speed_cfg.get("speed_limit_kmh", 60.0))

        # Homography (optional, loaded from config or set later)
        self.homography_matrix: np.ndarray | None = None
        homography_points = speed_cfg.get("homography", {})
        if homography_points.get("src_points") and homography_points.get("dst_points"):
            src = np.array(homography_points["src_points"], dtype=np.float32)
            dst = np.array(homography_points["dst_points"], dtype=np.float32)
            self.homography_matrix, _ = cv2.findHomography(src, dst)

        # Per-track state
        self._tracks: dict[Any, TrackSpeedState] = defaultdict(TrackSpeedState)

    def update(
        self,
        track_id: Any,
        bbox: tuple[int, int, int, int],
        frame_number: int,
        fps: float,
    ) -> float | None:
        """
        Update track with new bounding box and return smoothed speed (km/h).

        Args:
            track_id: Persistent track identifier.
            bbox: (x1, y1, x2, y2) bounding box.
            frame_number: Current frame index.
            fps: Video frame rate.

        Returns:
            Smoothed speed in km/h, or None if not enough data.
        """
        if not self.enabled or fps <= 0:
            return None

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        state = self._tracks[track_id]
        state.centroids.append((cx, cy))
        state.frame_numbers.append(frame_number)

        # Determine direction
        if len(state.centroids) >= 5:
            recent_y = [c[1] for c in list(state.centroids)[-5:]]
            if recent_y[-1] > recent_y[0]:
                state.direction = "approaching"  # moving down in frame
            else:
                state.direction = "receding"  # moving up in frame

        if self.method == "virtual_line":
            return self._virtual_line_speed(state, cy, frame_number, fps)
        elif self.method == "homography":
            return self._homography_speed(state, frame_number, fps)
        else:
            return self._centroid_displacement_speed(state, frame_number, fps)

    def _virtual_line_speed(
        self,
        state: TrackSpeedState,
        cy: float,
        frame_number: int,
        fps: float,
    ) -> float | None:
        """Paper 5 method: speed from crossing two virtual lines."""
        # Determine crossing direction (approaching = top→bottom, receding = bottom→top)
        if state.direction == "approaching":
            first_line = self.start_y
            second_line = self.stop_y
        else:
            first_line = self.stop_y
            second_line = self.start_y

        # Check first line crossing
        if state.crossed_start is None:
            if len(state.centroids) >= 2:
                prev_cy = state.centroids[-2][1]
                if (prev_cy <= first_line <= cy) or (prev_cy >= first_line >= cy):
                    state.crossed_start = frame_number

        # Check second line crossing
        if state.crossed_start is not None and state.crossed_stop is None:
            if len(state.centroids) >= 2:
                prev_cy = state.centroids[-2][1]
                if (prev_cy <= second_line <= cy) or (prev_cy >= second_line >= cy):
                    state.crossed_stop = frame_number

        # Calculate speed once both lines are crossed
        if state.crossed_start is not None and state.crossed_stop is not None:
            n_frames = abs(state.crossed_stop - state.crossed_start)
            if n_frames > 0:
                t_seconds = n_frames / fps
                state.speed_mps = self.distance_meters / t_seconds
                state.speed_kmh = state.speed_mps * 3.6
                state.smoothed_speeds.append(state.speed_kmh)

                # Reset for re-measurement if vehicle continues
                state.crossed_start = state.crossed_stop
                state.crossed_stop = None

        # Return smoothed speed
        if state.smoothed_speeds:
            window = list(state.smoothed_speeds)[-self.smoothing_window :]
            return sum(window) / len(window)
        return None

    def _centroid_displacement_speed(
        self,
        state: TrackSpeedState,
        frame_number: int,
        fps: float,
    ) -> float | None:
        """
        Fallback method: estimate speed from centroid displacement.
        Less accurate without calibration but works without virtual lines.
        Uses a rough pixels-per-meter assumption.
        """
        if len(state.centroids) < 5:
            return None

        # Use displacement over last N frames
        n = min(10, len(state.centroids))
        recent = list(state.centroids)[-n:]
        recent_frames = list(state.frame_numbers)[-n:]

        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        pixel_dist = (dx**2 + dy**2) ** 0.5

        frame_span = recent_frames[-1] - recent_frames[0]
        if frame_span <= 0:
            return None

        # Calibrated for CCTV highway perspective
        pixels_per_meter = 4.0  # ≈ 0.25 meters per pixel in the distance
        real_dist = pixel_dist / pixels_per_meter
        t_seconds = frame_span / fps
        speed_mps = real_dist / t_seconds
        speed_kmh = speed_mps * 3.6

        state.speed_kmh = speed_kmh
        state.smoothed_speeds.append(speed_kmh)

        window = list(state.smoothed_speeds)[-self.smoothing_window :]
        return sum(window) / len(window)

    def _homography_speed(
        self,
        state: TrackSpeedState,
        frame_number: int,
        fps: float,
    ) -> float | None:
        """Homography-based speed: transform pixel coords to real-world meters."""
        if self.homography_matrix is None or len(state.centroids) < 5:
            return None

        n = min(10, len(state.centroids))
        recent = list(state.centroids)[-n:]
        recent_frames = list(state.frame_numbers)[-n:]

        # Transform pixel points to real-world coords
        src_pt = np.array([[recent[0][0], recent[0][1]]], dtype=np.float32).reshape(
            -1, 1, 2
        )
        dst_pt = np.array([[recent[-1][0], recent[-1][1]]], dtype=np.float32).reshape(
            -1, 1, 2
        )

        import cv2

        src_world = cv2.perspectiveTransform(src_pt, self.homography_matrix)[0][0]
        dst_world = cv2.perspectiveTransform(dst_pt, self.homography_matrix)[0][0]

        dx = dst_world[0] - src_world[0]
        dy = dst_world[1] - src_world[1]
        real_dist = (dx**2 + dy**2) ** 0.5

        frame_span = recent_frames[-1] - recent_frames[0]
        if frame_span <= 0:
            return None

        t_seconds = frame_span / fps
        speed_mps = real_dist / t_seconds
        speed_kmh = speed_mps * 3.6

        state.speed_kmh = speed_kmh
        state.smoothed_speeds.append(speed_kmh)

        window = list(state.smoothed_speeds)[-self.smoothing_window :]
        return sum(window) / len(window)

    def get_speed(self, track_id: Any) -> float | None:
        """Get last smoothed speed for a track."""
        state = self._tracks.get(track_id)
        if state is None or not state.smoothed_speeds:
            return None
        window = list(state.smoothed_speeds)[-self.smoothing_window :]
        return sum(window) / len(window)

    def is_speeding(self, track_id: Any) -> bool:
        """Check if a track exceeds the configured speed limit."""
        speed = self.get_speed(track_id)
        return speed is not None and speed > self.speed_limit_kmh

    def get_direction(self, track_id: Any) -> str:
        """Get movement direction of a track."""
        state = self._tracks.get(track_id)
        return state.direction if state else ""

    def get_track_state(self, track_id: Any) -> TrackSpeedState | None:
        """Get full state for a track (for debugging/logging)."""
        return self._tracks.get(track_id)

    @property
    def virtual_lines(self) -> tuple[int, int]:
        """Return (start_y, stop_y) for rendering on video."""
        return (self.start_y, self.stop_y)
