"""
Quality filter for merged OCR dataset.
Removes images that are:
1. Too small (plate crop too tiny to read)
2. Too blurry (Laplacian variance below threshold)
3. Wrong aspect ratio (not plate-shaped)
"""
import cv2
import numpy as np
from pathlib import Path

OCR_DIR = Path("c:/projects/traffic-analytics-system/data/datasets/merged_ocr")

# Thresholds
MIN_WIDTH = 60       # Minimum plate width in pixels
MIN_HEIGHT = 15      # Minimum plate height in pixels
MIN_BLUR_SCORE = 30  # Laplacian variance (below = too blurry)
MAX_ASPECT = 12.0    # Max width/height ratio (too wide = likely wrong crop)
MIN_ASPECT = 1.5     # Min width/height ratio (too tall = likely wrong crop)

removed = 0
kept = 0

files = list(OCR_DIR.glob("*.jpg"))
print(f"Scanning {len(files)} images...")

for f in files:
    img = cv2.imread(str(f))
    if img is None:
        f.unlink()
        removed += 1
        continue

    h, w = img.shape[:2]

    # Check minimum size
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        f.unlink()
        removed += 1
        continue

    # Check aspect ratio
    aspect = w / h if h > 0 else 0
    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        f.unlink()
        removed += 1
        continue

    # Check blur using Laplacian variance
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < MIN_BLUR_SCORE:
        f.unlink()
        removed += 1
        continue

    kept += 1

print(f"\nQuality Filter Results:")
print(f"  Kept:    {kept}")
print(f"  Removed: {removed}")
print(f"  Total remaining: {kept}")
