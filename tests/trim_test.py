"""
Trim test: Does cropping off the left 15% of the plate crop fix the
leading-character hallucination (6IF, 54, 6NE, etc.)?

If yes → the bug is crop framing (IND strip / hologram bleeding into the bbox).
If no  → the bug is genuinely in EasyOCR's recognition model.
"""

import os, glob, cv2, easyocr

reader = easyocr.Reader(['en'], gpu=True)

image_paths = glob.glob("data/datasets/license_plates/images/train/*.jpg")[:30]

ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

print(f"{'File':<22} {'Full Crop':<20} {'Left-15% Trimmed':<20} {'Bbox x-start'}")
print("-" * 85)

for img_path in image_paths:
    fname = os.path.basename(img_path)
    label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()
    if not lines:
        continue

    parts = lines[0].strip().split()
    cx, cy, bw, bh = map(float, parts[1:5])

    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)

    pad_x = int((x2 - x1) * 0.1)
    pad_y = int((y2 - y1) * 0.1)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    cw = crop.shape[1]

    # Trimmed version: strip leftmost 15%
    left_trim = int(cw * 0.15)
    crop_trimmed = crop[:, left_trim:]

    # Save first 3 for visual inspection
    idx = int(fname.split("_")[-1].split(".")[0])
    if idx < 3:
        cv2.imwrite(f"RAWCHECK_{fname}", crop)
        cv2.imwrite(f"TRIMMED_{fname}", crop_trimmed)

    # OCR on full crop
    res_full = reader.readtext(crop, detail=1, allowlist=ALLOWLIST, decoder='greedy',
                               contrast_ths=0.1, adjust_contrast=0.5,
                               text_threshold=0.6, low_text=0.3, mag_ratio=2.0)
    text_full = ''.join(r[1] for r in res_full).upper().replace(' ', '') if res_full else '(none)'

    # Where does the first detected word start?
    bbox_x = ''
    if res_full:
        x_coords = [pt[0] for pt in res_full[0][0]]
        bbox_x = f"x={min(x_coords):.0f}/{cw}px"

    # OCR on trimmed crop
    res_trim = reader.readtext(crop_trimmed, detail=1, allowlist=ALLOWLIST, decoder='greedy',
                               contrast_ths=0.1, adjust_contrast=0.5,
                               text_threshold=0.6, low_text=0.3, mag_ratio=2.0)
    text_trim = ''.join(r[1] for r in res_trim).upper().replace(' ', '') if res_trim else '(none)'

    print(f"{fname:<22} {text_full:<20} {text_trim:<20} {bbox_x}")
