import os
import sys
import glob
from pathlib import Path

# Fix UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "data" / "datasets"

print("--- 1. INVESTIGATING HELMET_COMBINED CORRUPT LINES ---")
helmet_dir = DATA_DIR / "helmet_combined"
corrupt_examples = []
for split in ["train", "valid", "test"]:
    lbl_dir = helmet_dir / split / "labels"
    if not lbl_dir.exists():
        continue
    for f in lbl_dir.glob("*.txt"):
        lines = f.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                corrupt_examples.append((str(f.name), split, i+1, line, f"expected 5 tokens, got {len(parts)}"))
                if len(corrupt_examples) > 10:
                    break
            else:
                try:
                    c = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError as e:
                    corrupt_examples.append((str(f.name), split, i+1, line, f"parse error: {e}"))
                    if len(corrupt_examples) > 10:
                        break
        if len(corrupt_examples) > 10:
            break

print(f"Total corrupt examples found (showing up to 10):")
for ex in corrupt_examples[:10]:
    print(f"  File: {ex[0]} ({ex[1]}:L{ex[2]}) -> {ex[3]!r} Reason: {ex[4]}")


print("\n--- 2. INVESTIGATING DASHCOP_OCR & KAGGLE_SYNTHETIC LABELS ---")
dashcop_dir = DATA_DIR / "dashcop_ocr"
if dashcop_dir.exists():
    sample_files = list(dashcop_dir.glob("*"))[:10]
    print("Dashcop OCR sample files/dirs:")
    for sf in sample_files:
        print(f"  {sf.name} ({'dir' if sf.is_dir() else f'{sf.stat().st_size} bytes'})")

kaggle_dir = DATA_DIR / "kaggle_synthetic"
if kaggle_dir.exists():
    sample_files = list(kaggle_dir.glob("*"))[:10]
    print("\nKaggle Synthetic sample files/dirs:")
    for sf in sample_files:
        print(f"  {sf.name} ({'dir' if sf.is_dir() else f'{sf.stat().st_size} bytes'})")


print("\n--- 3. INVESTIGATING CCPD_YOLO ---")
ccpd_dir = DATA_DIR / "ccpd_yolo"
if ccpd_dir.exists():
    print("ccpd_yolo items:")
    for sf in ccpd_dir.glob("*"):
        print(f"  {sf.name} ({'dir' if sf.is_dir() else f'{sf.stat().st_size} bytes'})")


print("\n--- 4. INVESTIGATING VIDEOSET1 ---")
v_dir = DATA_DIR / "videoset1_videos"
x_dir = DATA_DIR / "videoset1_xml"
print(f"videoset1_videos exists: {v_dir.exists()}, items: {len(list(v_dir.glob('*'))) if v_dir.exists() else 0}")
print(f"videoset1_xml exists: {x_dir.exists()}, items: {len(list(x_dir.glob('*'))) if x_dir.exists() else 0}")


print("\n--- 5. CHECKING VEHICLE DETECTION ON HUGGING FACE ---")
with open("hf_audit_report.json", "r", encoding="utf-8") as f:
    import json
    hf_data = json.load(f)

veh_hf = hf_data.get("thundarstrom/traffic-vehicle-detection", {})
veh_files = [f["path"] for f in veh_hf.get("files", [])]
train_imgs = [f for f in veh_files if f.startswith("train/images/")]
train_lbls = [f for f in veh_files if f.startswith("train/labels/")]
test_imgs = [f for f in veh_files if f.startswith("test/images/")]
test_lbls = [f for f in veh_files if f.startswith("test/labels/")]
valid_imgs = [f for f in veh_files if f.startswith("valid/images/") or f.startswith("val/images/")]
valid_lbls = [f for f in veh_files if f.startswith("valid/labels/") or f.startswith("val/labels/")]

print(f"HF traffic-vehicle-detection counts:")
print(f"  train/images: {len(train_imgs)}, train/labels: {len(train_lbls)}")
print(f"  test/images : {len(test_imgs)}, test/labels : {len(test_lbls)}")
print(f"  valid/images: {len(valid_imgs)}, valid/labels: {len(valid_lbls)}")
