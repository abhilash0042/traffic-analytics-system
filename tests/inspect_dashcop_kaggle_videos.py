import os
import sys
from pathlib import Path

# Fix UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "data" / "datasets"

print("--- DASHCOP OCR FILES ---")
dashcop_imgs = list((DATA_DIR / "dashcop_ocr" / "images").glob("*"))[:10]
for f in dashcop_imgs:
    print(f"  {f.name}")

print("\n--- KAGGLE SYNTHETIC FILES ---")
kaggle_imgs = list((DATA_DIR / "kaggle_synthetic" / "generated").glob("*"))[:10]
for f in kaggle_imgs:
    print(f"  {f.name}")

print("\n--- VIDEOSET1 FILES ---")
v_files = list((DATA_DIR / "videoset1_videos").glob("*"))
print(f"Videos ({len(v_files)}):")
for f in v_files:
    print(f"  {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")

x_files = list((DATA_DIR / "videoset1_xml").glob("*"))
print(f"\nXMLs ({len(x_files)}):")
for f in x_files:
    print(f"  {f.name} ({f.stat().st_size / (1024):.2f} KB)")
