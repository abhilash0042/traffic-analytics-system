"""
Merge script: Processes all Kaggle Indian plate datasets, extracts cropped plate OCR images,
and merges with existing dashcop_ocr data into a unified dataset.

Handles annotation formats:
1. DataCluster (kaggle_indian_plates): <object><name>number_plate</name><attributes><number_plate_text>
2. Saisirishan (kaggle_indian_plates2): <object><name>KA19TR02</name> (plate text IS the name)
3. tkm22092 (kaggle_indian_plates3): TBD after download
"""
import cv2
import xml.etree.ElementTree as ET
import re
from pathlib import Path

BASE = Path("c:/projects/traffic-analytics-system/data/datasets")
OUTPUT_OCR = BASE / "merged_ocr"
OUTPUT_OCR.mkdir(parents=True, exist_ok=True)

# Regex: valid Indian plate text (2 letters, 2 digits, optional 1-3 letters, 4 digits)
# Looser regex - handles commercial plates, 1-letter series, temp plates etc.
PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z0-9]{1,8}$')

stats = {"total": 0, "valid": 0, "invalid": 0, "missing_img": 0}

def crop_and_save(img_path, xmin, ymin, xmax, ymax, plate_text, out_stem):
    """Crop the plate region from the image and save it to output_ocr."""
    if not img_path.exists():
        stats["missing_img"] += 1
        return
    img = cv2.imread(str(img_path))
    if img is None:
        stats["missing_img"] += 1
        return
    h, w = img.shape[:2]
    xmin, ymin = max(0, int(float(xmin))), max(0, int(float(ymin)))
    xmax, ymax = min(w, int(float(xmax))), min(h, int(float(ymax)))
    if xmax <= xmin or ymax <= ymin:
        stats["invalid"] += 1
        return
    crop = img[ymin:ymax, xmin:xmax]
    out_path = OUTPUT_OCR / f"{plate_text}_{out_stem}.jpg"
    cv2.imwrite(str(out_path), crop)
    stats["valid"] += 1
    stats["total"] += 1

def parse_datacluster(dataset_path):
    """DataCluster format: number_plate_text in <attribute> inside <object>."""
    print(f"\n[DataCluster] Scanning {dataset_path}")
    xml_files = list(dataset_path.rglob("*.xml"))
    img_dir = dataset_path / "number_plate_images_ocr" / "number_plate_images_ocr"
    print(f"  Found {len(xml_files)} xml files")
    for xf in xml_files:
        try:
            root = ET.parse(xf).getroot()
            filename = root.findtext("filename", "")
            img_path = img_dir / filename
            for obj in root.findall("object"):
                plate_text = None
                for attr in obj.findall(".//attribute"):
                    if attr.findtext("name") == "number_plate_text":
                        plate_text = attr.findtext("value", "").strip().upper().replace(" ","")
                bb = obj.find("bndbox")
                if plate_text and bb is not None and PLATE_REGEX.match(plate_text):
                    crop_and_save(img_path, bb.findtext("xmin"), bb.findtext("ymin"),
                                  bb.findtext("xmax"), bb.findtext("ymax"),
                                  plate_text, xf.stem)
                else:
                    stats["invalid"] += 1
        except Exception as e:
            print(f"  Error parsing {xf.name}: {e}")

def parse_saisirishan(dataset_path):
    """Saisirishan format: plate text IS the <name> field inside <object>."""
    print(f"\n[Saisirishan] Scanning {dataset_path}")
    xml_files = list(dataset_path.rglob("*.xml"))
    print(f"  Found {len(xml_files)} xml files")
    for xf in xml_files:
        try:
            root = ET.parse(xf).getroot()
            filename = root.findtext("filename", "")
            # Images are in same dir as xml
            img_path = xf.parent / filename
            if not img_path.exists():
                # try finding recursively
                candidates = list(dataset_path.rglob(filename))
                img_path = candidates[0] if candidates else img_path
            for obj in root.findall("object"):
                plate_text = obj.findtext("name", "").strip().upper().replace(" ","")
                bb = obj.find("bndbox")
                if plate_text and bb is not None and PLATE_REGEX.match(plate_text):
                    crop_and_save(img_path, bb.findtext("xmin"), bb.findtext("ymin"),
                                  bb.findtext("xmax"), bb.findtext("ymax"),
                                  plate_text, xf.stem)
                else:
                    stats["invalid"] += 1
        except Exception as e:
            print(f"  Error parsing {xf.name}: {e}")

def parse_yolo_dataset(dataset_path, dataset_name):
    """Crop plates from a YOLO-format dataset using label files."""
    print(f"\n[YOLO Dataset: {dataset_name}] Scanning {dataset_path}")
    import shutil
    images_dir = dataset_path / "images" / "train"
    labels_dir = dataset_path / "labels" / "train"
    if not images_dir.exists() or not labels_dir.exists():
        print(f"  Skipping: train split not found")
        return
    
    processed = 0
    for label_file in labels_dir.glob("*.txt"):
        img_file = images_dir / (label_file.stem + ".jpg")
        if not img_file.exists():
            continue
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h, w = img.shape[:2]
        with open(label_file) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            xmin = int((cx - bw/2) * w)
            ymin = int((cy - bh/2) * h)
            xmax = int((cx + bw/2) * w)
            ymax = int((cy + bh/2) * h)
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue
            crop = img[ymin:ymax, xmin:xmax]
            out_path = OUTPUT_OCR / f"{dataset_name}_{label_file.stem}_{i}.jpg"
            cv2.imwrite(str(out_path), crop)
            processed += 1
            stats['valid'] += 1
            stats['total'] += 1
    print(f"  Processed {processed} plate crops")


def copy_existing_ocr(src_dir):
    """Copy existing valid dashcop_ocr images that match Indian plate format."""
    print(f"\n[Existing OCR] Copying from {src_dir}")
    copied = 0
    for f in src_dir.glob("*.jpg"):
        plate_text = f.stem.split("_")[0].upper()
        if PLATE_REGEX.match(plate_text):
            import shutil
            shutil.copy2(f, OUTPUT_OCR / f.name)
            copied += 1
    print(f"  Copied {copied} valid images")
    stats["valid"] += copied
    stats["total"] += copied

def copy_filename_ocr(src_dir, ext="png"):
    """Copy images where filename IS the plate text (e.g. MH12AB1234.png)."""
    import shutil
    print(f"\n[Filename-as-Label] Copying from {src_dir}")
    copied = 0
    for f in Path(src_dir).rglob(f"*.{ext}"):
        plate_text = f.stem.upper().replace(" ", "").replace("-", "")
        if PLATE_REGEX.match(plate_text) and 6 <= len(plate_text) <= 12:
            out_path = OUTPUT_OCR / f"{plate_text}_{f.stem}_umar.jpg"
            img = cv2.imread(str(f))
            if img is not None:
                cv2.imwrite(str(out_path), img)
                copied += 1
                stats["valid"] += 1
                stats["total"] += 1
    print(f"  Copied {copied} valid images")


if __name__ == "__main__":
    # 1. Copy existing DashCop OCR crops (already cropped plate images)
    copy_existing_ocr(BASE / "dashcop_ocr" / "images")

    # 2. DataCluster dataset (Pascal VOC + number_plate_text attribute)
    parse_datacluster(BASE / "kaggle_indian_plates")

    # 3. Saisirishan indian vehicle dataset (Pascal VOC, plate text as <name>)
    parse_saisirishan(BASE / "kaggle_indian_plates2")

    # 4. perfect_indian_license_plates (YOLO format - crop from full image using labels)
    parse_yolo_dataset(BASE / "perfect_indian_license_plates", "perfect")

    # 5. license_plates_indian (YOLO format)
    parse_yolo_dataset(BASE / "license_plates_indian", "lpindian")

    # 6. abtexp synthetic Indian plates (18k, 100% valid format, filename = plate text)
    copy_filename_ocr(BASE / "kaggle_synthetic", ext="png")

    print(f"\n{'='*50}")
    print(f"MERGE COMPLETE")
    print(f"  Total saved: {stats['valid']}")
    print(f"  Invalid/skipped: {stats['invalid']}")
    print(f"  Missing images: {stats['missing_img']}")
    print(f"  Output dir: {OUTPUT_OCR}")

    # Final count
    final = list(OUTPUT_OCR.glob("*.jpg"))
    print(f"  Final image count in merged_ocr: {len(final)}")

