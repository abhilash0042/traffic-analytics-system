import os
import cv2
import easyocr
from anpr_pipeline import ANPREngine
import yaml

def main():
    config_path = "configs/pipeline_config_zone.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    engine = ANPREngine(None, config)
    reader = easyocr.Reader(['en'], gpu=True)

    test_files = [
        "data/datasets/license_plates/images/train/train_000000.jpg",
        "data/datasets/license_plates/images/train/train_000010.jpg",
        "data/datasets/license_plates/images/train/train_000016.jpg"
    ]

    for img_path in test_files:
        fname = os.path.basename(img_path)
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_path}")
            continue
            
        h, w = img.shape[:2]
        
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        if not lines:
            continue
            
        parts = lines[0].strip().split()
        cx, cy, bw, bh = map(float, parts[1:5])
        
        # Convert to pixel coordinates
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        # Pad slightly to ensure the whole plate is visible
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        crop = img[y1:y2, x1:x2]
        
        cw, ch = crop.shape[1], crop.shape[0]
        print(f"\n=== {fname} ===")
        print(f"Raw crop size: {cw}x{ch}")
        
        # CRITICAL: visually save the EXACT raw crop with no preprocessing at all
        raw_name = f"RAWCHECK_{fname}"
        cv2.imwrite(raw_name, crop)
        
        # Run EasyOCR on the completely untouched raw crop, no upscaling, no variants
        results = reader.readtext(crop, detail=1)
        print("Raw image OCR (no preprocessing at all):")
        for (bbox, text, conf) in results:
            print(f"  '{text}'  conf={conf:.3f}  bbox={bbox}")
            
        # Variants
        variants = engine.preprocess_variants(crop)
        variant_names = ["Upscaled", "CLAHE", "Threshold", "Sharpened", "Inverted CLAHE"]
        
        print("\nVariants OCR:")
        for v_name, variant in zip(variant_names, variants):
            v_res = reader.readtext(
                variant,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                decoder="beamsearch",
                beamWidth=10,
                batch_size=1,
            )
            print(f"  [{v_name}]")
            if not v_res:
                print("    (no reading)")
            for (bbox, text, conf) in v_res:
                print(f"    '{text}'  conf={conf:.3f}")

if __name__ == "__main__":
    main()
