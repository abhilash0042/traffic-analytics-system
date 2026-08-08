import os
import cv2
import yaml
import glob
from src.anpr_pipeline import ANPREngine

def main():
    config_path = "configs/pipeline_config_zone.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    engine = ANPREngine(None, config)
    
    # Get 30 images
    image_paths = glob.glob("data/datasets/license_plates/images/train/*.jpg")[:30]
    
    print(f"Testing OCR on {len(image_paths)} ground-truth crops...")
    print("-" * 50)
    
    success_count = 0
    total_found = 0
    
    for img_path in image_paths:
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        if not os.path.exists(label_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class cx cy bw bh (normalized)
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
                
                if crop.size == 0:
                    continue
                    
                total_found += 1
                
                # Run isolated OCR
                reading = engine.run_ocr(crop)
                
                if reading:
                    print(f"File: {os.path.basename(img_path)}")
                    print(f"  Extracted: '{reading.text}' (Conf: {reading.confidence:.2f})")
                    success_count += 1
                else:
                    print(f"File: {os.path.basename(img_path)}")
                    print(f"  Extracted: [NO VALID READING]")
                    
    print("-" * 50)
    print(f"Total Plates Tested: {total_found}")
    print(f"Valid OCR Readings: {success_count} ({(success_count/total_found)*100 if total_found else 0:.1f}%)")

if __name__ == "__main__":
    main()
