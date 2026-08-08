import os
import cv2
import easyocr
import csv
import glob

def main():
    reader = easyocr.Reader(['en'], gpu=True)

    configs = {
        'raw_beamsearch': dict(decoder='beamsearch'),
        'raw_greedy': dict(decoder='greedy'),
        'allowlist_greedy': dict(
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            decoder='greedy'
        ),
        'allowlist_beamsearch': dict(
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            decoder='beamsearch'
        ),
    }

    image_paths = glob.glob("data/datasets/license_plates/images/train/*.jpg")[:30]
    
    results_log = []
    
    print("Testing 4 OCR configurations on 30 crops...")
    
    for img_path in image_paths:
        fname = os.path.basename(img_path)
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        
        if not os.path.exists(label_path):
            continue
            
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        if not lines:
            continue
            
        parts = lines[0].strip().split()
        cx, cy, bw, bh = map(float, parts[1:5])
        
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        row = {'file': fname}
        
        for config_name, kwargs in configs.items():
            # EasyOCR allows extra kwargs
            result = reader.readtext(
                crop, 
                detail=1, 
                contrast_ths=0.1,
                adjust_contrast=0.5,
                text_threshold=0.6,
                low_text=0.3,
                mag_ratio=2.0,
                **kwargs
            )
            text = ''.join([r[1] for r in result]).upper().replace(' ', '') if result else ''
            row[config_name] = text
            
        results_log.append(row)

    with open('ocr_config_comparison.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_log[0].keys())
        writer.writeheader()
        writer.writerows(results_log)

    print("\nSample Results (First 10 images):")
    for row in results_log[:10]:
        print(f"--- {row['file']} ---")
        for k, v in row.items():
            if k != 'file':
                print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
