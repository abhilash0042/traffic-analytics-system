import os
import cv2
import glob
import csv
from src.anpr_pipeline import ANPREngine
from src.model_utils import load_config

from src.ai_pipeline import load_yolo_model

def main():
    # Load config and disable super-resolution since we found it hallucinates
    config = load_config('configs/pipeline_config_zone.yaml')
    config['anpr']['sr_enabled'] = False
    
    # Initialize our full pipeline
    print("Initializing ANPREngine...")
    plate_model, _, _ = load_yolo_model(config['models']['plate'], 'plate detector')
    engine = ANPREngine(plate_model, config)
    
    # Gather a sample of test images
    test_images = []
    
    # 1. Grab 5 images from Delhi (DL) in the State-wise dataset
    dl_images = glob.glob('data/datasets/indian_vehicles/State-wise_OLX/DL/*.jpg')[:5]
    test_images.extend(dl_images)
    
    # 2. Grab 5 images from Maharashtra (MH)
    mh_images = glob.glob('data/datasets/indian_vehicles/State-wise_OLX/MH/*.jpg')[:5]
    test_images.extend(mh_images)
    
    # 3. Grab 5 images from the DataCluster dataset
    dc_images = glob.glob('data/datasets/indian_vehicles_dc/Indian_vehicle_dataset/*.jpg')[:5]
    test_images.extend(dc_images)
    
    print(f"Running inference on {len(test_images)} images...")
    
    results = []
    
    for img_path in test_images:
        normalized_path = os.path.normpath(img_path)
        dataset_name = normalized_path.split(os.sep)[-3]
        fname = os.path.basename(img_path)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Run full pipeline: detection -> OCR
        boxes = engine.detect_plates_in_region(img)
        
        if not boxes:
            results.append({
                'dataset': dataset_name,
                'file': fname,
                'plate_text': '(no plate detected)',
                'confidence': 0.0
            })
        else:
            # Take the best plate detection
            best_box = max(boxes, key=lambda x: x[4])
            x1, y1, x2, y2, conf = best_box
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
                
            reading = engine.run_ocr(crop)
            
            results.append({
                'dataset': dataset_name,
                'file': fname,
                'plate_text': reading.text if reading else '(ocr failed)',
                'confidence': reading.confidence if reading else 0.0
            })
            
            # Save the cropped plate for visual inspection
            h, w = img.shape[:2]
            px, py = int((x2-x1)*0.1), int((y2-y1)*0.1)
            cx1, cy1 = max(0, x1-px), max(0, y1-py)
            cx2, cy2 = min(w, x2+px), min(h, y2+py)
            
            padded_crop = img[cy1:cy2, cx1:cx2]
            cv2.imwrite(f"TEST_CROP_{fname}", padded_crop)
                
    # Save results to CSV
    with open('indian_dataset_test_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'file', 'plate_text', 'confidence'])
        writer.writeheader()
        writer.writerows(results)
        
    print("\n--- Inference Results ---")
    for r in results:
        print(f"[{r['dataset']}] {r['file']}: {r['plate_text']} (conf: {r['confidence']:.2f})")

if __name__ == "__main__":
    main()
