import sys
import cv2
import yaml
from ultralytics import YOLO
from src.anpr_pipeline import ANPREngine

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_single_image.py <image_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    print(f"Loading image from {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        sys.exit(1)
        
    print("Loading config...")
    # Read config
    with open("configs/pipeline_config_zone.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print("Loading model...")
    try:
        model = YOLO("models/plate_detector.pt")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    print("Initializing ANPR Engine...")
    engine = ANPREngine(model, config)
    
    # Optional enhancement (from pipeline)
    print("Enhancing frame for detection...")
    engine.begin_frame(1, img)
    
    print("Detecting plates...")
    # use detect_plates_in_region directly
    detect_frame = engine._enhanced_frame if engine._enhanced_frame is not None else img
    plate_boxes = engine.detect_plates_in_region(detect_frame)
    
    print(f"Found {len(plate_boxes)} plate(s).")
    
    for i, box in enumerate(plate_boxes):
        x1, y1, x2, y2, conf = box
        print(f"\nPlate {i+1}: Box [{x1}, {y1}, {x2}, {y2}] with confidence {conf:.3f}")
        
        crop = img[y1:y2, x1:x2]
        
        print("Running OCR...")
        reading = engine.run_ocr(crop)
        
        if reading:
            print(f"Result: {reading.text} (Confidence: {reading.confidence:.3f})")
            print(f"Raw: {reading.raw_text}")
        else:
            print("OCR returned None (Validation failed or no text found).")
            
        cv2.imwrite(f"plate_crop_{i}.jpg", crop)

if __name__ == '__main__':
    main()
