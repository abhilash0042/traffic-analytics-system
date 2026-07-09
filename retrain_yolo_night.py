"""
YOLO retraining script with aggressive augmentation for night-time robustness.
This implements "Option 2" from the night-condition optimization plan,
which forces the model to learn to detect vehicles/plates under severe 
brightness and contrast variations using HSV augmentation.
"""

import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Retrain YOLO with aggressive night-time augmentations.")
    parser.add_argument("--data", type=str, default="data/plate_dataset.yaml", help="Path to the dataset YAML file.")
    parser.add_argument("--weights", type=str, default="weights/plate_finetuned.pt", help="Path to initial weights (e.g., current best plate detector).")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--device", type=str, default="0", help="GPU device to use (e.g., '0' or 'cpu').")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers (higher is faster).")
    args = parser.parse_args()

    # Check if dataset exists
    if not os.path.exists(args.data):
        print(f"Dataset configuration {args.data} not found.")
        print("Please update the --data argument to point to your training data.")
        return

    print("Starting YOLO retraining with aggressive night-condition augmentations...")
    print("Using high HSV variance (hsv_v, hsv_s) to simulate low-light scenarios.")

    model_path = args.weights
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found. Falling back to yolov8n.pt base model.")
        model_path = "yolov8n.pt"

    model = YOLO(model_path)

    # Train with aggressive augmentation
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=16,
        imgsz=640,
        device=args.device,
        workers=args.workers,
        name="plate_night_finetune",
        # Aggressive augmentation for night robustness
        hsv_h=0.015,           # Minor hue shift
        hsv_s=0.7,             # Aggressive saturation jitter (color washing out in dark)
        hsv_v=0.8,             # Very aggressive brightness jitter (simulates night/glare)
        degrees=10,            # Slight rotation (e.g. tilted plates)
        translate=0.1,         # Shift
        scale=0.5,             # Zoom in/out
        mosaic=1.0,            # Mosaic augmentation (helps with small objects)
        erasing=0.4,           # Random erasing (simulates occlusions or glare spots)
    )

    print(f"Training complete. Best model saved in {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    main()
