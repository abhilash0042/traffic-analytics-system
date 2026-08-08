import cv2
import numpy as np
import os
import torch
import warnings
warnings.filterwarnings('ignore')

from src.zero_dce import ZeroDCEEnhancer

def test_zero_dce():
    print("Loading Zero-DCE Enhancer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enhancer = ZeroDCEEnhancer(device=device)
    
    images = ["RAWCHECK_train_000001.jpg", "RAWCHECK_train_000002.jpg"]
    
    for idx, input_img_path in enumerate(images):
        if not os.path.exists(input_img_path):
            print(f"Cannot find {input_img_path}")
            continue
            
        img = cv2.imread(input_img_path)
        
        # Artificially darken the image to simulate night
        print(f"Creating simulated dark image for {input_img_path}...")
        dark_img = (img * 0.3).astype(np.uint8)
        
        print("Enhancing with Zero-DCE...")
        enhanced_img = enhancer.enhance(dark_img)
        
        # Create side-by-side comparison
        h, w = dark_img.shape[:2]
        # Resize to something manageable for display
        scale = min(1.0, 800 / w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        orig_resized = cv2.resize(img, (new_w, new_h))
        dark_resized = cv2.resize(dark_img, (new_w, new_h))
        enhanced_resized = cv2.resize(enhanced_img, (new_w, new_h))
        
        # Add labels
        cv2.putText(orig_resized, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(dark_resized, "Simulated Night (Input)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(enhanced_resized, "Zero-DCE Output", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        comparison = np.hstack((dark_resized, enhanced_resized))
        
        output_path = f"C:/Users/Abhilash/.gemini/antigravity-ide/brain/bfa34b96-a29e-4047-8e70-159f2e97cc33/zero_dce_comparison_{idx+2}.jpg"
        cv2.imwrite(output_path, comparison)
        print(f"Saved comparison to: {output_path}")

if __name__ == "__main__":
    test_zero_dce()
