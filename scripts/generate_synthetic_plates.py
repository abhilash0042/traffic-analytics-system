import os
import random
import string
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATASET_DIR = "data/datasets/lprnet_synthetic/images"
NUM_IMAGES = 2000

# Standard Indian State Codes
STATES = ['AN','AP','AR','AS','BR','CH','CG','DD','DL','DN',
          'GA','GJ','HR','HP','JK','JH','KA','KL','LA','LD',
          'MP','MH','MN','ML','MZ','NL','OD','PY','PB','RJ',
          'SK','TN','TS','TR','UP','UK','WB']

def generate_plate_string():
    state = random.choice(STATES)
    rto = f"{random.randint(1, 99):02d}"
    letters_len = random.choices([1, 2, 3], weights=[0.2, 0.7, 0.1])[0]
    letters = ''.join(random.choices(string.ascii_uppercase, k=letters_len))
    digits = f"{random.randint(1, 9999):04d}"
    return f"{state}{rto}{letters}{digits}"

def add_noise(image_np):
    # Simulate day/night/blur/bolts
    h, w = image_np.shape[:2]
    
    # 1. Random blur
    if random.random() < 0.4:
        k = random.choice([3, 5])
        image_np = cv2.GaussianBlur(image_np, (k, k), 0)
        
    # 2. Night simulation (darken)
    if random.random() < 0.3:
        image_np = cv2.convertScaleAbs(image_np, alpha=0.5, beta=10)
        
    # 3. Add random bolts (black dots)
    if random.random() < 0.8:
        for _ in range(random.randint(1, 4)):
            bx = random.randint(10, w - 10)
            by = random.randint(5, h - 5)
            cv2.circle(image_np, (bx, by), random.randint(2, 4), (30, 30, 30), -1)
            
    # 4. Add random shadows (gradient)
    if random.random() < 0.5:
        shadow = np.ones_like(image_np, dtype=np.float32)
        direction = random.choice(['left', 'right', 'top', 'bottom'])
        if direction == 'left':
            for x in range(w): shadow[:, x] *= (0.5 + 0.5 * (x / w))
        elif direction == 'right':
            for x in range(w): shadow[:, x] *= (1.0 - 0.5 * (x / w))
        image_np = (image_np * shadow).astype(np.uint8)
        
    return image_np

def generate_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)
    try:
        font = ImageFont.truetype("C:\\Windows\\Fonts\\arialbd.ttf", 40)
    except:
        font = ImageFont.load_default()
        print("Warning: Could not load Arial Bold. Using default font.")

    for i in range(NUM_IMAGES):
        plate_str = generate_plate_string()
        
        # Plate dimensions (94x24 is LPRNet standard, we'll generate 200x50 and let loader resize)
        w, h = 200, 50
        
        # Background color (white or slightly yellow/gray)
        bg_color = (
            random.randint(220, 255),
            random.randint(220, 255),
            random.randint(220, 255)
        )
        
        img = Image.new('RGB', (w, h), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Text color (black)
        text_color = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))
        
        # Draw text with random spacing to simulate real plates
        # Indian plates often have spaces: TS 07 JB 0405
        formatted_str = f"{plate_str[:2]} {plate_str[2:4]} {plate_str[4:-4]} {plate_str[-4:]}"
        
        # Randomly omit spaces to train robustness
        if random.random() < 0.3:
            formatted_str = plate_str
            
        draw.text((10 + random.randint(-5, 5), 5 + random.randint(-5, 5)), formatted_str, font=font, fill=text_color)
        
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_np = add_noise(img_np)
        
        # Format: text_uuid.jpg
        filename = f"{plate_str}_{i:05d}.jpg"
        cv2.imwrite(os.path.join(DATASET_DIR, filename), img_np)
        
        if (i + 1) % 500 == 0:
            print(f"Generated {i + 1}/{NUM_IMAGES} synthetic plates...")

    print(f"\nDone! Dataset saved to {DATASET_DIR}")

if __name__ == '__main__':
    generate_dataset()
