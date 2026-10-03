import os
import sys
import glob
import cv2
import numpy as np
from pathlib import Path
import json
import re
from collections import Counter, defaultdict

BASE = Path(r"c:\projects\traffic-analytics-system\data\datasets")

def analyze_yolo_dataset(name, root_dir, class_names):
    root = Path(root_dir)
    print(f"\n=======================================================")
    print(f"ANALYZING YOLO DATASET: {name}")
    print(f"Path: {root}")
    
    splits = {}
    # Check common split structures
    for s in ['train', 'valid', 'val', 'test']:
        # Structure A: images/train, labels/train
        img_dir = root / 'images' / s
        lbl_dir = root / 'labels' / s
        if not img_dir.exists():
            # Structure B: train/images, train/labels
            img_dir = root / s / 'images'
            lbl_dir = root / s / 'labels'
        if img_dir.exists():
            splits[s] = (img_dir, lbl_dir)
            
    if not splits:
        # Maybe flat images & labels
        img_dir = root / 'images'
        lbl_dir = root / 'labels'
        if img_dir.exists():
            splits['all'] = (img_dir, lbl_dir)

    total_images = 0
    total_labels = 0
    class_counts = Counter()
    box_sizes = {'small': 0, 'medium': 0, 'large': 0}
    empty_label_files = 0
    resolutions = []
    blur_scores = []
    brightness_scores = []
    
    # Sample up to 200 images for visual clarity stats
    sample_images = []

    for split_name, (img_dir, lbl_dir) in splits.items():
        imgs = list(img_dir.glob("*.[jJ][pP][gG]")) + list(img_dir.glob("*.[pP][nN][gG]")) + list(img_dir.glob("*.[jJ][pP][eE][gG]"))
        txts = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
        print(f"  Split '{split_name}': {len(imgs)} images, {len(txts)} label files")
        total_images += len(imgs)
        
        # sample images
        if imgs:
            step = max(1, len(imgs) // 50)
            sample_images.extend(imgs[::step][:50])

        for txt_file in txts:
            try:
                with open(txt_file, 'r') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                if not lines:
                    empty_label_files += 1
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        w = float(parts[3])
                        h = float(parts[4])
                        cls_name = class_names[cls_id] if (class_names and cls_id < len(class_names)) else str(cls_id)
                        class_counts[cls_name] += 1
                        total_labels += 1
                        
                        area_norm = w * h
                        if area_norm < 0.005:
                            box_sizes['small'] += 1
                        elif area_norm < 0.05:
                            box_sizes['medium'] += 1
                        else:
                            box_sizes['large'] += 1
            except Exception as e:
                pass

    # Sample image quality metrics
    print(f"  Sampling {len(sample_images)} images for clarity / resolution / lighting analysis...")
    for img_p in sample_images[:150]:
        img = cv2.imread(str(img_p))
        if img is not None:
            h, w = img.shape[:2]
            resolutions.append((w, h))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Laplacian variance
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_scores.append(lap_var)
            # Mean brightness
            brightness_scores.append(np.mean(gray))

    w_list = [r[0] for r in resolutions]
    h_list = [r[1] for r in resolutions]
    
    res_summary = f"{np.min(w_list)}x{np.min(h_list)} to {np.max(w_list)}x{np.max(h_list)} (Median: {int(np.median(w_list))}x{int(np.median(h_list))})" if resolutions else "N/A"
    avg_blur = np.mean(blur_scores) if blur_scores else 0
    pct_blurry = (sum(1 for b in blur_scores if b < 100) / len(blur_scores) * 100) if blur_scores else 0
    avg_brightness = np.mean(brightness_scores) if brightness_scores else 0
    pct_low_light = (sum(1 for b in brightness_scores if b < 60) / len(brightness_scores) * 100) if brightness_scores else 0
    pct_overexposed = (sum(1 for b in brightness_scores if b > 190) / len(brightness_scores) * 100) if brightness_scores else 0

    print(f"  Total Images: {total_images}")
    print(f"  Total Bounding Boxes: {total_labels}")
    print(f"  Empty / Background label files: {empty_label_files}")
    print(f"  Class Distribution: {dict(class_counts)}")
    print(f"  Box Size Distribution: Small (<0.5% area): {box_sizes['small']}, Med (0.5-5%): {box_sizes['medium']}, Large (>5%): {box_sizes['large']}")
    print(f"  Resolution range: {res_summary}")
    print(f"  Clarity (Laplacian Var): Mean={avg_blur:.1f} | Blurry (<100): {pct_blurry:.1f}%")
    print(f"  Lighting (0-255): Mean={avg_brightness:.1f} | Low-light (<60): {pct_low_light:.1f}% | Overexposed (>190): {pct_overexposed:.1f}%")
    
    return {
        'name': name,
        'type': 'YOLO Detection',
        'total_images': total_images,
        'total_labels': total_labels,
        'empty_label_files': empty_label_files,
        'class_counts': dict(class_counts),
        'box_sizes': box_sizes,
        'res_summary': res_summary,
        'avg_blur': avg_blur,
        'pct_blurry': pct_blurry,
        'avg_brightness': avg_brightness,
        'pct_low_light': pct_low_light,
        'pct_overexposed': pct_overexposed
    }

def analyze_ocr_dataset(name, root_dir):
    root = Path(root_dir)
    print(f"\n=======================================================")
    print(f"ANALYZING OCR DATASET: {name}")
    print(f"Path: {root}")
    
    imgs = list(root.rglob("*.[jJ][pP][gG]")) + list(root.rglob("*.[pP][nN][gG]")) + list(root.rglob("*.[jJ][pP][eE][gG]"))
    print(f"  Total OCR images: {len(imgs)}")
    
    char_counter = Counter()
    lengths = []
    valid_indian_format = 0
    resolutions = []
    blur_scores = []
    brightness_scores = []
    aspect_ratios = []
    
    # Check filename labels
    # Pattern: PLATE_rest.ext or PLATE.ext
    INDIAN_PLATE_PATTERN = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z0-9]{1,8}$')
    
    sample_imgs = imgs[::max(1, len(imgs)//200)][:200]
    
    for img_p in imgs:
        stem = img_p.stem
        # Extract plate text
        plate_text = stem.split('_')[0].upper().replace(" ", "").replace("-", "")
        lengths.append(len(plate_text))
        for c in plate_text:
            char_counter[c] += 1
        if INDIAN_PLATE_PATTERN.match(plate_text) and 7 <= len(plate_text) <= 11:
            valid_indian_format += 1
            
    for img_p in sample_imgs:
        img = cv2.imread(str(img_p))
        if img is not None:
            h, w = img.shape[:2]
            resolutions.append((w, h))
            aspect_ratios.append(w / max(1, h))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness_scores.append(np.mean(gray))
            
    w_list = [r[0] for r in resolutions]
    h_list = [r[1] for r in resolutions]
    
    res_summary = f"{np.min(w_list)}x{np.min(h_list)} to {np.max(w_list)}x{np.max(h_list)} (Median: {int(np.median(w_list))}x{int(np.median(h_list))})" if resolutions else "N/A"
    avg_ar = np.mean(aspect_ratios) if aspect_ratios else 0
    avg_blur = np.mean(blur_scores) if blur_scores else 0
    pct_blurry = (sum(1 for b in blur_scores if b < 100) / len(blur_scores) * 100) if blur_scores else 0
    avg_brightness = np.mean(brightness_scores) if brightness_scores else 0
    pct_valid_format = (valid_indian_format / len(imgs) * 100) if imgs else 0

    print(f"  Total Samples: {len(imgs)}")
    print(f"  Valid Indian Standard Text Format: {valid_indian_format} ({pct_valid_format:.1f}%)")
    print(f"  Text Length Range: {min(lengths) if lengths else 0} to {max(lengths) if lengths else 0} (Avg: {np.mean(lengths):.1f})")
    print(f"  Vocabulary Size: {len(char_counter)} unique chars")
    print(f"  Crop Resolutions: {res_summary}")
    print(f"  Average Aspect Ratio (W/H): {avg_ar:.2f}")
    print(f"  Clarity (Laplacian Var): Mean={avg_blur:.1f} | Low sharpness (<100): {pct_blurry:.1f}%")
    print(f"  Average Brightness: {avg_brightness:.1f}")
    
    return {
        'name': name,
        'type': 'OCR Recognition',
        'total_images': len(imgs),
        'pct_valid_format': pct_valid_format,
        'avg_length': np.mean(lengths) if lengths else 0,
        'vocab_size': len(char_counter),
        'res_summary': res_summary,
        'avg_ar': avg_ar,
        'avg_blur': avg_blur,
        'pct_blurry': pct_blurry,
        'avg_brightness': avg_brightness
    }

def analyze_video_dataset(name, root_dir):
    root = Path(root_dir)
    print(f"\n=======================================================")
    print(f"ANALYZING VIDEO DATASET: {name}")
    print(f"Path: {root}")
    vids = list(root.rglob("*.mp4")) + list(root.rglob("*.avi")) + list(root.rglob("*.mkv")) + list(root.rglob("*.mov"))
    print(f"  Total Video Files: {len(vids)}")
    
    vid_stats = []
    for v in vids:
        cap = cv2.VideoCapture(str(v))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = count / max(1, fps)
            vid_stats.append({
                'file': v.name,
                'resolution': f"{w}x{h}",
                'fps': fps,
                'frames': count,
                'duration_sec': duration
            })
            cap.release()
            print(f"    - {v.name}: {w}x{h} @ {fps:.1f} FPS, {count} frames (~{duration:.1f}s)")
    return {
        'name': name,
        'type': 'Video Footage',
        'total_videos': len(vids),
        'videos': vid_stats
    }

if __name__ == '__main__':
    results = {}
    
    # 1. Vehicle Detection
    results['vehicle_detection'] = analyze_yolo_dataset('vehicle_detection (UA-DETRAC)', BASE / 'vehicle_detection', ['bus', 'car', 'truck', 'van'])
    
    # 2. Indian License Plate Detection
    results['perfect_indian_plates'] = analyze_yolo_dataset('perfect_indian_license_plates', BASE / 'perfect_indian_license_plates', ['license_plate'])
    results['license_plates_indian'] = analyze_yolo_dataset('license_plates_indian (OLX)', BASE / 'license_plates_indian', ['license_plate'])
    
    # 3. Helmet Detection Datasets
    results['helmet_detection'] = analyze_yolo_dataset('helmet_detection (Main Balanced)', BASE / 'helmet_detection', ['helmet', 'no_helmet'])
    results['helmet_detection_extra_1'] = analyze_yolo_dataset('helmet_detection_extra (Combo)', BASE / 'helmet_detection_extra' / 'helmet_and_plates_combo', ['bike', 'helmet', 'no-helmet', 'number-plate'])
    results['helmet_detection_extra_2'] = analyze_yolo_dataset('helmet_detection_extra (Roboflow)', BASE / 'helmet_detection_extra' / 'license_plates_roboflow', ['helmet', 'plate', 'rider'])
    results['helmet_detection_new'] = analyze_yolo_dataset('helmet_detection_new', BASE / 'helmet_detection_new', ['helmet', 'no_helmet'])

    # 4. OCR Datasets
    results['merged_ocr'] = analyze_ocr_dataset('merged_ocr (Unified OCR)', BASE / 'merged_ocr')
    results['dashcop_ocr'] = analyze_ocr_dataset('dashcop_ocr (Real Dashcam OCR)', BASE / 'dashcop_ocr' / 'images')
    results['kaggle_synthetic'] = analyze_ocr_dataset('kaggle_synthetic (Synthetic Plates)', BASE / 'kaggle_synthetic')
    results['parseq_dataset'] = analyze_ocr_dataset('parseq_dataset', BASE / 'parseq_dataset')

    # 5. Raw / Extra Annotation Sets
    results['ccpd_yolo'] = analyze_yolo_dataset('ccpd_yolo (Chinese Plates Benchmark)', BASE / 'ccpd_yolo', ['license_plate'])
    results['dashcop_yolo'] = analyze_yolo_dataset('dashcop_yolo (Dashcam YOLO)', BASE / 'dashcop_yolo', ['license_plate'])

    # 6. Video datasets
    results['sample_videos'] = analyze_video_dataset('sample_videos', BASE / 'sample_videos')
    results['videoset1_videos'] = analyze_video_dataset('videoset1_videos', BASE / 'videoset1_videos')

    print("\n\nAll dataset scans finished successfully.")
