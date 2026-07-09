import os
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import random

def main():
    base_dir = Path("c:/projects/traffic-analytics-system/data/datasets")
    xml_dir = base_dir / "videoset1_xml_extracted" / "videoset1_xml_annots_with_rider_motor_poly"
    video_dir = base_dir / "videoset1_videos" / "videoset1_videos_part1"
    
    yolo_images_dir = base_dir / "dashcop_yolo" / "images" / "train"
    yolo_labels_dir = base_dir / "dashcop_yolo" / "labels" / "train"
    ocr_dir = base_dir / "dashcop_ocr" / "images"
    
    yolo_images_dir.mkdir(parents=True, exist_ok=True)
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)
    ocr_dir.mkdir(parents=True, exist_ok=True)
    
    # We want to extract ALL useful plates (max 50000)
    TARGET_SAMPLES = 50000
    extracted_count = 0
    
    xml_files = list(xml_dir.glob("*.xml"))
    random.shuffle(xml_files) # Shuffle to get plates from different videos
    
    for xml_file in xml_files:
        if extracted_count >= TARGET_SAMPLES:
            break
            
        video_filename = xml_file.stem + ".mp4"
        video_path = video_dir / video_filename
        
        if not video_path.exists():
            # Try avi
            video_filename = xml_file.stem + ".avi"
            video_path = video_dir / video_filename
            if not video_path.exists():
                print(f"Warning: Video not found for {xml_file.name}")
                continue
                
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Collect tracks with valid lp_number
        best_boxes = {} # track_id -> (frame_id, box_data, lp_number)
        
        for track in root.findall(".//track[@label='license_plate']"):
            track_id = track.get("id")
            for box in track.findall("box"):
                lp_attr = box.find(".//attribute[@name='lp_number']")
                if lp_attr is not None and lp_attr.text is not None:
                    lp_text = lp_attr.text.strip()
                    # A valid plate usually has > 6 chars and no '#'
                    if lp_text != '#' and len(lp_text) > 4:
                        xtl = float(box.get("xtl"))
                        ytl = float(box.get("ytl"))
                        xbr = float(box.get("xbr"))
                        ybr = float(box.get("ybr"))
                        area = (xbr - xtl) * (ybr - ytl)
                        
                        # Save the largest box for this track
                        if track_id not in best_boxes or area > best_boxes[track_id]['area']:
                            best_boxes[track_id] = {
                                'frame': int(box.get("frame")),
                                'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr,
                                'area': area,
                                'lp_text': lp_text
                            }
        
        if not best_boxes:
            continue
            
        print(f"Found {len(best_boxes)} valid plates in {xml_file.name}")
        
        # Open video and extract frames
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Group by frame to minimize video seeking
        frames_to_extract = {}
        for tid, data in best_boxes.items():
            if data['frame'] not in frames_to_extract:
                frames_to_extract[data['frame']] = []
            frames_to_extract[data['frame']].append((tid, data))
            
        sorted_frames = sorted(frames_to_extract.keys())
        current_frame_idx = 0
        
        for frame_idx in sorted_frames:
            if extracted_count >= TARGET_SAMPLES:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            for tid, data in frames_to_extract[frame_idx]:
                lp_text = data['lp_text'].replace(".", "").replace(" ", "").replace("-", "")
                xtl, ytl, xbr, ybr = data['xtl'], data['ytl'], data['xbr'], data['ybr']
                
                # Constrain to image bounds
                xtl = max(0, int(xtl))
                ytl = max(0, int(ytl))
                xbr = min(width, int(xbr))
                ybr = min(height, int(ybr))
                
                # Skip invalid boxes
                if xbr <= xtl or ybr <= ytl:
                    continue
                    
                # 1. Save YOLO Full Frame and Label
                yolo_img_name = f"dashcop_{xml_file.stem}_{tid}.jpg"
                yolo_lbl_name = f"dashcop_{xml_file.stem}_{tid}.txt"
                
                # Normalized YOLO format: class x_center y_center width height
                w_norm = (xbr - xtl) / width
                h_norm = (ybr - ytl) / height
                cx_norm = (xtl + (xbr - xtl) / 2) / width
                cy_norm = (ytl + (ybr - ytl) / 2) / height
                
                yolo_label_str = f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                
                cv2.imwrite(str(yolo_images_dir / yolo_img_name), frame)
                with open(yolo_labels_dir / yolo_lbl_name, "w") as f:
                    f.write(yolo_label_str + "\n")
                    
                # 2. Save OCR Crop
                crop = frame[ytl:ybr, xtl:xbr]
                ocr_img_name = f"{lp_text}_{xml_file.stem}_{tid}.jpg"
                cv2.imwrite(str(ocr_dir / ocr_img_name), crop)
                
                extracted_count += 1
                
        cap.release()
        
    print(f"Extraction complete! Successfully extracted {extracted_count} high-quality plate samples.")

if __name__ == "__main__":
    main()
