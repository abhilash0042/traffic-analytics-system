import xml.etree.ElementTree as ET
import cv2
import os
from pathlib import Path

def extract_demo(xml_file, video_file, output_path):
    print(f"Parsing {xml_file}...")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Find the first track with a license plate
    target_frame = -1
    target_box = None
    
    for track in root.findall("track"):
        if track.get("label") == "license_plate":
            # get the first box
            box = track.find("box")
            if box is not None:
                target_frame = int(box.get("frame"))
                target_box = {
                    "xtl": float(box.get("xtl")),
                    "ytl": float(box.get("ytl")),
                    "xbr": float(box.get("xbr")),
                    "ybr": float(box.get("ybr"))
                }
                break
                
    if target_frame == -1:
        print("No license plate found in XML.")
        return
        
    print(f"Found plate at frame {target_frame}. Extracting from video...")
    
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"Failed to open video: {video_file}")
        return
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    
    if ret:
        # Draw bounding box for context
        x1, y1 = int(target_box["xtl"]), int(target_box["ytl"])
        x2, y2 = int(target_box["xbr"]), int(target_box["ybr"])
        
        # Save full frame with bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, "License Plate", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imwrite(str(output_path), frame)
        print(f"Saved demo frame to {output_path}")
    else:
        print("Failed to read frame.")
        
    cap.release()

if __name__ == "__main__":
    # Use the 20221119161855_0060.xml because we know it has license plates from grep
    xml_path = Path(r"c:\projects\traffic-analytics-system\data\datasets\videoset1_xml_extracted\videoset1_xml_annots_with_rider_motor_poly\20211109123408_0060.xml")
    video_path = Path(r"c:\projects\traffic-analytics-system\data\datasets\videoset1_videos\videoset1_videos_part1\20211109123408_0060.mp4")
    output_path = Path(r"C:\Users\Abhilash\.gemini\antigravity-ide\brain\a2681e26-9cae-4c54-b247-70b434ab3228\dashcop_demo.jpg")
    
    if not video_path.exists():
        print(f"Video not extracted yet... {video_path}")
    else:
        extract_demo(xml_path, video_path, output_path)
