import os
import glob
import shutil
import random
import xml.etree.ElementTree as ET
import cv2

# Set random seed for reproducibility
random.seed(42)

def convert_xml_to_yolo(xml_path, img_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Read image size from xml
        size = root.find('size')
        if size is None:
            return None
            
        width_elem = size.find('width')
        height_elem = size.find('height')
        
        width_str = width_elem.text if width_elem is not None else None
        height_str = height_elem.text if height_elem is not None else None
        
        # If width or height is missing/zero, read image directly
        if not width_str or not height_str or int(width_str) == 0 or int(height_str) == 0:
            img = cv2.imread(img_path)
            if img is None:
                return None
            height, width = img.shape[:2]
        else:
            width = int(width_str)
            height = int(height_str)
            
        yolo_boxes = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # Coordinate conversion
            x_center = ((xmin + xmax) / 2.0) / width
            y_center = ((ymin + ymax) / 2.0) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            
            # Clip bounds to [0.0, 1.0]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            
            # YOLO format: class_id (0 for license plate) x_center y_center w h
            yolo_boxes.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
        return yolo_boxes
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None

def main():
    olx_dir = r"data/datasets/indian_vehicles/State-wise_OLX"
    output_dir = r"data/datasets/license_plates_indian"
    
    # 1. Gather all XML/JPG pairs
    xml_files = glob.glob(os.path.join(olx_dir, "**", "*.xml"), recursive=True)
    valid_pairs = []
    
    for xml_path in xml_files:
        img_path = xml_path.replace(".xml", ".jpg")
        if os.path.exists(img_path):
            valid_pairs.append((xml_path, img_path))
            
    print(f"Found {len(valid_pairs)} valid XML/JPG image pairs in {olx_dir}.")
    
    if not valid_pairs:
        print("No valid pairs found. Exiting.")
        return
        
    # 2. Shuffle pairs
    random.shuffle(valid_pairs)
    
    # 3. Split into Train (80%), Val (10%), Test (10%)
    total = len(valid_pairs)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    
    splits = {
        "train": valid_pairs[:train_end],
        "val": valid_pairs[train_end:val_end],
        "test": valid_pairs[val_end:]
    }
    
    # 4. Create directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)
        
    # 5. Process and copy files
    counts = {"train": 0, "val": 0, "test": 0}
    for split_name, pairs in splits.items():
        print(f"Processing {split_name} split ({len(pairs)} pairs)...")
        for xml_path, img_path in pairs:
            yolo_boxes = convert_xml_to_yolo(xml_path, img_path)
            if yolo_boxes is None or len(yolo_boxes) == 0:
                continue
                
            fname = os.path.basename(img_path)
            name_only = os.path.splitext(fname)[0]
            
            # Destination paths
            dst_img = os.path.join(output_dir, "images", split_name, fname)
            dst_lbl = os.path.join(output_dir, "labels", split_name, f"{name_only}.txt")
            
            # Copy image
            shutil.copy2(img_path, dst_img)
            
            # Write label txt file
            with open(dst_lbl, "w") as f:
                f.write("\n".join(yolo_boxes) + "\n")
                
            counts[split_name] += 1
            
    print("\nDataset preparation complete!")
    print(f"  Train: {counts['train']} images")
    print(f"  Val:   {counts['val']} images")
    print(f"  Test:  {counts['test']} images")
    
    # 6. Write data.yaml
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: license_plate
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"Wrote data.yaml to {yaml_path}")

if __name__ == "__main__":
    main()
