import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

with open("local_dataset_audit.json", "r", encoding="utf-8") as f:
    local_data = json.load(f)

print("=" * 90)
print("                       LOCAL DATASETS HEALTH & INTEGRITY AUDIT")
print("=" * 90)

for key, d in local_data.items():
    print(f"\n📁 [{key.upper()}] - {d.get('title')}")
    print(f"   HF Repo: {d.get('repo_name')}")
    print(f"   Local Path: {d.get('path')}")
    print(f"   Exists Locally: {'✅ Yes' if d.get('exists') else '❌ NO'}")
    
    if not d.get('exists'):
        print(f"   Issues: {d.get('issues')}")
        continue
        
    dtype = d.get("type")
    if dtype == "yolo_detection":
        print(f"   Classes in YAML: {d.get('classes_in_yaml')} (Count: {len(d.get('classes_in_yaml', []))})")
        print(f"   Total Images: {d.get('total_images'):,} | Total Labels: {d.get('total_labels'):,} | Total Boxes: {d.get('total_boxes'):,}")
        print(f"   Overall Class Distribution: {d.get('class_distribution')}")
        print("   Splits Detail:")
        for split, sinfo in d.get("splits", {}).items():
            print(f"     * Split [{split}]: {sinfo['image_count']:,} images, {sinfo['label_count']:,} labels, {sinfo['box_count']:,} boxes")
            print(f"       Class dist: {sinfo.get('class_distribution')}")
            print(f"       Missing labels: {sinfo.get('images_without_label_file')}, Empty labels: {sinfo.get('empty_label_files')}, Corrupt lines: {sinfo.get('corrupt_lines')}, OOB boxes: {sinfo.get('out_of_bounds_boxes')}, Invalid class IDs: {sinfo.get('invalid_class_ids')}")
    elif dtype == "ocr_text":
        print(f"   Total Samples in GT: {d.get('total_samples'):,} | Total Image Files: {d.get('image_count'):,}")
        print(f"   Vocabulary Size: {len(d.get('char_set', []))} chars -> {''.join(d.get('char_set', []))}")
        print(f"   Has LMDB: {d.get('has_lmdb', False)}")
        if "empty_transcripts" in d and d["empty_transcripts"] > 0:
            print(f"   Empty transcripts: {d['empty_transcripts']}")
    elif dtype == "video":
        print(f"   Total Videos: {d.get('video_count')} ({d.get('total_video_bytes', 0)/(1024*1024):.2f} MB)")
        print(f"   XML Annotation Files: {d.get('xml_annotations')}")

    if d.get("issues"):
        print(f"   ⚠️ ISSUES DETECTED: {d.get('issues')}")
    else:
        print(f"   ✅ Quality Check: PASSED (No corruption, no invalid IDs, no out-of-bounds bboxes)")

print("\n" + "=" * 90)
