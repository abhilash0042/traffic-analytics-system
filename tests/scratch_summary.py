import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

json_path = PROJECT_ROOT / "output/vehicle_cctv_results.json"
with open(json_path) as f:
    data = json.load(f)

tracks = set()
plates = set()
helmet_frames = 0

for frame in data:
    for t in frame['tracks']:
        tracks.add(t['track_id'])
        if t['plate_text']:
            plates.add(t['plate_text'])
        if t['helmet_label']:
            helmet_frames += 1

print(f"Unique tracks: {len(tracks)}")
print(f"Unique plates read: {len(plates)}")
print(f"Total helmet detection frames: {helmet_frames}")
if plates:
    print(f"Plates found: {list(plates)[:5]}")
