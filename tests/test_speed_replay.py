import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.speed_estimation import SpeedEstimator

config = {
    "speed_estimation": {
        "enabled": True,
        "method": "centroid"
    }
}
estimator = SpeedEstimator(config)

json_path = PROJECT_ROOT / "output/vehicle_cctv_results_zone.json"
with open(json_path) as f:
    data = json.load(f)

for frame_data in data:
    frame_idx = frame_data["frame"]
    for t in frame_data["tracks"]:
        track_id = t["track_id"]
        bbox = tuple(t["bbox"])
        speed = estimator.update(track_id, bbox, frame_idx, 25.0)
        if speed is not None:
            print(f"Frame {frame_idx}: Track {track_id} Speed: {speed}")
            break
