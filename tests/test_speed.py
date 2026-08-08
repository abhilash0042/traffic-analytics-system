import sys
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
print("Enabled:", estimator.enabled, "Method:", estimator.method)

track_id = 1
for i in range(1, 10):
    bbox = (100, 100+i*10, 200, 200+i*10)
    speed = estimator.update(track_id, bbox, i, 25.0)
    print(f"Frame {i}: Speed = {speed}")
