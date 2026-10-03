import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, ".")
from scripts.publish_to_hf import DATASET_CATALOG

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("=" * 75)
print(f"{'DATASET KEY':<18} | {'FOLDER NAME':<25} | {'SIZE (MB)':<10} | {'EST UPLOAD @ 500kB/s'}")
print("=" * 75)

for k, v in DATASET_CATALOG.items():
    p = v["source_path"]
    if p.exists():
        size_bytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        size_mb = size_bytes / (1024 * 1024)
        est_sec = (size_mb * 1024) / 500  # at 500 kB/s
        if est_sec < 60:
            est_str = f"{est_sec:.0f} sec"
        elif est_sec < 3600:
            est_str = f"{est_sec/60:.1f} min"
        else:
            est_str = f"{est_sec/3600:.1f} hrs"
        print(f"{k:<18} | {p.name:<25} | {size_mb:>8.1f} MB | {est_str}")
    else:
        print(f"{k:<18} | {p.name:<25} | {'NOT FOUND':>8}")

print("=" * 75)
