import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

report_path = Path("hf_audit_report.json")
if not report_path.exists():
    print("hf_audit_report.json not found")
    exit()

with open(report_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 80)
print(f"{'HUGGING FACE DATASET REPO':<45} | {'TOTAL SIZE':<12} | {'FILES':<8} | {'STATUS'}")
print("=" * 80)

for repo_id, info in data.items():
    if "error" in info:
        print(f"{repo_id:<45} | ERROR: {info['error']}")
        continue
    total_bytes = info.get("total_bytes", 0)
    files = info.get("files", [])
    file_names = [f["path"] for f in files]
    
    size_str = f"{total_bytes / (1024*1024):.2f} MB" if total_bytes > 1024*1024 else f"{total_bytes / 1024:.2f} KB"
    
    # Analyze status
    has_zip = any(f.endswith(".zip") or f.endswith(".tar.gz") for f in file_names)
    has_images = any(f.endswith(".jpg") or f.endswith(".png") for f in file_names)
    has_mp4 = any(f.endswith(".mp4") for f in file_names)
    has_mdb = any(f.endswith(".mdb") for f in file_names)
    
    if total_bytes < 10 * 1024 and not (has_zip or has_images or has_mp4 or has_mdb):
        status = "⚠️ EMPTY (Metadata/Card Only ~2KB)"
    elif has_zip:
        status = "✅ ARCHIVE UPLOADED (.zip)"
    elif has_images or has_mp4 or has_mdb:
        status = f"⚡ RAW FILES ({len(files)} files)"
    else:
        status = "❓ UNKNOWN"

    print(f"{repo_id:<45} | {size_str:<12} | {len(files):<8} | {status}")

print("=" * 80)
