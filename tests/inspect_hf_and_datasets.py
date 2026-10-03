import os
import sys
import json
from pathlib import Path

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from huggingface_hub import HfApi

def get_hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN=") or line.startswith("HUGGING_FACE_HUB_TOKEN="):
                val = line.split("=", 1)[1].strip().strip("\"'")
                if val:
                    return val
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        t = token_path.read_text(encoding="utf-8").strip()
        if t:
            return t
    return None

token = get_hf_token()
api = HfApi(token=token)
user = "thundarstrom"

print("=================================================================")
print(f"FETCHING ALL DATASETS UNDER USER/ORG: {user}")
print("=================================================================")

datasets = list(api.list_datasets(author=user))
print(f"Total datasets found: {len(datasets)}\n")

hf_summary = {}

for ds in datasets:
    repo_id = ds.id
    print(f"📂 Dataset: {repo_id}")
    print(f"   Private: {ds.private}")
    print(f"   Last Modified: {ds.last_modified}")
    
    try:
        tree = list(api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True))
        files_info = []
        total_bytes = 0
        for item in tree:
            item_type = getattr(item, "type", "file")
            size = getattr(item, "size", 0) or 0
            total_bytes += size
            files_info.append({
                "path": item.path,
                "size": size,
                "type": item_type
            })
            print(f"     - {item.path} ({size:,} bytes, {size/1024:.2f} KB)")
        
        print(f"   👉 Total Size: {total_bytes:,} bytes ({total_bytes/(1024*1024):.2f} MB)")
        hf_summary[repo_id] = {
            "files": files_info,
            "total_bytes": total_bytes,
            "has_data_archive": any(f["path"].endswith(".zip") or f["path"].endswith(".tar.gz") or f["path"].endswith(".mp4") for f in files_info)
        }
    except Exception as e:
        print(f"   ❌ Error listing files: {e}")
        hf_summary[repo_id] = {"error": str(e)}
    print()

output_path = Path("hf_audit_report.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(hf_summary, f, indent=2)

print(f"Saved audit summary to {output_path}")
