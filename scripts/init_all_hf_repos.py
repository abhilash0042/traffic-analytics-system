"""Initialize all 10 dataset repositories on Hugging Face with complete cards and LFS configs."""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from huggingface_hub import HfApi, create_repo
from scripts.publish_to_hf import DATASET_CATALOG, generate_dataset_card, GITATTRIBUTES_CONTENT, get_hf_token

def main():
    token = get_hf_token()
    if not token:
        print("[ERROR] HF_TOKEN not found in environment or .env")
        sys.exit(1)

    api = HfApi(token=token)
    namespace = "thundarstrom"

    print("=" * 65)
    print("🚀 INITIALIZING ALL 10 HUGGING FACE DATASET REPOSITORIES")
    print("=" * 65)

    for key, meta in DATASET_CATALOG.items():
        repo_name = meta["repo_name"]
        repo_id = f"{namespace}/{repo_name}"
        print(f"\n📦 Processing {repo_id}...")

        # 1. Create repo
        create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True, token=token)

        # 2. Upload README.md
        card = generate_dataset_card(key, namespace, meta)
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Initialize Dataset Card (README.md)",
        )

        # 3. Upload .gitattributes
        api.upload_file(
            path_or_fileobj=GITATTRIBUTES_CONTENT.encode("utf-8"),
            path_in_repo=".gitattributes",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Configure Git LFS attributes",
        )

        # 4. Upload data.yaml if exists
        yaml_p = meta["source_path"] / "data.yaml"
        if yaml_p.exists():
            api.upload_file(
                path_or_fileobj=str(yaml_p),
                path_in_repo="data.yaml",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                commit_message="Add YOLO data.yaml configuration",
            )

        print(f"   [OK] {repo_id} initialized!")

    print("\n" + "=" * 65)
    print("ALL 10 REPOSITORIES LIVE ON HUGGING FACE!")
    print(f"Profile: https://huggingface.co/{namespace}")
    print("=" * 65)

if __name__ == "__main__":
    main()
