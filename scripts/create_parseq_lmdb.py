"""
Run create_lmdb_dataset.py for train, val, and test splits.
"""
import os
import subprocess
from pathlib import Path

BASE = Path("c:/projects/traffic-analytics-system")
PARSEQ_TOOLS = BASE / "parseq/tools/create_lmdb_dataset.py"

DATA_BASE = BASE / "data/datasets/parseq_dataset"
LMDB_BASE = BASE / "data/datasets/parseq_lmdb"

def process():
    for split in ["train", "val", "test"]:
        input_path = DATA_BASE / split
        gt_file = input_path / "gt.txt"
        output_path = LMDB_BASE / split
        
        if not input_path.exists() or not gt_file.exists():
            print(f"Skipping {split} (input or gt.txt missing)")
            continue
            
        print(f"\nCreating LMDB for {split}...")
        cmd = [
            "python",
            str(PARSEQ_TOOLS),
            str(input_path),
            str(gt_file),
            str(output_path),
            "--checkValid=True"
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Finished {split} LMDB")

if __name__ == "__main__":
    process()
