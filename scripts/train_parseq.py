"""
Script to fine-tune PARSeq on our unified license plate dataset.
"""
import subprocess
from pathlib import Path
import os

BASE = Path("c:/projects/traffic-analytics-system")
PARSEQ_DIR = BASE / "parseq"
DATA_DIR = BASE / "data/datasets/parseq_lmdb"

def main():
    print("="*60)
    print("Starting PARSeq Fine-Tuning")
    print("="*60)
    
    # We will use parseq-tiny as it's lightweight but highly accurate
    cmd = [
        "python", str(PARSEQ_DIR / "train.py"),
        "+experiment=parseq-tiny",
        'model.charset_train="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"',
        f"data.root_dir={DATA_DIR.absolute()}",
        "trainer.max_epochs=30",
        "trainer.accelerator=gpu",
        "data.batch_size=128",
        # Use pretrained weights from PARSeq
        "model.pretrained=True"
    ]
    
    # Run the training
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(PARSEQ_DIR))

if __name__ == "__main__":
    main()
