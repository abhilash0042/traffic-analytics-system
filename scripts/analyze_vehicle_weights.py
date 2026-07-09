"""Compare downloaded vehicle_detector checkpoints and evaluate on UA-DETRAC test split."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_YAML = PROJECT_ROOT / "data/datasets/vehicle_detection/data.yaml"

CANDIDATES = [
    PROJECT_ROOT / "models/vehicle_detector.pt",
]


def checkpoint_info(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ta = ckpt.get("train_args") or {}
    return {
        "path": path,
        "size_mb": path.stat().st_size / 1e6,
        "mtime": path.stat().st_mtime,
        "epoch": ckpt.get("epoch"),
        "best_fitness": ckpt.get("best_fitness", ckpt.get("fitness")),
        "date": ckpt.get("date"),
        "epochs_planned": ta.get("epochs") if isinstance(ta, dict) else None,
        "imgsz": ta.get("imgsz") if isinstance(ta, dict) else None,
        "batch": ta.get("batch") if isinstance(ta, dict) else None,
    }


def main() -> None:
    existing = [p for p in CANDIDATES if p.is_file()]
    if not existing:
        print("No vehicle_detector*.pt files found.")
        sys.exit(1)

    print("=" * 72)
    print("DOWNLOADED CHECKPOINTS (newest first)")
    print("=" * 72)
    rows = sorted((checkpoint_info(p) for p in existing), key=lambda r: r["mtime"], reverse=True)
    for r in rows:
        p = r["path"]
        rel = p.relative_to(PROJECT_ROOT)
        print(f"\n{rel}")
        print(f"  size: {r['size_mb']:.1f} MB")
        print(f"  trained epoch: {r['epoch']} / {r['epochs_planned']}")
        print(f"  best_fitness: {r['best_fitness']}")
        print(f"  checkpoint date: {r['date']}")

    latest = rows[0]["path"]
    print("\n" + "=" * 72)
    print(f"FULL TEST EVAL — {latest.relative_to(PROJECT_ROOT)}")
    print("=" * 72)

    model = YOLO(str(latest))
    print(f"Classes: {model.names}")
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        device=0,
        batch=8,
        workers=0,
        verbose=False,
    )
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print("\nPer-class mAP50:")
    for i, name in model.names.items():
        ap = metrics.box.ap50[i] if metrics.box.ap50 is not None else float("nan")
        print(f"  {name:6s}  {ap:.4f}")

    # Quick sanity: compare mAP50 on a small batch vs installed models copy
    installed = PROJECT_ROOT / "models/vehicle_detector.pt"
    if installed.is_file() and installed.resolve() != latest.resolve():
        print("\n" + "-" * 72)
        print(f"Compare installed copy: {installed.relative_to(PROJECT_ROOT)}")
        m2 = YOLO(str(installed))
        m2_metrics = m2.val(data=str(DATA_YAML), split="test", device=0, batch=16, workers=0, verbose=False)
        print(f"  mAP50: {m2_metrics.box.map50:.4f}  (latest download: {metrics.box.map50:.4f})")
        delta = metrics.box.map50 - m2_metrics.box.map50
        print(f"  delta: {delta:+.4f}")


if __name__ == "__main__":
    main()
