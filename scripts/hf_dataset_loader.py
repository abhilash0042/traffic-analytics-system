"""Download public datasets from Hugging Face (no API key required)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "datasets"


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def warn(msg: str) -> None:
    print(f"  [!!] {msg}")


def folder_stats(path: Path) -> tuple[int, float]:
    if not path.is_dir():
        return 0, 0.0
    count = 0
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            count += 1
            total += file.stat().st_size
    return count, total / (1024 * 1024)


def write_data_yaml(
    root: Path,
    class_names: list[str],
    train_rel: str = "images/train",
    val_rel: str = "images/val",
    test_rel: str | None = "images/test",
) -> Path:
    data = {
        "path": str(root.resolve()),
        "train": train_rel,
        "val": val_rel,
        "nc": len(class_names),
        "names": class_names,
    }
    if test_rel:
        data["test"] = test_rel
    yaml_path = root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.dump(data, handle, default_flow_style=False, sort_keys=False)
    return yaml_path


def coco_bbox_to_yolo(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    return x_center, y_center, w / img_w, h / img_h


def export_hf_coco_detection(
    repo_id: str,
    config_name: str | None,
    save_dir: Path,
    class_names: list[str],
    split_map: dict[str, str] | None = None,
) -> bool:
    """Export a HuggingFace COCO-style detection dataset to YOLO layout."""
    try:
        from datasets import load_dataset
    except ImportError:
        warn("Install datasets: pip install datasets")
        return False

    split_map = split_map or {"train": "train", "valid": "val", "validation": "val", "test": "test"}
    save_dir.mkdir(parents=True, exist_ok=True)

    if folder_stats(save_dir)[0] > 50:
        ok(f"Already exported: {save_dir}")
        return True

    print(f"\nFetching from Hugging Face: {repo_id}")
    kwargs = {"path": repo_id}
    if config_name:
        kwargs["name"] = config_name
    dataset = load_dataset(**kwargs)

    for split_name, split_data in dataset.items():
        target_split = split_map.get(split_name, split_name)
        images_dir = save_dir / "images" / target_split
        labels_dir = save_dir / "labels" / target_split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Exporting split '{split_name}' -> {target_split} ({len(split_data)} images)")
        for index, sample in enumerate(split_data):
            if index > 0 and index % 500 == 0:
                print(f"    ... {index}/{len(split_data)} images")
            image = sample["image"]
            if hasattr(image, "convert"):
                image = image.convert("RGB")
            img_w, img_h = image.size
            image_path = images_dir / f"{target_split}_{index:06d}.jpg"
            image.save(image_path, quality=95)

            label_lines: list[str] = []
            objects = sample.get("objects") or {}
            bboxes = objects.get("bbox") or []
            categories = objects.get("category") or objects.get("label") or []

            for bbox, category in zip(bboxes, categories):
                if isinstance(category, str):
                    if category not in class_names:
                        continue
                    class_id = class_names.index(category)
                else:
                    class_id = int(category)
                    if class_id >= len(class_names):
                        continue
                x, y, w, h = coco_bbox_to_yolo(bbox, img_w, img_h)
                if w <= 0 or h <= 0:
                    continue
                label_lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            label_path = labels_dir / f"{target_split}_{index:06d}.txt"
            label_path.write_text("\n".join(label_lines), encoding="utf-8")

    yaml_path = write_data_yaml(save_dir, class_names)
    ok(f"Saved YOLO dataset -> {save_dir} ({yaml_path.name})")
    return True


def download_hf_snapshot(repo_id: str, save_dir: Path, min_files: int = 20) -> bool:
    """Download a dataset repo that already ships YOLO folders."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        warn("Install huggingface_hub: pip install huggingface_hub")
        return False

    if folder_stats(save_dir)[0] >= min_files:
        ok(f"Already downloaded: {save_dir}")
        return True

    print(f"\nDownloading Hugging Face snapshot: {repo_id}")
    cache_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
    cache_path = Path(cache_dir)

    if (cache_path / "data.yaml").is_file():
        if save_dir.exists():
            shutil.rmtree(save_dir)
        shutil.copytree(cache_path, save_dir)
        ok(f"Copied YOLO dataset -> {save_dir}")
        return True

    # Some repos nest the YOLO tree one level down
    for candidate in cache_path.rglob("data.yaml"):
        root = candidate.parent
        if folder_stats(root)[0] >= min_files:
            if save_dir.exists():
                shutil.rmtree(save_dir)
            shutil.copytree(root, save_dir)
            ok(f"Copied YOLO dataset -> {save_dir}")
            return True

    warn(f"Could not find YOLO layout in {repo_id}")
    return False


def download_license_plates_hf() -> bool:
    save_dir = DATA_DIR / "license_plates"
    return export_hf_coco_detection(
        repo_id="justjuu/license-plate-detection",
        config_name=None,
        save_dir=save_dir,
        class_names=["license_plate"],
    )


def download_helmet_hf() -> bool:
    save_dir = DATA_DIR / "helmet_detection"

    # Ready-made YOLO snapshot: helmet (0), head/no-helmet (1), vest (2)
    if download_hf_snapshot("jhboyo/ppe-dataset", save_dir, min_files=100):
        return True

    warn("Helmet dataset needs Roboflow API key or retry later (HF rate limits)")
    print("  Run: set ROBOFLOW_API_KEY=your_key && python download_datasets.py --helmets")
    return False


def download_vehicle_hf() -> bool:
    """Optional vehicle dataset — requires Roboflow UA-DETRAC 10K (too large for HF mirror)."""
    warn("Vehicle fine-tuning dataset is best from Roboflow UA-DETRAC 10K (~800 MB)")
    warn("Set ROBOFLOW_API_KEY and run: python download_datasets.py --vehicles")
    return False
