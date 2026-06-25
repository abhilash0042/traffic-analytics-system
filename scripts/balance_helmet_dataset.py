"""
Balance helmet detection train split by downsampling majority (helmet) class.

Strategy:
  - Keep every image that contains no_helmet (class 1)
  - Subsample helmet-only images until helmet/no_helmet instance ratio is met
  - Prefer dropping augmented copies (_ind_*) before originals

Usage:
    python scripts/balance_helmet_dataset.py
    python scripts/balance_helmet_dataset.py --target-ratio 1.1
    python scripts/balance_helmet_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_utils import load_config, resolve_path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUGMENT_MARKERS = ("_ind_blur", "_ind_motion", "_ind_night", "_ind_rain", "_ind_jpeg")


@dataclass
class LabelStats:
    label_path: Path
    image_path: Path | None
    helmet: int
    no_helmet: int

    @property
    def is_augmented(self) -> bool:
        return any(marker in self.label_path.stem for marker in AUGMENT_MARKERS)


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def info(msg: str) -> None:
    print(f"  [>>] {msg}")


def build_image_index(img_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    with os.scandir(img_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            suffix = Path(entry.name).suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                index[Path(entry.name).stem] = Path(entry.path)
    return index


def list_label_paths(label_dir: Path) -> list[Path]:
    paths: list[Path] = []
    with os.scandir(label_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".txt"):
                paths.append(Path(entry.path))
    return paths


def _read_one_label(label_path: Path, image_index: dict[str, Path]) -> LabelStats:
    helmet = no_helmet = 0
    with label_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            cls = line[0]
            if cls == "0":
                helmet += 1
            elif cls == "1":
                no_helmet += 1
    return LabelStats(
        label_path=label_path,
        image_path=image_index.get(label_path.stem),
        helmet=helmet,
        no_helmet=no_helmet,
    )


def _cache_path(label_dir: Path) -> Path:
    return label_dir / ".balance_stats.json"


def _load_cached_stats(label_dir: Path, image_index: dict[str, Path]) -> list[LabelStats] | None:
    cache_file = _cache_path(label_dir)
    if not cache_file.is_file():
        return None
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if payload.get("count") != len(image_index):
        return None
    rows: list[LabelStats] = []
    for item in payload["rows"]:
        label_path = label_dir / f"{item['stem']}.txt"
        if not label_path.is_file():
            return None
        rows.append(
            LabelStats(
                label_path=label_path,
                image_path=image_index.get(item["stem"]),
                helmet=int(item["helmet"]),
                no_helmet=int(item["no_helmet"]),
            )
        )
    return rows


def _save_cached_stats(label_dir: Path, rows: list[LabelStats]) -> None:
    payload = {
        "count": len(rows),
        "rows": [
            {"stem": row.label_path.stem, "helmet": row.helmet, "no_helmet": row.no_helmet}
            for row in rows
        ],
    }
    _cache_path(label_dir).write_text(json.dumps(payload), encoding="utf-8")


def read_label_stats(label_dir: Path, image_index: dict[str, Path]) -> list[LabelStats]:
    cached = _load_cached_stats(label_dir, image_index)
    if cached is not None:
        print(f"  loaded cached stats for {len(cached)} labels", flush=True)
        return cached

    label_paths = list_label_paths(label_dir)
    if not label_paths:
        return []
    workers = min(32, max(4, (len(label_paths) // 4000) + 4))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda path: _read_one_label(path, image_index), label_paths))
    _save_cached_stats(label_dir, rows)
    return rows


def summarize_rows(rows: list[LabelStats]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for row in rows:
        counts[0] += row.helmet
        counts[1] += row.no_helmet
    return counts


def balance_train(
    dataset_dir: Path,
    target_ratio: float,
    dry_run: bool,
    seed: int,
    stats: list[LabelStats] | None = None,
) -> tuple[int, int, Counter[int]]:
    random.seed(seed)

    img_dir = dataset_dir / "train" / "images"
    lbl_dir = dataset_dir / "train" / "labels"
    pruned_img = dataset_dir / "train" / "_pruned" / "images"
    pruned_lbl = dataset_dir / "train" / "_pruned" / "labels"

    if stats is None:
        stats = read_label_stats(lbl_dir, build_image_index(img_dir))
    if not stats:
        print("No train labels found.")
        return 0, 0, Counter()

    with_no_helmet = [row for row in stats if row.no_helmet > 0]
    helmet_only = [row for row in stats if row.no_helmet == 0 and row.helmet > 0]
    empty = [row for row in stats if row.helmet == 0 and row.no_helmet == 0]

    no_helmet_instances = sum(row.no_helmet for row in with_no_helmet)
    helmet_in_mixed = sum(row.helmet for row in with_no_helmet)
    target_helmet = int(no_helmet_instances * target_ratio)
    need_from_only = max(0, target_helmet - helmet_in_mixed)

    info(f"no_helmet instances (kept): {no_helmet_instances}")
    info(f"helmet in mixed images (kept): {helmet_in_mixed}")
    info(f"target helmet instances (ratio {target_ratio}): {target_helmet}")
    info(f"helmet-only budget: {need_from_only} from {len(helmet_only)} files")

    # Drop augmented helmet-only images first, then random originals.
    helmet_only.sort(key=lambda row: (row.is_augmented, random.random()))

    keep_only: list[LabelStats] = []
    acc = 0
    for row in helmet_only:
        if acc >= need_from_only:
            break
        keep_only.append(row)
        acc += row.helmet

    keep_rows = with_no_helmet + keep_only
    keep_set = {row.label_path.resolve() for row in keep_rows}
    remove_rows = [row for row in stats if row.label_path.resolve() not in keep_set]
    after_counts = summarize_rows(keep_rows)

    info(f"Keeping {len(keep_set)} label files; pruning {len(remove_rows)} (+ {len(empty)} empty)")

    if dry_run:
        print(
            f"\n  Dry run — would result in helmet={after_counts.get(0, 0)}, "
            f"no_helmet={after_counts.get(1, 0)}, "
            f"ratio={after_counts.get(0, 0) / max(after_counts.get(1, 0), 1):.2f}:1"
        )
        return len(remove_rows), len(empty), after_counts

    pruned_img.mkdir(parents=True, exist_ok=True)
    pruned_lbl.mkdir(parents=True, exist_ok=True)

    moved = 0
    total = len(remove_rows) + len(empty)
    for row in remove_rows + empty:
        if row.image_path and row.image_path.is_file():
            shutil.move(str(row.image_path), str(pruned_img / row.image_path.name))
        if row.label_path.is_file():
            shutil.move(str(row.label_path), str(pruned_lbl / row.label_path.name))
        moved += 1
        if moved % 2000 == 0 or moved == total:
            print(f"  [>>] Pruned {moved}/{total}...", flush=True)

    ok(f"Moved {moved} image/label pairs to train/_pruned/")
    return len(remove_rows), len(empty), after_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance helmet train split (downsample majority class)")
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=1.1,
        help="Target helmet/no_helmet instance ratio (default 1.1 — slight helmet majority)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help="Write completion summary to this file",
    )
    args = parser.parse_args()

    config = load_config()
    dataset_dir = resolve_path(config["training"]["helmet"]["dataset_dir"])

    print("\n" + "=" * 60)
    print("BALANCE HELMET TRAIN SPLIT")
    print("=" * 60)

    lbl_dir = dataset_dir / "train" / "labels"
    img_dir = dataset_dir / "train" / "images"
    print("Scanning labels...", flush=True)
    image_index = build_image_index(img_dir)
    all_stats = read_label_stats(lbl_dir, image_index)
    before = summarize_rows(all_stats)
    print(f"\nBefore: helmet={before.get(0, 0)}, no_helmet={before.get(1, 0)}, "
          f"ratio={before.get(0, 0) / max(before.get(1, 0), 1):.2f}:1")

    _, _, after = balance_train(
        dataset_dir, args.target_ratio, args.dry_run, args.seed, stats=all_stats
    )

    if not args.dry_run:
        train_images = sum(1 for row in all_stats if row.label_path.is_file())
        print(f"\nAfter: helmet={after.get(0, 0)}, no_helmet={after.get(1, 0)}, "
              f"ratio={after.get(0, 0) / max(after.get(1, 0), 1):.2f}:1")
        print(f"Train images: {train_images}")
        print("\nRetrain on balanced data:")
        print("  python train_models.py --helmets --continue-best")

        if args.status_file:
            args.status_file.parent.mkdir(parents=True, exist_ok=True)
            args.status_file.write_text(
                f"helmet={after.get(0, 0)}\nno_helmet={after.get(1, 0)}\n"
                f"ratio={after.get(0, 0) / max(after.get(1, 0), 1):.4f}\n"
                f"train_images={train_images}\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
