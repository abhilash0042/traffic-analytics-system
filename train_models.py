"""
Fine-tune YOLO11 detectors for the Traffic Analytics System.

Features:
  - Transfer learning from COCO pre-trained weights (never from scratch)
  - Pause anytime (Ctrl+C) and resume with --resume
  - Continue fine-tuning from best weights with --continue-best (RTX 5060 path)
  - Auto-sync best.pt -> models/ for ai_pipeline.py

Usage:
    python train_models.py --status
    python train_models.py --plates
    python train_models.py --helmets
    python train_models.py --vehicles
    python train_models.py --all
    python train_models.py --plates --resume
    python train_models.py --vehicles --continue-best

Full guide: docs/TRAINING_PLAN.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from model_utils import dataset_ready, find_data_yaml, load_config, resolve_path
from training_utils import (
    TASK_CLI,
    TASK_KEYS,
    build_train_kwargs,
    checkpoint_exists,
    format_cuda_status,
    format_progress_line,
    get_run_paths,
    read_training_progress,
    resolve_weights_path,
    resume_command,
    sync_finetuned_weights,
    verify_cuda_for_training,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def warn(msg: str) -> None:
    print(f"  [!!] {msg}")


def missing(msg: str) -> None:
    print(f"  [--] {msg}")


def info(msg: str) -> None:
    print(f"  [>>] {msg}")


def print_training_status(config: dict) -> None:
    print("\n" + "=" * 60)
    print("TRAINING DATA & MODEL STATUS")
    print("=" * 60)
    for line in format_cuda_status(config):
        print(f"  {line}")
    print(f"  Guide  : docs/TRAINING_PLAN.md")
    print("-" * 60)

    for key in TASK_KEYS:
        train_cfg = config["training"][key]
        dataset_dir = resolve_path(train_cfg["dataset_dir"])
        data_yaml = find_data_yaml(dataset_dir)
        finetuned = resolve_path(config["models"][key]["finetuned"])
        run_paths = get_run_paths(config, key)
        progress = read_training_progress(run_paths)

        label = key.replace("_", " ").title()
        print(f"\n  {label} detector")
        print(f"    Base weights : {train_cfg['base_weights']}")
        print(f"    Epochs/batch : {train_cfg['epochs']} / {train_cfg['batch']}")

        if finetuned.is_file():
            size_mb = finetuned.stat().st_size / (1024 * 1024)
            ok(f"Pipeline model : {finetuned.name} ({size_mb:.1f} MB)")
        else:
            missing(f"Pipeline model : not trained yet -> {finetuned.name}")

        if data_yaml:
            ok(f"Dataset        : {data_yaml}")
        else:
            missing(f"Dataset        : no data.yaml in {dataset_dir}")

        info(f"Run progress   : {format_progress_line(progress)}")
        if progress["has_checkpoint"]:
            info(f"Resume with    : {resume_command(key)}")

    print("\nRecommended order (RTX 4050):")
    print("  1. python download_datasets.py --plates --helmets --vehicles")
    print("  2. python train_models.py --plates")
    print("  3. python train_models.py --helmets")
    print("  4. python train_models.py --vehicles")
    print("  5. python ai_pipeline.py")
    print("\nPause anytime: Ctrl+C, then resume with --resume")


def _resolve_mode(
    config: dict,
    task_key: str,
    *,
    resume: bool,
    continue_best: bool,
    force_fresh: bool,
) -> str:
    if resume and continue_best:
        raise ValueError("Use either --resume or --continue-best, not both.")

    if resume:
        return "resume"

    if continue_best:
        return "continue"

    if force_fresh:
        return "fresh"

    if checkpoint_exists(config, task_key):
        return "needs_choice"

    return "fresh"


def train_one(
    task_key: str,
    config: dict,
    *,
    mode: str = "fresh",
    dry_run: bool = False,
) -> Path | None:
    train_cfg = config["training"][task_key]
    model_cfg = config["models"][task_key]

    if not train_cfg.get("enabled", True):
        warn(f"Skipping disabled task: {task_key}")
        return None

    dataset_dir = resolve_path(train_cfg["dataset_dir"])
    data_yaml = find_data_yaml(dataset_dir)
    if data_yaml is None:
        warn(f"No dataset for '{task_key}' at {dataset_dir}")
        return None

    target_path = resolve_path(model_cfg["finetuned"])
    run_paths = get_run_paths(config, task_key)
    progress = read_training_progress(run_paths)

    try:
        weights_path, mode_label = resolve_weights_path(config, task_key, mode=mode)
    except FileNotFoundError as exc:
        warn(str(exc))
        return None

    print("\n" + "=" * 60)
    print(f"TRAINING: {task_key.upper()} DETECTOR")
    print("=" * 60)
    print(f"  Mode         : {mode_label}")
    print(f"  Weights      : {weights_path}")
    print(f"  Save as      : {target_path.name}")
    print(f"  Dataset      : {data_yaml}")
    print(f"  Epochs       : {train_cfg['epochs']}")
    print(f"  Batch/imgsz  : {train_cfg['batch']} / {train_cfg['imgsz']}")
    print(f"  Run folder   : {run_paths.run_dir}")
    if progress["has_checkpoint"] and mode != "fresh":
        print(f"  Progress     : {format_progress_line(progress)}")

    if dry_run:
        warn("Dry run only — not training")
        return None

    cuda_ok, cuda_error = verify_cuda_for_training(config)
    if not cuda_ok:
        warn(cuda_error or "CUDA not ready for training.")
        return None

    from ultralytics import YOLO

    train_kwargs = build_train_kwargs(config, task_key, resume=(mode == "resume"))

    print("\n  Pause anytime with Ctrl+C. Resume with:")
    print(f"    {resume_command(task_key)}")
    print("-" * 60)

    interrupted = False
    results = None
    try:
        model = YOLO(str(weights_path))
        results = model.train(**train_kwargs)
    except KeyboardInterrupt:
        warn("Training paused by user (Ctrl+C).")
        if run_paths.last_pt.is_file():
            ok(f"Checkpoint saved -> {run_paths.last_pt}")
            info(f"Resume with: {resume_command(task_key)}")
            synced = sync_finetuned_weights(config, task_key)
            if synced:
                ok(f"Best weights synced -> {synced}")
                return synced
        else:
            warn("No last.pt found yet — restart this task without --resume.")
        return None

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if not best_path.is_file():
        best_path = run_paths.best_pt

    if not best_path.is_file():
        warn(f"Training finished but best.pt not found at {best_path}")
        return None

    synced = sync_finetuned_weights(config, task_key, best_path)
    if synced:
        ok(f"Saved fine-tuned weights -> {synced}")

    final_progress = read_training_progress(get_run_paths(config, task_key))
    if final_progress.get("best_map50") is not None:
        ok(f"Best validation mAP50: {final_progress['best_map50']:.4f}")

    return synced


def _tasks_from_args(args: argparse.Namespace) -> list[str]:
    tasks: list[str] = []
    if args.all or args.vehicles:
        tasks.append("vehicle")
    if args.all or args.plates:
        tasks.append("plate")
    if args.all or args.helmets:
        tasks.append("helmet")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO11 models for traffic analytics (pause/resume supported)"
    )
    parser.add_argument("--all", action="store_true", help="Train vehicle, plate, and helmet models")
    parser.add_argument("--vehicles", action="store_true", help="Train vehicle detector")
    parser.add_argument("--plates", action="store_true", help="Train license plate detector")
    parser.add_argument("--helmets", action="store_true", help="Train helmet detector")
    parser.add_argument("--status", action="store_true", help="Show dataset, checkpoint, and model status")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume paused training from last.pt (same run, same settings)",
    )
    parser.add_argument(
        "--continue-best",
        action="store_true",
        help="Fine-tune further from models/*.pt (new run — use after RTX 5060 upgrade)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Start fresh even if a checkpoint exists (ignores last.pt)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config/datasets without training")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "pipeline_config.yaml"),
        help="Path to pipeline config YAML",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    if args.status or not _tasks_from_args(args):
        print_training_status(config)
        if not _tasks_from_args(args):
            return

    tasks = _tasks_from_args(args)
    if args.all and not args.resume and not args.continue_best and not args.force:
        # Recommended order for --all fresh runs
        tasks = [t for t in ("plate", "helmet", "vehicle") if t in tasks]

    trained: list[Path] = []
    paused: list[Path] = []
    skipped = 0

    for task in tasks:
        if not dataset_ready(config["training"][task]["dataset_dir"]):
            warn(f"Dataset not ready for '{task}'. Download first, then re-run.")
            skipped += 1
            continue

        mode = _resolve_mode(
            config,
            task,
            resume=args.resume,
            continue_best=args.continue_best,
            force_fresh=args.force,
        )

        if mode == "needs_choice":
            run_paths = get_run_paths(config, task)
            progress = read_training_progress(run_paths)
            warn(
                f"Paused run found for '{task}' ({format_progress_line(progress)}). "
                f"Use --resume to continue or --force to restart."
            )
            info(f"  {resume_command(task)}")
            skipped += 1
            continue

        result = train_one(task, config, mode=mode, dry_run=args.dry_run)
        if result:
            progress = read_training_progress(get_run_paths(config, task))
            current = progress.get("current_epoch")
            total = progress.get("total_epochs")
            if current is not None and total and current < total:
                paused.append(result)
            else:
                trained.append(result)

    if args.dry_run:
        ok("Dry run complete — config and datasets look usable.")
        return

    print("\n" + "=" * 60)
    if trained:
        ok(f"Completed {len(trained)} model(s). Run: python ai_pipeline.py")
    if paused:
        info(f"Paused {len(paused)} run(s) — best weights saved to models/")
        info("Resume with: python train_models.py --plates --resume")
    if skipped:
        warn(f"Skipped {skipped} task(s). Run: python train_models.py --status")
    if not trained and not paused and not skipped:
        warn("No models trained.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
