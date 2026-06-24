"""Training helpers: checkpoints, resume, progress, and weight sync."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_utils import load_config, resolve_path

TASK_KEYS = ("plate", "helmet", "vehicle")
TASK_CLI = {"plate": "plates", "helmet": "helmets", "vehicle": "vehicles"}


@dataclass(frozen=True)
class RunPaths:
    task_key: str
    run_dir: Path
    weights_dir: Path
    last_pt: Path
    best_pt: Path
    results_csv: Path
    args_yaml: Path


def get_run_paths(config: dict[str, Any], task_key: str) -> RunPaths:
    train_cfg = config["training"][task_key]
    output_root = resolve_path(config["training"].get("output_dir", "runs/detect"))
    run_dir = output_root / train_cfg["project"] / train_cfg["name"]
    weights_dir = run_dir / "weights"
    return RunPaths(
        task_key=task_key,
        run_dir=run_dir,
        weights_dir=weights_dir,
        last_pt=weights_dir / "last.pt",
        best_pt=weights_dir / "best.pt",
        results_csv=run_dir / "results.csv",
        args_yaml=run_dir / "args.yaml",
    )


def checkpoint_exists(config: dict[str, Any], task_key: str) -> bool:
    return get_run_paths(config, task_key).last_pt.is_file()


def read_training_progress(run_paths: RunPaths) -> dict[str, Any]:
    """Parse Ultralytics results.csv for epoch count and best mAP50."""
    info: dict[str, Any] = {
        "has_checkpoint": run_paths.last_pt.is_file(),
        "has_best": run_paths.best_pt.is_file(),
        "current_epoch": None,
        "total_epochs": None,
        "best_map50": None,
        "last_map50": None,
    }

    if run_paths.args_yaml.is_file():
        try:
            import yaml

            args = yaml.safe_load(run_paths.args_yaml.read_text(encoding="utf-8"))
            if isinstance(args, dict):
                info["total_epochs"] = args.get("epochs")
        except Exception:
            pass

    if not run_paths.results_csv.is_file():
        return info

    try:
        with open(run_paths.results_csv, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return info

    if not rows:
        return info

    map_col = _find_map_column(rows[0].keys())
    last_row = rows[-1]
    info["current_epoch"] = _safe_int(last_row.get("epoch"))
    if map_col:
        maps = [_safe_float(row.get(map_col)) for row in rows]
        valid_maps = [m for m in maps if m is not None]
        if valid_maps:
            info["best_map50"] = max(valid_maps)
            info["last_map50"] = valid_maps[-1]

    return info


def _find_map_column(columns: Any) -> str | None:
    for name in columns:
        lower = name.lower()
        if "map50" in lower and "map50-95" not in lower:
            return name
    return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_train_kwargs(
    config: dict[str, Any],
    task_key: str,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    train_cfg = config["training"][task_key]
    defaults = config["training"].get("defaults", {})

    from model_utils import find_data_yaml

    data_yaml = find_data_yaml(train_cfg["dataset_dir"])
    if data_yaml is None:
        raise FileNotFoundError(f"No data.yaml for task '{task_key}'")

    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": int(train_cfg["epochs"]),
        "imgsz": int(train_cfg["imgsz"]),
        "batch": int(train_cfg["batch"]),
        "patience": int(train_cfg["patience"]),
        "close_mosaic": int(train_cfg["close_mosaic"]),
        "project": train_cfg["project"],
        "name": train_cfg["name"],
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        "resume": resume,
    }

    optional_keys = (
        "amp",
        "cache",
        "workers",
        "save_period",
        "seed",
        "cos_lr",
        "optimizer",
        "lr0",
        "lrf",
        "weight_decay",
        "warmup_epochs",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "fliplr",
        "mosaic",
        "mixup",
        "copy_paste",
        "deterministic",
        "fraction",
        "val",
        "plots",
    )

    for key in optional_keys:
        if key in train_cfg:
            kwargs[key] = train_cfg[key]
        elif key in defaults:
            kwargs[key] = defaults[key]

    device = config.get("training", {}).get("device")
    if device is not None:
        kwargs["device"] = device

    return kwargs


def resolve_weights_path(
    config: dict[str, Any],
    task_key: str,
    *,
    mode: str,
) -> tuple[Path | str, str]:
    """
    Return (weights_path, mode_label) for YOLO() constructor.

    mode:
      fresh   -> COCO base_weights from config
      resume  -> last.pt checkpoint
      continue -> models/<finetuned>.pt if exists, else base_weights
    """
    train_cfg = config["training"][task_key]
    model_cfg = config["models"][task_key]
    run_paths = get_run_paths(config, task_key)

    if mode == "resume":
        if not run_paths.last_pt.is_file():
            raise FileNotFoundError(
                f"No checkpoint to resume for '{task_key}'. Expected: {run_paths.last_pt}"
            )
        return run_paths.last_pt, "resume"

    if mode == "continue":
        finetuned = resolve_path(model_cfg["finetuned"])
        if finetuned.is_file():
            return finetuned, "continue-best"
        base = train_cfg["base_weights"]
        return base, "continue-base"

    return train_cfg["base_weights"], "fresh"


def sync_finetuned_weights(
    config: dict[str, Any],
    task_key: str,
    best_path: Path | None = None,
) -> Path | None:
    """Copy best.pt from run folder to models/ for the pipeline."""
    from model_utils import ensure_models_dir

    run_paths = get_run_paths(config, task_key)
    source = best_path or run_paths.best_pt
    if not source.is_file():
        return None

    target = resolve_path(config["models"][task_key]["finetuned"])
    ensure_models_dir(config)
    target.parent.mkdir(parents=True, exist_ok=True)

    import shutil

    shutil.copy2(source, target)
    return target


def format_progress_line(progress: dict[str, Any]) -> str:
    if not progress.get("has_checkpoint"):
        return "no checkpoint (fresh start available)"

    parts = ["checkpoint found"]
    if progress.get("current_epoch") is not None:
        epoch_str = str(progress["current_epoch"])
        if progress.get("total_epochs"):
            epoch_str += f"/{progress['total_epochs']}"
        parts.append(f"epoch {epoch_str}")

    if progress.get("best_map50") is not None:
        parts.append(f"best mAP50={progress['best_map50']:.4f}")

    return ", ".join(parts)


def resume_command(task_key: str) -> str:
    return f"python train_models.py --{TASK_CLI[task_key]} --resume"


def get_cuda_status() -> dict[str, Any]:
    import torch

    status: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": None,
        "is_cpu_build": "+cpu" in torch.__version__.lower(),
    }
    if status["cuda_available"] and status["cuda_device_count"] > 0:
        status["cuda_device_name"] = torch.cuda.get_device_name(0)
    return status


def format_cuda_status(config: dict[str, Any]) -> list[str]:
    device = config.get("training", {}).get("device")
    status = get_cuda_status()
    lines: list[str] = [f"PyTorch : {status['torch_version']}"]

    if status["cuda_available"]:
        name = status["cuda_device_name"] or "CUDA GPU"
        lines.append(f"GPU     : {name} (device {device})")
        return lines

    lines.append("GPU     : not available to PyTorch")
    if status["is_cpu_build"]:
        lines.append("Fix     : install CUDA PyTorch (see docs/TRAINING_PLAN.md)")
    else:
        lines.append("Fix     : update NVIDIA drivers or set training.device: cpu")
    return lines


def verify_cuda_for_training(config: dict[str, Any]) -> tuple[bool, str | None]:
    device = config.get("training", {}).get("device")
    if device in (None, "cpu", -1, "-1"):
        return True, None

    status = get_cuda_status()
    if status["cuda_available"]:
        return True, None

    if status["is_cpu_build"]:
        return False, (
            "PyTorch CPU-only build detected (torch+cpu). "
            "Install CUDA PyTorch, then re-run:\n"
            "  pip uninstall torch torchvision -y\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        )

    return False, (
        "CUDA device requested but torch.cuda.is_available() is False. "
        "Check NVIDIA drivers or set training.device: cpu in pipeline_config.yaml."
    )
