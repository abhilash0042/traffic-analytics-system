"""Shared helpers for loading config and resolving fine-tuned model paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pipeline_config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def first_existing_path(*candidates: str | Path) -> Path | None:
    for candidate in candidates:
        resolved = resolve_path(candidate)
        if resolved.is_file():
            return resolved
    return None


def resolve_model_path(model_cfg: dict[str, Any], role: str = "model") -> tuple[Path, str]:
    """
    Return (path, source_label) preferring fine-tuned weights over pretrained fallback.
    """
    finetuned = model_cfg.get("finetuned")
    pretrained = model_cfg.get("pretrained")
    fallback = model_cfg.get("fallback")

    finetuned_path = resolve_path(finetuned) if finetuned else None
    if finetuned_path and finetuned_path.is_file():
        return finetuned_path, "fine-tuned"

    found = first_existing_path(*(p for p in (pretrained, fallback) if p))
    if found is None:
        raise FileNotFoundError(
            f"No weights found for {role}. Expected one of: {finetuned}, {pretrained}, {fallback}"
        )

    if pretrained and found == resolve_path(pretrained):
        return found, "pretrained"
    return found, "fallback"


def has_finetuned_model(model_cfg: dict[str, Any]) -> bool:
    finetuned = model_cfg.get("finetuned")
    return bool(finetuned and resolve_path(finetuned).is_file())


def find_data_yaml(dataset_dir: str | Path) -> Path | None:
    """Locate a YOLO data.yaml inside a dataset directory (Roboflow/Kaggle layouts vary)."""
    root = resolve_path(dataset_dir)
    if not root.is_dir():
        return None

    direct = root / "data.yaml"
    if direct.is_file():
        return direct

    candidates: list[Path] = []
    for path in root.rglob("data.yaml"):
        candidates.append(path)

    if not candidates:
        return None

    def score(path: Path) -> tuple[int, int]:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        has_split = int("train:" in text and ("val:" in text or "valid:" in text))
        depth = len(path.parts)
        return (has_split, -depth)

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def dataset_ready(dataset_dir: str | Path) -> bool:
    return find_data_yaml(dataset_dir) is not None


def ensure_models_dir(config: dict[str, Any] | None = None) -> Path:
    config = config or load_config()
    models_dir = resolve_path(config["training"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir
