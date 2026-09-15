"""Reproducibility helpers: deterministic training (fixed internal seed)."""

from __future__ import annotations

import os
import random
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

DEFAULT_SEED = 2026
MANAGED_EXPERIMENT_DIRS = (
    "checkpoints",
    "logs",
    "metrics",
    "predictions",
    "visualizations",
)
MANAGED_EXPERIMENT_FILES = (
    "config_resolved.yaml",
    "training_summary.yaml",
    "training_summary.json",
    "post_train_summary.yaml",
)


def cudnn_flags(config: dict) -> tuple[bool, bool]:
    repro = config.get("reproducibility", {})
    benchmark = bool(repro.get("cudnn_benchmark", False))
    deterministic = bool(repro.get("cudnn_deterministic", True))
    return benchmark, deterministic


def set_seed(
    seed: int,
    *,
    cudnn_benchmark: bool = False,
    cudnn_deterministic: bool = True,
) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic


def make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    def _init(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return _init


def resolve_base_exp_dir(base_exp_dir: str | Path, root: Path | None = None) -> Path:
    """Return the single-run experiment directory."""
    path = Path(base_exp_dir)
    if not path.is_absolute():
        path = (root or Path.cwd()) / path
    return path


def resolve_exp_dir(base_exp_dir: str | Path, root: Path | None = None) -> Path:
    """Return the single-run experiment directory without a seed suffix."""
    return resolve_base_exp_dir(base_exp_dir, root=root)


def resolve_active_exp_dir(base_exp_dir: str | Path, root: Path | None = None) -> Path:
    """Resolve the directory that holds the single training run."""
    return resolve_base_exp_dir(base_exp_dir, root=root)


def prepare_experiment_dir(
    base_exp_dir: str | Path,
    *,
    root: Path | None = None,
    resume: bool = False,
) -> tuple[Path, tuple[Path, ...]]:
    """Prepare one run directory, replacing only generated training artifacts.

    A fresh run removes the known output directories and resolved-run files so
    checkpoints and CSV rows cannot be mixed across protocols. Source assets
    such as ``vendor/`` and experiment documentation are intentionally kept.
    Resumed runs preserve every artifact and append from the loaded checkpoint.
    """

    exp_dir = resolve_exp_dir(base_exp_dir, root=root).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        return exp_dir, ()

    removed: list[Path] = []
    for name in MANAGED_EXPERIMENT_DIRS:
        path = exp_dir / name
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            path.unlink()
            removed.append(path)
        elif path.is_dir():
            shutil.rmtree(path)
            removed.append(path)
    for name in MANAGED_EXPERIMENT_FILES:
        path = exp_dir / name
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
            removed.append(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
            removed.append(path)
    return exp_dir, tuple(removed)
