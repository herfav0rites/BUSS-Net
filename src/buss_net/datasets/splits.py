"""Train/test split helpers for the BUSS-Net protocol.

Protocol (2026-07 revision):
- Training uses ``data/processed/train``.
- A single evaluation pool lives under ``data/processed/test``.
- Per-epoch Dice on that pool selects ``best.pt`` (EMA when enabled; highest Dice).
- There is no separate validation split / locked hold-out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_EVAL_SPLIT_PATH = "data/processed/splits/eval_split.json"
DEFAULT_EVAL_SPLIT_SEED = 42
DEFAULT_TEST_ROOT = "data/processed/test"


def load_eval_split_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Eval split manifest not found: {manifest_path}. "
            "Expected data/processed/splits/eval_split.json"
        )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Backward compatible: merge legacy val+test maps into a single test pool.
    if "test_stems_by_dataset" not in data and "val_stems_by_dataset" in data:
        data["test_stems_by_dataset"] = data["val_stems_by_dataset"]
    if "test_stems_by_dataset" not in data:
        raise ValueError(f"Eval split manifest must contain test_stems_by_dataset: {manifest_path}")

    if "val_stems_by_dataset" in data and data["val_stems_by_dataset"]:
        merged: dict[str, list[str]] = {}
        for key in ("val_stems_by_dataset", "test_stems_by_dataset"):
            for dataset, stems in data.get(key, {}).items():
                bucket = set(merged.get(dataset, []))
                bucket.update(str(stem) for stem in stems)
                merged[dataset] = sorted(bucket)
        data["test_stems_by_dataset"] = merged
        data["val_stems_by_dataset"] = {}

    test_total = sum(len(stems) for stems in data["test_stems_by_dataset"].values())
    data["test_count"] = test_total
    data["total_samples"] = test_total
    data["val_count"] = 0
    return data


def stems_by_dataset_map(manifest: dict[str, Any], split: str = "test") -> dict[str, set[str]]:
    """Return per-dataset stems. ``split`` is always the merged test pool."""

    _ = split  # kept for call-site compatibility; val/test both map to test.
    return {
        str(dataset): {str(stem) for stem in stems}
        for dataset, stems in manifest["test_stems_by_dataset"].items()
        if stems
    }


def resolve_eval_split_path(dataset_cfg: dict[str, Any], root: Path) -> Path:
    rel = dataset_cfg.get("eval_split", dataset_cfg.get("val_split", DEFAULT_EVAL_SPLIT_PATH))
    return root / rel


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    """Deprecated legacy helper."""
    raise RuntimeError(
        f"Legacy train-pool val_split manifests are removed. Use {DEFAULT_EVAL_SPLIT_PATH}."
    )


def split_stems(manifest: dict[str, Any]) -> tuple[set[str], set[str]]:
    raise RuntimeError("Legacy split_stems() is removed. Use stems_by_dataset_map(manifest, 'test').")
