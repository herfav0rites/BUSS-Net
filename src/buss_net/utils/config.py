"""YAML config loading and deep/shallow merging."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load YAML, optionally deep-merging one or more relative ``_base_`` files."""

    config_path = Path(path).resolve()
    if config_path in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, config_path))
        raise ValueError(f"Cyclic YAML inheritance detected: {chain}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"YAML root must be a mapping: {config_path}")

    base_value = raw.pop("_base_", None)
    if base_value is None:
        return raw
    base_items = [base_value] if isinstance(base_value, (str, Path)) else list(base_value)
    merged: dict[str, Any] = {}
    for base_item in base_items:
        base_path = Path(base_item)
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        merged = deep_merge(merged, load_yaml(base_path, (*_stack, config_path)))
    return deep_merge(merged, raw)


def merge_config(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
