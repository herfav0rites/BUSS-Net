#!/usr/bin/env python3
"""Create the deterministic full-pool evaluation manifest used by training/testing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from buss_net.datasets.splits import DEFAULT_EVAL_SPLIT_PATH, DEFAULT_EVAL_SPLIT_SEED

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def create_eval_split(
    test_root: Path,
    *,
    out_path: Path,
    split_seed: int = DEFAULT_EVAL_SPLIT_SEED,
) -> dict[str, object]:
    stems_by_dataset: dict[str, list[str]] = {}
    for dataset in sorted(path for path in test_root.iterdir() if path.is_dir()):
        image_dir = dataset / "image"
        if not image_dir.is_dir():
            continue
        stems = sorted(path.stem for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if stems:
            stems_by_dataset[dataset.name] = stems
    if not stems_by_dataset:
        raise ValueError(f"No evaluation images found under {test_root}")
    counts = {name: len(stems) for name, stems in stems_by_dataset.items()}
    total = sum(counts.values())
    manifest: dict[str, object] = {
        "schema": "eval_split_v2",
        "split_seed": split_seed,
        "test_count": total,
        "total_samples": total,
        "source_test_root": str(test_root.as_posix()),
        "test_stems_by_dataset": stems_by_dataset,
        "test_counts_by_dataset": counts,
        "notes": "Full benchmark pool; epoch Dice selects best.pt (EMA when enabled).",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", type=Path, default=ROOT / "data/processed/test")
    parser.add_argument("--out", type=Path, default=ROOT / DEFAULT_EVAL_SPLIT_PATH)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_EVAL_SPLIT_SEED)
    args = parser.parse_args()
    manifest = create_eval_split(args.test_root, out_path=args.out, split_seed=args.split_seed)
    print(f"Wrote {args.out}: total={manifest['total_samples']}")


if __name__ == "__main__":
    main()
