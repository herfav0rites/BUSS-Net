#!/usr/bin/env python3
"""Resize and binarize the standard polyp segmentation dataset layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _paired_files(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected paired directories: {image_dir}, {mask_dir}")
    masks = {path.stem: path for path in mask_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    pairs = []
    for image in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
        if image.stem not in masks:
            raise ValueError(f"Mask missing for {image}")
        pairs.append((image, masks[image.stem]))
    if not pairs:
        raise ValueError(f"No image/mask pairs found in {image_dir}")
    return pairs


def _process_split(
    image_dir: Path, mask_dir: Path, out_images: Path, out_masks: Path, size: int
) -> int:
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    pairs = _paired_files(image_dir, mask_dir)
    for index, (image_path, mask_path) in enumerate(tqdm(pairs, desc=image_dir.parent.name), start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise OSError(f"Failed to read pair: {image_path}, {mask_path}")
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
        mask = ((mask > 127).astype("uint8") * 255)
        name = f"{index:04d}_{image_path.stem}.png"
        if not cv2.imwrite(str(out_images / name), image) or not cv2.imwrite(str(out_masks / name), mask):
            raise OSError(f"Failed to write processed pair: {name}")
    return len(pairs)


def prepare_dataset(raw_dir: str | Path, out_dir: str | Path, size: int = 352) -> None:
    raw = Path(raw_dir)
    out = Path(out_dir)
    train = raw / "TrainDataset"
    count = _process_split(
        train / "image", train / "mask", out / "train" / "image", out / "train" / "mask", size
    )
    print(f"Prepared train: {count}")
    test = raw / "TestDataset"
    if not test.is_dir():
        raise FileNotFoundError(f"TestDataset not found: {test}")
    for dataset in sorted(path for path in test.iterdir() if path.is_dir()):
        count = _process_split(
            dataset / "image",
            dataset / "mask",
            out / "test" / dataset.name / "image",
            out / "test" / dataset.name / "mask",
            size,
        )
        print(f"Prepared {dataset.name}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", "--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", "--out_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--size", type=int, default=352)
    args = parser.parse_args()
    prepare_dataset(args.raw_dir, args.out_dir, args.size)


if __name__ == "__main__":
    main()
