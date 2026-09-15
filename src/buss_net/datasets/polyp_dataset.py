"""PyTorch Dataset for paired colonoscopy image and polyp mask folders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import build_test_transform


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class PolypDataset(Dataset):
    """Load image/mask pairs from separate directories.

    The dataset expects matching file stems in ``image_dir`` and ``mask_dir``.
    Images are returned as normalized tensors and masks as binary ``float32``
    tensors with shape ``[1, H, W]``.
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        transform: Any | None = None,
        size: int = 352,
        include_stems: set[str] | None = None,
        exclude_stems: set[str] | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform or build_test_transform(size)

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        image_paths = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        mask_paths = sorted(p for p in self.mask_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        masks_by_stem = {p.stem: p for p in mask_paths}

        self.samples: list[tuple[Path, Path]] = []
        missing_masks: list[str] = []
        for image_path in image_paths:
            stem = image_path.stem
            if include_stems is not None and stem not in include_stems:
                continue
            if exclude_stems is not None and stem in exclude_stems:
                continue
            mask_path = masks_by_stem.get(stem)
            if mask_path is None:
                missing_masks.append(image_path.name)
            else:
                self.samples.append((image_path, mask_path))

        if missing_masks:
            preview = ", ".join(missing_masks[:5])
            raise ValueError(f"{len(missing_masks)} images have no matching mask in {self.mask_dir}: {preview}")

        if include_stems is not None:
            missing_in_split = sorted(include_stems - {image_path.stem for image_path, _ in self.samples})
            if missing_in_split:
                preview = ", ".join(missing_in_split[:5])
                raise ValueError(
                    f"{len(missing_in_split)} stems from the split manifest are missing in "
                    f"{self.image_dir}: {preview}"
                )
        elif exclude_stems is None:
            if len(self.samples) != len(mask_paths):
                image_stems = {p.stem for p in image_paths}
                extra_masks = [p.name for p in mask_paths if p.stem not in image_stems]
                preview = ", ".join(extra_masks[:5])
                raise ValueError(f"{len(extra_masks)} masks have no matching image in {self.image_dir}: {preview}")
        if not self.samples:
            raise ValueError(f"No image/mask pairs found in {self.image_dir} and {self.mask_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path = self.samples[index]

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")
        mask = (mask_gray > 127).astype(np.float32)
        original_size = torch.tensor(mask.shape, dtype=torch.long)

        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented["image"].float()
        mask_tensor = augmented["mask"]
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "name": image_path.stem,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "original_size": original_size,
        }
