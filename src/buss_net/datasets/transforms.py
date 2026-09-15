"""Albumentations transforms used by the polyp segmentation pipeline."""

from __future__ import annotations

from typing import Sequence

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN: Sequence[float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Sequence[float] = (0.229, 0.224, 0.225)


def build_minimal_train_transform(size: int = 352) -> A.Compose:
    """Minimal resize-only training transform."""

    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_train_transform(
    size: int = 352,
    profile: str = "default",
    crop_scale: tuple[float, float] = (0.75, 1.0),
) -> A.Compose:
    """Build the training-time augmentation pipeline from the design docs."""

    profile = profile.lower()
    if profile in {"none", "off", "false"}:
        return build_minimal_train_transform(size)
    if profile not in {"default", "strong"}:
        raise ValueError("Augmentation profile must be 'none', 'default', or 'strong'.")
    if profile == "strong":
        photometric = A.OneOf(
            [
                A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05, p=1.0),
                A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=18, val_shift_limit=18, p=1.0),
                A.RandomGamma(gamma_limit=(80, 125), p=1.0),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            ],
            p=0.45,
        )
        blur_p = 0.15
        dropout = A.CoarseDropout(max_holes=6, max_height=24, max_width=24, fill_value=0, mask_fill_value=None, p=0.25)
    else:
        photometric = A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3)
        blur_p = 0.2
        dropout = A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, mask_fill_value=None, p=0.2)
    return A.Compose(
        [
            A.Resize(size, size),
            A.RandomResizedCrop(height=size, width=size, scale=crop_scale, ratio=(0.9, 1.1), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                border_mode=0,
                value=0,
                mask_value=0,
                p=0.5,
            ),
            photometric,
            A.GaussianBlur(blur_limit=(3, 7), p=blur_p),
            dropout,
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_test_transform(size: int = 352) -> A.Compose:
    """Build deterministic validation/test transforms."""

    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
