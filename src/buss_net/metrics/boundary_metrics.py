"""Boundary quality metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _mask_boundary(mask: torch.Tensor) -> torch.Tensor:
    dilate = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    erode = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilate - erode).clamp(0, 1)


def boundary_f1_score(pred: torch.Tensor, target: torch.Tensor, tolerance: int = 2, eps: float = 1e-7) -> torch.Tensor:
    """Compute Boundary F1 with a small pixel tolerance."""

    pred_b = _mask_boundary(pred.float())
    target_b = _mask_boundary(target.float())
    if tolerance > 0:
        kernel = 2 * tolerance + 1
        pred_match = F.max_pool2d(pred_b, kernel_size=kernel, stride=1, padding=tolerance)
        target_match = F.max_pool2d(target_b, kernel_size=kernel, stride=1, padding=tolerance)
    else:
        pred_match = pred_b
        target_match = target_b
    dims = (1, 2, 3)
    pred_count = pred_b.sum(dim=dims)
    target_count = target_b.sum(dim=dims)
    precision = (pred_b * target_match).sum(dim=dims) / pred_count.clamp_min(eps)
    recall = (target_b * pred_match).sum(dim=dims) / target_count.clamp_min(eps)
    denominator = precision + recall
    f1 = torch.where(
        denominator > 0,
        2.0 * precision * recall / denominator.clamp_min(eps),
        torch.zeros_like(denominator),
    )
    # An empty prediction is perfect only when the reference boundary is also
    # empty.  If just one side is empty (or non-empty boundaries do not match),
    # Boundary F1 must remain zero.
    both_empty = (pred_count == 0) & (target_count == 0)
    return torch.where(both_empty, torch.ones_like(f1), f1).mean()
