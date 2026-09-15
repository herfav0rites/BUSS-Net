"""Binary segmentation metrics used in validation and testing."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .boundary_metrics import boundary_f1_score


@torch.no_grad()
def compute_binary_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    prob = torch.sigmoid(logits)
    pred = (prob > threshold).float()
    target = (target > 0.5).float()
    dims = (1, 2, 3)
    eps = 1e-7

    tp = (pred * target).sum(dim=dims)
    fp = (pred * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - pred) * target).sum(dim=dims)
    pred_sum = pred.sum(dim=dims)
    target_sum = target.sum(dim=dims)

    dice_den = pred_sum + target_sum
    iou_den = tp + fp + fn
    recall_den = tp + fn
    precision_den = tp + fp

    dice_per_image = torch.where(dice_den > 0, (2.0 * tp) / (dice_den + eps), torch.ones_like(tp))
    iou_per_image = torch.where(iou_den > 0, tp / (iou_den + eps), torch.ones_like(tp))
    recall_per_image = torch.where(recall_den > 0, tp / (recall_den + eps), torch.ones_like(tp))
    precision_per_image = torch.where(precision_den > 0, tp / (precision_den + eps), torch.zeros_like(tp))
    precision_per_image = torch.where((precision_den == 0) & (target_sum == 0), torch.ones_like(tp), precision_per_image)
    f2_per_image = torch.where(
        (4.0 * precision_per_image + recall_per_image) > 0,
        (5.0 * precision_per_image * recall_per_image) / (4.0 * precision_per_image + recall_per_image + eps),
        torch.zeros_like(tp),
    )

    dice = dice_per_image.mean()
    iou = iou_per_image.mean()
    recall = recall_per_image.mean()
    precision = precision_per_image.mean()
    f2 = f2_per_image.mean()
    b_f1 = boundary_f1_score(pred, target)
    mae = torch.abs(prob - target).mean()

    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "recall": float(recall.item()),
        "precision": float(precision.item()),
        "f2": float(f2.item()),
        "boundary_f1": float(b_f1.item()),
        "mae": float(mae.item()),
    }


@dataclass
class MetricAccumulator:
    totals: dict[str, float] = field(default_factory=dict)
    count: int = 0

    def update(self, metrics: dict[str, float], n: int = 1) -> None:
        for key, value in metrics.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value) * n
        self.count += n

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {key: value / self.count for key, value in self.totals.items()}
