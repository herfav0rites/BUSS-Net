"""BCE + Dice structure loss and total GeoFSS loss composition."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def mask_boundary(target: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Differentiation-free morphological gradient used as an edge target."""

    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("boundary kernel_size must be an odd integer >= 3.")
    padding = kernel_size // 2
    dilated = F.max_pool2d(target, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-target, kernel_size=kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0.0, 1.0)


class BoundarySupervisionLoss(nn.Module):
    """Class-balanced BCE plus Dice for the sparse boundary target."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        positives = target.sum(dim=(2, 3), keepdim=True)
        negatives = target.shape[-2] * target.shape[-1] - positives
        positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
        pixel_weight = 1.0 + target * (positive_weight - 1.0)
        bce = F.binary_cross_entropy_with_logits(logits, target, weight=pixel_weight, reduction="mean")
        probability = torch.sigmoid(logits)
        intersection = (probability * target).sum(dim=(2, 3))
        denominator = probability.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        return bce + dice


def _balanced_probability_bce(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Boundary-balanced BCE for differentiable routing probabilities."""

    positives = target.sum(dim=(2, 3), keepdim=True)
    negatives = target.shape[-2] * target.shape[-1] - positives
    positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
    pixel_weight = 1.0 + target * (positive_weight - 1.0)
    return F.binary_cross_entropy(
        probability.clamp(1e-5, 1.0 - 1e-5),
        target,
        weight=pixel_weight,
        reduction="mean",
    )


class StructureLoss(nn.Module):
    """0.5 BCE + 0.5 Dice loss for class-imbalanced polyp masks."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="mean")
        pred_prob = torch.sigmoid(pred)
        inter = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
        dice_loss = dice_loss.mean()
        loss = self.bce_weight * bce + self.dice_weight * dice_loss
        return loss, {"bce": bce.detach(), "dice": dice_loss.detach()}


class GeoFSSLoss(nn.Module):
    """Deep-supervised structure loss used by training experiments."""

    def __init__(
        self,
        ds_weights: dict[str, float] | None = None,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        boundary_weight: float = 0.0,
        boundary_kernel_size: int = 3,
        routing_weight: float = 0.0,
    ) -> None:
        super().__init__()
        initial_ds_weights = ds_weights or {
            "pred_final": 1.0,
            "pred_s4": 0.4,
            "pred_s3": 0.3,
            "pred_s2": 0.2,
            "pred_uggd_d1": 0.2,
            "pred_uggd_d2": 0.2,
            "pred_uggd_d3": 0.2,
            "pred_uggd_d4": 0.2,
        }
        self.base_ds_weights = {
            str(name): float(weight) for name, weight in initial_ds_weights.items()
        }
        self.ds_weights = dict(self.base_ds_weights)
        self.boundary_weight = float(boundary_weight)
        self.boundary_kernel_size = int(boundary_kernel_size)
        self.routing_weight = float(routing_weight)
        self.boundary_loss = BoundarySupervisionLoss()
        self.structure = StructureLoss(bce_weight=bce_weight, dice_weight=dice_weight)

    def set_ds_weights(self, weights: dict[str, float], *, normalize: bool = True) -> None:
        """Update deep-supervision weights without rebuilding the criterion."""

        cleaned = {str(name): max(0.0, float(weight)) for name, weight in weights.items()}
        total = sum(cleaned.values())
        if total <= 0.0:
            raise ValueError("At least one deep-supervision weight must be positive.")
        self.ds_weights = (
            {name: weight / total for name, weight in cleaned.items()}
            if normalize
            else cleaned
        )

    def set_boundary_weight(self, weight: float) -> None:
        """Update the auxiliary boundary-loss coefficient for the current epoch."""

        if weight < 0.0:
            raise ValueError("boundary weight must be non-negative.")
        self.boundary_weight = float(weight)

    def forward(self, outputs: dict[str, torch.Tensor], target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = target.new_tensor(0.0)
        logs: dict[str, torch.Tensor] = {}
        structure_total = target.new_tensor(0.0)
        bce_total = target.new_tensor(0.0)
        dice_total = target.new_tensor(0.0)

        for name, weight in self.ds_weights.items():
            if weight <= 0.0 or name not in outputs:
                continue
            pred = outputs[name]
            if pred.shape[-2:] != target.shape[-2:]:
                pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
            struct, parts = self.structure(pred, target)
            total = total + weight * struct
            structure_total = structure_total + weight * struct.detach()
            if "bce" in parts:
                bce_total = bce_total + weight * parts["bce"]
            if "dice" in parts:
                dice_total = dice_total + weight * parts["dice"]

        boundary_target = mask_boundary(target, self.boundary_kernel_size)
        boundary_total = target.new_tensor(0.0)
        if self.boundary_weight > 0.0 and "pred_boundary" in outputs:
            boundary_logits = outputs["pred_boundary"]
            if boundary_logits.shape[-2:] != target.shape[-2:]:
                # The guidance head is intentionally coarse. Pool the thin target
                # to its native grid instead of asking an upsampled coarse logit
                # to reproduce a two-pixel full-resolution contour.
                boundary_target = F.adaptive_max_pool2d(
                    boundary_target, output_size=boundary_logits.shape[-2:]
                )
            boundary_total = self.boundary_loss(boundary_logits, boundary_target)
            total = total + self.boundary_weight * boundary_total

        routing_total = target.new_tensor(0.0)
        if self.routing_weight > 0.0:
            route_terms: list[torch.Tensor] = []
            blg_identity = outputs.get("route_blg_identity")
            blg_local = outputs.get("route_blg_local")
            blg_global = outputs.get("route_blg_global")
            if blg_local is not None and blg_global is not None:
                route = (blg_local + blg_global).clamp(0.0, 1.0)
            elif blg_identity is not None:
                route = 1.0 - blg_identity
            else:
                route = None
            if route is not None:
                if route.shape[-2] <= boundary_target.shape[-2] and route.shape[-1] <= boundary_target.shape[-1]:
                    route_target = F.adaptive_max_pool2d(boundary_target, output_size=route.shape[-2:])
                else:
                    route_target = F.interpolate(boundary_target, size=route.shape[-2:], mode="nearest")
                # BLG should prefer the identity route in interiors and assign
                # residual local/global capacity around object boundaries.
                route_terms.append(_balanced_probability_bce(route, route_target))

            for name, route in outputs.items():
                if not name.startswith("route_uggd_d"):
                    continue
                if route.shape[-2] <= boundary_target.shape[-2] and route.shape[-1] <= boundary_target.shape[-1]:
                    route_target = F.adaptive_max_pool2d(boundary_target, output_size=route.shape[-2:])
                else:
                    route_target = F.interpolate(boundary_target, size=route.shape[-2:], mode="nearest")
                # Decoder disagreement is most useful around ambiguous object
                # contours, so supervise each UG-GD write gate with that proxy.
                route_terms.append(_balanced_probability_bce(route, route_target))
            if route_terms:
                routing_total = torch.stack(route_terms).mean()
                total = total + self.routing_weight * routing_total

        logs.update(
            {
                "total": total.detach(),
                "structure": structure_total,
                "bce": bce_total,
                "dice": dice_total,
                "boundary": boundary_total.detach(),
                "edge": routing_total.detach(),
            }
        )
        return total, logs
