"""TensorBoard logging helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


class TensorBoardLogger:
    def __init__(self, log_dir: str | Path, enabled: bool = True) -> None:
        self.writer = SummaryWriter(str(log_dir)) if enabled and SummaryWriter else None

    def log_scalars(self, prefix: str, scalars: Mapping[str, float], step: int) -> None:
        if not self.writer:
            return
        for key, value in scalars.items():
            self.writer.add_scalar(f"{prefix}/{key}", float(value), step)

    def log_lr(self, optimizer: torch.optim.Optimizer, step: int) -> None:
        if not self.writer:
            return
        for idx, group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f"LR/group_{idx}", float(group["lr"]), step)

    def log_prediction_grid(self, images: torch.Tensor, masks: torch.Tensor, logits: torch.Tensor, step: int, max_items: int = 4) -> None:
        if not self.writer:
            return
        count = min(max_items, images.shape[0])
        imgs = images[:count].detach().cpu()
        imgs = imgs * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imgs = imgs.clamp(0, 1)
        gts = masks[:count].detach().cpu().repeat(1, 3, 1, 1)
        preds = torch.sigmoid(logits[:count].detach().cpu()).repeat(1, 3, 1, 1)
        grid = torch.cat([imgs, gts, preds], dim=0)
        self.writer.add_images("Images/input_gt_pred", grid, step)

    def close(self) -> None:
        if self.writer:
            self.writer.close()
