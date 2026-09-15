"""Training anomaly monitoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrainMonitor:
    patience: int = 20
    min_delta: float = 1e-5
    best_dice: float = 0.0
    best_epoch: int = 0
    stagnant_epochs: int = 0

    def check_loss(self, loss: torch.Tensor, step: int) -> list[str]:
        warnings: list[str] = []
        value = float(loss.detach().item())
        if not torch.isfinite(loss) or value > 10:
            warnings.append(f"Loss explosion detected at step {step}: {value:.4f}")
        return warnings

    def update_validation(self, dice: float, epoch: int) -> tuple[bool, list[str]]:
        warnings: list[str] = []
        improved = dice > self.best_dice + self.min_delta
        if improved:
            self.best_dice = dice
            self.best_epoch = epoch
            self.stagnant_epochs = 0
        else:
            self.stagnant_epochs += 1
            if self.stagnant_epochs >= self.patience:
                warnings.append(f"Validation Dice stagnated for {self.stagnant_epochs} epochs")
        return improved, warnings

    def check_system(self, optimizer: torch.optim.Optimizer) -> list[str]:
        warnings: list[str] = []
        for idx, group in enumerate(optimizer.param_groups):
            lr = float(group["lr"])
            if lr <= 0 or lr > 1:
                warnings.append(f"Abnormal learning rate in group {idx}: {lr:.3e}")
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if used / max(total, 1e-6) > 0.95:
                warnings.append(f"GPU memory usage is critical: {used:.2f}/{total:.2f} GB")
        return warnings
