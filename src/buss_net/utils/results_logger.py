"""YOLO-style CSV / preview logging for training runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


def _unnormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def save_val_batch_grid(
    images: torch.Tensor,
    masks: torch.Tensor,
    logits: torch.Tensor,
    out_path: str | Path,
    max_items: int = 4,
) -> Path:
    """Save an RGB | GT | Pred montage to ``out_path``."""
    import cv2
    import numpy as np

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(max_items, int(images.shape[0]))
    imgs = _unnormalize(images[:count].detach())
    gts = masks[:count].detach()
    preds = torch.sigmoid(logits[:count].detach())
    if gts.ndim == 3:
        gts = gts.unsqueeze(1)
    if preds.ndim == 3:
        preds = preds.unsqueeze(1)
    if gts.shape[-2:] != imgs.shape[-2:]:
        gts = F.interpolate(gts.float(), size=imgs.shape[-2:], mode="nearest")
    if preds.shape[-2:] != imgs.shape[-2:]:
        preds = F.interpolate(preds, size=imgs.shape[-2:], mode="bilinear", align_corners=False)

    tiles = []
    for i in range(count):
        rgb = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        gt = (gts[i, 0].cpu().numpy() > 0.5).astype("uint8") * 255
        pr = (preds[i, 0].cpu().numpy() * 255).astype("uint8")
        gt_rgb = np.stack([gt, gt, gt], axis=-1)
        pr_rgb = np.stack([pr, pr, pr], axis=-1)
        tiles.append(np.concatenate([rgb, gt_rgb, pr_rgb], axis=1))
    canvas = np.concatenate(tiles, axis=0)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return out_path


def _float_series(rows: Sequence[Mapping[str, str]], key: str) -> list[float]:
    """Read a CSV column as floats; missing/blank cells become NaN."""

    values: list[float] = []
    for row in rows:
        raw = row.get(key, "")
        if raw is None or raw == "":
            values.append(float("nan"))
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return values


def plot_results_csv(csv_path: str | Path, out_path: str | Path | None = None) -> Path | None:
    """Render ``results.csv`` into a YOLO-style multi-metric ``results.png``."""

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    fieldnames = set(rows[0].keys())
    epochs = _float_series(rows, "epoch")

    # Prefer the real training total loss; accept legacy aliases.
    loss_candidates = (
        ("train/total", "total"),
        ("train/loss", "loss"),
        ("train/structure", "structure"),
        ("train/bce", "bce"),
        ("train/dice", "dice_loss"),
        ("train/boundary", "boundary"),
        ("train/edge", "edge"),
    )
    metric_candidates = (
        ("metrics/dice", "Dice"),
        ("metrics/iou", "IoU"),
        ("metrics/recall", "Recall"),
        ("metrics/precision", "Precision"),
        ("metrics/boundary_f1", "Boundary F1"),
        ("metrics/f2", "F2"),
        ("metrics/mae", "MAE"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    plotted_loss = False
    for key, label in loss_candidates:
        if key not in fieldnames:
            continue
        series = _float_series(rows, key)
        if all(v != v for v in series):  # all NaN
            continue
        # Skip constant-zero auxiliaries (e.g. unused edge loss) to reduce clutter.
        finite = [v for v in series if v == v]
        if finite and max(abs(v) for v in finite) < 1e-12:
            continue
        # Emphasize the primary total/loss curve.
        lw = 2.0 if key in {"train/total", "train/loss"} else 1.2
        axes[0].plot(epochs, series, label=label, linewidth=lw)
        plotted_loss = True
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(True, alpha=0.3)
    if plotted_loss:
        axes[0].legend(loc="best", fontsize=8)

    plotted_metric = False
    for key, label in metric_candidates:
        if key not in fieldnames:
            continue
        series = _float_series(rows, key)
        if all(v != v for v in series):
            continue
        lw = 2.0 if key == "metrics/dice" else 1.2
        axes[1].plot(epochs, series, label=label, linewidth=lw)
        plotted_metric = True
    axes[1].set_title("Val metrics")
    axes[1].set_xlabel("epoch")
    axes[1].grid(True, alpha=0.3)
    if plotted_metric:
        axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    out = Path(out_path) if out_path is not None else csv_path.with_name("results.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


class YoloResultsLogger:
    """Append epoch metrics to ``logs/results.csv`` and save val preview images."""

    def __init__(self, exp_dir: str | Path, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.exp_dir = Path(exp_dir)
        self.log_dir = self.exp_dir / "logs"
        self.csv_path = self.log_dir / "results.csv"
        self._fieldnames: list[str] | None = None
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_epoch(
        self,
        epoch: int,
        *,
        train_losses: Mapping[str, float],
        val_metrics: Mapping[str, float],
        per_dataset: Mapping[str, Mapping[str, float]] | None = None,
        lr: float = 0.0,
        epoch_time: float = 0.0,
    ) -> None:
        if not self.enabled:
            return
        row: dict[str, Any] = {
            "epoch": float(epoch),
            "time": float(epoch_time),
            "lr": float(lr),
        }
        for key, value in train_losses.items():
            row[f"train/{key}"] = float(value)
        for key, value in val_metrics.items():
            row[f"metrics/{key}"] = float(value)
        if per_dataset:
            for ds_name, metrics in sorted(per_dataset.items()):
                if "dice" in metrics:
                    row[f"metrics/dice_{ds_name}"] = float(metrics["dice"])

        if self._fieldnames is None:
            if self.csv_path.is_file():
                with self.csv_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    self._fieldnames = list(reader.fieldnames or [])
            if not self._fieldnames:
                self._fieldnames = list(row.keys())
            else:
                for key in row:
                    if key not in self._fieldnames:
                        self._fieldnames.append(key)

        assert self._fieldnames is not None
        write_header = not self.csv_path.is_file() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in self._fieldnames})

        try:
            self._plot_results()
        except Exception as exc:
            # Keep training alive, but surface plot failures instead of silencing them.
            print(f"[YoloResultsLogger] results.png plot failed: {exc}")

    def save_val_batch(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        logits: torch.Tensor,
        epoch: int,
        max_items: int = 4,
    ) -> None:
        if not self.enabled:
            return
        try:
            save_val_batch_grid(
                images,
                masks,
                logits,
                self.log_dir / f"val_batch{epoch:03d}.jpg",
                max_items=max_items,
            )
        except Exception:
            return

    def _plot_results(self) -> None:
        plot_results_csv(self.csv_path, self.log_dir / "results.png")

    def regenerate_plot(self) -> Path | None:
        """Rebuild ``results.png`` from the existing CSV (post-hoc fix)."""

        return plot_results_csv(self.csv_path, self.log_dir / "results.png")
