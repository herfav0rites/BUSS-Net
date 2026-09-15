"""Rich console logging."""

from __future__ import annotations

from collections.abc import Mapping

try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None
    Table = None


class RichLogger:
    def __init__(self) -> None:
        self.console = Console() if Console else None

    def log(self, message: str) -> None:
        if self.console:
            self.console.print(message)
        else:
            print(message)

    def epoch_summary(
        self,
        epoch: int,
        epochs: int,
        train_losses: Mapping[str, float],
        val_metrics: Mapping[str, float],
        best_dice: float,
        lr: float,
        epoch_time: float,
        warnings: list[str] | None = None,
    ) -> None:
        if not self.console or not Table:
            print(f"Epoch {epoch}/{epochs} loss={train_losses.get('total', 0):.4f} dice={val_metrics.get('dice', 0):.4f}")
            return

        table = Table(title=f"Epoch {epoch:03d}/{epochs:03d} | {epoch_time:.1f}s | lr={lr:.2e}")
        table.add_column("Split", style="cyan")
        table.add_column("Total")
        table.add_column("BCE")
        table.add_column("DiceLoss")
        table.add_column("Boundary")
        table.add_column("Edge")
        table.add_column("Dice")
        table.add_column("IoU")
        table.add_column("Recall")
        table.add_column("BdyF1")
        table.add_column("MAE")
        table.add_row(
            "Train",
            f"{train_losses.get('total', 0):.4f}",
            f"{train_losses.get('bce', 0):.4f}",
            f"{train_losses.get('dice', 0):.4f}",
            f"{train_losses.get('boundary', 0):.4f}",
            f"{train_losses.get('edge', 0):.4f}",
            "-",
            "-",
            "-",
            "-",
            "-",
        )
        table.add_row(
            "Valid",
            "-",
            "-",
            "-",
            "-",
            "-",
            f"{val_metrics.get('dice', 0):.4f}",
            f"{val_metrics.get('iou', 0):.4f}",
            f"{val_metrics.get('recall', 0):.4f}",
            f"{val_metrics.get('boundary_f1', 0):.4f}",
            f"{val_metrics.get('mae', 0):.4f}",
        )
        self.console.print(table)
        self.console.print(f"[bold green]Best mean Dice:[/] {best_dice:.4f}")
        for warning in warnings or []:
            self.console.print(f"[bold yellow]WARNING:[/] {warning}")

    def dataset_metrics(self, per_dataset: Mapping[str, Mapping[str, float]]) -> None:
        if not per_dataset:
            return
        if not self.console or not Table:
            print(per_dataset)
            return
        table = Table(title="Validation by Dataset")
        for col in ["Dataset", "Dice", "IoU", "Recall", "Precision", "F2", "BdyF1", "MAE"]:
            table.add_column(col)
        for name, metrics in per_dataset.items():
            table.add_row(
                name,
                f"{metrics.get('dice', 0):.4f}",
                f"{metrics.get('iou', 0):.4f}",
                f"{metrics.get('recall', 0):.4f}",
                f"{metrics.get('precision', 0):.4f}",
                f"{metrics.get('f2', 0):.4f}",
                f"{metrics.get('boundary_f1', 0):.4f}",
                f"{metrics.get('mae', 0):.4f}",
            )
        self.console.print(table)
