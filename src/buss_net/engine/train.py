"""Training and validation loops."""

from __future__ import annotations

from contextlib import contextmanager
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from buss_net.losses import GeoFSSLoss
from buss_net.metrics import MetricAccumulator, compute_binary_metrics
from buss_net.utils import RichLogger, TensorBoardLogger, TrainMonitor, YoloResultsLogger


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_dist_avail_and_initialized() or dist.get_rank() == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP / torch.compile wrappers to reach the trainable module."""

    current = model
    # torch.compile wraps the module as OptimizedModule(_orig_mod=...).
    while hasattr(current, "_orig_mod"):
        current = current._orig_mod  # type: ignore[attr-defined]
    if hasattr(current, "module"):
        current = current.module
    return current


def optimizer_reference_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the base LR even when added modules use differential groups."""

    for group in optimizer.param_groups:
        if group.get("group_name") == "base" and float(group.get("weight_decay", 0.0)) > 0.0:
            return float(group["lr"])
    for group in optimizer.param_groups:
        if group.get("group_name") == "base":
            return float(group["lr"])
    return float(optimizer.param_groups[0]["lr"])


class ModelEMA:
    """Exponential moving average of model weights for steadier validation."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1).")
        self.decay = decay
        self.shadow = {
            key: value.detach().clone()
            for key, value in unwrap_model(model).state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        current = unwrap_model(model).state_dict()
        for key, value in current.items():
            if key not in self.shadow:
                self.shadow[key] = value.detach().clone()
                continue
            if torch.is_floating_point(value):
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(value.detach())

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {key: value.detach().clone() for key, value in self.shadow.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state.get("decay", self.decay))
        shadow = state.get("shadow", {})
        self.shadow = {key: value.detach().clone() for key, value in shadow.items()}

    @contextmanager
    def apply_to(self, model: torch.nn.Module):
        target = unwrap_model(model)
        current = target.state_dict()
        backup = {key: value.detach().clone() for key, value in current.items()}
        try:
            for key, value in current.items():
                if key in self.shadow:
                    value.copy_(self.shadow[key].to(device=value.device, dtype=value.dtype))
            yield target
        finally:
            restored = target.state_dict()
            for key, value in restored.items():
                value.copy_(backup[key].to(device=value.device, dtype=value.dtype))


def _to_device(
    batch: dict[str, Any],
    device: torch.device,
    *,
    channels_last: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    if channels_last and images.ndim == 4 and images.shape[1] in {1, 3, 4}:
        images = images.contiguous(memory_format=torch.channels_last)
    return images, masks


def _average_logs(totals: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in totals.items()}


def _nonfinite_gradient_names(model: torch.nn.Module, limit: int = 8) -> list[str]:
    names: list[str] = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            names.append(name)
            if len(names) >= limit:
                break
    return names


def _clip_gradients(
    model: torch.nn.Module,
    *,
    max_norm: float,
    mode: str = "norm",
) -> list[str]:
    if max_norm <= 0:
        return []
    if mode == "value":
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-max_norm, max_norm)
        return []
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, error_if_nonfinite=True)
    except RuntimeError as exc:
        return _nonfinite_gradient_names(model) or [str(exc)]
    return []


def _resize_training_pair(
    images: torch.Tensor,
    masks: torch.Tensor,
    size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize images smoothly while preserving binary segmentation targets."""

    if images.shape[-2:] == size:
        return images, masks
    return (
        F.interpolate(images, size=size, mode="bilinear", align_corners=False),
        F.interpolate(masks, size=size, mode="nearest"),
    )


def _forward_for_training(
    model: torch.nn.Module,
    images: torch.Tensor,
    criterion: GeoFSSLoss,
) -> dict[str, torch.Tensor]:
    if criterion.routing_weight <= 0.0:
        return model(images)
    if not bool(getattr(unwrap_model(model), "supports_routing_loss", False)):
        raise ValueError("routing_weight > 0 requires a model with differentiable routing outputs.")
    return model(images, return_routing=True)


def _multiply_gradients(model: torch.nn.Module, factor: float) -> None:
    if factor == 1.0:
        return
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(factor)


def _perform_optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scaler: GradScaler | None,
    amp: bool,
    grad_clip: float,
    grad_clip_mode: str,
    epoch: int,
    step: int,
    progress: Any,
    ema: ModelEMA | None,
) -> bool:
    if amp and scaler is not None:
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            bad_names = _clip_gradients(model, max_norm=grad_clip, mode=grad_clip_mode)
            if bad_names:
                rank = dist.get_rank() if is_dist_avail_and_initialized() else 0
                progress.write(
                    f"[rank {rank}] skipped non-finite gradient batch at epoch={epoch}, step={step}; "
                    f"first_bad_params={bad_names}"
                )
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                return False
        scaler.step(optimizer)
        scaler.update()
    else:
        bad_names = _clip_gradients(model, max_norm=grad_clip, mode=grad_clip_mode)
        if bad_names:
            rank = dist.get_rank() if is_dist_avail_and_initialized() else 0
            progress.write(
                f"[rank {rank}] skipped non-finite gradient batch at epoch={epoch}, step={step}; "
                f"first_bad_params={bad_names}"
            )
            optimizer.zero_grad(set_to_none=True)
            return False
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if ema is not None:
        ema.update(model)
    return True


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: GeoFSSLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    amp: bool = True,
    grad_clip: float = 1.0,
    grad_clip_mode: str = "norm",
    multi_scale_rates: list[float] | None = None,
    train_size: int = 352,
    epoch: int = 1,
    limit_batches: int | None = None,
    monitor: TrainMonitor | None = None,
    ema: ModelEMA | None = None,
    grad_accum_steps: int = 1,
    channels_last: bool = False,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    size_rates = multi_scale_rates or [1.0]
    if not size_rates:
        raise ValueError("multi_scale_rates must contain at least one scale.")
    grad_accum_steps = max(1, int(grad_accum_steps))
    total_steps = len(loader)
    if limit_batches is not None:
        total_steps = min(total_steps, max(0, int(limit_batches)))
    progress = tqdm(loader, desc=f"train {epoch}", ncols=100, disable=not is_main_process())
    optimizer.zero_grad(set_to_none=True)
    accumulated_batches = 0
    for step, batch in enumerate(progress, start=1):
        if step > total_steps:
            break
        images, masks = _to_device(batch, device, channels_last=channels_last)
        batch_loss_value = 0.0
        batch_logs: dict[str, float] = {}
        for rate in size_rates:
            trainsize = int(round(train_size * rate / 32) * 32)
            trainsize = max(32, trainsize)
            scaled_images, scaled_masks = _resize_training_pair(
                images,
                masks,
                (trainsize, trainsize),
            )

            with autocast(enabled=amp and device.type == "cuda"):
                outputs = _forward_for_training(model, scaled_images, criterion)
                loss, logs = criterion(outputs, scaled_masks)

            if monitor and is_main_process() and rate == 1.0:
                warnings = monitor.check_loss(loss, step)
                if warnings:
                    raise FloatingPointError("; ".join(warnings))

            normalized_loss = loss / (grad_accum_steps * len(size_rates))
            if amp and scaler is not None:
                scaler.scale(normalized_loss).backward()
            else:
                normalized_loss.backward()
            batch_loss_value += float(loss.item()) / len(size_rates)
            for key, value in logs.items():
                batch_logs[key] = batch_logs.get(key, 0.0) + float(value.item()) / len(size_rates)

        batch_size = images.shape[0]
        count += batch_size
        for key, value in batch_logs.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size

        accumulated_batches += 1
        if accumulated_batches == grad_accum_steps or step == total_steps:
            if accumulated_batches < grad_accum_steps:
                # Losses were divided by the full accumulation size. Correct
                # the final partial window so it remains an average, not a
                # smaller update that silently underweights the tail batches.
                _multiply_gradients(model, grad_accum_steps / accumulated_batches)
            _perform_optimizer_step(
                model,
                optimizer,
                scaler=scaler,
                amp=amp and device.type == "cuda",
                grad_clip=grad_clip,
                grad_clip_mode=grad_clip_mode,
                epoch=epoch,
                step=step,
                progress=progress,
                ema=ema,
            )
            accumulated_batches = 0

        progress.set_postfix(loss=f"{batch_loss_value:.4f}")

    if is_dist_avail_and_initialized():
        keys = sorted(totals)
        payload = torch.tensor([totals[key] for key in keys] + [float(count)], device=device)
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        count = int(payload[-1].item())
        totals = {key: float(payload[idx].item()) for idx, key in enumerate(keys)}
    return _average_logs(totals, count)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loaders: dict[str, torch.utils.data.DataLoader],
    criterion: GeoFSSLoss,
    device: torch.device,
    amp: bool = True,
    limit_batches: int | None = None,
    channels_last: bool = False,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    model.eval()
    overall = MetricAccumulator()
    per_dataset: dict[str, dict[str, float]] = {}

    for name, loader in loaders.items():
        acc = MetricAccumulator()
        progress = tqdm(loader, desc=f"valid {name}", ncols=100, disable=not is_main_process())
        for step, batch in enumerate(progress, start=1):
            if limit_batches is not None and step > limit_batches:
                break
            images, masks = _to_device(batch, device, channels_last=channels_last)
            with autocast(enabled=amp and device.type == "cuda"):
                outputs = model(images)
                _loss, _ = criterion(outputs, masks)
            logits = outputs["pred_final"]
            metrics = compute_binary_metrics(logits, masks)
            batch_size = images.shape[0]
            acc.update(metrics, n=batch_size)
            overall.update(metrics, n=batch_size)
            progress.set_postfix(dice=f"{metrics['dice']:.4f}")
        per_dataset[name] = acc.compute()

    return overall.compute(), per_dataset


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: GradScaler | None,
    epoch: int,
    best_dice: float,
    best_epoch: int,
    config: dict[str, Any],
    ema: ModelEMA | None = None,
    model_state: dict[str, torch.Tensor] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_dice": best_dice,
            "best_epoch": best_epoch,
            "model": model_state if model_state is not None else unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "ema": ema.state_dict() if ema else None,
            "config": config,
        },
        path,
    )


def _linear_schedule(
    epoch: int,
    *,
    start_epoch: int,
    end_epoch: int,
    start_value: float,
    end_value: float,
) -> float:
    """Inclusive, restart-safe linear schedule evaluated once per epoch."""

    if end_epoch < start_epoch:
        raise ValueError("schedule end_epoch must be >= start_epoch.")
    if epoch <= start_epoch:
        return float(start_value)
    if epoch >= end_epoch:
        return float(end_value)
    progress = (epoch - start_epoch) / max(1, end_epoch - start_epoch)
    return float(start_value + progress * (end_value - start_value))


def apply_epoch_schedules(
    model: torch.nn.Module,
    criterion: GeoFSSLoss,
    config: dict[str, Any],
    epoch: int,
) -> dict[str, float]:
    """Apply residual, deep-supervision, and boundary schedules for one epoch."""

    train_cfg = config.get("train", {})
    values: dict[str, float] = {}

    residual_cfg = train_cfg.get("residual_warmup", {})
    if bool(residual_cfg.get("enabled", False)):
        factor = _linear_schedule(
            epoch,
            start_epoch=int(residual_cfg.get("start_epoch", 1)),
            end_epoch=int(residual_cfg.get("end_epoch", 30)),
            start_value=float(residual_cfg.get("start_factor", 0.0)),
            end_value=float(residual_cfg.get("end_factor", 1.0)),
        )
        if not 0.0 <= factor <= 1.0:
            raise ValueError("residual warm-up factor must remain in [0, 1].")
        setter = getattr(unwrap_model(model), "set_residual_warmup_factor", None)
        if setter is not None:
            setter(factor)
        values["schedule/residual_factor"] = factor

    ds_cfg = train_cfg.get("dynamic_deep_supervision", {})
    if bool(ds_cfg.get("enabled", False)):
        auxiliary_factor = _linear_schedule(
            epoch,
            start_epoch=int(ds_cfg.get("decay_start_epoch", 80)),
            end_epoch=int(ds_cfg.get("decay_end_epoch", 160)),
            start_value=float(ds_cfg.get("start_factor", 1.0)),
            end_value=float(ds_cfg.get("end_factor", 0.0)),
        )
        final_name = str(ds_cfg.get("final_output", "pred_final"))
        scheduled_weights = {
            name: weight if name == final_name else weight * auxiliary_factor
            for name, weight in criterion.base_ds_weights.items()
        }
        normalize = bool(config.get("model", {}).get("decoder", {}).get("normalize_ds_weights", True))
        criterion.set_ds_weights(scheduled_weights, normalize=normalize)
        values["schedule/deep_supervision_factor"] = auxiliary_factor
        values["schedule/final_loss_weight"] = float(criterion.ds_weights.get(final_name, 0.0))

    boundary_cfg = train_cfg.get("boundary_decay", {})
    if bool(boundary_cfg.get("enabled", False)):
        boundary_weight = _linear_schedule(
            epoch,
            start_epoch=int(boundary_cfg.get("decay_start_epoch", 80)),
            end_epoch=int(boundary_cfg.get("decay_end_epoch", 180)),
            start_value=float(boundary_cfg.get("start_weight", criterion.boundary_weight)),
            end_value=float(boundary_cfg.get("end_weight", 0.0)),
        )
        criterion.set_boundary_weight(boundary_weight)
        values["schedule/boundary_weight"] = boundary_weight

    return values


def fit(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loaders: dict[str, torch.utils.data.DataLoader],
    criterion: GeoFSSLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    epochs: int,
    exp_dir: str | Path,
    config: dict[str, Any],
    amp: bool = True,
    grad_clip: float = 1.0,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
    start_epoch: int = 1,
    initial_best_dice: float = 0.0,
    initial_best_epoch: int = 0,
    scaler_state: dict[str, Any] | None = None,
    ema_state: dict[str, Any] | None = None,
) -> dict[str, float]:
    exp_dir = Path(exp_dir)
    ckpt_dir = exp_dir / "checkpoints"
    logger = RichLogger()
    tb = TensorBoardLogger(exp_dir / "logs" / "tensorboard", enabled=is_main_process())
    results_logger = YoloResultsLogger(exp_dir, enabled=is_main_process())
    monitor = TrainMonitor(
        patience=int(config.get("train", {}).get("early_stop_patience", 20)),
        min_delta=float(config.get("train", {}).get("early_stop_min_delta", 1e-5)),
        best_dice=initial_best_dice,
        best_epoch=initial_best_epoch,
    )
    scaler = GradScaler(enabled=amp and device.type == "cuda") if amp else None
    if scaler is not None and scaler_state:
        scaler.load_state_dict(scaler_state)
    ema_cfg = config.get("train", {}).get("ema", {})
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema_decay = float(ema_cfg.get("decay", 0.999))
    ema_eval_start_epoch = int(ema_cfg.get("eval_start_epoch", 30))
    ema = ModelEMA(model, decay=ema_decay) if ema_enabled else None
    if ema is not None and ema_state:
        ema.load_state_dict(ema_state)
    ema_best_dice = 0.0
    ema_best_epoch = 0
    train_cfg = config.get("train", {})
    multi_scale_rates = [float(rate) for rate in train_cfg.get("multi_scale_rates", [1.0])]
    train_size = int(config.get("input", {}).get("size", train_cfg.get("input_size", 352)))
    grad_clip_mode = str(train_cfg.get("gradient_clip_mode", "norm"))
    scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()
    channels_last = bool(train_cfg.get("channels_last", False))
    # Model is already placed on device (and optionally channels_last) in scripts/train.py.
    # Keep a no-op .to(device) here so resume / DDP paths stay consistent.
    model.to(device)
    image_interval = int(config.get("logging", {}).get("image_interval", 10))
    save_val_batches = bool(config.get("logging", {}).get("save_val_batches", True))
    for epoch in range(start_epoch, epochs + 1):
        schedule_values = apply_epoch_schedules(model, criterion, config, epoch)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        start = time.time()
        train_losses = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            amp=amp,
            grad_clip=grad_clip,
            grad_clip_mode=grad_clip_mode,
            multi_scale_rates=multi_scale_rates,
            train_size=train_size,
            epoch=epoch,
            limit_batches=limit_train_batches,
            monitor=monitor,
            ema=ema,
            grad_accum_steps=max(1, int(train_cfg.get("gradient_accum_steps", 1))),
            channels_last=channels_last,
        )
        train_losses.update(schedule_values)
        if is_dist_avail_and_initialized():
            dist.barrier()

        # Epoch Dice on the merged eval pool selects best.pt (EMA when enabled).
        val_metrics: dict[str, float] = {}
        per_dataset: dict[str, dict[str, float]] = {}
        if is_main_process():
            val_metrics, per_dataset = validate(
                model,
                val_loaders,
                criterion,
                device,
                amp=amp,
                limit_batches=limit_val_batches,
                channels_last=channels_last,
            )
        if is_dist_avail_and_initialized():
            dist.barrier()

        ema_metrics: dict[str, float] = {}
        ema_per_dataset: dict[str, dict[str, float]] = {}
        if ema is not None and is_main_process() and epoch >= ema_eval_start_epoch:
            with ema.apply_to(model) as ema_model:
                ema_metrics, ema_per_dataset = validate(
                    ema_model,
                    val_loaders,
                    criterion,
                    device,
                    amp=amp,
                    limit_batches=limit_val_batches,
                    channels_last=channels_last,
                )

        scheduler_metric = val_metrics.get("dice", 0.0) if is_main_process() else 0.0
        if is_dist_avail_and_initialized():
            metric_tensor = torch.tensor([float(scheduler_metric)], device=device)
            dist.broadcast(metric_tensor, src=0)
            scheduler_metric = float(metric_tensor.item())
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(scheduler_metric)
            elif scheduler_name in {"nnunet_poly", "nnunet-polylr", "poly"}:
                scheduler.step(epoch)
            else:
                scheduler.step()

        should_stop = False
        if is_main_process():
            improved, warnings = monitor.update_validation(val_metrics.get("dice", 0.0), epoch)
            warnings.extend(monitor.check_system(optimizer))
            # Formal artifacts: only best.pt / last.pt. With EMA on, both store EMA shadow
            # weights; best.pt is selected by EMA eval-pool Dice (not raw model Dice).
            ema_improved = False
            if ema is not None and ema_metrics:
                ema_dice = float(ema_metrics.get("dice", 0.0))
                ema_improved = ema_dice > ema_best_dice + monitor.min_delta
                if ema_improved:
                    ema_best_dice = ema_dice
                    ema_best_epoch = epoch
                ema_shadow = ema.state_dict()["shadow"]
                save_checkpoint(
                    ckpt_dir / "last.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    ema_best_dice,
                    ema_best_epoch,
                    config,
                    ema=ema,
                    model_state=ema_shadow,
                )
                if ema_improved:
                    save_checkpoint(
                        ckpt_dir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        ema_best_dice,
                        ema_best_epoch,
                        config,
                        ema=ema,
                        model_state=ema_shadow,
                    )
            else:
                save_checkpoint(
                    ckpt_dir / "last.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    monitor.best_dice,
                    monitor.best_epoch,
                    config,
                    ema=ema,
                )
                if improved:
                    save_checkpoint(
                        ckpt_dir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        monitor.best_dice,
                        monitor.best_epoch,
                        config,
                        ema=ema,
                    )

            tb.log_scalars("Loss/Train", train_losses, epoch)
            tb.log_scalars("Metrics/Test", val_metrics, epoch)
            if ema_metrics:
                tb.log_scalars("Metrics/Test_EMA", ema_metrics, epoch)
            tb.log_lr(optimizer, epoch)
            if val_loaders and (epoch == 1 or (image_interval > 0 and epoch % image_interval == 0)):
                first_loader = next(iter(val_loaders.values()))
                sample = next(iter(first_loader))
                sample_images, sample_masks = _to_device(
                    sample, device, channels_last=channels_last
                )
                eval_model = model
                eval_model.eval()
                with torch.no_grad(), autocast(enabled=amp and device.type == "cuda"):
                    sample_outputs = eval_model(sample_images)
                tb.log_prediction_grid(sample_images, sample_masks, sample_outputs["pred_final"], epoch)
                if save_val_batches:
                    results_logger.save_val_batch(sample_images, sample_masks, sample_outputs["pred_final"], epoch)
            results_logger.log_epoch(
                epoch,
                train_losses=train_losses,
                val_metrics=val_metrics,
                per_dataset=per_dataset,
                lr=optimizer_reference_lr(optimizer),
                epoch_time=time.time() - start,
            )
            report_best = ema_best_dice if (ema is not None and ema_best_epoch > 0) else monitor.best_dice
            logger.epoch_summary(
                epoch=epoch,
                epochs=epochs,
                train_losses=train_losses,
                val_metrics=val_metrics,
                best_dice=report_best,
                lr=optimizer_reference_lr(optimizer),
                epoch_time=time.time() - start,
                warnings=warnings,
            )
            logger.dataset_metrics(per_dataset)
            if ema_metrics:
                logger.log(f"EMA mean Dice: {ema_metrics.get('dice', 0.0):.4f} | EMA best: {ema_best_dice:.4f} @ epoch {ema_best_epoch}")
                logger.dataset_metrics(ema_per_dataset)
            # Default off: train fixed epochs; best.pt is still the max-Dice epoch (EMA when on).
            early_stop_enabled = bool(config.get("train", {}).get("early_stop_enabled", False))
            if ema_improved:
                monitor.stagnant_epochs = 0
            should_stop = early_stop_enabled and monitor.stagnant_epochs >= monitor.patience
            if should_stop:
                logger.log(f"Early stopping at epoch {epoch}: test Dice did not improve for {monitor.patience} epochs.")

        if is_dist_avail_and_initialized():
            stop_tensor = torch.tensor([int(should_stop)], device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())
        if should_stop:
            break
    tb.close()
    return {"best_dice": monitor.best_dice, "best_epoch": float(monitor.best_epoch)}
