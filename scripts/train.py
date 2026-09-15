#!/usr/bin/env python3
"""Train BUSS-Net or a configured comparison model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from buss_net.datasets import PolypDataset, build_test_transform, build_train_transform
from buss_net.datasets.splits import load_eval_split_manifest, resolve_eval_split_path, stems_by_dataset_map
from buss_net.engine import fit
from buss_net.engine.train import unwrap_model
from buss_net.losses import GeoFSSLoss
from buss_net.models import build_model
from buss_net.utils.config import load_yaml
from buss_net.utils.reproducibility import (
    DEFAULT_SEED,
    cudnn_flags,
    make_worker_init_fn,
    prepare_experiment_dir,
    set_seed,
)


def build_loaders(
    config: dict[str, Any], device: torch.device, seed: int
) -> tuple[DataLoader, dict[str, DataLoader]]:
    train_cfg = config["train"]
    dataset_cfg = config["dataset"]
    size = int(train_cfg.get("input_size", 352))
    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(train_cfg.get("num_workers", 4))
    manifest = load_eval_split_manifest(resolve_eval_split_path(dataset_cfg, ROOT))
    eval_stems = stems_by_dataset_map(manifest, "test")
    worker_init = make_worker_init_fn(seed) if num_workers > 0 else None

    train_dataset = PolypDataset(
        ROOT / dataset_cfg["train_image_dir"],
        ROOT / dataset_cfg["train_mask_dir"],
        transform=build_train_transform(size),
        size=size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init,
    )
    test_root = ROOT / dataset_cfg.get("test_root", "data/processed/test")
    eval_loaders: dict[str, DataLoader] = {}
    for name in sorted(eval_stems):
        stems = eval_stems[name]
        if not stems:
            continue
        dataset = PolypDataset(
            test_root / name / "image",
            test_root / name / "mask",
            transform=build_test_transform(size),
            size=size,
            include_stems=stems,
        )
        eval_loaders[name] = DataLoader(
            dataset,
            batch_size=max(1, min(batch_size, 8)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init,
        )
    return train_loader, eval_loaders


def build_criterion(config: dict[str, Any]) -> GeoFSSLoss:
    loss_cfg = config.get("loss", {})
    structure_cfg = loss_cfg.get("structure", {})
    boundary_cfg = loss_cfg.get("boundary", {})
    routing_cfg = loss_cfg.get("routing", {})
    model_cfg = config.get("model", {})
    decoder_cfg = model_cfg.get("decoder", {})
    uggd_cfg = model_cfg.get("uggd", {})
    raw_weights = decoder_cfg.get("ds_weights", {"pred_final": 1.0})
    if not bool(decoder_cfg.get("deep_supervision", False)):
        weights = {"pred_final": 1.0}
    elif isinstance(raw_weights, dict):
        weights = {str(name): float(weight) for name, weight in raw_weights.items()}
    else:
        weights = {
            "pred_final": float(raw_weights[0]),
            "pred_s4": float(raw_weights[1] if len(raw_weights) > 1 else 0.4),
        }
    if bool(decoder_cfg.get("deep_supervision", False)) and bool(uggd_cfg.get("enabled", False)):
        auxiliary_weight = float(uggd_cfg.get("auxiliary_loss_weight", 0.2))
        decoder_indices = uggd_cfg.get("decoder_indices", (1, 2, 3, 4))
        for position in range(1, len(decoder_indices) + 1):
            weights.setdefault(f"pred_uggd_d{position}", auxiliary_weight)
        if bool(uggd_cfg.get("detach_uncertainty", True)) and not any(
            weights.get(f"pred_uggd_d{position}", 0.0) > 0.0
            for position in range(1, len(decoder_indices) + 1)
        ):
            raise ValueError(
                "UG-GD with detach_uncertainty=true requires at least one positive "
                "pred_uggd_d* auxiliary-loss weight."
            )
    criterion = GeoFSSLoss(
        ds_weights=weights,
        bce_weight=float(structure_cfg.get("bce_weight", 1.0)),
        dice_weight=float(structure_cfg.get("dice_weight", 1.0)),
        boundary_weight=float(boundary_cfg.get("weight", 0.0)),
        boundary_kernel_size=int(boundary_cfg.get("kernel_size", 3)),
        routing_weight=float(routing_cfg.get("weight", 0.0)),
    )
    criterion.set_ds_weights(
        weights,
        normalize=bool(decoder_cfg.get("normalize_ds_weights", True)),
    )
    return criterion


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config["train"]
    name = str(train_cfg.get("optimizer", "AdamW")).lower()
    base_lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    differential_cfg = train_cfg.get("differential_lr", {})
    differential_enabled = bool(differential_cfg.get("enabled", False))
    default_multiplier = float(differential_cfg.get("default_multiplier", 1.0))
    if default_multiplier <= 0.0:
        raise ValueError("train.differential_lr.default_multiplier must be positive.")
    configured_groups = differential_cfg.get("groups", {})
    if differential_enabled and not isinstance(configured_groups, dict):
        raise ValueError("train.differential_lr.groups must be a mapping.")

    group_rules: list[tuple[str, tuple[str, ...], float]] = []
    if differential_enabled:
        for group_name, group_cfg in configured_groups.items():
            if not isinstance(group_cfg, dict):
                raise ValueError(f"differential LR group {group_name!r} must be a mapping.")
            prefixes = tuple(str(value) for value in group_cfg.get("prefixes", ()))
            multiplier = float(group_cfg.get("multiplier", 1.0))
            if not prefixes:
                raise ValueError(f"differential LR group {group_name!r} has no prefixes.")
            if multiplier <= 0.0:
                raise ValueError(f"differential LR multiplier for {group_name!r} must be positive.")
            group_rules.append((str(group_name), prefixes, multiplier))

    grouped: dict[tuple[str, float, bool], list[torch.nn.Parameter]] = {}
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr_group, multiplier = "base", default_multiplier
        for candidate_name, prefixes, candidate_multiplier in group_rules:
            if parameter_name.startswith(prefixes):
                lr_group, multiplier = candidate_name, candidate_multiplier
                break
        no_decay = bool(getattr(parameter, "_no_weight_decay", False))
        grouped.setdefault((lr_group, multiplier, no_decay), []).append(parameter)

    parameters: list[dict[str, Any]] = []
    for (group_name, multiplier, no_decay), group_parameters in grouped.items():
        parameters.append(
            {
                "params": group_parameters,
                "lr": base_lr * multiplier,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "group_name": group_name,
                "lr_multiplier": multiplier,
            }
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=base_lr,
            momentum=float(train_cfg.get("momentum", 0.9)),
            nesterov=bool(train_cfg.get("nesterov", False)),
        )
    if name == "adam":
        return torch.optim.Adam(parameters, lr=base_lr)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=base_lr)
    raise ValueError(f"Unsupported optimizer: {train_cfg.get('optimizer')}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler | None:
    train_cfg = config["train"]
    name = str(train_cfg.get("scheduler", "none")).lower()
    epochs = int(train_cfg.get("epochs", 100))
    if name in {"nnunet_poly", "nnunet-polylr", "poly"}:
        exponent = float(train_cfg.get("poly_exponent", 0.9))
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: max(0.0, 1.0 - float(epoch) / max(epochs, 1)) ** exponent,
        )
    if name in {"cosine", "cosineannealing"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name in {"none", "constant"}:
        return None
    raise ValueError(f"Unsupported scheduler: {train_cfg.get('scheduler')}")


def apply_runtime_accelerators(
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Optionally convert convolutions to channels_last and wrap with torch.compile."""

    train_cfg = config.get("train", {})
    model = model.to(device)

    channels_last = bool(train_cfg.get("channels_last", False))
    if channels_last and device.type == "cuda":
        # Only 4-D conv / norm weights accept channels_last; Linear/1-D params stay NCHW.
        converted = 0
        for module in model.modules():
            if isinstance(
                module,
                (
                    torch.nn.Conv2d,
                    torch.nn.ConvTranspose2d,
                    torch.nn.BatchNorm2d,
                    torch.nn.InstanceNorm2d,
                ),
            ):
                module.to(memory_format=torch.channels_last)
                converted += 1
        print(f"runtime=channels_last conv_modules={converted}", flush=True)

    compile_cfg = train_cfg.get("compile", {})
    compile_enabled = bool(compile_cfg) if isinstance(compile_cfg, bool) else bool(
        compile_cfg.get("enabled", False)
    )
    if not compile_enabled or device.type != "cuda":
        return model
    if isinstance(compile_cfg, bool):
        compile_cfg = {"enabled": True}

    cache_dir = str(
        compile_cfg.get(
            "cache_dir",
            Path.home() / ".cache" / "torchinductor" / "buss_net",
        )
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)
    os.environ.setdefault("TRITON_CACHE_DIR", str(Path(cache_dir) / "triton"))

    mode = str(compile_cfg.get("mode", "default"))
    fullgraph = bool(compile_cfg.get("fullgraph", False))
    dynamic = bool(compile_cfg.get("dynamic", True))
    suppress_errors = bool(compile_cfg.get("suppress_errors", True))
    try:
        import torch._dynamo as dynamo

        if suppress_errors:
            dynamo.config.suppress_errors = True
        compiled = torch.compile(
            model,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        # Force one warm-up so inductor/triton failures surface before training.
        input_size = int(train_cfg.get("input_size", 416))
        warm = torch.zeros(1, 3, input_size, input_size, device=device)
        if channels_last:
            warm = warm.contiguous(memory_format=torch.channels_last)
        model.eval()
        with torch.inference_mode():
            _ = compiled(warm)
        model.train()
        print(
            f"runtime=torch.compile mode={mode} fullgraph={fullgraph} "
            f"dynamic={dynamic} suppress_errors={suppress_errors} cache={cache_dir}",
            flush=True,
        )
        return compiled  # type: ignore[return-value]
    except Exception as exc:  # pragma: no cover - depends on local torch/inductor
        print(f"runtime=torch.compile_failed fallback_eager reason={exc}", flush=True)
        model.train()
        return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a configured segmentation model.")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "comparisons" / "C00_buss-net.yaml"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_yaml(config_path)
    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = args.epochs
    config.setdefault("experiment", {})["training_protocol"] = "v2_multiscale_mean_nearest_mask"
    seed = int(config.get("reproducibility", {}).get("seed", DEFAULT_SEED))
    benchmark, deterministic = cudnn_flags(config)
    set_seed(seed, cudnn_benchmark=benchmark, cudnn_deterministic=deterministic)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    exp_dir, removed_artifacts = prepare_experiment_dir(
        ROOT / config["output"]["exp_dir"],
        resume=args.resume is not None,
    )
    if removed_artifacts:
        print(
            "fresh_run_overwrite=" + ",".join(path.name for path in removed_artifacts),
            flush=True,
        )
    config["output"]["exp_dir"] = str(exp_dir.relative_to(ROOT))
    (exp_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    train_loader, eval_loaders = build_loaders(config, device, seed)
    model = build_model(config)
    model = apply_runtime_accelerators(model, config, device)
    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    start_epoch, best_dice, best_epoch = 1, 0.0, 0
    scaler_state = None
    ema_state = None
    if args.resume is not None:
        resume_path = args.resume if args.resume.is_absolute() else ROOT / args.resume
        checkpoint = torch.load(resume_path, map_location=device)
        # Checkpoints store unwrapped weights; torch.compile wraps as OptimizedModule.
        unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_dice = float(checkpoint.get("best_dice", 0.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        checkpoint_semantics = checkpoint.get("checkpoint_semantics")
        if checkpoint_semantics == "ema":
            if not bool(config.get("train", {}).get("ema", {}).get("enabled", False)):
                raise ValueError(
                    "Cannot resume an EMA-only checkpoint with train.ema.enabled=false."
                )
            ema_state = {
                "decay": float(checkpoint.get("ema_decay", config["train"]["ema"].get("decay", 0.999))),
                "shadow": checkpoint["model"],
            }
            print(
                "resume_mode=ema_warm_start (optimizer, scheduler, and scaler state are intentionally reset)",
                flush=True,
            )
        else:
            # Backward compatibility for legacy raw checkpoints.
            if checkpoint.get("optimizer"):
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scheduler is not None and checkpoint.get("scheduler"):
                scheduler.load_state_dict(checkpoint["scheduler"])
            scaler_state = checkpoint.get("scaler")
            ema_state = checkpoint.get("ema")

    train_cfg = config["train"]
    print(
        f"model={config['model']['name']} device={device} exp_dir={exp_dir.relative_to(ROOT)} "
        f"params={sum(parameter.numel() for parameter in model.parameters()):,}",
        flush=True,
    )
    optimizer_summary = ", ".join(
        f"{group.get('group_name', 'base')}:{group['lr']:.3g}"
        f"/wd={group['weight_decay']:.3g}/n={sum(p.numel() for p in group['params']):,}"
        for group in optimizer.param_groups
    )
    print(f"optimizer_groups={optimizer_summary}", flush=True)
    result = fit(
        model=model,
        train_loader=train_loader,
        val_loaders=eval_loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=int(train_cfg.get("epochs", 100)),
        exp_dir=exp_dir,
        config=config,
        amp=bool(train_cfg.get("amp", False)),
        grad_clip=float(train_cfg.get("gradient_clip", 0.0)),
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        start_epoch=start_epoch,
        initial_best_dice=best_dice,
        initial_best_epoch=best_epoch,
        scaler_state=scaler_state,
        ema_state=ema_state,
    )
    print(f"Training done: best_dice={result['best_dice']:.4f} @ epoch {int(result['best_epoch'])}")


if __name__ == "__main__":
    main()
