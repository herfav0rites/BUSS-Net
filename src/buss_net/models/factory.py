"""Model factory for the minimal BUSS-Net release."""

from __future__ import annotations

from typing import Any

from torch import nn

_RANDOM_ONLY = {
    "buss_net",
    "bussnet",
    "geofss_net",  # legacy alias
    "geofssnet",  # legacy alias
}


def _normalize_model_name(name: str) -> str:
    return str(name).lower().replace("-", "_").replace(" ", "")


def _enforce_initialization_policy(config: dict[str, Any], model_name: str) -> None:
    """Enforce the C01-aligned initialization policy: random only, no external weights."""

    model_cfg = config.get("model", config)
    init_cfg = model_cfg.get("initialization", {})
    if not isinstance(init_cfg, dict):
        raise ValueError("model.initialization must be a mapping.")
    mode = str(init_cfg.get("mode", "random")).lower()
    allow_external = bool(init_cfg.get("allow_external_weights", False))

    if mode != "random" or allow_external:
        raise ValueError(
            f"{model_name} requires random initialization "
            "(no external pretrained weights) under the C01-aligned protocol."
        )
    if model_cfg.get("warm_start"):
        raise ValueError("model.warm_start is disabled under the random-init protocol.")
    if model_name not in _RANDOM_ONLY:
        # Still refuse external weights for unknown names.
        return


def build_model(config: dict[str, Any] | None = None) -> nn.Module:
    config = config or {}
    model_cfg = config.get("model", config)
    name = _normalize_model_name(model_cfg.get("name", "buss-net"))
    _enforce_initialization_policy(config, name)

    if name in {"buss_net", "bussnet", "geofss_net", "geofssnet"}:
        from .buss_net import build_buss_net

        return build_buss_net(config)

    raise ValueError(f"Unknown model name: {name}")
