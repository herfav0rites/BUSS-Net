"""C01 nnU-Net baseline built from the official MIC-DKFZ sources.

The project deliberately fixes the architecture so C01 and the derived C00
share the same six-stage 2D U-Net topology.  It does not claim to run the full
nnU-Net experiment planner, preprocessing or post-processing pipeline.
"""

from __future__ import annotations

import sys
import importlib.util
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from ._adapt import single_scale
from ._vendor import vendor_root


# The official nnU-Net planner uses 32 base features and caps 2-D networks at
# 512 features. The previous adapter accidentally used the 3-D cap of 320.
NNUNET_FEATURES = (32, 64, 128, 256, 512, 512)
NNUNET_N_STAGES = len(NNUNET_FEATURES)
NNUNET_KERNEL_SIZES = ((3, 3),) * NNUNET_N_STAGES
NNUNET_STRIDES = ((1, 1), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2))
NNUNET_CONVS_PER_STAGE = (2,) * NNUNET_N_STAGES
NNUNET_CONVS_PER_DECODER_STAGE = (2,) * (NNUNET_N_STAGES - 1)


class NNUNetBaseline(nn.Module):
    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out_size = x.shape[2:]
        logits = self.core(x)
        if isinstance(logits, (tuple, list)):
            names = ("pred_final", "pred_ds1", "pred_ds2", "pred_ds3", "pred_ds4")
            return {
                name: F.interpolate(value, size=out_size, mode="bilinear", align_corners=False)
                if value.shape[-2:] != out_size
                else value
                for name, value in zip(names, logits)
            }
        if isinstance(logits, dict):
            logits = logits.get("pred_final", next(iter(logits.values())))
        return single_scale(logits, out_size, output_type="logits")


def ensure_nnunet_sources() -> tuple[object, object]:
    """Put the official nnU-Net and network-architecture sources on ``sys.path``."""

    root = vendor_root("nnunet")
    nnunet_pkg = root / "nnunetv2"
    installed_nnunet = importlib.util.find_spec("nnunetv2") is not None
    if not nnunet_pkg.is_dir() and not installed_nnunet:
        raise FileNotFoundError(
            f"nnUNet source missing at {root}. "
            "Clone https://github.com/MIC-DKFZ/nnUNet into experiments/comparisons/C01_nnunet/vendor/nnUNet."
        )
    dna_root = root.parent / "dynamic-network-architectures"
    dna_pkg = dna_root / "dynamic_network_architectures"
    installed_dna = importlib.util.find_spec("dynamic_network_architectures") is not None
    if not dna_pkg.is_dir() and not installed_dna:
        raise FileNotFoundError(
            f"dynamic-network-architectures source missing at {dna_root}. "
            "Clone https://github.com/MIC-DKFZ/dynamic-network-architectures there."
        )
    for source_root in (root, dna_root):
        if not source_root.is_dir():
            continue
        source_str = str(source_root)
        if source_str not in sys.path:
            sys.path.insert(0, source_str)
    return root, dna_root


def plainconv_architecture_kwargs() -> dict[str, object]:
    """Return the topology shared by C01 and C00."""

    ensure_nnunet_sources()
    from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm  # type: ignore[import-untyped]

    conv_op = nn.Conv2d
    return {
        "input_channels": 3,
        "n_stages": NNUNET_N_STAGES,
        "features_per_stage": list(NNUNET_FEATURES),
        "conv_op": conv_op,
        "kernel_sizes": [list(v) for v in NNUNET_KERNEL_SIZES],
        "strides": [list(v) for v in NNUNET_STRIDES],
        "n_conv_per_stage": list(NNUNET_CONVS_PER_STAGE),
        "conv_bias": True,
        "norm_op": get_matching_instancenorm(conv_op),
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": nn.LeakyReLU,
        "nonlin_kwargs": {"inplace": True},
        "nonlin_first": False,
    }


def _build_plainconv_unet(num_classes: int, *, deep_supervision: bool = False) -> nn.Module:
    ensure_nnunet_sources()
    from dynamic_network_architectures.architectures.unet import PlainConvUNet  # type: ignore[import-untyped]

    return PlainConvUNet(
        **plainconv_architecture_kwargs(),
        num_classes=num_classes,
        n_conv_per_stage_decoder=list(NNUNET_CONVS_PER_DECODER_STAGE),
        deep_supervision=deep_supervision,
    )


def build(config: dict[str, Any] | None = None) -> NNUNetBaseline:
    config = config or {}
    model_cfg = config.get("model", config)
    num_classes = int(model_cfg.get("num_classes", 1))
    decoder_cfg = model_cfg.get("decoder", {})
    core = _build_plainconv_unet(
        num_classes,
        deep_supervision=bool(decoder_cfg.get("deep_supervision", False)),
    )
    return NNUNetBaseline(core)
