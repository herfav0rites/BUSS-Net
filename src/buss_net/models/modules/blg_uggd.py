"""BLG and UG-GD modules used by BUSS-Net.

BLG refines a semantic bottleneck using mid-level boundary guidance. UG-GD
gathers multi-stage features, estimates prediction disagreement, and writes
scale-specific residuals back under uncertainty control.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn
import torch.nn.functional as F


def _zero_conv(conv: nn.Conv2d) -> None:
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def _near_zero_conv(conv: nn.Conv2d, std: float = 1e-3) -> None:
    """Initialize a residual projection near zero without blocking gradients."""

    nn.init.normal_(conv.weight, mean=0.0, std=std)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class LocalGeometryBranch(nn.Module):
    """Large/dilated depthwise convolutions for local contour geometry."""

    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=5,
                padding=2,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )
        self.project = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        _near_zero_conv(self.project)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.local(self.reduce(x)))


class GlobalContinuityBranch(nn.Module):
    """Lightweight bidirectional row/column cumulative scanner.

    Each position receives prefix and suffix context along both spatial axes.
    This gives an image-wide receptive field without an optional Mamba or
    Transformer dependency.
    """

    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )
        self.project = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        _near_zero_conv(self.project)

    @staticmethod
    def _bidirectional_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
        length = x.shape[dim]
        shape = [1] * x.ndim
        shape[dim] = length
        counts = torch.arange(1, length + 1, device=x.device, dtype=x.dtype).view(shape)
        prefix = torch.cumsum(x, dim=dim) / counts
        reversed_x = torch.flip(x, dims=(dim,))
        suffix = torch.flip(torch.cumsum(reversed_x, dim=dim) / counts, dims=(dim,))
        return 0.5 * (prefix + suffix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        row_context = self._bidirectional_mean(x, dim=2)
        column_context = self._bidirectional_mean(x, dim=3)
        return self.project(self.mix(0.5 * (row_context + column_context)))


class BoundaryGuidanceHead(nn.Module):
    """Predict boundary logits from a mid-shallow encoder feature."""

    def __init__(self, in_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.head(feature)


class BLG(nn.Module):
    """Boundary-conditioned Local-Global bottleneck (formal name: BLG)."""

    def __init__(
        self,
        channels: int,
        guidance_channels: int,
        hidden_channels: int = 128,
        gate_groups: int = 4,
        residual_scale: float = 1.0,
        use_internal_global: bool = True,
        router_identity_bias: float = 1.0,
    ) -> None:
        super().__init__()
        if channels % gate_groups != 0:
            raise ValueError(f"channels ({channels}) must be divisible by gate_groups ({gate_groups}).")
        if not 0.0 <= residual_scale <= 1.0:
            raise ValueError("residual_scale must be in [0, 1].")

        hidden_channels = min(channels, int(hidden_channels))
        self.channels = int(channels)
        self.gate_groups = int(gate_groups)
        self.router_identity_bias = float(router_identity_bias)
        self.local_branch = LocalGeometryBranch(channels, hidden_channels)
        self.global_branch = (
            GlobalContinuityBranch(channels, hidden_channels)
            if use_internal_global
            else None
        )
        self.boundary_head = BoundaryGuidanceHead(guidance_channels)
        self.local_norm = nn.GroupNorm(1, channels)
        self.global_norm = nn.GroupNorm(1, channels)

        router_in = channels * 3 + 1
        router_hidden = max(32, hidden_channels)
        self.local_context = nn.Conv2d(
            router_in,
            router_in,
            kernel_size=3,
            padding=1,
            groups=router_in,
            bias=True,
        )
        _zero_conv(self.local_context)
        self.router = nn.Sequential(
            nn.Conv2d(router_in, router_hidden, kernel_size=1, bias=False),
            nn.GroupNorm(1, router_hidden),
            nn.GELU(),
            nn.Conv2d(router_hidden, 3 * gate_groups, kernel_size=1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.register_buffer("_residual_warmup_factor", torch.tensor(1.0), persistent=False)
        self._init_router()

    def set_residual_warmup_factor(self, factor: float) -> None:
        if not 0.0 <= factor <= 1.0:
            raise ValueError("residual warm-up factor must be in [0, 1].")
        self._residual_warmup_factor.fill_(float(factor))

    def _init_router(self) -> None:
        final = self.router[-1]
        assert isinstance(final, nn.Conv2d)
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        assert final.bias is not None
        with torch.no_grad():
            final.bias.zero_()
            final.bias[: self.gate_groups].fill_(self.router_identity_bias)

    def forward(
        self,
        x: torch.Tensor,
        guidance_feature: torch.Tensor,
        *,
        external_global_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta_local = self.local_branch(x)
        if external_global_delta is None:
            if self.global_branch is None:
                raise ValueError(
                    "BLG requires external_global_delta when its internal global expert is disabled."
                )
            delta_global = self.global_branch(x)
            external_global = False
        else:
            if external_global_delta.shape != x.shape:
                raise ValueError("external_global_delta must have the same shape as the BLG input.")
            delta_global = external_global_delta
            external_global = True
        boundary_logits = self.boundary_head(guidance_feature)
        boundary = torch.sigmoid(
            F.interpolate(boundary_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        )

        gate_input = torch.cat(
            [x, self.local_norm(delta_local), self.global_norm(delta_global), boundary], dim=1
        )
        gate_input = gate_input + self.local_context(gate_input)
        b, _, h, w = x.shape
        logits = self.router(gate_input).view(b, 3, self.gate_groups, h, w)
        weights = torch.softmax(logits, dim=1)
        group_width = self.channels // self.gate_groups
        weight_local = weights[:, 1].repeat_interleave(group_width, dim=1)
        weight_global = weights[:, 2].repeat_interleave(group_width, dim=1)
        scale = (
            self.residual_scale.clamp(0.0, 1.0)
            * self._residual_warmup_factor
        ).to(dtype=x.dtype)
        local_update = scale * weight_local * delta_local
        # A GeoFSS external expert already carries its own per-channel
        # LayerScale. Do not multiply it by BLG's residual scale a second time.
        global_update = weight_global * delta_global
        if not external_global:
            global_update = scale * global_update
        output = x + local_update + global_update
        return output, boundary_logits, weights


class UncertaintyGuidedGatherDistribute(nn.Module):
    """Gather a feature pyramid and selectively distribute its consensus."""

    def __init__(
        self,
        stage_channels: Sequence[int],
        gather_channels: int = 64,
        gate_groups: int = 4,
        distribution_bottleneck_ratio: float = 0.5,
        max_distribution_scale: float = 2.0,
        use_entropy: bool = True,
        detach_uncertainty: bool = True,
        variance_weight: float = 0.5,
        explicit_uncertainty_modulation: bool = True,
        uncertainty_gain: float = 1.0,
        residual_scale: float = 1.0,
        gather_index: int | None = None,
    ) -> None:
        super().__init__()
        if len(stage_channels) < 2:
            raise ValueError("UG-GD requires at least two feature stages.")
        if not 0.0 <= residual_scale <= 1.0:
            raise ValueError("residual_scale must be in [0, 1].")
        if gate_groups < 1:
            raise ValueError("gate_groups must be positive.")
        if any(int(channels) % gate_groups != 0 for channels in stage_channels):
            raise ValueError("Every UG-GD stage channel count must be divisible by gate_groups.")
        if not 0.0 < distribution_bottleneck_ratio <= 1.0:
            raise ValueError("distribution_bottleneck_ratio must be in (0, 1].")
        if max_distribution_scale < 1.0:
            raise ValueError("max_distribution_scale must be at least 1.0.")
        if not 0.0 <= variance_weight <= 1.0:
            raise ValueError("variance_weight must be in [0, 1].")
        if uncertainty_gain < 0.0:
            raise ValueError("uncertainty_gain must be non-negative.")

        self.stage_channels = tuple(int(c) for c in stage_channels)
        default_gather_index = max(0, len(self.stage_channels) // 2 - 1)
        self.gather_index = default_gather_index if gather_index is None else int(gather_index)
        if not 0 <= self.gather_index < len(self.stage_channels):
            raise ValueError(
                f"gather_index must be in [0, {len(self.stage_channels) - 1}], got {self.gather_index}."
            )
        self.use_entropy = bool(use_entropy)
        self.detach_uncertainty = bool(detach_uncertainty)
        self.gate_groups = int(gate_groups)
        self.distribution_bottleneck_ratio = float(distribution_bottleneck_ratio)
        self.max_distribution_scale = float(max_distribution_scale)
        self.variance_weight = float(variance_weight)
        self.explicit_uncertainty_modulation = bool(explicit_uncertainty_modulation)
        self.align_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, gather_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(1, gather_channels),
                    nn.GELU(),
                )
                for channels in self.stage_channels
            ]
        )

        gather_out = nn.Conv2d(gather_channels, gather_channels, kernel_size=1)
        _near_zero_conv(gather_out)
        self.gather_fusion = nn.Sequential(
            nn.Conv2d(gather_channels * len(self.stage_channels), gather_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, gather_channels),
            nn.GELU(),
            nn.Conv2d(
                gather_channels,
                gather_channels,
                kernel_size=3,
                padding=1,
                groups=gather_channels,
                bias=False,
            ),
            nn.GroupNorm(1, gather_channels),
            nn.GELU(),
            gather_out,
        )
        self.prediction_heads = nn.ModuleList(
            [nn.Conv2d(channels, 1, kernel_size=1) for channels in self.stage_channels]
        )
        self.scale_experts = nn.ModuleList()
        self.delta_heads = nn.ModuleList()
        self.gate_heads = nn.ModuleList()
        for channels in self.stage_channels:
            distribution_channels = max(
                self.gate_groups,
                int(round(channels * self.distribution_bottleneck_ratio)),
            )
            self.scale_experts.append(
                nn.Sequential(
                    nn.Conv2d(gather_channels, channels, kernel_size=1, bias=False),
                    nn.GroupNorm(1, channels),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, kernel_size=1),
                )
            )
            self.delta_heads.append(
                nn.Sequential(
                    nn.Conv2d(channels * 2, distribution_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(1, distribution_channels),
                    nn.GELU(),
                    nn.Conv2d(
                        distribution_channels,
                        distribution_channels,
                        kernel_size=3,
                        padding=1,
                        groups=distribution_channels,
                        bias=False,
                    ),
                    nn.Conv2d(distribution_channels, channels, kernel_size=1),
                )
            )
            delta_out = self.delta_heads[-1][-1]
            assert isinstance(delta_out, nn.Conv2d)
            _near_zero_conv(delta_out)
            gate_out = nn.Conv2d(distribution_channels, self.gate_groups, kernel_size=1)
            nn.init.normal_(gate_out.weight, mean=0.0, std=1e-3)
            nn.init.constant_(gate_out.bias, -2.0)
            self.gate_heads.append(
                nn.Sequential(
                    nn.Conv2d(channels * 2 + 1, distribution_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(1, distribution_channels),
                    nn.GELU(),
                    gate_out,
                )
            )
        inverse_softplus_gain = (
            math.log(math.expm1(float(uncertainty_gain))) if uncertainty_gain > 0.0 else -20.0
        )
        self.uncertainty_gains = nn.Parameter(
            torch.full((len(self.stage_channels),), inverse_softplus_gain)
        )
        self.residual_scales = nn.Parameter(
            torch.full((len(self.stage_channels),), float(residual_scale))
        )
        self.register_buffer("_residual_warmup_factor", torch.tensor(1.0), persistent=False)

    def set_residual_warmup_factor(self, factor: float) -> None:
        if not 0.0 <= factor <= 1.0:
            raise ValueError("residual warm-up factor must be in [0, 1].")
        self._residual_warmup_factor.fill_(float(factor))

    @staticmethod
    def _resize_feature(feature: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if feature.shape[-2:] == size:
            return feature
        if feature.shape[-2] >= size[0] and feature.shape[-1] >= size[1]:
            return F.adaptive_avg_pool2d(feature, output_size=size)
        return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)

    def compute_uncertainty(
        self,
        logits_list: Sequence[torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        probabilities = [
            torch.sigmoid(F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False))
            for logits in logits_list
        ]
        stack = torch.stack(probabilities, dim=1)
        mean_probability = stack.mean(dim=1)
        variance = ((stack - mean_probability.unsqueeze(1)) ** 2).mean(dim=1)
        normalized_variance = (variance / 0.25).clamp(0.0, 1.0)
        if self.use_entropy:
            eps = torch.finfo(mean_probability.dtype).eps
            probability = mean_probability.clamp(eps, 1.0 - eps)
            entropy = -(probability * probability.log() + (1.0 - probability) * (1.0 - probability).log())
            normalized_entropy = (entropy / math.log(2.0)).clamp(0.0, 1.0)
            uncertainty = (
                self.variance_weight * normalized_variance
                + (1.0 - self.variance_weight) * normalized_entropy
            )
        else:
            uncertainty = normalized_variance
        if self.detach_uncertainty:
            uncertainty = uncertainty.detach()
        return uncertainty

    def _distribution_size(
        self,
        feature_size: tuple[int, int],
        gather_size: tuple[int, int],
    ) -> tuple[int, int]:
        """Cap consensus corrections while preserving the native identity feature."""

        return tuple(
            min(feature_extent, max(1, int(round(gather_extent * self.max_distribution_scale))))
            for feature_extent, gather_extent in zip(feature_size, gather_size)
        )

    def forward(
        self,
        stage_features: Sequence[torch.Tensor],
        refine_indices: Sequence[int] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        if len(stage_features) != len(self.stage_channels):
            raise ValueError(
                f"Expected {len(self.stage_channels)} stage features, got {len(stage_features)}."
            )
        gather_size = tuple(stage_features[self.gather_index].shape[-2:])
        aligned = [
            project(self._resize_feature(feature, gather_size))
            for feature, project in zip(stage_features, self.align_proj)
        ]
        consensus = self.gather_fusion(torch.cat(aligned, dim=1))
        pre_logits = [head(feature) for feature, head in zip(stage_features, self.prediction_heads)]
        # Decoder features are ordered low-to-high resolution in BUSS-Net.
        uncertainty_size = self._distribution_size(
            tuple(stage_features[-1].shape[-2:]),
            gather_size,
        )
        uncertainty = self.compute_uncertainty(pre_logits, uncertainty_size)
        active_indices = (
            set(range(len(stage_features)))
            if refine_indices is None
            else {int(index) for index in refine_indices}
        )
        if any(index < 0 or index >= len(stage_features) for index in active_indices):
            raise ValueError("refine_indices contains an out-of-range stage index.")

        refined: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        for index, (feature, expert, delta_head, gate_head) in enumerate(
            zip(stage_features, self.scale_experts, self.delta_heads, self.gate_heads)
        ):
            if index not in active_indices:
                refined.append(feature)
                continue
            feature_size = tuple(feature.shape[-2:])
            distribution_size = self._distribution_size(feature_size, gather_size)
            working_feature = self._resize_feature(feature, distribution_size)
            global_feature = self._resize_feature(expert(consensus), distribution_size)
            uncertainty_i = self._resize_feature(uncertainty, distribution_size)
            joined = torch.cat([working_feature, global_feature], dim=1)
            delta = delta_head(joined)
            gate_logits = gate_head(torch.cat([joined, uncertainty_i], dim=1))
            if self.explicit_uncertainty_modulation:
                gain = F.softplus(self.uncertainty_gains[index]).to(dtype=feature.dtype)
                gate_logits = gate_logits + gain * (2.0 * uncertainty_i - 1.0)
            group_gate = torch.sigmoid(gate_logits)
            group_width = self.stage_channels[index] // self.gate_groups
            gate = group_gate.repeat_interleave(group_width, dim=1)
            scale = (
                self.residual_scales[index].clamp(0.0, 1.0)
                * self._residual_warmup_factor
            ).to(dtype=feature.dtype)
            correction = gate * delta
            if distribution_size != feature_size:
                correction = F.interpolate(
                    correction,
                    size=feature_size,
                    mode="bilinear",
                    align_corners=False,
                )
            refined.append(feature + scale * correction)
            gates.append(group_gate)
        return refined, pre_logits, uncertainty, gates
