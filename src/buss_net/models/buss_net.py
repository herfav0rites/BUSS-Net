"""BUSS-Net: nnU-Net + GeoFSS bridges + BLG + UG-GD.

The encoder/decoder topology is the fixed C01 PlainConvUNet.  Original C00
GeoFSS bridges enhance the S4 and S5 encoder outputs with geometrically
scheduled state-space scans and frequency FFNs.  BLG then refines the S5
bottleneck using an S3 boundary cue, while UG-GD gathers four decoder stages
and distributes uncertainty-conditioned consensus before prediction.

The GeoFSS implementation is adapted from the MIT-licensed EVSSM source at
commit 5098a5276640694a39a941119f9e8bcc3ece9fb0.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .baselines.nnunet import (
    NNUNET_CONVS_PER_DECODER_STAGE,
    ensure_nnunet_sources,
    plainconv_architecture_kwargs,
)
from .modules import BLG, UncertaintyGuidedGatherDistribute


try:  # pragma: no cover - depends on the CUDA/compiler environment.
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except Exception:  # pragma: no cover
    selective_scan_fn = None


class RMSLayerNorm2d(nn.Module):
    """Channel-wise biased RMS normalization used by GeoFSS blocks."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.weight._no_weight_decay = True
        self.bias._no_weight_decay = True
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 2, 3, 1)
        y = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + self.eps)
        y = y * self.weight + self.bias
        return y.permute(0, 3, 1, 2).contiguous()


class EfficientFrequencyFFN(nn.Module):
    """EVSSM-style gated FFN with shape-safe local FFT filtering."""

    def __init__(
        self,
        channels: int,
        expansion: float = 3.0,
        patch_size: int = 8,
        use_frequency_filter: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if patch_size < 2:
            raise ValueError("frequency patch_size must be >= 2.")
        hidden = max(1, int(channels * expansion))
        self.patch_size = int(patch_size)
        self.use_frequency_filter = bool(use_frequency_filter)
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=bias)
        self.depthwise = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            padding=1,
            groups=hidden * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=bias)
        if self.use_frequency_filter:
            # Optimize a bounded residual response around identity. Directly
            # optimizing an all-one filter allowed SGD weight decay to dominate
            # the task gradient in the previous implementation.
            self.frequency_filter = nn.Parameter(
                torch.zeros(channels, 1, 1, self.patch_size, self.patch_size // 2 + 1)
            )
            self.frequency_filter._no_weight_decay = True
        else:
            self.register_parameter("frequency_filter", None)

    def frequency_response(self) -> torch.Tensor | None:
        if self.frequency_filter is None:
            return None
        return 1.0 + torch.tanh(self.frequency_filter)

    def _pad_to_patch(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        h, w = x.shape[-2:]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h == 0 and pad_w == 0:
            return x, (h, w)
        mode = "reflect" if pad_h < h and pad_w < w else "replicate"
        return F.pad(x, (0, pad_w, 0, pad_h), mode=mode), (h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.depthwise(x).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        if not self.use_frequency_filter:
            return x

        original_dtype = x.dtype
        x, (original_h, original_w) = self._pad_to_patch(x)
        b, c, h, w = x.shape
        p = self.patch_size
        patches = x.float().reshape(b, c, h // p, p, w // p, p)
        patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        response = self.frequency_response()
        assert response is not None
        spectrum = torch.fft.rfft2(patches, dim=(-2, -1)) * response.float()
        patches = torch.fft.irfft2(spectrum, s=(p, p), dim=(-2, -1))
        x = patches.permute(0, 1, 2, 4, 3, 5).contiguous().reshape(b, c, h, w)
        return x[:, :, :original_h, :original_w].to(dtype=original_dtype)


class SelectiveScan2d(nn.Module):
    """Single-direction Mamba scan with local modeling of delta/B/C."""

    def __init__(
        self,
        channels: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: float = 2.0,
        parameter_conv_kernel: int = 7,
        use_parameter_conv: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if selective_scan_fn is None:
            raise ImportError(
                "mamba_ssm selective_scan_fn is unavailable. Use the WSL LMS environment "
                "or set model.geofss.backend: bigru for diagnostics."
            )
        if parameter_conv_kernel % 2 == 0:
            raise ValueError("parameter_conv_kernel must be odd.")
        self.channels = int(channels)
        self.d_state = int(d_state)
        self.inner = int(expand * channels)
        self.dt_rank = math.ceil(channels / 16)
        self.in_proj = nn.Linear(channels, self.inner * 2, bias=bias)
        self.local_conv = nn.Conv2d(
            self.inner,
            self.inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.inner,
            bias=True,
        )
        self.x_proj_weight = nn.Parameter(
            nn.Linear(self.inner, self.dt_rank + 2 * self.d_state, bias=False).weight.unsqueeze(0)
        )
        parameter_channels = self.dt_rank + 2 * self.d_state
        self.parameter_conv = (
            nn.Conv1d(
                parameter_channels,
                parameter_channels,
                kernel_size=parameter_conv_kernel,
                padding=parameter_conv_kernel // 2,
                groups=parameter_channels,
            )
            if use_parameter_conv
            else nn.Identity()
        )
        dt_proj = self._init_dt_projection(self.dt_rank, self.inner)
        self.dt_proj_weight = nn.Parameter(dt_proj.weight.unsqueeze(0))
        self.dt_proj_bias = nn.Parameter(dt_proj.bias.unsqueeze(0))
        self.dt_proj_bias._no_weight_decay = True
        self.a_logs = self._init_a_logs(self.d_state, self.inner)
        self.ds = nn.Parameter(torch.ones(self.inner))
        self.ds._no_weight_decay = True
        self.out_norm = nn.LayerNorm(self.inner)
        self.out_norm.weight._no_weight_decay = True
        self.out_norm.bias._no_weight_decay = True
        self.out_proj = nn.Linear(self.inner, channels, bias=bias)

    @staticmethod
    def _init_dt_projection(dt_rank: int, inner: int) -> nn.Linear:
        projection = nn.Linear(dt_rank, inner, bias=True)
        std = dt_rank**-0.5
        nn.init.uniform_(projection.weight, -std, std)
        dt = torch.exp(torch.rand(inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        dt = dt.clamp(min=1e-4)
        with torch.no_grad():
            projection.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        projection.bias._no_reinit = True
        return projection

    @staticmethod
    def _init_a_logs(d_state: int, inner: int) -> nn.Parameter:
        values = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(inner, 1)
        parameter = nn.Parameter(torch.log(values))
        parameter._no_weight_decay = True
        return parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        b, _, h, w = x.shape
        length = h * w
        xz = self.in_proj(x.permute(0, 2, 3, 1).contiguous())
        scan_input, gate = xz.chunk(2, dim=-1)
        scan_input = scan_input.permute(0, 3, 1, 2).contiguous()
        scan_input = F.gelu(self.local_conv(scan_input))

        with torch.autocast(device_type=x.device.type, enabled=False):
            sequence = scan_input.float().reshape(b, self.inner, length)
            parameters = torch.einsum("b d l, k c d -> b k c l", sequence, self.x_proj_weight.float())
            parameters = self.parameter_conv(parameters[:, 0]).unsqueeze(1)
            dts, bs, cs = torch.split(parameters, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_proj_weight.float())
            output = selective_scan_fn(
                sequence,
                dts.reshape(b, self.inner, length).contiguous(),
                -torch.exp(self.a_logs.float()),
                bs.float(),
                cs.float(),
                self.ds.float(),
                z=None,
                delta_bias=self.dt_proj_bias.float().reshape(-1),
                delta_softplus=True,
                return_last_state=False,
            )
        output = output.transpose(1, 2).reshape(b, h, w, self.inner)
        output = self.out_norm(output) * F.gelu(gate.float())
        output = self.out_proj(output).permute(0, 3, 1, 2).contiguous()
        return output.to(dtype=original_dtype)


class DiagnosticGRUScan(nn.Module):
    """Portable scan fallback for CPU tests; not used by the official config."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.gru = nn.GRU(channels, channels, batch_first=True)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        sequence = x.flatten(2).transpose(1, 2)
        output, _ = self.gru(self.norm(sequence.float()))
        output = self.proj(output).to(dtype=x.dtype)
        return output.transpose(1, 2).reshape(b, c, h, w)


def _grid_index_plan(height: int, width: int, patch: int) -> list[tuple[int, int, int, int]]:
    """Return overlapping window corners matching EVSSM's grid schedule."""

    k1, k2 = min(height, patch), min(width, patch)
    num_row = (height - 1) // k1 + 1
    num_col = (width - 1) // k2 + 1
    step_j = k2 if num_col == 1 else math.ceil((width - k2) / (num_col - 1) - 1e-8)
    step_i = k1 if num_row == 1 else math.ceil((height - k1) / (num_row - 1) - 1e-8)
    windows: list[tuple[int, int, int, int]] = []
    i, last_i = 0, False
    while i < height and not last_i:
        if i + k1 >= height:
            i, last_i = height - k1, True
        j, last_j = 0, False
        while j < width and not last_j:
            if j + k2 >= width:
                j, last_j = width - k2, True
            windows.append((i, j, i + k1, j + k2))
            j += step_j
        i += step_i
    return windows


def apply_overlap_grid(module: nn.Module, x: torch.Tensor, patch: int) -> torch.Tensor:
    """Run a scan on overlapping windows and average overlap regions."""

    batch, channels, height, width = x.shape
    if height <= patch and width <= patch:
        return module(x)
    windows = _grid_index_plan(height, width, patch)
    parts = [x[:, :, i0:i1, j0:j1] for i0, j0, i1, j1 in windows]
    scanned = module(torch.cat(parts, dim=0))
    per_window = scanned.reshape(len(windows), batch, channels, scanned.shape[-2], scanned.shape[-1])
    predictions = x.new_zeros(batch, channels, height, width)
    counts = x.new_zeros(batch, 1, height, width)
    for index, (i0, j0, i1, j1) in enumerate(windows):
        predictions[:, :, i0:i1, j0:j1] = predictions[:, :, i0:i1, j0:j1] + per_window[index]
        counts[:, :, i0:i1, j0:j1] = counts[:, :, i0:i1, j0:j1] + 1.0
    return predictions / counts.clamp_min(1.0)


class GeoFSSBlock(nn.Module):
    """Geometrically scheduled state-space scan followed by frequency FFN."""

    def __init__(
        self,
        channels: int,
        index: int,
        backend: str = "mamba",
        d_state: int = 8,
        d_conv: int = 3,
        expand: float = 2.0,
        parameter_conv_kernel: int = 7,
        use_geometric_transform: bool = True,
        use_parameter_conv: bool = True,
        ffn_expansion: float = 3.0,
        frequency_patch_size: int = 8,
        use_frequency_filter: bool = True,
        use_grid_scan: bool = False,
        scan_patch_size: int = 32,
    ) -> None:
        super().__init__()
        self.index = int(index)
        self.use_geometric_transform = bool(use_geometric_transform)
        self.use_grid_scan = bool(use_grid_scan)
        self.scan_patch_size = int(scan_patch_size)
        if self.scan_patch_size < 1:
            raise ValueError("scan_patch_size must be >= 1.")
        self.norm1 = RMSLayerNorm2d(channels)
        if backend == "mamba":
            self.scan = SelectiveScan2d(
                channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                parameter_conv_kernel=parameter_conv_kernel,
                use_parameter_conv=use_parameter_conv,
            )
        elif backend == "bigru":
            self.scan = DiagnosticGRUScan(channels)
        else:
            raise ValueError(f"GeoFSS backend must be 'mamba' or 'bigru', got {backend!r}.")
        self.norm2 = RMSLayerNorm2d(channels)
        self.ffn = EfficientFrequencyFFN(
            channels,
            expansion=ffn_expansion,
            patch_size=frequency_patch_size,
            use_frequency_filter=use_frequency_filter,
        )

    def geometric_transform(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_geometric_transform:
            return x
        if self.index % 2 == 0:
            return x.transpose(-2, -1).contiguous()
        return torch.flip(x, dims=(-2, -1)).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.geometric_transform(x)
        normalized = self.norm1(x)
        scanned = (
            apply_overlap_grid(self.scan, normalized, self.scan_patch_size)
            if self.use_grid_scan
            else self.scan(normalized)
        )
        x = x + scanned
        return x + self.ffn(self.norm2(x))


class GeoFSSBridge(nn.Module):
    """GeoFSS cycle that writes only the innovation produced by its blocks.

    A linear input projection, a normally initialized output projection, and a
    small per-channel LayerScale replace the old combination of InstanceNorm,
    near-zero output initialization, scalar residual scale, and long warm-up.
    """

    def __init__(
        self,
        in_channels: int = 512,
        bridge_channels: int = 256,
        num_blocks: int = 4,
        layer_scale_init: float = 0.02,
        **block_kwargs: Any,
    ) -> None:
        super().__init__()
        if num_blocks < 4 or num_blocks % 4 != 0:
            raise ValueError("num_blocks must be a positive multiple of four.")
        if layer_scale_init < 0.0:
            raise ValueError("layer_scale_init must be non-negative.")
        self.project_in = nn.Conv2d(
            in_channels, bridge_channels, kernel_size=1, bias=False
        )
        self.blocks = nn.ModuleList(
            [GeoFSSBlock(bridge_channels, index=index, **block_kwargs) for index in range(num_blocks)]
        )
        self.project_out = nn.Conv2d(
            bridge_channels, in_channels, kernel_size=1, bias=False
        )
        self.layer_scale = nn.Parameter(
            torch.full((1, in_channels, 1, 1), float(layer_scale_init))
        )
        self.layer_scale._no_weight_decay = True

    def candidate_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Return the unscaled GeoFSS innovation in the input channel space."""

        original_shape = x.shape[-2:]
        projected_input = self.project_in(x)
        transformed = projected_input
        for block in self.blocks:
            transformed = block(transformed)
        if transformed.shape[-2:] != original_shape:
            raise RuntimeError(
                "GeoFSS geometric cycle did not restore shape: "
                f"{transformed.shape[-2:]} vs {original_shape}."
            )
        return self.project_out(transformed - projected_input)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.candidate_delta(x)
        return self.layer_scale.to(dtype=delta.dtype) * delta

    def forward(
        self,
        x: torch.Tensor,
        spatial_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        delta = self.residual(x)
        if spatial_gate is not None:
            if spatial_gate.shape[-2:] != x.shape[-2:]:
                raise ValueError("spatial_gate must match the GeoFSS feature resolution.")
            delta = delta * spatial_gate.to(dtype=delta.dtype)
        return x + delta


class BUSSNet(nn.Module):
    """Official improved C00 built on the fixed nnU-Net PlainConvUNet."""

    supports_routing_loss = True

    def __init__(
        self,
        num_classes: int = 1,
        use_geofss: bool = True,
        stages: tuple[int, ...] = (4, 5),
        stage4_bridge_channels: int = 128,
        stage5_bridge_channels: int = 256,
        stage4_location: str = "decoder_post_fusion",
        stage4_frequency_patch_size: int | None = None,
        stage5_frequency_patch_size: int | None = None,
        stage4_use_frequency_filter: bool | None = None,
        stage5_use_frequency_filter: bool | None = None,
        stage4_layer_scale_init: float | None = None,
        stage5_layer_scale_init: float | None = None,
        use_blg: bool = True,
        blg_hidden_channels: int = 128,
        blg_gate_groups: int = 4,
        blg_residual_scale: float = 1.0,
        blg_use_geofss_global_expert: bool = True,
        blg_router_identity_bias: float = 1.0,
        use_uggd: bool = True,
        uggd_decoder_indices: tuple[int, ...] = (1, 2, 3, 4),
        uggd_gather_channels: int = 64,
        uggd_gather_index: int = 1,
        uggd_gate_groups: int = 4,
        uggd_distribution_bottleneck_ratio: float = 0.5,
        uggd_max_distribution_scale: float = 2.0,
        uggd_use_entropy: bool = True,
        uggd_detach_uncertainty: bool = True,
        uggd_variance_weight: float = 0.5,
        uggd_explicit_uncertainty_modulation: bool = True,
        uggd_uncertainty_gain: float = 1.0,
        uggd_residual_scale: float = 1.0,
        **bridge_kwargs: Any,
    ) -> None:
        super().__init__()
        requested_stages = tuple(sorted(set(int(stage) for stage in stages)))
        if any(stage not in {4, 5} for stage in requested_stages):
            raise ValueError(f"GeoFSS stages must be a subset of (4, 5), got {requested_stages}.")
        if use_geofss and not requested_stages:
            raise ValueError("GeoFSS stages cannot be empty when GeoFSS is enabled.")
        if stage4_location not in {"encoder_skip", "decoder_post_fusion"}:
            raise ValueError(
                "stage4_location must be 'encoder_skip' or 'decoder_post_fusion'."
            )
        selected_stages = requested_stages if use_geofss else ()
        self.use_geofss = bool(use_geofss)
        self.geofss_stages = selected_stages
        self.stage4_location = str(stage4_location)

        common_frequency_patch_size = int(bridge_kwargs.pop("frequency_patch_size", 2))
        common_use_frequency_filter = bool(bridge_kwargs.pop("use_frequency_filter", True))
        common_layer_scale_init = float(bridge_kwargs.pop("layer_scale_init", 0.02))
        stage4_frequency_patch_size = int(
            common_frequency_patch_size
            if stage4_frequency_patch_size is None
            else stage4_frequency_patch_size
        )
        stage5_frequency_patch_size = int(
            common_frequency_patch_size
            if stage5_frequency_patch_size is None
            else stage5_frequency_patch_size
        )
        stage4_use_frequency_filter = bool(
            False if stage4_use_frequency_filter is None else stage4_use_frequency_filter
        )
        stage5_use_frequency_filter = bool(
            common_use_frequency_filter
            if stage5_use_frequency_filter is None
            else stage5_use_frequency_filter
        )
        stage4_layer_scale_init = float(
            common_layer_scale_init
            if stage4_layer_scale_init is None
            else stage4_layer_scale_init
        )
        stage5_layer_scale_init = float(
            common_layer_scale_init
            if stage5_layer_scale_init is None
            else stage5_layer_scale_init
        )
        ensure_nnunet_sources()
        from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder  # type: ignore[import-untyped]
        from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder  # type: ignore[import-untyped]

        architecture = plainconv_architecture_kwargs()
        self.encoder = PlainConvEncoder(**architecture, return_skips=True)
        features = [int(value) for value in architecture["features_per_stage"]]
        self.stage4_bridge = (
            GeoFSSBridge(
                features[-2],
                stage4_bridge_channels,
                frequency_patch_size=stage4_frequency_patch_size,
                use_frequency_filter=stage4_use_frequency_filter,
                layer_scale_init=stage4_layer_scale_init,
                **bridge_kwargs,
            )
            if 4 in selected_stages
            else nn.Identity()
        )
        self.stage5_bridge = (
            GeoFSSBridge(
                features[-1],
                stage5_bridge_channels,
                frequency_patch_size=stage5_frequency_patch_size,
                use_frequency_filter=stage5_use_frequency_filter,
                layer_scale_init=stage5_layer_scale_init,
                **bridge_kwargs,
            )
            if 5 in selected_stages
            else nn.Identity()
        )
        self.use_blg = bool(use_blg)
        self.blg_uses_stage5_geofss = bool(
            self.use_blg and 5 in selected_stages and blg_use_geofss_global_expert
        )
        self.blg = (
            BLG(
                channels=features[-1],
                guidance_channels=features[-3],
                hidden_channels=blg_hidden_channels,
                gate_groups=blg_gate_groups,
                residual_scale=blg_residual_scale,
                use_internal_global=not self.blg_uses_stage5_geofss,
                router_identity_bias=blg_router_identity_bias,
            )
            if self.use_blg
            else None
        )
        self.decoder = UNetDecoder(
            self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=list(NNUNET_CONVS_PER_DECODER_STAGE),
            deep_supervision=False,
        )
        decoder_channels = [features[-(index + 2)] for index in range(len(features) - 1)]
        self.uggd_decoder_indices = tuple(int(index) for index in uggd_decoder_indices)
        if any(index < 0 or index >= len(decoder_channels) for index in self.uggd_decoder_indices):
            raise ValueError(
                f"UG-GD decoder indices must be in [0, {len(decoder_channels) - 1}], "
                f"got {self.uggd_decoder_indices}."
            )
        selected_channels = [decoder_channels[index] for index in self.uggd_decoder_indices]
        self.use_uggd = bool(use_uggd)
        self.uggd = (
            UncertaintyGuidedGatherDistribute(
                stage_channels=selected_channels,
                gather_channels=uggd_gather_channels,
                gather_index=uggd_gather_index,
                gate_groups=uggd_gate_groups,
                distribution_bottleneck_ratio=uggd_distribution_bottleneck_ratio,
                max_distribution_scale=uggd_max_distribution_scale,
                use_entropy=uggd_use_entropy,
                detach_uncertainty=uggd_detach_uncertainty,
                variance_weight=uggd_variance_weight,
                explicit_uncertainty_modulation=uggd_explicit_uncertainty_modulation,
                uncertainty_gain=uggd_uncertainty_gain,
                residual_scale=uggd_residual_scale,
            )
            if self.use_uggd
            else None
        )

    def set_residual_warmup_factor(self, factor: float) -> None:
        """Ramp legacy routed branches; GeoFSS uses LayerScale only."""

        for module in (self.blg, self.uggd):
            setter = getattr(module, "set_residual_warmup_factor", None)
            if setter is not None:
                setter(factor)

    def _decode_features(self, skips: list[torch.Tensor]) -> list[torch.Tensor]:
        low_resolution = skips[-1]
        decoder_features: list[torch.Tensor] = []
        for index, stage in enumerate(self.decoder.stages):
            x = self.decoder.transpconvs[index](low_resolution)
            x = torch.cat((x, skips[-(index + 2)]), dim=1)
            x = stage(x)
            if (
                index == 0
                and self.stage4_location == "decoder_post_fusion"
                and not isinstance(self.stage4_bridge, nn.Identity)
            ):
                x = self.stage4_bridge(x)
            decoder_features.append(x)
            low_resolution = x
        return decoder_features

    @staticmethod
    def _upsample(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_diagnostics: bool = False,
        return_auxiliary: bool | None = None,
        return_routing: bool = False,
    ) -> dict[str, torch.Tensor]:
        return_auxiliary = self.training if return_auxiliary is None else bool(return_auxiliary)
        out_size = tuple(x.shape[-2:])
        skips = list(self.encoder(x))
        if self.stage4_location == "encoder_skip":
            skips[-2] = self.stage4_bridge(skips[-2])

        stage5_global_delta: torch.Tensor | None = None
        if not isinstance(self.stage5_bridge, nn.Identity):
            if self.blg_uses_stage5_geofss:
                stage5_global_delta = self.stage5_bridge.residual(skips[-1])
            else:
                skips[-1] = self.stage5_bridge(skips[-1])

        boundary_logits: torch.Tensor | None = None
        blg_routes: torch.Tensor | None = None
        if self.blg is not None:
            skips[-1], boundary_logits, blg_routes = self.blg(
                skips[-1],
                skips[-3],
                external_global_delta=stage5_global_delta,
            )

        decoder_features = self._decode_features(skips)
        selected = [decoder_features[index] for index in self.uggd_decoder_indices]
        uggd_logits: list[torch.Tensor] = []
        uncertainty: torch.Tensor | None = None
        uggd_gates: list[torch.Tensor] = []
        if self.uggd is not None:
            # Earlier decoder refinements supervise auxiliary heads but cannot
            # alter an already decoded final feature. Skip them for lean
            # inference while retaining the final-stage consensus correction.
            refine_indices = (
                None
                if (return_auxiliary or return_diagnostics or return_routing)
                else (len(selected) - 1,)
            )
            selected, uggd_logits, uncertainty, uggd_gates = self.uggd(
                selected,
                refine_indices=refine_indices,
            )
            for index, feature in zip(self.uggd_decoder_indices, selected):
                decoder_features[index] = feature

        final_logits = self.decoder.seg_layers[-1](decoder_features[-1])
        outputs: dict[str, torch.Tensor] = {
            "pred_final": self._upsample(final_logits, out_size),
        }
        if return_auxiliary:
            selected_refined_logits = [
                self.decoder.seg_layers[index](decoder_features[index])
                for index in self.uggd_decoder_indices
            ]
            outputs.update(
                {
                    "pred_s4": self._upsample(selected_refined_logits[0], out_size),
                    "pred_s3": self._upsample(
                        selected_refined_logits[min(1, len(selected_refined_logits) - 1)], out_size
                    ),
                    "pred_s2": self._upsample(
                        selected_refined_logits[min(2, len(selected_refined_logits) - 1)], out_size
                    ),
                }
            )
            for position, logits in enumerate(uggd_logits, start=1):
                outputs[f"pred_uggd_d{position}"] = self._upsample(logits, out_size)
            if boundary_logits is not None:
                # Preserve the native S3 resolution for coherent boundary supervision.
                outputs["pred_boundary"] = boundary_logits
        if return_diagnostics and uncertainty is not None:
            outputs["uncertainty"] = self._upsample(uncertainty.detach(), out_size)
        if return_routing and blg_routes is not None:
            for index, name in enumerate(("identity", "local", "global")):
                # Keep native-resolution routes attached to autograd for the
                # optional routing objective. Diagnostics below remain detached.
                outputs[f"route_blg_{name}"] = blg_routes[:, index].mean(dim=1, keepdim=True)
        elif return_diagnostics and blg_routes is not None:
            for index, name in enumerate(("identity", "local", "global")):
                route = blg_routes[:, index].detach().mean(dim=1, keepdim=True)
                outputs[f"route_blg_{name}"] = self._upsample(route, out_size)
        if return_routing:
            for position, gate in enumerate(uggd_gates, start=1):
                outputs[f"route_uggd_d{position}"] = gate.mean(dim=1, keepdim=True)
        elif return_diagnostics:
            for position, gate in enumerate(uggd_gates, start=1):
                outputs[f"route_uggd_d{position}"] = self._upsample(
                    gate.detach().mean(dim=1, keepdim=True), out_size
                )
        return outputs


def build_buss_net(config: dict[str, Any] | None = None) -> BUSSNet:
    config = config or {}
    model_cfg = config.get("model", config)
    geofss_cfg = model_cfg.get("geofss", {})
    blg_cfg = model_cfg.get("blg", {})
    uggd_cfg = model_cfg.get("uggd", {})
    stage4_cfg = geofss_cfg.get("stage4", {})
    stage5_cfg = geofss_cfg.get("stage5", {})
    common_frequency_patch_size = int(geofss_cfg.get("frequency_patch_size", 2))
    common_use_frequency_filter = bool(geofss_cfg.get("use_frequency_filter", True))
    common_layer_scale_init = float(geofss_cfg.get("layer_scale_init", 0.02))
    global_expert = str(blg_cfg.get("global_expert", "geofss_s5")).lower()
    return BUSSNet(
        num_classes=int(model_cfg.get("num_classes", 1)),
        use_geofss=bool(geofss_cfg.get("enabled", True)),
        stages=tuple(int(stage) for stage in geofss_cfg.get("stages", [4, 5])),
        stage4_bridge_channels=int(
            stage4_cfg.get("bridge_channels", geofss_cfg.get("stage4_bridge_channels", 128))
        ),
        stage5_bridge_channels=int(
            stage5_cfg.get("bridge_channels", geofss_cfg.get("stage5_bridge_channels", 256))
        ),
        stage4_location=str(stage4_cfg.get("location", "decoder_post_fusion")),
        stage4_frequency_patch_size=int(
            stage4_cfg.get("frequency_patch_size", common_frequency_patch_size)
        ),
        stage5_frequency_patch_size=int(
            stage5_cfg.get("frequency_patch_size", common_frequency_patch_size)
        ),
        stage4_use_frequency_filter=bool(
            stage4_cfg.get("use_frequency_filter", common_use_frequency_filter)
        ),
        stage5_use_frequency_filter=bool(
            stage5_cfg.get("use_frequency_filter", common_use_frequency_filter)
        ),
        stage4_layer_scale_init=float(
            stage4_cfg.get("layer_scale_init", common_layer_scale_init)
        ),
        stage5_layer_scale_init=float(
            stage5_cfg.get("layer_scale_init", common_layer_scale_init)
        ),
        num_blocks=int(geofss_cfg.get("num_blocks", 4)),
        backend=str(geofss_cfg.get("backend", "mamba")),
        d_state=int(geofss_cfg.get("d_state", 8)),
        d_conv=int(geofss_cfg.get("d_conv", 3)),
        expand=float(geofss_cfg.get("expand", 2.0)),
        parameter_conv_kernel=int(geofss_cfg.get("parameter_conv_kernel", 7)),
        use_geometric_transform=bool(geofss_cfg.get("use_geometric_transform", True)),
        use_parameter_conv=bool(geofss_cfg.get("use_parameter_conv", True)),
        ffn_expansion=float(geofss_cfg.get("ffn_expansion", 3.0)),
        use_grid_scan=bool(geofss_cfg.get("use_grid_scan", False)),
        scan_patch_size=int(geofss_cfg.get("scan_patch_size", 32)),
        use_blg=bool(blg_cfg.get("enabled", True)),
        blg_hidden_channels=int(blg_cfg.get("hidden_channels", 128)),
        blg_gate_groups=int(blg_cfg.get("gate_groups", 4)),
        blg_residual_scale=float(blg_cfg.get("residual_scale", 1.0)),
        blg_use_geofss_global_expert=global_expert == "geofss_s5",
        blg_router_identity_bias=float(blg_cfg.get("router_identity_bias", 1.0)),
        use_uggd=bool(uggd_cfg.get("enabled", True)),
        uggd_decoder_indices=tuple(int(index) for index in uggd_cfg.get("decoder_indices", [1, 2, 3, 4])),
        uggd_gather_channels=int(uggd_cfg.get("gather_channels", 64)),
        uggd_gather_index=int(uggd_cfg.get("gather_index", 1)),
        uggd_gate_groups=int(uggd_cfg.get("gate_groups", 4)),
        uggd_distribution_bottleneck_ratio=float(uggd_cfg.get("bottleneck_ratio", 0.5)),
        uggd_max_distribution_scale=float(uggd_cfg.get("max_distribution_scale", 2.0)),
        uggd_use_entropy=bool(uggd_cfg.get("use_entropy", True)),
        uggd_detach_uncertainty=bool(uggd_cfg.get("detach_uncertainty", True)),
        uggd_variance_weight=float(uggd_cfg.get("variance_weight", 0.5)),
        uggd_explicit_uncertainty_modulation=bool(
            uggd_cfg.get("explicit_uncertainty_modulation", True)
        ),
        uggd_uncertainty_gain=float(uggd_cfg.get("uncertainty_gain", 1.0)),
        uggd_residual_scale=float(uggd_cfg.get("residual_scale", 1.0)),
    )
