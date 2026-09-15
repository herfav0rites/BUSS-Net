"""Adapt upstream baseline outputs to the BUSS-Net training-dict contract."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

OutputType = Literal["logits", "probabilities"]


def _to_logits(t: torch.Tensor, *, output_type: OutputType) -> torch.Tensor:
    """Convert an explicitly declared baseline output to logits."""

    if t.dtype in {torch.float16, torch.bfloat16}:
        t = t.float()
    if output_type == "logits":
        return t
    if output_type == "probabilities":
        return torch.logit(t.clamp(1e-6, 1.0 - 1e-6))
    raise ValueError(f"Unsupported baseline output_type: {output_type}")


def single_scale(
    output: torch.Tensor,
    out_size: tuple[int, int],
    *,
    output_type: OutputType = "logits",
) -> dict[str, torch.Tensor]:
    logits = _to_logits(output, output_type=output_type)
    if logits.shape[-2:] != out_size:
        logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
    return {
        "pred_final": logits,
        "pred_s4": logits,
        "pred_s3": logits,
        "pred_s2": logits,
    }


def from_named_logits(
    final: torch.Tensor,
    s4: torch.Tensor | None = None,
    s3: torch.Tensor | None = None,
    s2: torch.Tensor | None = None,
    out_size: tuple[int, int] | None = None,
    *,
    output_type: OutputType = "logits",
) -> dict[str, torch.Tensor]:
    out_size = out_size or tuple(final.shape[-2:])
    preds = {
        "pred_final": _to_logits(final, output_type=output_type),
        "pred_s4": _to_logits(s4 if s4 is not None else final, output_type=output_type),
        "pred_s3": _to_logits(s3 if s3 is not None else final, output_type=output_type),
        "pred_s2": _to_logits(s2 if s2 is not None else final, output_type=output_type),
    }
    for key, value in preds.items():
        if value.shape[-2:] != out_size:
            preds[key] = F.interpolate(value, size=out_size, mode="bilinear", align_corners=False)
    return preds


def from_sequence(
    outputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
    out_size: tuple[int, int],
    *,
    output_type: OutputType = "logits",
) -> dict[str, torch.Tensor]:
    if not outputs:
        raise ValueError("Baseline returned no predictions.")
    seq = list(outputs)
    if len(seq) >= 5:
        final, s4, s3, s2 = seq[-1], seq[-2], seq[-3], seq[-4]
    elif len(seq) == 4:
        final, s4, s3, s2 = seq[3], seq[2], seq[1], seq[0]
    elif len(seq) == 3:
        final, s4, s3, s2 = seq[2], seq[1], seq[0], seq[0]
    elif len(seq) == 2:
        final, s4, s3, s2 = seq[1], seq[0], seq[0], seq[0]
    else:
        final = seq[0]
        s4 = s3 = s2 = final
    return from_named_logits(final, s4, s3, s2, out_size=out_size, output_type=output_type)
