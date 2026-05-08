"""Ternary quantization utilities for Chimera 2.1."""

from __future__ import annotations
from typing import Literal

import torch

from .target_bits import entropy_from_ternary_ratios


StatMode = Literal["mean", "median"]


def _check_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating point tensor")


def _safe_gamma(reference: torch.Tensor, gamma: torch.Tensor | float) -> torch.Tensor:
    gamma_tensor = torch.as_tensor(gamma, dtype=reference.dtype, device=reference.device)
    eps = torch.finfo(reference.dtype).eps
    return gamma_tensor.clamp_min(eps)


def abs_stat(U: torch.Tensor, mode: StatMode = "mean") -> torch.Tensor:
    """Return a scalar absolute-value statistic."""
    _check_tensor("U", U)
    values = U.detach().abs()
    if mode == "mean":
        return values.mean()
    if mode == "median":
        return values.median()
    raise ValueError("mode must be 'mean' or 'median'")


def ema_update(old: torch.Tensor, new: torch.Tensor, rho: float) -> torch.Tensor:
    """Compute an exponential moving average update."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    return old.mul(rho).add(new, alpha=1.0 - rho)


def quantize_weight_strict_bitnet(
    U: torch.Tensor,
    gamma: torch.Tensor | float,
) -> torch.Tensor:
    """Quantize weights with strict BitNet-style ternary rounding."""
    _check_tensor("U", U)
    gamma_safe = _safe_gamma(U, gamma)
    return torch.clamp(torch.round(U / gamma_safe), -1, 1).to(dtype=U.dtype)


def quantize_weight_chimera(
    U: torch.Tensor,
    gamma: torch.Tensor | float,
    alpha: float,
) -> torch.Tensor:
    """Quantize weights with a dynamic ternary threshold."""
    _check_tensor("U", U)
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    gamma_safe = _safe_gamma(U, gamma)
    keep = (U.abs() > alpha * gamma_safe).to(dtype=U.dtype)
    return torch.sign(U) * keep


def ternary_stats(Wq: torch.Tensor) -> dict[str, float]:
    """Return ternary ratios and entropy effective bitwidth."""
    if not isinstance(Wq, torch.Tensor):
        raise TypeError("Wq must be a torch.Tensor")
    if Wq.numel() == 0:
        raise ValueError("Wq must be non-empty")
    total = float(Wq.numel())
    zero_ratio = float((Wq == 0).sum().item()) / total
    plus_ratio = float((Wq > 0).sum().item()) / total
    minus_ratio = float((Wq < 0).sum().item()) / total
    effective_bits = entropy_from_ternary_ratios(zero_ratio, plus_ratio, minus_ratio)
    return {
        "zero_ratio": zero_ratio,
        "plus_ratio": plus_ratio,
        "minus_ratio": minus_ratio,
        "effective_bits": effective_bits,
    }
