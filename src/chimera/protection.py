"""Experimental protection-gate primitives for Chimera research rounds."""

from __future__ import annotations

import torch


def lr_neutral_energy_kappa(
    psi: torch.Tensor,
    d: torch.Tensor,
    *,
    lambda_gate: float = 3.0,
    alpha: float = 0.375,
    eps: float = 1.0e-20,
) -> torch.Tensor:
    """Return a bounded L2-energy-neutral multiplicative protection gate.

    The gate uses centered trust scores z=tanh(lambda*psi) with d^2 weights:

        mu = sum_i d_i^2 z_i / sum_i d_i^2
        kappa_i^2 = 1 + alpha (z_i - mu)

    Therefore sum_i d_i^2 kappa_i^2 == sum_i d_i^2 up to floating-point
    error, so the gate redistributes update energy without inflating the scalar
    learning-rate budget. For 0 <= alpha <= 3/8 and psi in [-1, 1],
    kappa lies in [0.5, sqrt(1.75)] which is inside Chimera's [0.5, 2] bounds.

    ``psi`` and ``d`` must have identical shapes. If d is identically zero,
    the neutral gate is the identity.
    """
    if psi.shape != d.shape:
        raise ValueError("psi and d must have identical shapes")
    if lambda_gate < 0:
        raise ValueError("lambda_gate must be non-negative")
    if not 0.0 <= alpha <= 0.375:
        raise ValueError("alpha must be in [0, 0.375] for the stated bound guarantee")

    weights = d.square()
    total_weight = weights.sum()
    if float(total_weight.item()) <= eps:
        return torch.ones_like(d)

    z = torch.tanh(lambda_gate * psi)
    mu = (weights * z).sum() / total_weight
    kappa_sq = 1.0 + alpha * (z - mu)
    # The clamp is only a floating-point guard; the alpha constraint implies
    # kappa_sq >= 0.25 analytically.
    return kappa_sq.clamp_min(0.25).sqrt()
