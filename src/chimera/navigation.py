"""Experimental Navigation primitives for Chimera N-1 audits.

These helpers are intentionally not wired into :class:`Chimera21`.  They let
experiments replace only the candidate direction d_t while keeping the
Protection side fixed.
"""

from __future__ import annotations

import torch

NAVIGATION_MODES = (
    "adam_diag",
    "momentum_raw",
    "momentum_sign",
    "tensor_rms_momentum",
    "grad_diag",
)


def navigation_direction(
    mode: str,
    *,
    grad: torch.Tensor,
    m_hat: torch.Tensor,
    v_hat: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Return one N-1 candidate direction using a shared ``m_hat/v_hat`` state.

    ``adam_diag`` is the current Chimera navigator.  The other modes are
    controlled ablations of numerator momentum and coordinatewise second-
    moment geometry.
    """
    if mode not in NAVIGATION_MODES:
        raise ValueError(f"mode must be one of {NAVIGATION_MODES}")
    if eps <= 0:
        raise ValueError("eps must be positive")

    sqrt_v_hat = v_hat.sqrt()
    if mode == "adam_diag":
        return m_hat / (sqrt_v_hat + eps)
    if mode == "momentum_raw":
        return m_hat
    if mode == "momentum_sign":
        return torch.sign(m_hat)
    if mode == "tensor_rms_momentum":
        return m_hat / (v_hat.mean().sqrt() + eps)
    return grad / (sqrt_v_hat + eps)


def power_diagonal_direction(
    *,
    m_hat: torch.Tensor,
    v_hat: torch.Tensor,
    power: float,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Interpolate diagonal preconditioning strength.

    d(power) = m_hat / (sqrt(v_hat) + eps)**power.

    ``power=1`` is the current Adam-like Chimera navigator and ``power=0`` is
    raw momentum. Scalar learning rate must be retuned for each power because
    the direction scale changes with ``power``.
    """
    if power < 0:
        raise ValueError("power must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")
    return m_hat / (v_hat.sqrt() + eps).pow(power)
