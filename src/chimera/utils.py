"""Small utilities used by Chimera 2.1 modules and tests."""

from __future__ import annotations

import random

import numpy as np
import torch


def maybe_seed(seed: int | None) -> None:
    """Seed Python, NumPy, and Torch when a seed is provided."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_2d_input(x: torch.Tensor) -> torch.Tensor:
    """Validate and return a 2D input tensor."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 2:
        raise ValueError(f"expected input shape [batch, in_features], got {tuple(x.shape)}")
    return x


def count_ternary_values(Wq: torch.Tensor) -> dict[int, int]:
    """Count -1, 0, and +1 entries in a ternary tensor."""
    if not isinstance(Wq, torch.Tensor):
        raise TypeError("Wq must be a torch.Tensor")
    return {
        -1: int((Wq < 0).sum().item()),
        0: int((Wq == 0).sum().item()),
        1: int((Wq > 0).sum().item()),
    }
