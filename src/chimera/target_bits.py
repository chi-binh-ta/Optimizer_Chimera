"""Target sparsity helpers for effective-bit ternary quantization."""

from __future__ import annotations

from dataclasses import dataclass
import math


Branch = str


def entropy_from_ternary_ratios(zero: float, plus: float, minus: float) -> float:
    """Return Shannon entropy for ternary ratios."""
    ratios = (float(zero), float(plus), float(minus))
    if any(p < 0.0 for p in ratios):
        raise ValueError("ternary ratios must be non-negative")
    total = sum(ratios)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("ternary ratios must sum to 1")
    return -sum(p * math.log2(p) for p in ratios if p > 0.0)


def _symmetric_entropy(zero_ratio: float) -> float:
    side = (1.0 - zero_ratio) / 2.0
    return entropy_from_ternary_ratios(zero_ratio, side, side)


def symmetric_zero_ratio_for_entropy(target_bits: float, branch: Branch = "sparse") -> float:
    """Find zero ratio for symmetric plus/minus entropy on a branch."""
    if branch not in {"sparse", "dense"}:
        raise ValueError("branch must be 'sparse' or 'dense'")
    max_entropy = math.log2(3.0)
    target = float(target_bits)
    if target < 0.0 or target > max_entropy:
        raise ValueError("target_bits must be in [0, log2(3)]")

    if branch == "sparse":
        lo, hi = 1.0 / 3.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if _symmetric_entropy(mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    if target < 1.0:
        raise ValueError("dense branch cannot represent entropy below 1 bit")
    lo, hi = 0.0, 1.0 / 3.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _symmetric_entropy(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@dataclass
class TargetBitsController:
    """Adjust alpha to move zero ratio toward an entropy-derived target."""

    target_bits: float = 1.32
    branch: Branch = "sparse"
    target_zero_ratio: float | None = None
    tolerance: float = 0.02
    step_size: float = 0.05
    alpha_min: float = 0.0
    alpha_max: float = 1.5
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.target_bits < 0:
            raise ValueError("target_bits must be non-negative")
        if self.branch not in {"sparse", "dense"}:
            raise ValueError("branch must be 'sparse' or 'dense'")
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if self.step_size < 0:
            raise ValueError("step_size must be non-negative")
        if self.alpha_max < self.alpha_min:
            raise ValueError("alpha_max must be >= alpha_min")
        if self.target_zero_ratio is None:
            self.target_zero_ratio = symmetric_zero_ratio_for_entropy(
                self.target_bits,
                self.branch,
            )
        if not 0.0 <= self.target_zero_ratio <= 1.0:
            raise ValueError("target_zero_ratio must be in [0, 1]")

    def update(self, alpha: float, current_zero_ratio: float) -> float:
        """Return adjusted alpha from sparse/dense branch zero-ratio error."""
        zero_ratio = float(current_zero_ratio)
        if not 0.0 <= zero_ratio <= 1.0:
            raise ValueError("current_zero_ratio must be in [0, 1]")
        error = zero_ratio - float(self.target_zero_ratio)
        next_alpha = float(alpha)
        if error < -self.tolerance:
            next_alpha += self.step_size
        elif error > self.tolerance:
            next_alpha -= self.step_size
        return min(max(next_alpha, self.alpha_min), self.alpha_max)


SparseBranchController = TargetBitsController
