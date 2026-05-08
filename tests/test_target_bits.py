import pytest

from chimera import (
    TargetBitsController,
    entropy_from_ternary_ratios,
    symmetric_zero_ratio_for_entropy,
)


def test_controller_increases_alpha_when_zero_ratio_below_sparse_target() -> None:
    controller = TargetBitsController(
        target_bits=1.32,
        branch="sparse",
        tolerance=0.01,
        step_size=0.05,
        alpha_min=0.0,
        alpha_max=1.0,
    )

    assert controller.update(alpha=0.5, current_zero_ratio=0.30) == pytest.approx(0.55)


def test_controller_decreases_alpha_when_zero_ratio_above_sparse_target() -> None:
    controller = TargetBitsController(
        target_bits=1.32,
        branch="sparse",
        tolerance=0.01,
        step_size=0.05,
        alpha_min=0.0,
        alpha_max=1.0,
    )

    assert controller.update(alpha=0.5, current_zero_ratio=0.90) == pytest.approx(0.45)


def test_controller_clips_alpha() -> None:
    controller = TargetBitsController(
        target_bits=1.32,
        branch="sparse",
        tolerance=0.0,
        step_size=0.25,
        alpha_min=0.1,
        alpha_max=0.6,
    )

    assert controller.update(alpha=0.55, current_zero_ratio=0.0) == pytest.approx(0.6)
    assert controller.update(alpha=0.12, current_zero_ratio=1.0) == pytest.approx(0.1)


def test_sparse_zero_ratio_for_target_bits_is_high_sparsity_branch() -> None:
    zero_ratio = symmetric_zero_ratio_for_entropy(1.32, branch="sparse")
    assert 0.62 <= zero_ratio <= 0.64


def test_dense_zero_ratio_for_target_bits_is_low_sparsity_branch() -> None:
    zero_ratio = symmetric_zero_ratio_for_entropy(1.32, branch="dense")
    assert 0.075 <= zero_ratio <= 0.10


def test_balanced_ternary_entropy_is_log2_three() -> None:
    entropy = entropy_from_ternary_ratios(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    assert entropy == pytest.approx(1.5849625007)
