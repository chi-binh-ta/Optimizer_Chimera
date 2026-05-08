import torch
import pytest

from chimera.quantization import (
    abs_stat,
    quantize_weight_chimera,
    quantize_weight_strict_bitnet,
    ternary_stats,
)


def test_strict_quantizer_is_ternary() -> None:
    U = torch.tensor([-2.0, -0.2, 0.0, 0.3, 1.8])
    Wq = quantize_weight_strict_bitnet(U, gamma=0.5)
    assert set(Wq.tolist()).issubset({-1.0, 0.0, 1.0})


def test_chimera_quantizer_is_ternary() -> None:
    U = torch.tensor([-2.0, -0.2, 0.0, 0.3, 1.8])
    Wq = quantize_weight_chimera(U, gamma=0.5, alpha=0.6)
    assert set(Wq.tolist()).issubset({-1.0, 0.0, 1.0})


def test_strict_vs_chimera_quant_outputs() -> None:
    U = torch.tensor([[-1.2, -0.1, 0.0], [0.2, 0.8, 1.7]])
    strict = quantize_weight_strict_bitnet(U, gamma=0.5)
    chimera = quantize_weight_chimera(U, gamma=0.5, alpha=0.6)

    assert strict.shape == U.shape
    assert chimera.shape == U.shape
    assert set(strict.flatten().tolist()).issubset({-1.0, 0.0, 1.0})
    assert set(chimera.flatten().tolist()).issubset({-1.0, 0.0, 1.0})


def test_abs_stat_mean_and_median_are_finite_scalars() -> None:
    U = torch.tensor([-2.0, -0.2, 0.0, 0.3, 1.8])
    for mode in ("mean", "median"):
        stat = abs_stat(U, mode=mode)
        assert stat.ndim == 0
        assert torch.isfinite(stat)


def test_effective_bits_all_zero() -> None:
    stats = ternary_stats(torch.zeros(10))
    assert stats["effective_bits"] == pytest.approx(0.0)


def test_effective_bits_balanced_ternary() -> None:
    Wq = torch.tensor([-1.0, 0.0, 1.0] * 4)
    stats = ternary_stats(Wq)
    assert stats["effective_bits"] == pytest.approx(torch.log2(torch.tensor(3.0)).item())


def test_effective_bits_near_target_distribution() -> None:
    Wq = torch.tensor([0.0] * 630 + [1.0] * 185 + [-1.0] * 185)
    stats = ternary_stats(Wq)
    assert stats["effective_bits"] == pytest.approx(1.32, abs=0.02)
