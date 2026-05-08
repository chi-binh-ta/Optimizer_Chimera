import torch
import pytest

from chimera import BitLinear
from chimera.optimizer import load_config
from chimera.quantization import ternary_stats


def test_bitlinear_forward_shape_and_stats() -> None:
    layer = BitLinear(8, 4, warmup_steps=4)
    x = torch.randn(3, 8)

    y = layer(x)
    stats = layer.get_last_stats()

    assert y.shape == (3, 4)
    assert {"gamma", "alpha", "zero_ratio", "plus_ratio", "minus_ratio"} <= stats.keys()
    assert stats["gamma"] > 0.0
    assert 0.0 <= stats["zero_ratio"] <= 1.0


def test_alpha_warmup_progression() -> None:
    config = load_config("configs/default.yaml")
    alpha_min = float(config["alpha_min"])
    alpha_target = float(config["alpha_target"])
    warmup_steps = 4
    layer = BitLinear(
        2,
        2,
        alpha_min=alpha_min,
        alpha_target=alpha_target,
        warmup_steps=warmup_steps,
    )

    alphas = []
    for _ in range(warmup_steps + 3):
        alphas.append(layer.current_alpha())
        layer.forward_step.add_(1)

    assert alphas[0] == pytest.approx(alpha_min)
    assert alphas[warmup_steps] == pytest.approx(alpha_target)
    assert alphas[-1] == pytest.approx(alpha_target)
    assert all(left <= right for left, right in zip(alphas, alphas[1:]))


def test_eighth_forward_uses_alpha_target() -> None:
    layer = BitLinear(4, 2, warmup_steps=8, alpha_min=0.0, alpha_target=0.7)
    x = torch.randn(2, 4)

    alphas = []
    for _ in range(8):
        layer(x)
        alphas.append(layer.get_last_stats()["alpha"])

    assert alphas[0] == pytest.approx(0.0)
    assert alphas[-1] == pytest.approx(0.7)


def test_bitlinear_stats_ranges() -> None:
    layer = BitLinear(8, 4, warmup_steps=4)
    y = layer(torch.randn(3, 8))
    stats = layer.get_last_stats()
    total = stats["zero_ratio"] + stats["plus_ratio"] + stats["minus_ratio"]

    assert y.shape == (3, 4)
    for key in ("zero_ratio", "plus_ratio", "minus_ratio"):
        assert 0.0 <= stats[key] <= 1.0
    assert total == pytest.approx(1.0)


def test_bitlinear_quant_modes_have_valid_ternary_stats() -> None:
    for mode in ("strict_bitnet", "chimera"):
        layer = BitLinear(4, 3, warmup_steps=4, quant_mode=mode)
        Wq = layer.quantize_weight()
        stats = ternary_stats(Wq)
        total = stats["zero_ratio"] + stats["plus_ratio"] + stats["minus_ratio"]

        assert Wq.shape == layer.weight_master.shape
        assert set(Wq.flatten().tolist()).issubset({-1.0, 0.0, 1.0})
        assert total == pytest.approx(1.0)


def test_bitlinear_eval_does_not_mutate_quantizer_state() -> None:
    layer = BitLinear(4, 3, warmup_steps=4)
    x = torch.randn(2, 4)

    layer.train()
    layer(x)
    assert int(layer.forward_step.item()) == 1

    with torch.no_grad():
        layer.scale_ema.fill_(0.123)
    step_before = int(layer.forward_step.item())
    scale_before = layer.scale_ema.clone()

    layer.eval()
    layer(x)

    assert int(layer.forward_step.item()) == step_before
    assert torch.equal(layer.scale_ema, scale_before)
