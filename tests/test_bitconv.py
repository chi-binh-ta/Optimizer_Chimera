import pytest
import torch

from chimera import BitConv2d


def test_bitconv2d_output_shape() -> None:
    layer = BitConv2d(3, 6, kernel_size=3, padding=1)
    y = layer(torch.randn(2, 3, 8, 8))

    assert y.shape == (2, 6, 8, 8)


def test_bitconv2d_quant_modes_stats() -> None:
    for mode in ("strict_bitnet", "chimera"):
        layer = BitConv2d(3, 4, kernel_size=3, padding=1, quant_mode=mode)
        layer(torch.randn(2, 3, 8, 8))
        stats = layer.get_last_stats()
        total = stats["zero_ratio"] + stats["plus_ratio"] + stats["minus_ratio"]
        Wq = layer.quantize_weight()

        assert set(Wq.flatten().tolist()).issubset({-1.0, 0.0, 1.0})
        assert total == pytest.approx(1.0)
        assert 0.0 <= stats["effective_bits"] <= torch.log2(torch.tensor(3.0)).item()


def test_bitconv2d_effective_bits_range() -> None:
    layer = BitConv2d(1, 2, kernel_size=3, padding=1, quant_mode="chimera")
    layer(torch.randn(1, 1, 4, 4))
    stats = layer.get_last_stats()

    assert 0.0 <= stats["effective_bits"] <= 1.585


def test_bitconv2d_eval_does_not_mutate_quantizer_state() -> None:
    layer = BitConv2d(3, 4, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 8, 8)

    layer.train()
    layer(x)
    assert int(layer.forward_step.item()) == 1

    with torch.no_grad():
        layer.scale_ema.fill_(0.321)
    step_before = int(layer.forward_step.item())
    scale_before = layer.scale_ema.clone()

    layer.eval()
    layer(x)

    assert int(layer.forward_step.item()) == step_before
    assert torch.equal(layer.scale_ema, scale_before)
