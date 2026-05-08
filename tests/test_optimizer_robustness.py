import pytest
import torch

from chimera import Chimera21, optimizer_state_memory_summary
from chimera.optimizer import _apply_kappa_gate


def _psi_after_zero_step(policy: str, *, idle_decay: float = 0.2) -> float:
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = Chimera21(
        [p],
        lr=0.01,
        beta1=0.9,
        beta2=0.9,
        rho_psi=0.5,
        zero_grad_policy=policy,
        idle_decay=idle_decay,
    )

    p.grad = torch.tensor([1.0])
    opt.step()
    p.grad = torch.tensor([0.0])
    opt.step()

    return float(opt.state[p]["psi"].item())


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("standard", 0.25),
        ("freeze", 0.5),
        ("decay", 0.1),
        ("inertia", 0.75),
    ],
)
def test_zero_grad_policies(policy: str, expected: float) -> None:
    assert _psi_after_zero_step(policy) == pytest.approx(expected)


def test_kappa_noise_gate_suppresses_high_noise_ratio() -> None:
    kappa_raw = torch.tensor([2.0])
    high_noise = torch.tensor([100.0])

    kappa_eff = _apply_kappa_gate(
        kappa_raw,
        high_noise,
        mode="noise_ratio",
        noise_threshold=1.0,
        gate_sharpness=10.0,
    )

    assert kappa_eff.item() == pytest.approx(1.0)


def test_kappa_noise_gate_preserves_low_noise_ratio() -> None:
    kappa_raw = torch.tensor([2.0])
    low_noise = torch.tensor([0.0])

    kappa_eff = _apply_kappa_gate(
        kappa_raw,
        low_noise,
        mode="noise_ratio",
        noise_threshold=1.0,
        gate_sharpness=10.0,
    )

    assert kappa_eff.item() == pytest.approx(2.0, rel=1e-3)


def test_psi_int8_storage_runs_and_dequantizes_to_unit_range() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    opt = Chimera21([p], lr=0.01, psi_storage="int8")

    p.grad = torch.tensor([0.5, -0.25])
    opt.step()

    psi_storage = opt.state[p]["psi"]
    psi_dequantized = psi_storage.float() / 127.0

    assert psi_storage.dtype == torch.int8
    assert float(psi_dequantized.min().item()) >= -1.0
    assert float(psi_dequantized.max().item()) <= 1.0


def test_optimizer_diagnostics_are_recorded_when_enabled() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0, 0.5]))
    opt = Chimera21([p], lr=0.01, log_diagnostics=True)

    p.grad = torch.tensor([0.5, 0.0, -0.25])
    opt.step()

    diagnostics = opt.last_diagnostics
    expected = {
        "mean_kappa",
        "std_kappa",
        "min_kappa",
        "max_kappa",
        "mean_noise_ratio",
        "mean_effective_lr",
        "active_mean_effective_lr",
        "applied_update_norm",
        "param_update_ratio",
        "mean_update_abs",
        "max_update_abs",
        "zero_grad_ratio",
        "active_grad_ratio",
        "psi_abs_mean",
        "psi_saturation_ratio",
        "collision_score",
    }

    assert expected <= diagnostics.keys()
    assert diagnostics["zero_grad_ratio"] == pytest.approx(1.0 / 3.0)
    assert diagnostics["active_grad_ratio"] == pytest.approx(2.0 / 3.0)


def test_active_effective_lr_ignores_zero_gradient_entries() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    opt = Chimera21([p], lr=0.01, log_diagnostics=True)

    p.grad = torch.tensor([0.0, 1.0])
    opt.step()

    diagnostics = opt.last_diagnostics
    assert diagnostics["mean_effective_lr"] > 1000.0
    assert diagnostics["active_mean_effective_lr"] < 0.02


def test_optimizer_state_memory_summary_counts_int8_psi_smaller_than_fp32() -> None:
    p_fp32 = torch.nn.Parameter(torch.ones(16))
    p_int8 = torch.nn.Parameter(torch.ones(16))
    opt_fp32 = Chimera21([p_fp32], psi_storage="fp32")
    opt_int8 = Chimera21([p_int8], psi_storage="int8")

    p_fp32.grad = torch.ones_like(p_fp32)
    p_int8.grad = torch.ones_like(p_int8)
    opt_fp32.step()
    opt_int8.step()

    fp32_summary = optimizer_state_memory_summary(opt_fp32)
    int8_summary = optimizer_state_memory_summary(opt_int8)

    assert fp32_summary["psi"] == 16 * 4
    assert int8_summary["psi"] == 16
    assert int8_summary["psi"] < fp32_summary["psi"]


def test_adam_memory_summary_maps_exp_avg_to_m_and_v() -> None:
    p = torch.nn.Parameter(torch.ones(8))
    opt = torch.optim.Adam([p], lr=0.01)

    p.grad = torch.ones_like(p)
    opt.step()
    summary = optimizer_state_memory_summary(opt)

    assert summary["m"] == 8 * 4
    assert summary["v"] == 8 * 4
    assert summary["total"] >= summary["m"] + summary["v"]
