import pytest
import torch

from chimera.protection import lr_neutral_energy_kappa


def test_lr_neutral_gate_preserves_l2_update_energy() -> None:
    psi = torch.tensor([-1.0, -0.25, 0.1, 0.9])
    d = torch.tensor([0.2, -1.5, 0.7, 2.0])
    kappa = lr_neutral_energy_kappa(psi, d, lambda_gate=3.0, alpha=0.375)
    before = d.square().sum()
    after = (kappa * d).square().sum()
    assert after.item() == pytest.approx(before.item(), rel=1e-6, abs=1e-7)


def test_lr_neutral_gate_respects_chimera_bounds() -> None:
    psi = torch.linspace(-1.0, 1.0, 257)
    d = torch.linspace(-3.0, 3.0, 257)
    kappa = lr_neutral_energy_kappa(psi, d, lambda_gate=8.0, alpha=0.375)
    assert float(kappa.min()) >= 0.5
    assert float(kappa.max()) <= 2.0


def test_uniform_trust_reduces_to_identity() -> None:
    psi = torch.full((32,), 0.7)
    d = torch.randn(32)
    kappa = lr_neutral_energy_kappa(psi, d, lambda_gate=3.0, alpha=0.375)
    assert torch.allclose(kappa, torch.ones_like(kappa), atol=1e-6, rtol=1e-6)


def test_zero_candidate_update_reduces_to_identity() -> None:
    psi = torch.linspace(-1.0, 1.0, 16)
    d = torch.zeros(16)
    kappa = lr_neutral_energy_kappa(psi, d)
    assert torch.equal(kappa, torch.ones_like(kappa))


def test_gate_is_permutation_equivariant() -> None:
    psi = torch.tensor([-0.8, 0.2, 0.9, -0.1])
    d = torch.tensor([0.5, -2.0, 1.0, 0.25])
    perm = torch.tensor([2, 0, 3, 1])
    original = lr_neutral_energy_kappa(psi, d, lambda_gate=2.0, alpha=0.3)
    permuted = lr_neutral_energy_kappa(psi[perm], d[perm], lambda_gate=2.0, alpha=0.3)
    assert torch.allclose(permuted, original[perm], atol=1e-6, rtol=1e-6)


def test_invalid_alpha_rejected() -> None:
    with pytest.raises(ValueError):
        lr_neutral_energy_kappa(torch.zeros(2), torch.ones(2), alpha=0.4)
