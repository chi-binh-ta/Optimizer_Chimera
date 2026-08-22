import pytest
import torch

from chimera import Chimera21, optimizer_state_memory_summary
from chimera.optimizer import _step_trust_statistic


def test_invalid_trust_mode_rejected() -> None:
    p = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError):
        Chimera21([p], trust_mode="bad")


def test_raw_default_backward_compatible_first_step() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    opt = Chimera21([p], lr=0.01, rho_psi=0.5)
    p.grad = torch.tensor([0.5, -0.25])
    opt.step()
    assert torch.allclose(opt.state[p]["psi"], torch.tensor([0.5, 0.5]))


def test_step_trust_is_bounded_and_zero_gradient_is_neutral() -> None:
    grad = torch.tensor([1.0, -1.0, 0.0])
    param = torch.tensor([1.0, 2.0, 3.0])
    m_hat = torch.tensor([0.5, 0.5, 0.5])
    sqrt_v_hat = torch.ones(3)
    d = torch.tensor([0.5, -0.5, 0.5])
    trust = _step_trust_statistic(
        grad=grad,
        param=param,
        m_hat=m_hat,
        sqrt_v_hat=sqrt_v_hat,
        d=d,
        lr=0.1,
        eps_opt=1.0e-8,
    )
    assert float(trust.min().item()) >= -1.0
    assert float(trust.max().item()) <= 1.0
    assert trust[2].item() == 0.0


def test_step_trust_wrong_direction_is_not_positive() -> None:
    trust = _step_trust_statistic(
        grad=torch.tensor([1.0]),
        param=torch.tensor([1.0]),
        m_hat=torch.tensor([-1.0]),
        sqrt_v_hat=torch.tensor([1.0]),
        d=torch.tensor([-1.0]),
        lr=0.01,
        eps_opt=1.0e-8,
    )
    assert trust.item() <= 0.0


def test_step_trust_high_coherence_small_step_is_high() -> None:
    trust = _step_trust_statistic(
        grad=torch.tensor([1.0]),
        param=torch.tensor([10.0]),
        m_hat=torch.tensor([1.0]),
        sqrt_v_hat=torch.tensor([1.0]),
        d=torch.tensor([1.0]),
        lr=1.0e-3,
        eps_opt=1.0e-8,
    )
    assert trust.item() > 0.99


def test_step_trust_adds_no_persistent_tensor_state() -> None:
    p_raw = torch.nn.Parameter(torch.ones(16))
    p_p25 = torch.nn.Parameter(torch.ones(16))
    raw = Chimera21([p_raw])
    p25 = Chimera21([p_p25], trust_mode="step_trust")
    p_raw.grad = torch.ones_like(p_raw)
    p_p25.grad = torch.ones_like(p_p25)
    raw.step()
    p25.step()
    assert optimizer_state_memory_summary(raw) == optimizer_state_memory_summary(p25)


def test_alternating_gradient_does_not_create_false_confidence() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = Chimera21(
        [p],
        lr=1.0e-3,
        beta1=0.9,
        beta2=0.999,
        rho_psi=0.95,
        trust_mode="step_trust",
    )
    for step in range(200):
        p.grad = torch.tensor([1.0 if step % 2 == 0 else -1.0])
        opt.step()
    assert abs(float(opt.state[p]["psi"].item())) < 0.1
