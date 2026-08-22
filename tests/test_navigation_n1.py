import pytest
import torch

from chimera.navigation import navigation_direction, power_diagonal_direction


def test_adam_diag_matches_current_chimera_formula() -> None:
    grad = torch.tensor([1.0, -2.0])
    m_hat = torch.tensor([0.5, -0.25])
    v_hat = torch.tensor([4.0, 0.25])
    eps = 1.0e-8
    got = navigation_direction(
        "adam_diag", grad=grad, m_hat=m_hat, v_hat=v_hat, eps=eps
    )
    expected = m_hat / (v_hat.sqrt() + eps)
    assert torch.allclose(got, expected)


def test_power_one_equals_adam_diag() -> None:
    m_hat = torch.tensor([0.5, -0.25, 1.0])
    v_hat = torch.tensor([4.0, 0.25, 9.0])
    expected = m_hat / (v_hat.sqrt() + 1.0e-8)
    got = power_diagonal_direction(
        m_hat=m_hat, v_hat=v_hat, power=1.0, eps=1.0e-8
    )
    assert torch.allclose(got, expected)


def test_power_zero_equals_raw_momentum() -> None:
    m_hat = torch.tensor([0.5, -0.25, 1.0])
    v_hat = torch.tensor([4.0, 0.25, 9.0])
    got = power_diagonal_direction(
        m_hat=m_hat, v_hat=v_hat, power=0.0, eps=1.0e-8
    )
    assert torch.allclose(got, m_hat)


def test_momentum_sign_preserves_only_sign() -> None:
    grad = torch.ones(4)
    m_hat = torch.tensor([-3.0, -0.1, 0.0, 8.0])
    v_hat = torch.ones(4)
    got = navigation_direction(
        "momentum_sign", grad=grad, m_hat=m_hat, v_hat=v_hat
    )
    assert torch.equal(got, torch.tensor([-1.0, -1.0, 0.0, 1.0]))


def test_tensor_rms_uses_one_scalar_denominator_per_tensor() -> None:
    grad = torch.ones(3)
    m_hat = torch.tensor([1.0, 2.0, 3.0])
    v_hat = torch.tensor([1.0, 4.0, 9.0])
    got = navigation_direction(
        "tensor_rms_momentum", grad=grad, m_hat=m_hat, v_hat=v_hat, eps=1.0e-8
    )
    expected = m_hat / (v_hat.mean().sqrt() + 1.0e-8)
    assert torch.allclose(got, expected)


def test_grad_diag_removes_momentum_from_numerator() -> None:
    grad = torch.tensor([2.0, -3.0])
    m_hat = torch.tensor([-5.0, 7.0])
    v_hat = torch.tensor([4.0, 9.0])
    got = navigation_direction(
        "grad_diag", grad=grad, m_hat=m_hat, v_hat=v_hat, eps=1.0e-8
    )
    expected = grad / (v_hat.sqrt() + 1.0e-8)
    assert torch.allclose(got, expected)


def test_invalid_navigation_mode_rejected() -> None:
    with pytest.raises(ValueError):
        navigation_direction(
            "bad", grad=torch.ones(1), m_hat=torch.ones(1), v_hat=torch.ones(1)
        )
