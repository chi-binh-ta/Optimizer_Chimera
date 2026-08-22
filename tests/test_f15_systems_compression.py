import torch

from chimera.systems_compression import ChimeraF15Exact


class SlowFrozenF1(torch.optim.Optimizer):
    """Literal frozen F-1 reference used only for equivalence testing."""

    def __init__(self, params, lr=1e-2):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            prepared = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["psi"] = torch.zeros_like(p)
                state["step"] += 1
                state["m"].mul_(0.9).add_(g, alpha=0.1)
                state["v"].mul_(0.999).addcmul_(g, g, value=0.001)
                m_hat = state["m"] / (1.0 - 0.9 ** state["step"])
                v_hat = state["v"] / (1.0 - 0.999 ** state["step"])
                sqrt_v = v_hat.sqrt()
                d = m_hat / (sqrt_v + 1e-8)
                coherence = (m_hat.abs() / (sqrt_v + 1e-8)).clamp(0.0, 1.0)
                base_step = lr * d.abs()
                exposure = base_step / (p.abs() + 2.0 * base_step + 1e-8)
                trust = (
                    torch.sign(g * d) * coherence.square() * (1.0 - exposure)
                    - exposure
                ).clamp(-1.0, 1.0)
                trust = torch.where(g == 0, torch.zeros_like(trust), trust)
                state["psi"].mul_(0.3).add_(trust, alpha=0.7)
                prepared.append((p, d, state["psi"]))

            d_flat = torch.cat([d.reshape(-1) for _, d, _ in prepared])
            psi_flat = torch.cat([psi.reshape(-1) for _, _, psi in prepared])
            weights = d_flat.square()
            z = torch.tanh(3.0 * psi_flat)
            mu = (weights * z).sum() / weights.sum()
            kappa = (1.0 + 0.375 * (z - mu)).clamp_min(0.25).sqrt()
            offset = 0
            for p, d, _ in prepared:
                n = d.numel()
                p.add_(kappa[offset : offset + n].view_as(d) * d, alpha=-lr)
                offset += n


def test_f15_exact_matches_frozen_reference_bitwise() -> None:
    p1_ref = torch.nn.Parameter(torch.tensor([1.0, -0.5, 0.0, 2.0]))
    p2_ref = torch.nn.Parameter(torch.tensor([[0.2, -1.0], [0.0, 0.7]]))
    p1_fast = torch.nn.Parameter(p1_ref.detach().clone())
    p2_fast = torch.nn.Parameter(p2_ref.detach().clone())
    ref = SlowFrozenF1([p1_ref, p2_ref], lr=0.03)
    fast = ChimeraF15Exact([p1_fast, p2_fast], lr=0.03)

    generator = torch.Generator().manual_seed(123)
    for step in range(100):
        g1 = torch.randn(p1_ref.shape, generator=generator)
        g2 = torch.randn(p2_ref.shape, generator=generator)
        if step % 7 == 0:
            g1[2] = 0.0
            g2[1, 0] = 0.0
        p1_ref.grad = g1.clone()
        p2_ref.grad = g2.clone()
        p1_fast.grad = g1.clone()
        p2_fast.grad = g2.clone()
        ref.step()
        fast.step()

        assert torch.equal(p1_ref, p1_fast)
        assert torch.equal(p2_ref, p2_fast)


def test_f15_state_is_only_m_v_psi() -> None:
    p = torch.nn.Parameter(torch.ones(32))
    opt = ChimeraF15Exact([p])
    p.grad = torch.ones_like(p)
    opt.step()
    tensor_keys = {k for k, v in opt.state[p].items() if isinstance(v, torch.Tensor)}
    assert tensor_keys == {"m", "v", "psi"}
