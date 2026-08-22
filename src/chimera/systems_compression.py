"""F-1.5 systems-only implementation of the frozen Chimera candidate.

This module does not change the P-3.5 Protection mathematics or the
N_{gamma=.5} Adam-like Navigation mathematics.  It only changes dataflow:
foreach state updates, reuse of temporary tensors, and a flat group reduction
for the LR-neutral energy gate.
"""
from __future__ import annotations

from typing import Iterable

import torch


class ChimeraF15Exact(torch.optim.Optimizer):
    """Exact-trajectory systems rewrite of frozen Chimera F-1.

    Defaults are the frozen F-1 candidate:
      beta1=0.9, beta2=0.999, eps=1e-8
      rho_psi=0.3, lambda_gate=3.0, alpha=0.375

    The implementation assumes parameters inside one group share device/dtype,
    as required by torch foreach kernels. It keeps psi in parameter dtype
    (FP32 in the audited configuration).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        rho_psi: float = 0.3,
        lambda_gate: float = 3.0,
        alpha: float = 0.375,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("betas must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not 0 <= rho_psi <= 1:
            raise ValueError("rho_psi must be in [0, 1]")
        if lambda_gate < 0:
            raise ValueError("lambda_gate must be non-negative")
        if not 0 <= alpha <= 0.375:
            raise ValueError("alpha must be in [0, 0.375]")
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            rho_psi=rho_psi,
            lambda_gate=lambda_gate,
            alpha=alpha,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1 = float(group["beta1"])
            beta2 = float(group["beta2"])
            eps = float(group["eps"])
            rho = float(group["rho_psi"])
            lam = float(group["lambda_gate"])
            alpha = float(group["alpha"])

            params = []
            grads = []
            ms = []
            vs = []
            psis = []
            bc1 = []
            bc2 = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("ChimeraF15Exact does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["psi"] = torch.zeros_like(p)
                state["step"] += 1
                step = state["step"]
                params.append(p)
                grads.append(p.grad)
                ms.append(state["m"])
                vs.append(state["v"])
                psis.append(state["psi"])
                bc1.append(1.0 - beta1**step)
                bc2.append(1.0 - beta2**step)

            if not params:
                continue

            # Frozen Navigation: d = m_hat / (sqrt(v_hat) + eps).
            torch._foreach_mul_(ms, beta1)
            torch._foreach_add_(ms, grads, alpha=1.0 - beta1)
            torch._foreach_mul_(vs, beta2)
            torch._foreach_addcmul_(vs, grads, grads, value=1.0 - beta2)
            m_hat = torch._foreach_div(ms, bc1)
            v_hat = torch._foreach_div(vs, bc2)
            sqrt_v_hat = torch._foreach_sqrt(v_hat)
            denom = torch._foreach_add(sqrt_v_hat, eps)
            d = torch._foreach_div(m_hat, denom)

            # Frozen P-2.6 step_trust_local. Since d=m_hat/denom,
            # coherence=abs(d) exactly in real arithmetic.
            coherence_sq = torch._foreach_abs(d)
            torch._foreach_clamp_max_(coherence_sq, 1.0)
            torch._foreach_mul_(coherence_sq, coherence_sq)
            direction = torch._foreach_mul(grads, d)
            torch._foreach_sign_(direction)
            base_step = torch._foreach_abs(d)
            torch._foreach_mul_(base_step, lr)
            exposure_denom = torch._foreach_abs(params)
            torch._foreach_add_(exposure_denom, base_step, alpha=2.0)
            torch._foreach_add_(exposure_denom, eps)
            exposure = torch._foreach_div(base_step, exposure_denom)
            one_minus_exposure = torch._foreach_mul(exposure, -1.0)
            torch._foreach_add_(one_minus_exposure, 1.0)
            trust = torch._foreach_mul(direction, coherence_sq)
            torch._foreach_mul_(trust, one_minus_exposure)
            torch._foreach_sub_(trust, exposure)
            for trust_i, grad_i in zip(trust, grads):
                trust_i.mul_(grad_i.ne(0))
            torch._foreach_mul_(psis, rho)
            torch._foreach_add_(psis, trust, alpha=1.0 - rho)

            # Frozen P-3.5 global energy-neutral gate. Keep the same flattened
            # reduction order as F-1 so the audited FP32 trajectory is exact.
            d_flat = torch.cat([x.reshape(-1) for x in d])
            z_flat = torch.cat([x.reshape(-1) for x in psis])
            z_flat.mul_(lam).tanh_()
            weights = d_flat.square()
            mu = (weights * z_flat).sum() / weights.sum().clamp_min(1e-20)
            z_flat.sub_(mu).mul_(alpha).add_(1.0).sqrt_()  # z_flat now kappa

            updates = []
            offset = 0
            for d_i in d:
                n = d_i.numel()
                updates.append(d_i * z_flat[offset : offset + n].view_as(d_i))
                offset += n
            torch._foreach_add_(params, updates, alpha=-lr)

        return loss
