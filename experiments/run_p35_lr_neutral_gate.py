"""P-3.5: LR-neutral Protection Gate audit.

The P-2.6 step_trust_local statistic is fixed. This round replaces the
un-normalized exponential kappa map with a global parameter-group gate that
redistributes, but does not inflate, the L2 update-energy budget.

Default frontier learning rates were selected on tune seeds in the P-3.5
stress audit and are exposed as CLI arguments rather than changing Chimera's
main default configuration.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from chimera.protection import lr_neutral_energy_kappa
from chimera.optimizer import load_config
from experiments.compare_optimizers import (
    compute_loss,
    make_model,
    make_noise_schedule,
    make_problem,
)

PROBLEMS = ("regression", "sparse_relu", "noisy_quadratic", "saddle")
DEFAULT_LRS = {
    "regression": 0.18,
    "sparse_relu": 0.12,
    "noisy_quadratic": 0.012,
    "saddle": 0.08,
}


class LRNeutralChimera(torch.optim.Optimizer):
    """Experimental P-3.5 optimizer used only by this audit harness."""

    def __init__(
        self,
        params,
        *,
        lr: float,
        beta1: float,
        beta2: float,
        eps_opt: float,
        rho_psi: float,
        lambda_gate: float,
        alpha: float,
        enabled: bool,
    ) -> None:
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps_opt=eps_opt,
            rho_psi=rho_psi,
            lambda_gate=lambda_gate,
            alpha=alpha,
            enabled=enabled,
        )
        super().__init__(params, defaults)
        self.last_diagnostics: dict[str, float] = {}

    @torch.no_grad()
    def step(self, closure=None):
        del closure
        collisions: list[float] = []
        norm_errors: list[float] = []
        min_kappa: list[float] = []
        max_kappa: list[float] = []

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps_opt = group["eps_opt"]
            rho_psi = group["rho_psi"]
            lambda_gate = group["lambda_gate"]
            alpha = group["alpha"]
            enabled = group["enabled"]

            prepared = []
            flat_psi = []
            flat_d = []
            base_sq = 0.0

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["m"] = torch.zeros_like(parameter)
                    state["v"] = torch.zeros_like(parameter)
                    state["psi"] = torch.zeros_like(parameter)
                    state["step"] = 0

                m = state["m"]
                v = state["v"]
                psi = state["psi"]
                state["step"] += 1
                step = state["step"]

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                m_hat = m / (1.0 - beta1**step)
                v_hat = v / (1.0 - beta2**step)
                sqrt_v_hat = v_hat.sqrt()
                d = m_hat / (sqrt_v_hat + eps_opt)

                # Fixed P-2.6 step_trust_local statistic.
                coherence = (m_hat.abs() / (sqrt_v_hat + eps_opt)).clamp(0.0, 1.0)
                direction = torch.sign(grad * d)
                base_step = lr * d.abs()
                exposure = base_step / (parameter.detach().abs() + 2.0 * base_step + eps_opt)
                trust = direction * coherence.square() * (1.0 - exposure) - exposure
                trust = trust.clamp(-1.0, 1.0)
                trust = torch.where(grad == 0, torch.zeros_like(trust), trust)
                psi.mul_(rho_psi).add_(trust, alpha=1.0 - rho_psi)

                prepared.append((parameter, m_hat, sqrt_v_hat, d, psi))
                flat_psi.append(psi.reshape(-1))
                flat_d.append(d.reshape(-1))
                base_sq += float(d.square().sum().item())

            if not prepared:
                continue

            if enabled:
                psi_group = torch.cat(flat_psi)
                d_group = torch.cat(flat_d)
                kappa_group = lr_neutral_energy_kappa(
                    psi_group,
                    d_group,
                    lambda_gate=lambda_gate,
                    alpha=alpha,
                )
            else:
                kappa_group = torch.ones(sum(d.numel() for _, _, _, d, _ in prepared), device=prepared[0][3].device)

            offset = 0
            applied_sq = 0.0
            for parameter, m_hat, sqrt_v_hat, d, _psi in prepared:
                count = d.numel()
                kappa = kappa_group[offset : offset + count].view_as(d)
                offset += count
                update = kappa * d
                parameter.add_(update, alpha=-lr)
                applied_sq += float(update.square().sum().item())
                noise_ratio = sqrt_v_hat / (m_hat.abs() + eps_opt)
                collisions.append(float((kappa.log().abs() * noise_ratio).mean().item()))
                min_kappa.append(float(kappa.min().item()))
                max_kappa.append(float(kappa.max().item()))

            denom = base_sq**0.5 + 1.0e-20
            norm_errors.append(abs(applied_sq**0.5 - base_sq**0.5) / denom)

        self.last_diagnostics = {
            "mean_collision": float(sum(collisions) / max(len(collisions), 1)),
            "max_norm_error": float(max(norm_errors, default=0.0)),
            "min_kappa": float(min(min_kappa, default=1.0)),
            "max_kappa": float(max(max_kappa, default=1.0)),
        }


def clean_eval(model, x, target, problem: str) -> float:
    with torch.no_grad():
        if problem == "noisy_quadratic":
            return float(sum(0.5 * p.pow(2).mean() for p in model.parameters()).item())
        prediction = model(x)
        loss = torch.nn.functional.mse_loss(prediction, target)
        if problem == "saddle":
            loss = loss + 0.01 * sum(
                (p.pow(4) - p.pow(2)).mean() for p in model.parameters()
            )
        return float(loss.item())


def run_one(*, seed: int, problem: str, lr: float, steps: int, enabled: bool,
            rho_psi: float, lambda_gate: float, alpha: float, config: dict) -> dict:
    x, target = make_problem(
        seed=seed, batch_size=64, in_features=10, problem=problem,
        noise_scale=2.0, sparsity=0.7,
    )
    model = make_model(seed=seed, in_features=10, hidden_features=8, problem=problem)
    noise_schedule = make_noise_schedule(model=model, seed=seed, steps=steps, noise_scale=2.0)
    optimizer = LRNeutralChimera(
        model.parameters(), lr=lr,
        beta1=float(config["beta1"]), beta2=float(config["beta2"]),
        eps_opt=float(config["eps_opt"]), rho_psi=rho_psi,
        lambda_gate=lambda_gate, alpha=alpha, enabled=enabled,
    )
    loss_fn = torch.nn.MSELoss()
    previous = clean_eval(model, x, target, problem)
    increases = 0
    collisions = []
    norm_errors = []
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        loss = compute_loss(
            model=model, x=x, target=target, loss_fn=loss_fn,
            problem=problem, step=step, noise_schedule=noise_schedule,
        )
        loss.backward()
        optimizer.step()
        current = clean_eval(model, x, target, problem)
        increases += int(current > previous + 1.0e-12)
        previous = current
        collisions.append(optimizer.last_diagnostics["mean_collision"])
        norm_errors.append(optimizer.last_diagnostics["max_norm_error"])

    return {
        "seed": seed,
        "problem": problem,
        "mode": "lr_neutral" if enabled else "scalar_frontier",
        "lr": lr,
        "rho_psi": rho_psi,
        "lambda_gate": lambda_gate,
        "alpha": alpha,
        "eval_loss": previous,
        "loss_increase_rate": increases / steps,
        "mean_collision": sum(collisions) / len(collisions),
        "max_norm_error": max(norm_errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--rho-psi", type=float, default=0.1)
    parser.add_argument("--lambda-gate", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.375)
    parser.add_argument("--out", type=Path, default=Path("outputs/p35_lr_neutral.csv"))
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "default.yaml")
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for problem in PROBLEMS:
            lr = DEFAULT_LRS[problem]
            for enabled in (False, True):
                rows.append(run_one(
                    seed=seed, problem=problem, lr=lr, steps=args.steps,
                    enabled=enabled, rho_psi=args.rho_psi,
                    lambda_gate=args.lambda_gate, alpha=args.alpha, config=config,
                ))

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    pivot = frame.pivot_table(index=["seed", "problem"], columns="mode", values="eval_loss").reset_index()
    pivot["ratio"] = pivot["lr_neutral"] / pivot["scalar_frontier"]
    print(frame.groupby(["mode", "problem"])[["eval_loss", "loss_increase_rate", "mean_collision"]].mean())
    print("mean_ratio=", float(pivot["ratio"].mean()))
    print("median_ratio=", float(pivot["ratio"].median()))
    print("improved_fraction=", float((pivot["ratio"] < 1.0).mean()))


if __name__ == "__main__":
    main()
