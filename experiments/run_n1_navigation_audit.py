"""N-1: audit Chimera Navigation with P-3.5 Protection frozen.

Protection is fixed to:
- P-2.6 step_trust_local statistic,
- rho_psi = 0.3,
- global L2-energy-neutral gate,
- lambda_gate = 3.0,
- alpha = 0.375.

Only the candidate direction d_t changes. Scalar LR must be tuned separately
for each navigator/problem before held-out comparison because the candidate
directions have different scales.
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

from chimera.navigation import NAVIGATION_MODES, navigation_direction, power_diagonal_direction
from chimera.optimizer import load_config
from chimera.protection import lr_neutral_energy_kappa
from experiments.compare_optimizers import compute_loss, make_model, make_noise_schedule, make_problem

PROBLEMS = ("regression", "sparse_relu", "noisy_quadratic", "saddle")


def clean_eval(model, x, target, problem: str) -> float:
    with torch.no_grad():
        if problem == "noisy_quadratic":
            return float(sum(0.5 * p.pow(2).mean() for p in model.parameters()).item())
        prediction = model(x)
        loss = torch.nn.functional.mse_loss(prediction, target)
        if problem == "saddle":
            loss = loss + 0.01 * sum((p.pow(4) - p.pow(2)).mean() for p in model.parameters())
        return float(loss.item())


class N1NavigationOptimizer(torch.optim.Optimizer):
    """Research-only optimizer that changes Navigation and freezes Protection."""

    def __init__(self, params, *, lr: float, beta1: float, beta2: float, eps_opt: float,
                 navigator: str, precondition_power: float | None = None,
                 rho_psi: float = 0.3, lambda_gate: float = 3.0,
                 alpha: float = 0.375) -> None:
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, eps_opt=eps_opt,
            navigator=navigator, precondition_power=precondition_power,
            rho_psi=rho_psi, lambda_gate=lambda_gate, alpha=alpha,
        )
        super().__init__(params, defaults)
        self.last_diagnostics: dict[str, float] = {}

    @torch.no_grad()
    def step(self, closure=None):
        del closure
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps_opt = group["eps_opt"]
            navigator = group["navigator"]
            power = group["precondition_power"]
            rho = group["rho_psi"]
            lam = group["lambda_gate"]
            alpha = group["alpha"]

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

                state["step"] += 1
                step = state["step"]
                m = state["m"]
                v = state["v"]
                psi = state["psi"]
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                m_hat = m / (1.0 - beta1**step)
                v_hat = v / (1.0 - beta2**step)
                sqrt_v_hat = v_hat.sqrt()

                if power is None:
                    d = navigation_direction(
                        navigator, grad=grad, m_hat=m_hat, v_hat=v_hat, eps=eps_opt
                    )
                else:
                    d = power_diagonal_direction(
                        m_hat=m_hat, v_hat=v_hat, power=power, eps=eps_opt
                    )

                # Frozen P-2.6 step_trust_local statistic.
                coherence = (m_hat.abs() / (sqrt_v_hat + eps_opt)).clamp(0.0, 1.0)
                direction = torch.sign(grad * d)
                base_step = lr * d.abs()
                exposure = base_step / (parameter.detach().abs() + 2.0 * base_step + eps_opt)
                trust = direction * coherence.square() * (1.0 - exposure) - exposure
                trust = trust.clamp(-1.0, 1.0)
                trust = torch.where(grad == 0, torch.zeros_like(trust), trust)
                psi.mul_(rho).add_(trust, alpha=1.0 - rho)

                prepared.append((parameter, m_hat, sqrt_v_hat, d))
                flat_psi.append(psi.reshape(-1))
                flat_d.append(d.reshape(-1))
                base_sq += float(d.square().sum().item())

            if not prepared:
                continue

            psi_group = torch.cat(flat_psi)
            d_group = torch.cat(flat_d)
            kappa_group = lr_neutral_energy_kappa(
                psi_group, d_group, lambda_gate=lam, alpha=alpha
            )

            offset = 0
            applied_sq = 0.0
            collision_sum = 0.0
            count_total = 0
            for parameter, m_hat, sqrt_v_hat, d in prepared:
                count = d.numel()
                kappa = kappa_group[offset : offset + count].view_as(d)
                offset += count
                update = kappa * d
                parameter.add_(update, alpha=-lr)
                applied_sq += float(update.square().sum().item())
                noise_ratio = sqrt_v_hat / (m_hat.abs() + eps_opt)
                collision_sum += float((kappa.log().abs() * noise_ratio).sum().item())
                count_total += count

            base_norm = base_sq**0.5
            applied_norm = applied_sq**0.5
            self.last_diagnostics = {
                "collision": collision_sum / max(count_total, 1),
                "norm_error": abs(applied_norm - base_norm) / (base_norm + 1.0e-20),
            }


def run_one(*, seed: int, problem: str, navigator: str, lr: float,
            steps: int, power: float | None, config: dict) -> dict:
    x, target = make_problem(
        seed=seed, batch_size=64, in_features=10, problem=problem,
        noise_scale=2.0, sparsity=0.7,
    )
    model = make_model(seed=seed, in_features=10, hidden_features=8, problem=problem)
    noise_schedule = make_noise_schedule(model=model, seed=seed, steps=steps, noise_scale=2.0)
    optimizer = N1NavigationOptimizer(
        model.parameters(), lr=lr,
        beta1=float(config["beta1"]), beta2=float(config["beta2"]),
        eps_opt=float(config["eps_opt"]), navigator=navigator,
        precondition_power=power,
    )
    previous = clean_eval(model, x, target, problem)
    increases = 0
    collisions = []
    norm_errors = []
    loss_fn = torch.nn.MSELoss()
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
        collisions.append(optimizer.last_diagnostics["collision"])
        norm_errors.append(optimizer.last_diagnostics["norm_error"])

    return {
        "seed": seed, "problem": problem, "navigator": navigator,
        "precondition_power": power, "lr": lr, "eval_loss": previous,
        "loss_increase_rate": increases / steps,
        "mean_collision": sum(collisions) / len(collisions),
        "max_norm_error": max(norm_errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--navigator", choices=NAVIGATION_MODES, default="adam_diag")
    parser.add_argument("--precondition-power", type=float, default=None)
    parser.add_argument("--problem", choices=PROBLEMS, default="regression")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed-start", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--out", type=Path, default=Path("outputs/n1_navigation.csv"))
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "default.yaml")
    rows = [
        run_one(
            seed=seed, problem=args.problem, navigator=args.navigator,
            lr=args.lr, steps=args.steps, power=args.precondition_power,
            config=config,
        )
        for seed in range(args.seed_start, args.seed_start + args.seeds)
    ]
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))
    print("mean_eval_loss=", float(frame["eval_loss"].mean()))
    print("mean_loss_increase_rate=", float(frame["loss_increase_rate"].mean()))
    print("mean_collision=", float(frame["mean_collision"].mean()))


if __name__ == "__main__":
    main()
