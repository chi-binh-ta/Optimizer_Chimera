"""P-3 calibration harness for Chimera step_trust_local.

Keeps the P-2.6 statistic fixed and calibrates only rho_psi/lambda_gate.
Uses a deterministic evaluation objective for noisy_quadratic because its
training loss contains a stochastic linear term and may be negative.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from chimera import Chimera21
from chimera.optimizer import _dequantize_psi, load_config
from experiments.compare_optimizers import (
    compute_loss,
    make_model,
    make_noise_schedule,
    make_problem,
)

PROBLEMS = ("regression", "sparse_relu", "noisy_quadratic", "saddle")


def sampled_collision(optimizer: Chimera21) -> float:
    total = 0.0
    count = 0
    for group in optimizer.param_groups:
        beta1 = group["beta1"]
        beta2 = group["beta2"]
        eps = group["eps_opt"]
        lam = group["lambda_gate"]
        kmin = group["kappa_min"]
        kmax = group["kappa_max"]
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if not state or "m" not in state:
                continue
            step = state["step"]
            m_hat = state["m"] / (1.0 - beta1**step)
            v_hat = state["v"] / (1.0 - beta2**step)
            psi = _dequantize_psi(state["psi"], dtype=parameter.dtype)
            noise_ratio = v_hat.sqrt() / (m_hat.abs() + eps)
            kappa = torch.exp(lam * psi).clamp(kmin, kmax)
            total += float((kappa.log().abs() * noise_ratio).sum().item())
            count += kappa.numel()
    return total / max(count, 1)


def run_one(
    *, seed: int, problem: str, rho_psi: float, lambda_gate: float,
    steps: int, lr: float, noise_scale: float, sparsity: float,
    config: dict,
) -> dict:
    x, target = make_problem(
        seed=seed, batch_size=64, in_features=10, problem=problem,
        noise_scale=noise_scale, sparsity=sparsity,
    )
    base = make_model(seed=seed, in_features=10, hidden_features=8, problem=problem)
    initial_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    noise_schedule = make_noise_schedule(
        model=base, seed=seed, steps=steps, noise_scale=noise_scale,
    )
    model = make_model(seed=seed, in_features=10, hidden_features=8, problem=problem)
    model.load_state_dict(initial_state)
    optimizer = Chimera21(
        model.parameters(), lr=lr,
        beta1=float(config["beta1"]), beta2=float(config["beta2"]),
        eps_opt=float(config["eps_opt"]), weight_decay=float(config["weight_decay"]),
        rho_psi=rho_psi, lambda_gate=lambda_gate,
        kappa_min=float(config["kappa_min"]), kappa_max=float(config["kappa_max"]),
        trust_mode="step_trust_local", log_diagnostics=False,
    )
    loss_fn = torch.nn.MSELoss()
    collisions = []
    train_loss = None
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        train_loss = compute_loss(
            model=model, x=x, target=target, loss_fn=loss_fn,
            problem=problem, step=step, noise_schedule=noise_schedule,
        )
        train_loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == steps:
            collisions.append(sampled_collision(optimizer))

    with torch.no_grad():
        if problem == "noisy_quadratic":
            eval_loss = sum(0.5 * p.pow(2).mean() for p in model.parameters())
        else:
            eval_loss = compute_loss(
                model=model, x=x, target=target, loss_fn=loss_fn,
                problem=problem, step=steps, noise_schedule=noise_schedule,
            )

    return {
        "seed": seed, "problem": problem, "rho_psi": rho_psi,
        "lambda_gate": lambda_gate, "lr": lr,
        "train_loss": float(train_loss.item()), "eval_loss": float(eval_loss.item()),
        "mean_collision": float(np.mean(collisions)),
        "max_collision": float(np.max(collisions)),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        frame[frame["lambda_gate"] == 0.0]
        .groupby(["seed", "problem"], as_index=False)["eval_loss"].mean()
        .rename(columns={"eval_loss": "baseline_eval_loss"})
    )
    paired = frame.merge(baseline, on=["seed", "problem"], how="left")
    paired["eval_ratio"] = paired["eval_loss"] / paired["baseline_eval_loss"]
    aggregate = paired.groupby(["rho_psi", "lambda_gate"], as_index=False).agg(
        mean_eval_ratio=("eval_ratio", "mean"),
        median_eval_ratio=("eval_ratio", "median"),
        worst_eval_ratio=("eval_ratio", "max"),
        mean_collision=("mean_collision", "mean"),
        worst_collision=("max_collision", "max"),
    )
    by_problem = (
        paired.groupby(["rho_psi", "lambda_gate", "problem"])["eval_ratio"]
        .mean().unstack().reset_index()
    )
    return aggregate.merge(by_problem, on=["rho_psi", "lambda_gate"])


def parse_floats(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rhos", default="0.3,0.4,0.95")
    parser.add_argument("--lambdas", default="0,0.1,1.0,1.2,1.5")
    parser.add_argument("--seed-start", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-scale", type=float, default=2.0)
    parser.add_argument("--sparsity", type=float, default=0.7)
    parser.add_argument("--out", type=Path, default=Path("outputs/p3_calibration.csv"))
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "default.yaml")
    rows = []
    for rho in parse_floats(args.rhos):
        for lam in parse_floats(args.lambdas):
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                for problem in PROBLEMS:
                    rows.append(run_one(
                        seed=seed, problem=problem, rho_psi=rho, lambda_gate=lam,
                        steps=args.steps, lr=args.lr, noise_scale=args.noise_scale,
                        sparsity=args.sparsity, config=config,
                    ))
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = summarize(frame)
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.sort_values(["mean_eval_ratio", "mean_collision"]).to_string(index=False))


if __name__ == "__main__":
    main()
