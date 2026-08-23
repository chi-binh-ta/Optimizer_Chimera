"""P-B1: coordinate Protection-Lite versus tensor-block radial Protection.

This round freezes Navigation, tau, hard-clamp mapping and alpha.  The only
architectural change is

    coordinatewise kappa_i * d_i -> one scalar s_b * d_b per parameter tensor.

The script reuses the repository stress problems from compare_optimizers.py.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.compare_optimizers import (
    compute_loss,
    make_model,
    make_noise_schedule,
    make_problem,
)

BETA1 = 0.9
BETA2 = 0.999
EPS = 1.0e-8
SLOPE = 3.0
ALPHA = 0.375
PROBLEMS = ("regression", "sparse_relu", "noisy_quadratic", "saddle")
VARIANTS = ("adam", "coord_lite", "block_tensor")


class PBLite(torch.optim.Optimizer):
    """Frozen Adam Navigation + Protection-Lite evidence, with selectable P geometry."""

    def __init__(self, params, lr: float, mode: str):
        if mode not in ("coord", "block"):
            raise ValueError("mode must be 'coord' or 'block'")
        super().__init__(list(params), dict(lr=lr, mode=mode))
        self.last_diag = {}

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            mode = group["mode"]
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
                state["step"] += 1

                m, v = state["m"], state["v"]
                m.mul_(BETA1).add_(g, alpha=1.0 - BETA1)
                v.mul_(BETA2).addcmul_(g, g, value=1.0 - BETA2)
                mh = m / (1.0 - BETA1 ** state["step"])
                vh = v / (1.0 - BETA2 ** state["step"])
                d = mh / (vh.sqrt() + EPS)

                # Frozen Protection-Lite evidence.
                c = d.abs().clamp(0.0, 1.0)
                base_step = lr * d.abs()
                exposure = base_step / (p.detach().abs() + 2.0 * base_step + EPS)
                tau = (
                    torch.sign(g * d) * c.square() * (1.0 - exposure) - exposure
                ).clamp(-1.0, 1.0)
                tau = torch.where(g == 0, torch.zeros_like(tau), tau)
                z = torch.clamp(SLOPE * tau, -1.0, 1.0)
                w = d.square()
                W = w.sum()
                Q = (w * z).sum()
                prepared.append((p, d, z, W, Q))

            if not prepared:
                continue

            total_W = sum((x[3] for x in prepared), torch.zeros_like(prepared[0][3]))
            total_Q = sum((x[4] for x in prepared), torch.zeros_like(prepared[0][4]))
            q_bar = total_Q / total_W.clamp_min(1.0e-20)

            before = float(total_W)
            after = 0.0
            cosines = []

            for p, d, z, W, Q in prepared:
                if mode == "coord":
                    scale = (1.0 + ALPHA * (z - q_bar)).clamp_min(0.25).sqrt()
                else:
                    q_b = Q / W.clamp_min(1.0e-20)
                    scale = (1.0 + ALPHA * (q_b - q_bar)).clamp_min(0.25).sqrt()

                update = scale * d
                p.add_(update, alpha=-lr)
                after += float(update.square().sum())
                denom = float(d.norm() * update.norm())
                if denom > 0:
                    cosines.append(float((d * update).sum() / denom))

            self.last_diag = {
                "norm_error": abs(after**0.5 - before**0.5) / (before**0.5 + 1.0e-20),
                "mean_block_cosine": float(np.mean(cosines)) if cosines else 1.0,
            }


def clean_eval(model, x, target, problem):
    with torch.no_grad():
        if problem == "noisy_quadratic":
            return float(sum(0.5 * p.square().mean() for p in model.parameters()))
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, target)
        if problem == "saddle":
            loss = loss + 0.01 * sum(
                (p.pow(4) - p.pow(2)).mean() for p in model.parameters()
            )
        return float(loss)


def make_optimizer(variant, params, lr):
    if variant == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(BETA1, BETA2), eps=EPS, foreach=True)
    return PBLite(params, lr, "coord" if variant == "coord_lite" else "block")


def run_one(variant, seed, problem, lr, steps=200):
    x, target = make_problem(seed, problem=problem)
    model = make_model(seed, problem=problem)
    noise = make_noise_schedule(model, seed, steps)
    optimizer = make_optimizer(variant, model.parameters(), lr)
    initial = clean_eval(model, x, target, problem)
    prev = initial
    increases = 0

    for step in range(1, steps + 1):
        model.zero_grad(set_to_none=True)
        loss = compute_loss(model, x, target, problem, step, noise)
        loss.backward()
        optimizer.step()
        cur = clean_eval(model, x, target, problem)
        increases += int(cur > prev + 1.0e-12)
        prev = cur

    diag = getattr(optimizer, "last_diag", {})
    return {
        "variant": variant,
        "seed": seed,
        "problem": problem,
        "lr": lr,
        "initial_loss": initial,
        "eval_loss": prev,
        "increase_rate": increases / steps,
        "norm_error": diag.get("norm_error", 0.0),
        "mean_block_cosine": diag.get("mean_block_cosine", np.nan),
    }


def main():
    out = Path("outputs/pb1_block_radial")
    out.mkdir(parents=True, exist_ok=True)

    centers = {
        "adam": {"noisy_quadratic": .012, "regression": .27, "saddle": .12, "sparse_relu": .1875},
        "coord_lite": {"noisy_quadratic": .012, "regression": .12, "saddle": .08, "sparse_relu": .15},
        "block_tensor": {"noisy_quadratic": .012, "regression": .12, "saddle": .08, "sparse_relu": .15},
    }
    factors = (.5, .75, 1.0, 1.5, 2.0)

    tune = []
    for variant in VARIANTS:
        for problem in PROBLEMS:
            grid = sorted({round(centers[variant][problem] * f, 8) for f in factors})
            for lr in grid:
                for seed in range(800, 804):
                    tune.append(run_one(variant, seed, problem, lr, steps=80))
    tune_df = pd.DataFrame(tune)
    tune_df.to_csv(out / "pb1_tune_raw.csv", index=False)
    frontier = (
        tune_df.groupby(["variant", "problem", "lr"], as_index=False)
        .eval_loss.mean()
        .sort_values(["variant", "problem", "eval_loss"])
        .groupby(["variant", "problem"], as_index=False)
        .first()
    )
    frontier.to_csv(out / "pb1_lr_frontier.csv", index=False)

    held = []
    for _, row in frontier.iterrows():
        for seed in range(900, 912):
            held.append(run_one(row.variant, seed, row.problem, float(row.lr), steps=200))
    held_df = pd.DataFrame(held)
    held_df.to_csv(out / "pb1_heldout_raw.csv", index=False)

    coord = held_df[held_df.variant == "coord_lite"][["seed", "problem", "initial_loss", "eval_loss"]].rename(columns={"eval_loss": "coord_loss"})
    paired = held_df.merge(coord, on=["seed", "problem", "initial_loss"])
    denom = paired.initial_loss.abs().clip(lower=1.0e-12)
    paired["norm_delta_vs_coord"] = (paired.eval_loss - paired.coord_loss) / denom
    paired.to_csv(out / "pb1_paired_vs_coord.csv", index=False)

    print(frontier.to_string(index=False))
    print()
    print(
        paired.groupby("variant", as_index=False)
        .agg(
            mean_delta=("norm_delta_vs_coord", "mean"),
            median_delta=("norm_delta_vs_coord", "median"),
            win_fraction=("norm_delta_vs_coord", lambda x: float((x < 0).mean())),
            max_norm_error=("norm_error", "max"),
            mean_block_cosine=("mean_block_cosine", "mean"),
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
