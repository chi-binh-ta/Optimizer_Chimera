from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from compare_optimizers import (
        compute_loss,
        make_model,
        make_noise_schedule,
        make_problem,
    )
except ModuleNotFoundError:  # pragma: no cover
    from experiments.compare_optimizers import (
        compute_loss,
        make_model,
        make_noise_schedule,
        make_problem,
    )

from chimera import Chimera21, optimizer_state_memory_bytes
from chimera.optimizer import load_config

PROBLEMS = ("regression", "sparse_relu", "noisy_quadratic", "saddle")
MODES = ("raw_agreement", "step_trust")


def build_optimizer(model: torch.nn.Module, *, lr: float, config: dict, trust_mode: str) -> Chimera21:
    return Chimera21(
        model.parameters(),
        lr=lr,
        beta1=float(config["beta1"]),
        beta2=float(config["beta2"]),
        eps_opt=float(config["eps_opt"]),
        weight_decay=float(config["weight_decay"]),
        rho_psi=float(config["rho_psi"]),
        lambda_gate=float(config["lambda_gate"]),
        kappa_min=float(config["kappa_min"]),
        kappa_max=float(config["kappa_max"]),
        log_diagnostics=True,
        trust_mode=trust_mode,
    )


def run_one(args: argparse.Namespace, config: dict, problem: str, trust_mode: str) -> dict:
    x, target = make_problem(
        seed=args.seed,
        batch_size=args.batch_size,
        in_features=args.in_features,
        problem=problem,
        noise_scale=args.noise_scale,
        sparsity=args.sparsity,
    )
    base = make_model(
        seed=args.seed,
        in_features=args.in_features,
        hidden_features=args.hidden_features,
        problem=problem,
    )
    initial_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    noise_schedule = make_noise_schedule(
        model=base,
        seed=args.seed,
        steps=args.steps,
        noise_scale=args.noise_scale,
    )

    model = make_model(
        seed=args.seed,
        in_features=args.in_features,
        hidden_features=args.hidden_features,
        problem=problem,
    )
    model.load_state_dict(initial_state)
    optimizer = build_optimizer(model, lr=args.lr, config=config, trust_mode=trust_mode)
    loss_fn = torch.nn.MSELoss()
    times = []
    final_loss = None

    for step in range(1, args.steps + 1):
        start = time.perf_counter()
        optimizer.zero_grad()
        loss = compute_loss(
            model=model,
            x=x,
            target=target,
            loss_fn=loss_fn,
            problem=problem,
            step=step,
            noise_schedule=noise_schedule,
        )
        loss.backward()
        optimizer.step()
        times.append((time.perf_counter() - start) * 1000.0)
        final_loss = float(loss.item())

    diagnostics = optimizer.last_diagnostics
    warm = times[args.timing_warmup_steps :] or times
    return {
        "problem": problem,
        "trust_mode": trust_mode,
        "seed": args.seed,
        "steps": args.steps,
        "final_loss": final_loss,
        "mean_step_time_ms_after_warmup": sum(warm) / len(warm),
        "mean_kappa": diagnostics.get("mean_kappa"),
        "mean_noise_ratio": diagnostics.get("mean_noise_ratio"),
        "collision_score": diagnostics.get("collision_score"),
        "mean_trust_stat": diagnostics.get("mean_trust_stat"),
        "mean_abs_trust_stat": diagnostics.get("mean_abs_trust_stat"),
        "optimizer_state_bytes": optimizer_state_memory_bytes(optimizer),
    }


def main() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    parser = argparse.ArgumentParser(description="P-2.5 raw-agreement vs step-trust stress audit.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--in-features", type=int, default=10)
    parser.add_argument("--hidden-features", type=int, default=8)
    parser.add_argument("--lr", type=float, default=float(config["lr"]))
    parser.add_argument("--noise-scale", type=float, default=2.0)
    parser.add_argument("--sparsity", type=float, default=0.7)
    parser.add_argument("--timing-warmup-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/p25_step_trust_stress.jsonl"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for problem in PROBLEMS:
            for trust_mode in MODES:
                record = run_one(args, config, problem, trust_mode)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
