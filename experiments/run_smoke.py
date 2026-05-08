from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from chimera import BitLinear, Chimera21
from chimera.logging_utils import log_benchmark_record, make_benchmark_record
from chimera.optimizer import load_config
from chimera.utils import maybe_seed


SEED = 7
SCRIPT = "run_smoke"


def main() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    benchmark = config.get("benchmark", {})
    steps = int(benchmark.get("steps", 8))
    log_steps = {1, max(1, steps // 2), steps}
    lr = float(config["lr"])

    maybe_seed(SEED)
    layer = BitLinear(
        int(benchmark.get("in_features", 8)),
        int(benchmark.get("out_features", 4)),
        stat_mode=config["stat_mode"],
        rho_s=float(config["rho_s"]),
        alpha_min=float(config["alpha_min"]),
        alpha_target=float(config["alpha_target"]),
        warmup_steps=steps,
        quant_mode="chimera",
        eps_gamma=float(config["eps_gamma"]),
        eps_beta=float(config["eps_beta"]),
        ste_clip=float(config["ste_clip"]),
    )
    optimizer = Chimera21(
        layer.parameters(),
        lr=lr,
        beta1=float(config["beta1"]),
        beta2=float(config["beta2"]),
        eps_opt=float(config["eps_opt"]),
        weight_decay=float(config["weight_decay"]),
        rho_psi=float(config["rho_psi"]),
        lambda_gate=float(config["lambda_gate"]),
        kappa_min=float(config["kappa_min"]),
        kappa_max=float(config["kappa_max"]),
    )
    loss_fn = torch.nn.MSELoss()

    x = torch.randn(int(benchmark.get("batch_size", 16)), layer.in_features)
    target = torch.randn(int(benchmark.get("batch_size", 16)), layer.out_features)

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        pred = layer(x)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

        if step in log_steps:
            stats = layer.get_last_stats()
            record = make_benchmark_record(
                run_id=f"{SCRIPT}-seed-{SEED}",
                script=SCRIPT,
                mode="chimera",
                seed=SEED,
                init_seed=SEED,
                step=step,
                loss=float(loss.item()),
                stats=stats,
                lr=lr,
                config=config,
            )
            log_benchmark_record(prefix="smoke", record=record)

    print("[smoke] status=passed")


if __name__ == "__main__":
    main()
