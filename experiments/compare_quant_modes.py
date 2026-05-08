from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chimera import BitLinear, Chimera21
from chimera.logging_utils import (
    log_benchmark_record,
    make_benchmark_record,
    make_summary_record,
)
from chimera.optimizer import load_config
from chimera.utils import maybe_seed


SCRIPT = "compare_quant_modes"
SEED_BASE = 11
INIT_SEED_OFFSET = 18


@dataclass(frozen=True)
class RunResult:
    mode: str
    seed: int
    final_loss: float
    final_stats: dict[str, float]


def _benchmark_defaults(config: dict[str, Any]) -> dict[str, Any]:
    defaults = config.get("benchmark", {})
    if not isinstance(defaults, dict):
        raise ValueError("config field 'benchmark' must be a mapping")
    return defaults


def build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    defaults = _benchmark_defaults(config)
    parser = argparse.ArgumentParser(description="Compare Chimera quantization modes.")
    parser.add_argument("--steps", type=int, default=int(defaults.get("steps", 8)))
    parser.add_argument("--seeds", type=int, default=int(defaults.get("seeds", 1)))
    parser.add_argument("--batch-size", type=int, default=int(defaults.get("batch_size", 16)))
    parser.add_argument("--in-features", type=int, default=int(defaults.get("in_features", 8)))
    parser.add_argument("--out-features", type=int, default=int(defaults.get("out_features", 4)))
    parser.add_argument("--lr", type=float, default=float(config["lr"]))
    parser.add_argument("--log-jsonl", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("steps", "seeds", "batch_size", "in_features", "out_features"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.lr < 0:
        raise ValueError("--lr must be non-negative")
    if args.device != "cpu":
        raise ValueError("Optimizer Chimera v2 benchmark harness is CPU-only; use --device cpu")


def build_layer(
    mode: str,
    config: dict[str, Any],
    *,
    in_features: int,
    out_features: int,
    steps: int,
) -> BitLinear:
    return BitLinear(
        in_features,
        out_features,
        stat_mode=config["stat_mode"],
        rho_s=float(config["rho_s"]),
        alpha_min=float(config["alpha_min"]),
        alpha_target=float(config["alpha_target"]),
        warmup_steps=steps,
        quant_mode=mode,
        eps_gamma=float(config["eps_gamma"]),
        eps_beta=float(config["eps_beta"]),
        ste_clip=float(config["ste_clip"]),
    )


def build_optimizer(layer: BitLinear, config: dict[str, Any], *, lr: float) -> Chimera21:
    return Chimera21(
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


def _log_summary(
    results: list[RunResult],
    *,
    config: dict[str, Any],
    lr: float,
    jsonl_path: Path | None,
) -> None:
    modes = sorted({result.mode for result in results})
    for mode in modes:
        mode_results = [result for result in results if result.mode == mode]
        losses = np.array([result.final_loss for result in mode_results], dtype=float)
        zero_ratios = np.array(
            [result.final_stats["zero_ratio"] for result in mode_results],
            dtype=float,
        )
        plus_ratios = np.array(
            [result.final_stats["plus_ratio"] for result in mode_results],
            dtype=float,
        )
        minus_ratios = np.array(
            [result.final_stats["minus_ratio"] for result in mode_results],
            dtype=float,
        )
        effective_bits = np.array(
            [result.final_stats["effective_bits"] for result in mode_results],
            dtype=float,
        )
        record = make_summary_record(
            run_id=f"{SCRIPT}-summary-mode-{mode}",
            script=SCRIPT,
            mode=mode,
            lr=lr,
            config=config,
            mean_final_loss=float(losses.mean()),
            std_final_loss=float(losses.std()),
            mean_zero_ratio=float(zero_ratios.mean()),
            std_zero_ratio=float(zero_ratios.std()),
            mean_plus_ratio=float(plus_ratios.mean()),
            std_plus_ratio=float(plus_ratios.std()),
            mean_minus_ratio=float(minus_ratios.mean()),
            std_minus_ratio=float(minus_ratios.std()),
            mean_effective_bits=float(effective_bits.mean()),
            std_effective_bits=float(effective_bits.std()),
            optimizer_name="chimera21",
        )
        log_benchmark_record(
            prefix="compare",
            record=record,
            jsonl_path=jsonl_path,
            echo=True,
        )


def run_mode(
    *,
    mode: str,
    seed: int,
    init_seed: int,
    config: dict[str, Any],
    steps: int,
    batch_size: int,
    in_features: int,
    out_features: int,
    lr: float,
    jsonl_path: Path | None = None,
    echo: bool = True,
) -> RunResult:
    maybe_seed(seed)
    x = torch.randn(batch_size, in_features)
    target = torch.randn(batch_size, out_features)

    maybe_seed(init_seed)
    layer = build_layer(
        mode,
        config,
        in_features=in_features,
        out_features=out_features,
        steps=steps,
    )
    optimizer = build_optimizer(layer, config, lr=lr)
    loss_fn = torch.nn.MSELoss()
    log_steps = {1, max(1, steps // 2), steps}
    final_loss = float("nan")

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        pred = layer(x)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

        if step in log_steps:
            stats = layer.get_last_stats()
            record = make_benchmark_record(
                run_id=f"{SCRIPT}-mode-{mode}-seed-{seed}",
                script=SCRIPT,
                mode=mode,
                seed=seed,
                init_seed=init_seed,
                step=step,
                loss=final_loss,
                stats=stats,
                lr=lr,
                config=config,
                record_type="final" if step == steps else "step",
            )
            log_benchmark_record(
                prefix="compare",
                record=record,
                jsonl_path=jsonl_path,
                final=(step == steps),
                echo=echo,
            )

    return RunResult(
        mode=mode,
        seed=seed,
        final_loss=final_loss,
        final_stats=layer.get_last_stats(),
    )


def run_benchmark(
    *,
    config: dict[str, Any],
    args: argparse.Namespace,
    echo: bool = True,
) -> list[RunResult]:
    _validate_args(args)
    if args.log_jsonl is not None:
        args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.log_jsonl.write_text("", encoding="utf-8")
    quant_modes = list(config.get("quant_modes", ["strict_bitnet", "chimera"]))
    if "strict_bitnet" not in quant_modes or "chimera" not in quant_modes:
        raise ValueError("config quant_modes must include strict_bitnet and chimera")

    results: list[RunResult] = []
    for seed_offset in range(args.seeds):
        seed = SEED_BASE + seed_offset
        init_seed = seed + INIT_SEED_OFFSET
        for mode in quant_modes:
            results.append(
                run_mode(
                    mode=str(mode),
                    seed=seed,
                    init_seed=init_seed,
                    config=config,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    in_features=args.in_features,
                    out_features=args.out_features,
                    lr=args.lr,
                    jsonl_path=args.log_jsonl,
                    echo=echo,
                )
            )
    if echo:
        _log_summary(
            results,
            config=config,
            lr=float(args.lr),
            jsonl_path=args.log_jsonl,
        )
        print("[compare] status=passed")
    return results


def main(argv: list[str] | None = None) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    parser = build_parser(config)
    args = parser.parse_args(argv)
    run_benchmark(config=config, args=args)


if __name__ == "__main__":
    main()
