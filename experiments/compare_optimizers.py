from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chimera import (
    Chimera21,
    optimizer_state_memory_bytes,
    optimizer_state_memory_summary,
)
from chimera.logging_utils import timestamp_utc
from chimera.optimizer import load_config
from chimera.utils import maybe_seed


SCRIPT = "compare_optimizers"


def build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare optimizer robustness variants.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--in-features", type=int, default=10)
    parser.add_argument("--hidden-features", type=int, default=8)
    parser.add_argument("--lr", type=float, default=float(config["lr"]))
    parser.add_argument(
        "--problem",
        choices=("regression", "sparse_relu", "noisy_quadratic", "saddle"),
        default="regression",
    )
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--timing-warmup-steps", type=int, default=0)
    parser.add_argument("--log-jsonl", type=Path, default=None)
    return parser


def make_problem(
    *,
    seed: int,
    batch_size: int,
    in_features: int,
    problem: str = "regression",
    noise_scale: float = 1.0,
    sparsity: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic synthetic inputs and targets."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, in_features, generator=generator)
    if problem == "regression":
        x[:, ::3] = 0.0
    elif problem == "sparse_relu":
        mask = torch.rand(batch_size, in_features, generator=generator) > sparsity
        x = x.clamp_min(0.0) * mask.to(x.dtype)
    true_w = torch.randn(in_features, 1, generator=generator)
    noise = noise_scale * 0.05 * torch.randn(batch_size, 1, generator=generator)
    target = x @ true_w + noise
    return x, target


def make_model(
    *,
    seed: int,
    in_features: int,
    hidden_features: int,
    problem: str = "regression",
) -> torch.nn.Module:
    """Create a small deterministic regression model."""
    maybe_seed(seed)
    activation: torch.nn.Module = (
        torch.nn.ReLU() if problem == "sparse_relu" else torch.nn.Tanh()
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(in_features, hidden_features),
        activation,
        torch.nn.Linear(hidden_features, 1),
    )
    if problem == "sparse_relu":
        with torch.no_grad():
            first = model[0]
            if isinstance(first, torch.nn.Linear):
                first.bias.fill_(-0.75)
    return model


def make_noise_schedule(
    *,
    model: torch.nn.Module,
    seed: int,
    steps: int,
    noise_scale: float,
) -> list[list[torch.Tensor]]:
    """Create deterministic gradient-noise tensors for stress problems."""
    generator = torch.Generator().manual_seed(seed + 10_000)
    schedule: list[list[torch.Tensor]] = []
    for _ in range(steps):
        step_noise = []
        for parameter in model.parameters():
            step_noise.append(
                noise_scale
                * torch.randn(
                    tuple(parameter.shape),
                    generator=generator,
                    dtype=parameter.dtype,
                )
            )
        schedule.append(step_noise)
    return schedule


def compute_loss(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    loss_fn: torch.nn.Module,
    problem: str,
    step: int,
    noise_schedule: list[list[torch.Tensor]],
) -> torch.Tensor:
    """Compute one problem-specific synthetic loss."""
    if problem == "noisy_quadratic":
        loss = torch.zeros((), dtype=x.dtype)
        for parameter, noise in zip(model.parameters(), noise_schedule[step - 1]):
            loss = loss + 0.5 * parameter.pow(2).mean()
            loss = loss + (parameter * noise.to(parameter)).mean()
        return loss
    prediction = model(x)
    loss = loss_fn(prediction, target)
    if problem == "saddle":
        saddle_term = torch.zeros((), dtype=loss.dtype)
        for parameter in model.parameters():
            saddle_term = saddle_term + (parameter.pow(4) - parameter.pow(2)).mean()
        loss = loss + 0.01 * saddle_term
    return loss


def make_optimizer(
    name: str,
    parameters,
    *,
    lr: float,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    """Build one optimizer variant."""
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr)

    kwargs = {
        "lr": lr,
        "beta1": float(config["beta1"]),
        "beta2": float(config["beta2"]),
        "eps_opt": float(config["eps_opt"]),
        "weight_decay": float(config["weight_decay"]),
        "rho_psi": float(config["rho_psi"]),
        "lambda_gate": float(config["lambda_gate"]),
        "kappa_min": float(config["kappa_min"]),
        "kappa_max": float(config["kappa_max"]),
        "log_diagnostics": True,
    }
    if name == "chimera21":
        return Chimera21(parameters, **kwargs)
    if name == "chimera21_crossgate":
        return Chimera21(
            parameters,
            **kwargs,
            kappa_gate_mode="noise_ratio",
            noise_threshold=float(config.get("noise_threshold", 2.0)),
            gate_sharpness=float(config.get("gate_sharpness", 4.0)),
        )
    if name == "chimera21_zero_freeze":
        return Chimera21(parameters, **kwargs, zero_grad_policy="freeze")
    if name == "chimera21_zero_decay":
        return Chimera21(parameters, **kwargs, zero_grad_policy="decay")
    if name == "chimera21_zero_inertia":
        return Chimera21(parameters, **kwargs, zero_grad_policy="inertia")
    if name == "chimera21_psi_int8":
        return Chimera21(parameters, **kwargs, psi_storage="int8")
    raise ValueError(f"unknown optimizer variant: {name}")


def optimizer_metadata(name: str, optimizer: torch.optim.Optimizer) -> dict[str, str]:
    """Return optimizer policy metadata for logging."""
    if not isinstance(optimizer, Chimera21):
        return {
            "psi_storage": "fp32",
            "zero_grad_policy": "n/a",
            "kappa_gate_mode": "n/a",
        }
    group = optimizer.param_groups[0]
    return {
        "psi_storage": str(group["psi_storage"]),
        "zero_grad_policy": str(group["zero_grad_policy"]),
        "kappa_gate_mode": str(group["kappa_gate_mode"]),
    }


def psi_fp32_reference_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Return the fp32 psi bytes implied by Chimera state shapes."""
    if not isinstance(optimizer, Chimera21):
        return 0
    total = 0
    for state in optimizer.state.values():
        psi = state.get("psi")
        if isinstance(psi, torch.Tensor):
            total += int(psi.numel() * 4)
    return total


def diagnostic_record(
    *,
    optimizer_name: str,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    lr: float,
    seed: int,
    problem: str,
    step_time_ms: float,
) -> dict[str, Any]:
    """Build one optimizer diagnostic JSONL record."""
    diagnostics = (
        optimizer.last_diagnostics
        if isinstance(optimizer, Chimera21)
        else {}
    )
    metadata = optimizer_metadata(optimizer_name, optimizer)
    memory = optimizer_state_memory_summary(optimizer)
    psi_reference = psi_fp32_reference_bytes(optimizer)
    psi_bytes = int(memory["psi"])
    return {
        "run_id": f"{SCRIPT}-{optimizer_name}-seed-{seed}",
        "script": SCRIPT,
        "record_type": "step",
        "problem": problem,
        "optimizer_name": optimizer_name,
        "step": int(step),
        "loss": float(loss),
        "step_time_ms": float(step_time_ms),
        "mean_kappa": diagnostics.get("mean_kappa"),
        "mean_noise_ratio": diagnostics.get("mean_noise_ratio"),
        "active_mean_effective_lr": diagnostics.get("active_mean_effective_lr"),
        "applied_update_norm": diagnostics.get("applied_update_norm"),
        "param_update_ratio": diagnostics.get("param_update_ratio"),
        "mean_update_abs": diagnostics.get("mean_update_abs"),
        "max_update_abs": diagnostics.get("max_update_abs"),
        "collision_score": diagnostics.get("collision_score"),
        "zero_grad_ratio": diagnostics.get("zero_grad_ratio"),
        "optimizer_state_bytes": optimizer_state_memory_bytes(optimizer),
        "psi_state_bytes": psi_bytes,
        "psi_memory_ratio_vs_fp32": (
            None if psi_reference == 0 else float(psi_bytes) / float(psi_reference)
        ),
        "psi_storage": metadata["psi_storage"],
        "zero_grad_policy": metadata["zero_grad_policy"],
        "kappa_gate_mode": metadata["kappa_gate_mode"],
        "lr": float(lr),
        "seed": int(seed),
        "timestamp_utc": timestamp_utc(),
    }


def _mean_defined(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record[key] for record in records if record.get(key) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _max_defined(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record[key] for record in records if record.get(key) is not None]
    if not values:
        return None
    return float(max(values))


def _median(values: list[float]) -> float | None:
    """Return median for a non-empty list, or None."""
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _timing_after_warmup(
    records: list[dict[str, Any]],
    timing_warmup_steps: int,
) -> list[float]:
    """Return step timings after warmup, falling back to all records if empty."""
    values = [
        float(record["step_time_ms"])
        for record in records
        if int(record["step"]) > timing_warmup_steps
    ]
    if values:
        return values
    return [float(record["step_time_ms"]) for record in records]


def summary_record(
    records: list[dict[str, Any]],
    *,
    timing_warmup_steps: int = 0,
) -> dict[str, Any]:
    """Build one optimizer stress summary record."""
    final = records[-1]
    timing_values = _timing_after_warmup(records, timing_warmup_steps)
    return {
        "run_id": final["run_id"],
        "script": SCRIPT,
        "record_type": "summary",
        "problem": final["problem"],
        "optimizer_name": final["optimizer_name"],
        "final_loss": float(final["loss"]),
        "mean_step_time_ms": _mean_defined(records, "step_time_ms"),
        "mean_step_time_ms_after_warmup": float(sum(timing_values) / len(timing_values)),
        "median_step_time_ms_after_warmup": _median(timing_values),
        "timing_warmup_steps": int(timing_warmup_steps),
        "mean_collision_score": _mean_defined(records, "collision_score"),
        "max_collision_score": _max_defined(records, "collision_score"),
        "mean_zero_grad_ratio": _mean_defined(records, "zero_grad_ratio"),
        "mean_active_effective_lr": _mean_defined(records, "active_mean_effective_lr"),
        "optimizer_state_bytes": int(final["optimizer_state_bytes"]),
        "psi_state_bytes": int(final["psi_state_bytes"]),
        "psi_memory_ratio_vs_fp32": final["psi_memory_ratio_vs_fp32"],
        "psi_storage": final["psi_storage"],
        "zero_grad_policy": final["zero_grad_policy"],
        "kappa_gate_mode": final["kappa_gate_mode"],
        "lr": final["lr"],
        "seed": final["seed"],
        "timestamp_utc": timestamp_utc(),
    }


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one optimizer record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_variant(
    *,
    optimizer_name: str,
    args: argparse.Namespace,
    config: dict[str, Any],
    x: torch.Tensor,
    target: torch.Tensor,
    initial_state: dict[str, torch.Tensor],
    noise_schedule: list[list[torch.Tensor]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train one optimizer variant and emit records."""
    model = make_model(
        seed=args.seed,
        in_features=args.in_features,
        hidden_features=args.hidden_features,
        problem=args.problem,
    )
    model.load_state_dict(initial_state)
    optimizer = make_optimizer(
        optimizer_name,
        model.parameters(),
        lr=float(args.lr),
        config=config,
    )
    loss_fn = torch.nn.MSELoss()
    final_record: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for step in range(1, args.steps + 1):
        step_start = time.perf_counter()
        optimizer.zero_grad()
        loss = compute_loss(
            model=model,
            x=x,
            target=target,
            loss_fn=loss_fn,
            problem=args.problem,
            step=step,
            noise_schedule=noise_schedule,
        )
        loss.backward()
        optimizer.step()
        step_time_ms = max((time.perf_counter() - step_start) * 1000.0, 1.0e-9)
        record = diagnostic_record(
            optimizer_name=optimizer_name,
            optimizer=optimizer,
            step=step,
            loss=float(loss.item()),
            lr=float(args.lr),
            seed=int(args.seed),
            problem=args.problem,
            step_time_ms=step_time_ms,
        )
        final_record = record
        records.append(record)
        if args.log_jsonl is not None:
            write_jsonl(args.log_jsonl, record)
        if step in {1, args.steps}:
            message = (
                "[optimizer] optimizer_name={optimizer_name} step={step} "
                "loss={loss:.6f} mean_kappa={mean_kappa} "
                "mean_noise_ratio={mean_noise_ratio} collision_score={collision_score}"
            ).format(**record)
            print(message)
    summary = summary_record(
        records,
        timing_warmup_steps=int(args.timing_warmup_steps),
    )
    if args.log_jsonl is not None:
        write_jsonl(args.log_jsonl, summary)
    return final_record, summary


def run(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    if args.steps <= 0:
        raise SystemExit("--steps must be positive.")
    if args.batch_size <= 0 or args.in_features <= 0 or args.hidden_features <= 0:
        raise SystemExit("--batch-size, --in-features, and --hidden-features must be positive.")
    if args.lr < 0:
        raise SystemExit("--lr must be non-negative.")
    if not 0.0 <= args.sparsity <= 1.0:
        raise SystemExit("--sparsity must be in [0, 1].")
    if args.timing_warmup_steps < 0:
        raise SystemExit("--timing-warmup-steps must be non-negative.")
    if args.log_jsonl is not None:
        args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if not getattr(args, "append_log", False):
            args.log_jsonl.write_text("", encoding="utf-8")

    maybe_seed(args.seed)
    x, target = make_problem(
        seed=args.seed,
        batch_size=args.batch_size,
        in_features=args.in_features,
        problem=args.problem,
        noise_scale=args.noise_scale,
        sparsity=args.sparsity,
    )
    base_model = make_model(
        seed=args.seed,
        in_features=args.in_features,
        hidden_features=args.hidden_features,
        problem=args.problem,
    )
    noise_schedule = make_noise_schedule(
        model=base_model,
        seed=args.seed,
        steps=args.steps,
        noise_scale=args.noise_scale,
    )
    initial_state = {
        key: value.detach().clone()
        for key, value in base_model.state_dict().items()
    }
    variants = (
        "adam",
        "chimera21",
        "chimera21_crossgate",
        "chimera21_zero_freeze",
        "chimera21_zero_decay",
        "chimera21_zero_inertia",
        "chimera21_psi_int8",
    )
    summaries = []
    for name in variants:
        _, summary = run_variant(
            optimizer_name=name,
            args=args,
            config=config,
            x=x,
            target=target,
            initial_state=initial_state,
            noise_schedule=noise_schedule,
        )
        summaries.append(summary)
    print("[optimizer_summary] optimizer_name final_loss mean_step_time_ms mean_step_time_ms_after_warmup median_step_time_ms_after_warmup mean_collision_score mean_zero_grad_ratio mean_active_effective_lr optimizer_state_bytes psi_state_bytes")
    for summary in summaries:
        print(
            "[optimizer_summary] {optimizer_name} {final_loss:.6f} {mean_step_time_ms} "
            "{mean_step_time_ms_after_warmup} {median_step_time_ms_after_warmup} "
            "{mean_collision_score} {mean_zero_grad_ratio} {mean_active_effective_lr} "
            "{optimizer_state_bytes} {psi_state_bytes}".format(**summary)
        )
    print("[optimizer] status=passed")
    return summaries


def main(argv: list[str] | None = None) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    parser = build_parser(config)
    args = parser.parse_args(argv)
    run(args, config)


if __name__ == "__main__":
    main()
