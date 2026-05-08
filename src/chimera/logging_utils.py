"""Small benchmark logging helpers for Chimera experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JSONL_FIELDS = (
    "run_id",
    "script",
    "mode",
    "record_type",
    "seed",
    "init_seed",
    "step",
    "loss",
    "gamma",
    "alpha",
    "zero_ratio",
    "plus_ratio",
    "minus_ratio",
    "effective_bits",
    "alpha_used",
    "lr",
    "optimizer_name",
    "target_bits",
    "stat_mode",
    "rho_s",
    "rho_psi",
    "lambda_gate",
    "kappa_min",
    "kappa_max",
    "timestamp_utc",
)


def timestamp_utc() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_benchmark_record(
    *,
    run_id: str,
    script: str,
    mode: str,
    seed: int,
    init_seed: int,
    step: int,
    loss: float,
    stats: dict[str, float],
    lr: float,
    config: dict[str, Any],
    record_type: str = "step",
    optimizer_name: str = "chimera21",
    target_bits: float | None = None,
) -> dict[str, Any]:
    """Build one normalized benchmark log record."""
    if record_type not in {"step", "batch", "epoch", "eval", "final", "summary"}:
        raise ValueError("invalid record_type")
    record = {
        "run_id": run_id,
        "script": script,
        "mode": mode,
        "record_type": record_type,
        "seed": int(seed),
        "init_seed": int(init_seed),
        "step": int(step),
        "loss": float(loss),
        "gamma": float(stats["gamma"]),
        "alpha": float(stats["alpha"]),
        "zero_ratio": float(stats["zero_ratio"]),
        "plus_ratio": float(stats["plus_ratio"]),
        "minus_ratio": float(stats["minus_ratio"]),
        "effective_bits": float(stats["effective_bits"]),
        "alpha_used": mode == "chimera",
        "lr": float(lr),
        "optimizer_name": optimizer_name,
        "target_bits": None if target_bits is None else float(target_bits),
        "stat_mode": str(config["stat_mode"]),
        "rho_s": float(config["rho_s"]),
        "rho_psi": float(config["rho_psi"]),
        "lambda_gate": float(config["lambda_gate"]),
        "kappa_min": float(config["kappa_min"]),
        "kappa_max": float(config["kappa_max"]),
        "timestamp_utc": timestamp_utc(),
    }
    missing = set(JSONL_FIELDS) - set(record)
    if missing:
        raise ValueError(f"missing benchmark log fields: {sorted(missing)}")
    return record


def make_summary_record(
    *,
    run_id: str,
    script: str,
    mode: str,
    lr: float,
    config: dict[str, Any],
    mean_final_loss: float,
    std_final_loss: float,
    mean_zero_ratio: float,
    std_zero_ratio: float,
    mean_plus_ratio: float,
    std_plus_ratio: float,
    mean_minus_ratio: float,
    std_minus_ratio: float,
    mean_effective_bits: float,
    std_effective_bits: float,
    optimizer_name: str = "chimera21",
    target_bits: float | None = None,
) -> dict[str, Any]:
    """Build one aggregate summary record for JSONL output."""
    record = {
        "run_id": run_id,
        "script": script,
        "mode": mode,
        "record_type": "summary",
        "seed": None,
        "init_seed": None,
        "step": None,
        "loss": float(mean_final_loss),
        "gamma": None,
        "alpha": None,
        "zero_ratio": float(mean_zero_ratio),
        "plus_ratio": None,
        "minus_ratio": None,
        "effective_bits": float(mean_effective_bits),
        "alpha_used": mode == "chimera",
        "lr": float(lr),
        "optimizer_name": optimizer_name,
        "target_bits": None if target_bits is None else float(target_bits),
        "stat_mode": str(config["stat_mode"]),
        "rho_s": float(config["rho_s"]),
        "rho_psi": float(config["rho_psi"]),
        "lambda_gate": float(config["lambda_gate"]),
        "kappa_min": float(config["kappa_min"]),
        "kappa_max": float(config["kappa_max"]),
        "timestamp_utc": timestamp_utc(),
        "mean_final_loss": float(mean_final_loss),
        "std_final_loss": float(std_final_loss),
        "mean_zero_ratio": float(mean_zero_ratio),
        "std_zero_ratio": float(std_zero_ratio),
        "mean_plus_ratio": float(mean_plus_ratio),
        "std_plus_ratio": float(std_plus_ratio),
        "mean_minus_ratio": float(mean_minus_ratio),
        "std_minus_ratio": float(std_minus_ratio),
        "mean_effective_bits": float(mean_effective_bits),
        "std_effective_bits": float(std_effective_bits),
    }
    missing = set(JSONL_FIELDS) - set(record)
    if missing:
        raise ValueError(f"missing summary log fields: {sorted(missing)}")
    return record


def format_step_record(prefix: str, record: dict[str, Any]) -> str:
    """Format one human-readable step record."""
    return (
        "[{prefix}] record_type={record_type} mode={mode} seed={seed} step={step} loss={loss:.6f} "
        "gamma={gamma:.6f} alpha={alpha:.6f} zero_ratio={zero_ratio:.6f} "
        "plus_ratio={plus_ratio:.6f} minus_ratio={minus_ratio:.6f} "
        "effective_bits={effective_bits:.6f} lr={lr:.6f}"
    ).format(prefix=prefix, **record)


def format_final_record(prefix: str, record: dict[str, Any]) -> str:
    """Format one human-readable final record."""
    return (
        "[{prefix}] record_type={record_type} mode={mode} seed={seed} final_loss={loss:.6f} "
        "final_gamma={gamma:.6f} final_alpha={alpha:.6f} "
        "final_zero_ratio={zero_ratio:.6f} final_plus_ratio={plus_ratio:.6f} "
        "final_minus_ratio={minus_ratio:.6f} final_effective_bits={effective_bits:.6f} "
        "lr={lr:.6f}"
    ).format(prefix=prefix, **record)


def format_summary_record(prefix: str, record: dict[str, Any]) -> str:
    """Format one human-readable summary record."""
    return (
        "[{prefix}] record_type={record_type} mode={mode} mean_final_loss={mean_final_loss:.6f} "
        "std_final_loss={std_final_loss:.6f} mean_zero_ratio={mean_zero_ratio:.6f} "
        "std_zero_ratio={std_zero_ratio:.6f} mean_plus_ratio={mean_plus_ratio:.6f} "
        "mean_minus_ratio={mean_minus_ratio:.6f} "
        "mean_effective_bits={mean_effective_bits:.6f} "
        "std_effective_bits={std_effective_bits:.6f}"
    ).format(prefix=prefix, **record)


def _console_prefix(prefix: str, record: dict[str, Any]) -> str:
    record_type = str(record.get("record_type", "step"))
    if prefix == "resnet_cifar":
        return f"{prefix}_{record_type}"
    if record_type == "summary":
        return f"{prefix}_summary"
    return prefix


def write_jsonl_record(path: str | Path, record: dict[str, Any]) -> None:
    """Append one record to a JSONL file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def log_benchmark_record(
    *,
    prefix: str,
    record: dict[str, Any],
    jsonl_path: str | Path | None = None,
    final: bool = False,
    echo: bool = True,
) -> None:
    """Emit a console record and optionally append it to JSONL."""
    if echo:
        console_prefix = _console_prefix(prefix, record)
        if record.get("record_type") == "summary":
            formatter = format_summary_record
        else:
            formatter = format_final_record if final else format_step_record
        print(formatter(console_prefix, record))
    if jsonl_path is not None:
        write_jsonl_record(jsonl_path, record)
