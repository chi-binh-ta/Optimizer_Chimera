from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from compare_optimizers import run as run_compare
except ModuleNotFoundError:  # pragma: no cover - used when imported in tests
    from experiments.compare_optimizers import run as run_compare

from chimera.optimizer import load_config


PROBLEMS = ("regression", "sparse_relu", "noisy_quadratic", "saddle")


def build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    """Build CLI for the full optimizer stress suite."""
    parser = argparse.ArgumentParser(description="Run all optimizer robustness stress problems.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--in-features", type=int, default=10)
    parser.add_argument("--hidden-features", type=int, default=8)
    parser.add_argument("--lr", type=float, default=float(config["lr"]))
    parser.add_argument("--noise-scale", type=float, default=2.0)
    parser.add_argument("--sparsity", type=float, default=0.7)
    parser.add_argument("--timing-warmup-steps", type=int, default=0)
    parser.add_argument("--log-jsonl", type=Path, default=Path("outputs/optimizer_stress_suite.jsonl"))
    return parser


def _problem_args(args: argparse.Namespace, problem: str) -> argparse.Namespace:
    """Create compare_optimizers args for one problem."""
    return argparse.Namespace(
        steps=args.steps,
        seed=args.seed,
        batch_size=args.batch_size,
        in_features=args.in_features,
        hidden_features=args.hidden_features,
        lr=args.lr,
        problem=problem,
        noise_scale=args.noise_scale,
        sparsity=args.sparsity,
        timing_warmup_steps=args.timing_warmup_steps,
        log_jsonl=args.log_jsonl,
        append_log=True,
    )


def run(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Run every optimizer stress problem into one JSONL file."""
    if args.log_jsonl is not None:
        args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.log_jsonl.write_text("", encoding="utf-8")
    all_summaries: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        summaries = run_compare(_problem_args(args, problem), config)
        all_summaries.extend(summaries)

    print("[optimizer_stress_suite] problem optimizer_name final_loss mean_step_time_ms mean_step_time_ms_after_warmup median_step_time_ms_after_warmup mean_collision_score mean_zero_grad_ratio optimizer_state_bytes psi_state_bytes")
    for summary in all_summaries:
        print(
            "[optimizer_stress_suite] {problem} {optimizer_name} {final_loss:.6f} "
            "{mean_step_time_ms} {mean_step_time_ms_after_warmup} "
            "{median_step_time_ms_after_warmup} {mean_collision_score} {mean_zero_grad_ratio} "
            "{optimizer_state_bytes} {psi_state_bytes}".format(**summary)
        )
    print("[optimizer_stress_suite] status=passed")
    return all_summaries


def main(argv: list[str] | None = None) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    parser = build_parser(config)
    args = parser.parse_args(argv)
    run(args, config)


if __name__ == "__main__":
    main()
