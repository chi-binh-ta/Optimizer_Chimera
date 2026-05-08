from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chimera.optimizer import load_config
try:
    from train_resnet_cifar import build_parser, run
except ModuleNotFoundError:  # pragma: no cover - used when imported as a package in tests
    from experiments.train_resnet_cifar import build_parser, run


def build_ablation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CIFAR Chimera ablation modes.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--target-bits", type=float, default=1.32)
    parser.add_argument("--target-branch", choices=("sparse", "dense"), default="sparse")
    parser.add_argument("--target-zero-ratio", type=float, default=None)
    parser.add_argument("--log-jsonl", type=Path, default=Path("outputs/cifar_ablation.jsonl"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--first-last-fp32", action="store_true")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--controller-frequency", choices=("epoch", "batch"), default="epoch")
    parser.add_argument(
        "--controller-affects",
        choices=("alpha_target", "alpha_override"),
        default="alpha_override",
    )
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    return parser


def main(argv: list[str] | None = None) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    args = build_ablation_parser().parse_args(argv)
    args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.log_jsonl.write_text("", encoding="utf-8")
    lr = float(config["lr"]) if args.lr is None else float(args.lr)

    base_parser = build_parser(config)
    runs = [
        ("fp32_baseline", "sgd", "fp32_baseline"),
        ("strict_bitnet", "chimera21", "strict_bitnet"),
        ("chimera", "chimera21", "chimera_sparse"),
    ]
    for index, (mode, optimizer_name, suffix) in enumerate(runs):
        run_args = base_parser.parse_args(
            [
                "--dataset-root",
                str(args.dataset_root),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--lr",
                str(lr),
                "--mode",
                mode,
                "--optimizer",
                optimizer_name,
                "--target-bits",
                str(args.target_bits),
                "--target-branch",
                args.target_branch,
                "--log-jsonl",
                str(args.log_jsonl),
                "--device",
                args.device,
                "--seed",
                str(args.seed + index),
                "--max-batches",
                str(args.max_batches),
                "--eval-max-batches",
                str(args.eval_max_batches),
                "--controller-frequency",
                args.controller_frequency,
                "--controller-affects",
                args.controller_affects,
                "--num-threads",
                str(args.num_threads),
                "--checkpoint-dir",
                str(args.checkpoint_dir),
                "--run-name",
                f"ablation_{suffix}_seed{args.seed + index}",
            ]
        )
        run_args.log_jsonl = args.log_jsonl
        run_args.download = args.download
        run_args.synthetic = args.synthetic
        run_args.first_last_fp32 = args.first_last_fp32
        run_args.save_checkpoint = args.save_checkpoint
        run_args.target_zero_ratio = args.target_zero_ratio
        run_args.append_log = True
        run(args=run_args, config=config)
    print("[cifar_ablation] status=passed")


if __name__ == "__main__":
    main()
