from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chimera import Chimera21, TargetBitsController, entropy_from_ternary_ratios
from chimera.logging_utils import (
    log_benchmark_record,
    make_benchmark_record,
    make_summary_record,
    timestamp_utc,
)
from chimera.models.resnet_cifar import ChimeraResNet20
from chimera.optimizer import load_config
from chimera.utils import maybe_seed


SCRIPT = "train_resnet_cifar"


def build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    benchmark = config.get("benchmark", {})
    if not isinstance(benchmark, dict):
        benchmark = {}
    parser = argparse.ArgumentParser(description="CPU-safe CIFAR ResNet Chimera trainer.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=int(benchmark.get("batch_size", 16)))
    parser.add_argument("--eval-batch-size", type=int, default=int(benchmark.get("batch_size", 16)))
    parser.add_argument("--lr", type=float, default=float(config["lr"]))
    parser.add_argument(
        "--mode",
        choices=("fp32_baseline", "strict_bitnet", "chimera"),
        default="chimera",
    )
    parser.add_argument("--target-bits", type=float, default=float(config.get("target_bits", 1.32)))
    parser.add_argument(
        "--target-branch",
        choices=("sparse", "dense"),
        default=str(config.get("target_branch", "sparse")),
    )
    parser.add_argument("--target-zero-ratio", type=float, default=None)
    parser.add_argument(
        "--controller-frequency",
        choices=("epoch", "batch"),
        default="epoch",
    )
    parser.add_argument(
        "--controller-affects",
        choices=("alpha_target", "alpha_override"),
        default="alpha_target",
    )
    parser.add_argument("--log-jsonl", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--optimizer", choices=("chimera21", "sgd", "adamw"), default=None)
    parser.add_argument("--first-last-fp32", action="store_true")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--run-name", type=str, default=None)
    return parser


def quant_config_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract BitConv2d/BitLinear quantization config from default config."""
    keys = (
        "stat_mode",
        "rho_s",
        "alpha_min",
        "alpha_target",
        "warmup_steps",
        "eps_gamma",
        "eps_beta",
        "ste_clip",
    )
    return {key: config[key] for key in keys if key in config}


def resolve_optimizer_name(mode: str, optimizer_name: str | None) -> str:
    """Resolve default optimizer by model mode."""
    if optimizer_name is not None:
        return optimizer_name
    if mode == "fp32_baseline":
        return "sgd"
    return "chimera21"


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    optimizer_name: str,
    lr: float,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    """Build the requested optimizer."""
    if optimizer_name == "chimera21":
        return Chimera21(
            parameters,
            lr=lr,
            beta1=float(config["beta1"]),
            beta2=float(config["beta2"]),
            eps_opt=float(config["eps_opt"]),
            weight_decay=float(config["weight_decay"]),
            rho_psi=float(config["rho_psi"]),
            lambda_gate=float(config["lambda_gate"]),
            kappa_min=float(config["kappa_min"]),
            kappa_max=float(config["kappa_max"]),
            log_diagnostics=bool(config.get("log_diagnostics", False)),
            zero_grad_policy=str(config.get("zero_grad_policy", "standard")),
            idle_decay=float(config.get("idle_decay", 0.95)),
            kappa_gate_mode=str(config.get("kappa_gate_mode", "none")),
            noise_threshold=float(config.get("noise_threshold", 2.0)),
            gate_sharpness=float(config.get("gate_sharpness", 4.0)),
            psi_storage=str(config.get("psi_storage", "fp32")),
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=float(config["weight_decay"]))
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=float(config["weight_decay"]))
    raise ValueError("optimizer_name must be chimera21, sgd, or adamw")


def _load_cifar10(dataset_root: Path, *, train: bool, download: bool):
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - optional local dependency
        raise SystemExit(
            "torchvision is not installed. Install it manually to run CIFAR experiments."
        ) from exc

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    try:
        return datasets.CIFAR10(
            root=str(dataset_root),
            train=train,
            transform=transform,
            download=download,
        )
    except RuntimeError as exc:  # pragma: no cover - depends on local data
        raise SystemExit(
            "CIFAR-10 is unavailable at the requested dataset root. "
            "Pass --download to download explicitly."
        ) from exc


def _quant_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    modules = []
    for module in model.modules():
        if callable(getattr(module, "get_last_stats", None)):
            modules.append(module)
    return modules


def model_stats(model: torch.nn.Module) -> dict[str, float]:
    """Return unweighted and global ternary stats across quantized layers."""
    modules = _quant_modules(model)
    if not modules:
        return {
            "gamma": 0.0,
            "alpha": 0.0,
            "zero_ratio": 0.0,
            "plus_ratio": 0.0,
            "minus_ratio": 0.0,
            "effective_bits": 0.0,
            "alpha_used_mean": 0.0,
            "alpha_target_mean": 0.0,
            "mean_zero_ratio": 0.0,
            "mean_plus_ratio": 0.0,
            "mean_minus_ratio": 0.0,
            "unweighted_mean_zero_ratio": 0.0,
            "unweighted_mean_plus_ratio": 0.0,
            "unweighted_mean_minus_ratio": 0.0,
            "unweighted_mean_effective_bits": 0.0,
            "global_zero_ratio": 0.0,
            "global_plus_ratio": 0.0,
            "global_minus_ratio": 0.0,
            "global_effective_bits": 0.0,
        }
    stats = [module.get_last_stats() for module in modules]
    weights = [float(getattr(module, "weight_master").numel()) for module in modules]
    total = sum(weights)
    global_zero = sum(item["zero_ratio"] * weight for item, weight in zip(stats, weights)) / total
    global_plus = sum(item["plus_ratio"] * weight for item, weight in zip(stats, weights)) / total
    global_minus = sum(item["minus_ratio"] * weight for item, weight in zip(stats, weights)) / total
    global_effective_bits = entropy_from_ternary_ratios(global_zero, global_plus, global_minus)
    alpha_used_mean = float(sum(item["alpha"] for item in stats) / len(stats))
    alpha_target_mean = float(
        sum(float(getattr(module, "alpha_target", 0.0)) for module in modules) / len(modules)
    )
    unweighted_zero = float(sum(item["zero_ratio"] for item in stats) / len(stats))
    unweighted_plus = float(sum(item["plus_ratio"] for item in stats) / len(stats))
    unweighted_minus = float(sum(item["minus_ratio"] for item in stats) / len(stats))
    unweighted_bits = float(sum(item["effective_bits"] for item in stats) / len(stats))
    result = {
        "gamma": float(sum(item["gamma"] for item in stats) / len(stats)),
        "alpha": alpha_used_mean,
        "zero_ratio": float(global_zero),
        "plus_ratio": float(global_plus),
        "minus_ratio": float(global_minus),
        "effective_bits": float(global_effective_bits),
        "alpha_used_mean": alpha_used_mean,
        "alpha_target_mean": alpha_target_mean,
        "mean_zero_ratio": float(global_zero),
        "mean_plus_ratio": float(global_plus),
        "mean_minus_ratio": float(global_minus),
        "unweighted_mean_zero_ratio": unweighted_zero,
        "unweighted_mean_plus_ratio": unweighted_plus,
        "unweighted_mean_minus_ratio": unweighted_minus,
        "unweighted_mean_effective_bits": unweighted_bits,
        "global_zero_ratio": float(global_zero),
        "global_plus_ratio": float(global_plus),
        "global_minus_ratio": float(global_minus),
        "global_effective_bits": float(global_effective_bits),
    }
    return result


def apply_target_bits_control(
    model: torch.nn.Module,
    controller: TargetBitsController,
    *,
    current_zero_ratio: float,
    affects: str,
) -> tuple[float, float, float]:
    """Update controlled alpha and return new mean, delta, and error."""
    modules = _quant_modules(model)
    if not modules:
        return 0.0, 0.0, 0.0
    if affects not in {"alpha_target", "alpha_override"}:
        raise ValueError("affects must be alpha_target or alpha_override")
    next_alphas = []
    old_alphas = []
    error = float(current_zero_ratio) - float(controller.target_zero_ratio)
    for module in modules:
        if affects == "alpha_override":
            override = getattr(module, "alpha_override", None)
            current = float(override if override is not None else getattr(module, "alpha_target", 0.0))
        else:
            current = float(getattr(module, "alpha_target", 0.0))
        next_alpha = controller.update(current, current_zero_ratio)
        old_alphas.append(current)
        if affects == "alpha_override":
            setattr(module, "alpha_override", next_alpha)
        else:
            setattr(module, "alpha_target", next_alpha)
        next_alphas.append(next_alpha)
    old_mean = float(sum(old_alphas) / len(old_alphas))
    new_mean = float(sum(next_alphas) / len(next_alphas))
    return new_mean, new_mean - old_mean, error


def initialize_alpha_override(model: torch.nn.Module, alpha: float) -> None:
    """Set direct alpha override on all quantized execution layers."""
    for module in _quant_modules(model):
        setattr(module, "alpha_override", float(alpha))


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == target).sum().item()) / float(target.numel())


def resolve_device(device_name: str) -> torch.device:
    """Resolve a requested torch device with clear CUDA errors."""
    try:
        device = torch.device(device_name)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Invalid --device value: {device_name}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested with --device cuda, but torch.cuda.is_available() is false.")
    if device.type not in {"cpu", "cuda"}:
        raise SystemExit("Only --device cpu and --device cuda are supported.")
    return device


def _model_device(model: torch.nn.Module) -> torch.device:
    """Return the current model device."""
    return next(model.parameters()).device


def _epoch_record(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    optimizer_name: str,
    epoch: int,
    step: int,
    train_loss: float,
    train_accuracy: float,
    stats: dict[str, float],
    record_type: str,
) -> dict[str, Any]:
    record = make_benchmark_record(
        run_id=f"{SCRIPT}-mode-{args.mode}-seed-{args.seed}",
        script=SCRIPT,
        mode=args.mode,
        seed=args.seed,
        init_seed=args.seed,
        step=step,
        loss=train_loss,
        stats=stats,
        lr=float(args.lr),
        config=config,
        record_type=record_type,
        optimizer_name=optimizer_name,
        target_bits=float(args.target_bits),
    )
    record.update(
        {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "mean_zero_ratio": float(stats["zero_ratio"]),
            "mean_plus_ratio": float(stats["plus_ratio"]),
            "mean_minus_ratio": float(stats["minus_ratio"]),
            "mean_effective_bits": float(stats["effective_bits"]),
            "unweighted_mean_effective_bits": float(stats["unweighted_mean_effective_bits"]),
            "unweighted_mean_zero_ratio": float(stats["unweighted_mean_zero_ratio"]),
            "unweighted_mean_plus_ratio": float(stats["unweighted_mean_plus_ratio"]),
            "unweighted_mean_minus_ratio": float(stats["unweighted_mean_minus_ratio"]),
            "global_zero_ratio": float(stats["global_zero_ratio"]),
            "global_plus_ratio": float(stats["global_plus_ratio"]),
            "global_minus_ratio": float(stats["global_minus_ratio"]),
            "global_effective_bits": float(stats["global_effective_bits"]),
            "target_bits": float(args.target_bits),
            "target_branch": args.target_branch,
            "target_zero_ratio": None
            if args.target_zero_ratio is None
            else float(args.target_zero_ratio),
            "controller_frequency": args.controller_frequency,
            "controller_affects": args.controller_affects,
            "controller_error": float(stats.get("controller_error", 0.0)),
            "controller_update_delta": float(stats.get("controller_update_delta", 0.0)),
            "alpha_used_mean": float(stats["alpha_used_mean"]),
            "alpha_target_mean": float(stats["alpha_target_mean"]),
            "alpha_control_mean": float(stats.get("alpha_control_mean", stats["alpha_used_mean"])),
            "alpha_next_mean": float(stats.get("alpha_next_mean", stats["alpha_target_mean"])),
            "test_loss": None,
            "test_accuracy": None,
            "split": "train",
        }
    )
    return record


@torch.no_grad()
def evaluate(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    max_batches: int,
) -> tuple[float, float, dict[str, float]]:
    """Evaluate on a loader and return loss, accuracy, and current model stats."""
    total_loss = 0.0
    total_correct = 0.0
    total_seen = 0
    was_training = model.training
    device = _model_device(model)
    model.eval()
    try:
        for batch_idx, (x, y) in enumerate(loader, start=1):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            batch_size = int(y.numel())
            total_loss += float(loss.item()) * batch_size
            total_correct += _accuracy(logits, y) * batch_size
            total_seen += batch_size
            if max_batches and batch_idx >= max_batches:
                break
        stats = model_stats(model)
    finally:
        model.train(was_training)
    return (
        total_loss / max(total_seen, 1),
        total_correct / max(total_seen, 1),
        stats,
    )


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    args: argparse.Namespace,
    config: dict[str, Any],
    optimizer_name: str,
    epoch: int,
    global_step: int,
    controller: TargetBitsController | None = None,
    echo: bool = True,
) -> tuple[int, dict[str, Any]]:
    """Train one epoch and log batch/epoch records."""
    total_loss = 0.0
    total_correct = 0.0
    total_seen = 0
    last_stats = model_stats(model)
    device = _model_device(model)

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        global_step += 1

        batch_size = int(y.numel())
        total_loss += float(loss.item()) * batch_size
        total_correct += _accuracy(logits, y) * batch_size
        total_seen += batch_size
        last_stats = model_stats(model)
        last_stats["controller_error"] = (
            last_stats["global_zero_ratio"] - float(args.target_zero_ratio)
            if args.target_zero_ratio is not None
            else 0.0
        )
        last_stats["controller_update_delta"] = 0.0

        if (
            controller is not None
            and args.mode == "chimera"
            and args.controller_frequency == "batch"
        ):
            new_alpha, delta, error = apply_target_bits_control(
                model,
                controller,
                current_zero_ratio=last_stats["global_zero_ratio"],
                affects=args.controller_affects,
            )
            last_stats["controller_error"] = error
            last_stats["controller_update_delta"] = delta
            last_stats["alpha_target_mean"] = new_alpha

        batch_record = make_benchmark_record(
            run_id=f"{SCRIPT}-mode-{args.mode}-seed-{args.seed}",
            script=SCRIPT,
            mode=args.mode,
            seed=args.seed,
            init_seed=args.seed,
            step=global_step,
            loss=float(loss.item()),
            stats=last_stats,
            lr=float(args.lr),
            config=config,
            record_type="batch",
            optimizer_name=optimizer_name,
            target_bits=float(args.target_bits),
        )
        batch_record.update(
            {
                "epoch": int(epoch),
                "batch_idx": int(batch_idx),
                "train_loss": float(loss.item()),
                "train_accuracy": _accuracy(logits, y),
                "mean_zero_ratio": float(last_stats["zero_ratio"]),
                "mean_plus_ratio": float(last_stats["plus_ratio"]),
                "mean_minus_ratio": float(last_stats["minus_ratio"]),
                "mean_effective_bits": float(last_stats["effective_bits"]),
                "unweighted_mean_effective_bits": float(
                    last_stats["unweighted_mean_effective_bits"]
                ),
                "unweighted_mean_zero_ratio": float(last_stats["unweighted_mean_zero_ratio"]),
                "unweighted_mean_plus_ratio": float(last_stats["unweighted_mean_plus_ratio"]),
                "unweighted_mean_minus_ratio": float(last_stats["unweighted_mean_minus_ratio"]),
                "global_zero_ratio": float(last_stats["global_zero_ratio"]),
                "global_plus_ratio": float(last_stats["global_plus_ratio"]),
                "global_minus_ratio": float(last_stats["global_minus_ratio"]),
                "global_effective_bits": float(last_stats["global_effective_bits"]),
                "target_branch": args.target_branch,
                "target_zero_ratio": None
                if args.target_zero_ratio is None
                else float(args.target_zero_ratio),
                "controller_frequency": args.controller_frequency,
                "controller_affects": args.controller_affects,
                "controller_error": float(last_stats["controller_error"]),
                "controller_update_delta": float(last_stats["controller_update_delta"]),
                "alpha_used_mean": float(last_stats["alpha_used_mean"]),
                "alpha_target_mean": float(last_stats["alpha_target_mean"]),
                "alpha_control_mean": float(last_stats["alpha_used_mean"]),
                "alpha_next_mean": float(last_stats["alpha_target_mean"]),
                "test_loss": None,
                "test_accuracy": None,
                "split": "train",
            }
        )
        log_benchmark_record(
            prefix="resnet_cifar",
            record=batch_record,
            jsonl_path=args.log_jsonl,
            echo=echo and batch_idx == 1,
        )

        if args.max_batches and batch_idx >= args.max_batches:
            break

    train_loss = total_loss / max(total_seen, 1)
    train_accuracy = total_correct / max(total_seen, 1)
    stats = dict(last_stats)
    stats["controller_update_delta"] = 0.0
    stats["controller_error"] = (
        stats["global_zero_ratio"] - float(args.target_zero_ratio)
        if args.target_zero_ratio is not None
        else 0.0
    )
    if controller is not None and args.mode == "chimera":
        if args.controller_frequency == "epoch":
            next_alpha, delta, error = apply_target_bits_control(
                model,
                controller,
                current_zero_ratio=stats["global_zero_ratio"],
                affects=args.controller_affects,
            )
            stats["alpha_target_mean"] = next_alpha
            stats["controller_update_delta"] = delta
            stats["controller_error"] = error
            stats["alpha_next_mean"] = next_alpha
        else:
            stats["alpha_target_mean"] = float(last_stats["alpha_target_mean"])
            stats["alpha_next_mean"] = float(last_stats["alpha_target_mean"])
    stats["alpha_control_mean"] = float(stats["alpha_used_mean"])
    stats.setdefault("alpha_next_mean", float(stats["alpha_target_mean"]))

    epoch_record = _epoch_record(
        args=args,
        config=config,
        optimizer_name=optimizer_name,
        epoch=epoch,
        step=global_step,
        train_loss=train_loss,
        train_accuracy=train_accuracy,
        stats=stats,
        record_type="epoch",
    )
    return global_step, epoch_record


def run_training(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    train_loader: DataLoader,
    test_loader: DataLoader | None = None,
) -> list[dict[str, Any]]:
    """Run training over a provided loader."""
    torch.set_num_threads(int(args.num_threads))
    maybe_seed(args.seed)
    optimizer_name = resolve_optimizer_name(args.mode, args.optimizer)
    device = resolve_device(str(args.device))
    model = ChimeraResNet20(
        mode=args.mode,
        quant_config=quant_config_from_config(config),
        first_last_fp32=bool(args.first_last_fp32),
    ).to(device)
    if args.mode == "chimera" and args.controller_affects == "alpha_override":
        initialize_alpha_override(model, float(config["alpha_target"]))
    optimizer = build_optimizer(
        model.parameters(),
        optimizer_name=optimizer_name,
        lr=float(args.lr),
        config=config,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    controller_config = config.get("target_bits_controller", {})
    if not isinstance(controller_config, dict):
        controller_config = {}
    controller = TargetBitsController(
        target_bits=float(args.target_bits),
        branch=args.target_branch,
        target_zero_ratio=args.target_zero_ratio,
        tolerance=float(controller_config.get("tolerance", 0.02)),
        step_size=float(controller_config.get("step_size", 0.05)),
        alpha_min=float(config["alpha_min"]),
        alpha_max=float(controller_config.get("alpha_max", 1.5)),
    )
    if args.target_zero_ratio is None:
        args.target_zero_ratio = float(controller.target_zero_ratio)
    records: list[dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        global_step, epoch_record = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            args=args,
            config=config,
            optimizer_name=optimizer_name,
            epoch=epoch,
            global_step=global_step,
            controller=controller,
        )
        if test_loader is not None:
            test_loss, test_accuracy, eval_stats = evaluate(
                model=model,
                loader=test_loader,
                loss_fn=loss_fn,
                max_batches=int(getattr(args, "eval_max_batches", 0)),
            )
            epoch_record["test_loss"] = float(test_loss)
            epoch_record["test_accuracy"] = float(test_accuracy)
            eval_stats = dict(eval_stats)
            eval_stats["controller_error"] = (
                eval_stats["global_zero_ratio"] - float(args.target_zero_ratio)
                if args.target_zero_ratio is not None
                else 0.0
            )
            eval_stats["controller_update_delta"] = 0.0
            eval_stats["alpha_control_mean"] = float(eval_stats["alpha_used_mean"])
            eval_stats["alpha_next_mean"] = float(epoch_record["alpha_next_mean"])
            eval_record = _epoch_record(
                args=args,
                config=config,
                optimizer_name=optimizer_name,
                epoch=epoch,
                step=global_step,
                train_loss=float(test_loss),
                train_accuracy=float(test_accuracy),
                stats=eval_stats,
                record_type="eval",
            )
            eval_record.update(
                {
                    "split": "test",
                    "loss": float(test_loss),
                    "train_loss": None,
                    "train_accuracy": None,
                    "test_loss": float(test_loss),
                    "test_accuracy": float(test_accuracy),
                    "timestamp_utc": timestamp_utc(),
                }
            )
        log_benchmark_record(
            prefix="resnet_cifar",
            record=epoch_record,
            jsonl_path=args.log_jsonl,
            echo=True,
        )
        if test_loader is not None:
            log_benchmark_record(
                prefix="resnet_cifar",
                record=eval_record,
                jsonl_path=args.log_jsonl,
                echo=True,
            )
        records.append(epoch_record)

    final_record = dict(records[-1])
    final_record["record_type"] = "final"
    final_record["split"] = "summary"
    final_record["timestamp_utc"] = timestamp_utc()
    log_benchmark_record(
        prefix="resnet_cifar",
        record=final_record,
        jsonl_path=args.log_jsonl,
        final=True,
        echo=True,
    )
    summary = make_summary_record(
        run_id=f"{SCRIPT}-summary-mode-{args.mode}-seed-{args.seed}",
        script=SCRIPT,
        mode=args.mode,
        lr=float(args.lr),
        config=config,
        mean_final_loss=float(final_record["train_loss"]),
        std_final_loss=0.0,
        mean_zero_ratio=float(final_record["mean_zero_ratio"]),
        std_zero_ratio=0.0,
        mean_plus_ratio=float(final_record["mean_plus_ratio"]),
        std_plus_ratio=0.0,
        mean_minus_ratio=float(final_record["mean_minus_ratio"]),
        std_minus_ratio=0.0,
        mean_effective_bits=float(final_record["mean_effective_bits"]),
        std_effective_bits=0.0,
        optimizer_name=optimizer_name,
        target_bits=float(args.target_bits),
    )
    summary.update(
        {
            "train_loss": float(final_record["train_loss"]),
            "train_accuracy": float(final_record["train_accuracy"]),
            "target_bits": float(args.target_bits),
            "target_branch": args.target_branch,
            "target_zero_ratio": final_record["target_zero_ratio"],
            "alpha_used_mean": float(final_record["alpha_used_mean"]),
            "alpha_target_mean": float(final_record["alpha_target_mean"]),
            "alpha_control_mean": float(final_record["alpha_control_mean"]),
            "alpha_next_mean": float(final_record["alpha_next_mean"]),
            "controller_frequency": args.controller_frequency,
            "controller_affects": args.controller_affects,
            "controller_error": float(final_record["controller_error"]),
            "controller_update_delta": float(final_record["controller_update_delta"]),
            "global_zero_ratio": float(final_record["global_zero_ratio"]),
            "global_effective_bits": float(final_record["global_effective_bits"]),
            "test_loss": final_record.get("test_loss"),
            "test_accuracy": final_record.get("test_accuracy"),
            "split": "summary",
        }
    )
    log_benchmark_record(
        prefix="resnet_cifar",
        record=summary,
        jsonl_path=args.log_jsonl,
        echo=True,
    )
    if getattr(args, "save_checkpoint", False):
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            args=args,
            final_metrics=final_record,
        )
    return records


def save_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    args: argparse.Namespace,
    final_metrics: dict[str, Any],
) -> Path:
    """Save a benchmark checkpoint with metadata."""
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"{SCRIPT}_{args.mode}_seed{args.seed}"
    path = args.checkpoint_dir / f"{run_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "args": vars(args),
            "final_metrics": final_metrics,
        },
        path,
    )
    return path


def build_synthetic_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """Build deterministic synthetic train and test loaders."""
    train_batches = max(int(args.max_batches), 1)
    eval_batches = max(int(args.eval_max_batches), 1)
    train_x = torch.randn(train_batches * args.batch_size, 3, 32, 32)
    train_y = torch.arange(train_batches * args.batch_size) % 10
    test_x = torch.randn(eval_batches * args.eval_batch_size, 3, 32, 32)
    test_y = torch.arange(eval_batches * args.eval_batch_size) % 10
    train_loader = DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        batch_size=args.eval_batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def run(args: argparse.Namespace, config: dict[str, Any]) -> None:
    resolve_device(str(args.device))
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise SystemExit("--epochs, --batch-size, and --eval-batch-size must be positive.")
    if args.lr < 0:
        raise SystemExit("--lr must be non-negative.")
    if args.num_threads <= 0:
        raise SystemExit("--num-threads must be positive.")

    torch.set_num_threads(int(args.num_threads))
    maybe_seed(args.seed)
    if args.log_jsonl is not None:
        args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if not getattr(args, "append_log", False):
            args.log_jsonl.write_text("", encoding="utf-8")
    if args.synthetic:
        train_loader, test_loader = build_synthetic_loaders(args)
    else:
        train_dataset = _load_cifar10(args.dataset_root, train=True, download=args.download)
        test_dataset = _load_cifar10(args.dataset_root, train=False, download=args.download)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
        )
    run_training(args=args, config=config, train_loader=train_loader, test_loader=test_loader)
    print("[resnet_cifar] status=passed")


def main(argv: list[str] | None = None) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    parser = build_parser(config)
    args = parser.parse_args(argv)
    run(args, config)


if __name__ == "__main__":
    main()
