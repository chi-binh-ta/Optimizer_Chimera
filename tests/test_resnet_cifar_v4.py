import argparse
import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from chimera.bitconv import BitConv2d
from chimera.bitlinear import BitLinear
from chimera.models.resnet_cifar import ChimeraResNet20
from chimera.optimizer import load_config
from experiments.train_resnet_cifar import model_stats, run_training


def _first_bitconv(model: nn.Module) -> BitConv2d:
    for module in model.modules():
        if isinstance(module, BitConv2d):
            return module
    raise AssertionError("expected at least one BitConv2d")


def test_resnet_quant_config_propagates_alpha_target() -> None:
    model = ChimeraResNet20(
        mode="chimera",
        quant_config={
            "alpha_target": 0.42,
            "alpha_min": 0.1,
            "warmup_steps": 3,
            "rho_s": 0.9,
            "stat_mode": "mean",
            "eps_gamma": 1e-5,
            "eps_beta": 1e-5,
            "ste_clip": 1.0,
        },
    )
    layer = _first_bitconv(model)

    assert layer.alpha_target == pytest.approx(0.42)
    assert layer.alpha_min == pytest.approx(0.1)
    assert layer.warmup_steps == 3


def test_resnet_first_last_fp32_module_types() -> None:
    model = ChimeraResNet20(mode="chimera", first_last_fp32=True)

    assert isinstance(model.conv1, nn.Conv2d)
    assert isinstance(model.fc, nn.Linear)
    assert any(isinstance(module, BitConv2d) for module in model.layer1.modules())
    assert not isinstance(model.fc, BitLinear)


def test_synthetic_one_batch_training_logs_epoch_final_summary(tmp_path) -> None:
    config = load_config("configs/default.yaml")
    path = tmp_path / "resnet.jsonl"
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    args = argparse.Namespace(
        epochs=1,
        batch_size=2,
        lr=float(config["lr"]),
        mode="chimera",
        target_bits=1.32,
        log_jsonl=path,
        device="cpu",
        seed=3,
        download=False,
        max_batches=1,
        optimizer="chimera21",
        first_last_fp32=True,
        num_threads=1,
        target_branch="sparse",
        target_zero_ratio=None,
        controller_frequency="epoch",
        controller_affects="alpha_target",
        synthetic=True,
    )

    records = run_training(args=args, config=config, train_loader=loader, test_loader=loader)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    record_types = {line["record_type"] for line in lines}
    epoch = next(line for line in lines if line["record_type"] == "epoch")
    final = next(line for line in lines if line["record_type"] == "final")

    assert records
    assert {"batch", "epoch", "final", "summary"} <= record_types
    assert epoch["optimizer_name"] == "chimera21"
    assert epoch["target_bits"] == pytest.approx(1.32)
    assert "train_loss" in epoch
    assert "train_accuracy" in epoch
    assert "mean_effective_bits" in epoch
    assert "alpha_used_mean" in epoch
    assert "alpha_target_mean" in epoch
    assert "controller_error" in epoch
    assert "target_zero_ratio" in epoch
    assert "alpha_mean" not in epoch
    assert "global_zero_ratio" in epoch
    assert final["record_type"] == "final"


def test_model_stats_global_ratios_sum_to_one() -> None:
    model = ChimeraResNet20(mode="chimera", first_last_fp32=True)
    model(torch.randn(2, 3, 32, 32))
    stats = model_stats(model)
    total = (
        stats["global_zero_ratio"]
        + stats["global_plus_ratio"]
        + stats["global_minus_ratio"]
    )

    assert total == pytest.approx(1.0)
    assert stats["effective_bits"] == pytest.approx(stats["global_effective_bits"])


def test_synthetic_resnet_moves_zero_ratio_toward_sparse_target(tmp_path) -> None:
    config = load_config("configs/default.yaml")
    path = tmp_path / "resnet_move.jsonl"
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    args = argparse.Namespace(
        epochs=2,
        batch_size=2,
        lr=float(config["lr"]),
        mode="chimera",
        target_bits=1.32,
        log_jsonl=path,
        device="cpu",
        seed=4,
        download=False,
        max_batches=1,
        optimizer="chimera21",
        first_last_fp32=True,
        num_threads=1,
        target_branch="sparse",
        target_zero_ratio=None,
        controller_frequency="epoch",
        controller_affects="alpha_target",
        synthetic=True,
    )

    run_training(args=args, config=config, train_loader=loader, test_loader=loader)
    epochs = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if '"record_type": "epoch"' in line
    ]

    assert len(epochs) == 2
    target = epochs[-1]["target_zero_ratio"]
    assert 0.62 <= target <= 0.64
    assert abs(target - epochs[1]["global_zero_ratio"]) <= abs(
        target - epochs[0]["global_zero_ratio"]
    )


def test_alpha_override_moves_alpha_used_faster_than_alpha_target(tmp_path) -> None:
    config = load_config("configs/default.yaml")
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])

    def run_case(affects: str, path_name: str) -> dict:
        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        args = argparse.Namespace(
            epochs=1,
            batch_size=2,
            lr=float(config["lr"]),
            mode="chimera",
            target_bits=1.32,
            log_jsonl=tmp_path / path_name,
            device="cpu",
            seed=8,
            download=False,
            max_batches=1,
            optimizer="chimera21",
            first_last_fp32=True,
            num_threads=1,
            target_branch="sparse",
            target_zero_ratio=None,
            controller_frequency="batch",
            controller_affects=affects,
            synthetic=True,
        )
        run_training(args=args, config=config, train_loader=loader, test_loader=loader)
        return next(
            json.loads(line)
            for line in args.log_jsonl.read_text(encoding="utf-8").splitlines()
            if '"record_type": "batch"' in line
        )

    target_only = run_case("alpha_target", "target.jsonl")
    override = run_case("alpha_override", "override.jsonl")

    assert override["alpha_used_mean"] > target_only["alpha_used_mean"]


def test_batch_controller_logs_error_and_increases_alpha_when_below_target(tmp_path) -> None:
    config = load_config("configs/default.yaml")
    path = tmp_path / "batch_control.jsonl"
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    args = argparse.Namespace(
        epochs=1,
        batch_size=2,
        lr=float(config["lr"]),
        mode="chimera",
        target_bits=1.32,
        log_jsonl=path,
        device="cpu",
        seed=9,
        download=False,
        max_batches=1,
        optimizer="chimera21",
        first_last_fp32=True,
        num_threads=1,
        target_branch="sparse",
        target_zero_ratio=None,
        controller_frequency="batch",
        controller_affects="alpha_override",
        synthetic=True,
    )

    run_training(args=args, config=config, train_loader=loader, test_loader=loader)
    batch = next(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if '"record_type": "batch"' in line
    )

    assert batch["controller_error"] < 0.0
    assert batch["controller_update_delta"] > 0.0
    assert batch["alpha_target_mean"] > batch["alpha_used_mean"]
