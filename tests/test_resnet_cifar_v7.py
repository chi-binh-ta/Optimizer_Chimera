import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from chimera.optimizer import load_config
from experiments.run_cifar_ablation import main as run_ablation_main
from experiments.train_resnet_cifar import resolve_device, run_training


def _args(tmp_path: Path, *, save_checkpoint: bool = False) -> argparse.Namespace:
    config = load_config("configs/default.yaml")
    return argparse.Namespace(
        epochs=1,
        batch_size=2,
        eval_batch_size=2,
        lr=float(config["lr"]),
        mode="chimera",
        target_bits=1.32,
        target_branch="sparse",
        target_zero_ratio=None,
        log_jsonl=tmp_path / "resnet_v7.jsonl",
        device="cpu",
        seed=13,
        download=False,
        max_batches=1,
        eval_max_batches=1,
        optimizer="chimera21",
        first_last_fp32=True,
        num_threads=1,
        controller_frequency="epoch",
        controller_affects="alpha_override",
        synthetic=True,
        save_checkpoint=save_checkpoint,
        checkpoint_dir=tmp_path / "ckpt",
        run_name="unit_resnet",
    )


def _loader() -> DataLoader:
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    return DataLoader(TensorDataset(x, y), batch_size=2)


def test_synthetic_train_and_test_loop_logs_eval_fields(tmp_path) -> None:
    config = load_config("configs/default.yaml")
    args = _args(tmp_path)

    run_training(args=args, config=config, train_loader=_loader(), test_loader=_loader())
    lines = [json.loads(line) for line in args.log_jsonl.read_text(encoding="utf-8").splitlines()]
    epoch = next(line for line in lines if line["record_type"] == "epoch")
    eval_record = next(line for line in lines if line["record_type"] == "eval")
    final = next(line for line in lines if line["record_type"] == "final")

    assert epoch["test_loss"] is not None
    assert epoch["test_accuracy"] is not None
    assert epoch["split"] == "train"
    assert eval_record["split"] == "test"
    assert eval_record["test_loss"] is not None
    assert eval_record["test_accuracy"] is not None
    assert "alpha_control_mean" in epoch
    assert "alpha_next_mean" in epoch
    assert final["test_loss"] is not None


def test_final_record_has_fresh_timestamp(tmp_path) -> None:
    config = load_config("configs/default.yaml")
    args = _args(tmp_path)

    run_training(args=args, config=config, train_loader=_loader(), test_loader=_loader())
    lines = [json.loads(line) for line in args.log_jsonl.read_text(encoding="utf-8").splitlines()]
    epoch = next(line for line in lines if line["record_type"] == "epoch")
    final = next(line for line in lines if line["record_type"] == "final")

    assert final["timestamp_utc"] != epoch["timestamp_utc"]


def test_checkpoint_save_load_metadata_shape(tmp_path) -> None:
    config = load_config("configs/default.yaml")
    args = _args(tmp_path, save_checkpoint=True)

    run_training(args=args, config=config, train_loader=_loader(), test_loader=_loader())
    checkpoint = torch.load(args.checkpoint_dir / "unit_resnet.pt", weights_only=False)

    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert checkpoint["config"]["lr"] == config["lr"]
    assert checkpoint["args"]["run_name"] == "unit_resnet"
    assert checkpoint["final_metrics"]["record_type"] == "final"


def test_run_cifar_ablation_synthetic_smoke_records_all_modes(tmp_path) -> None:
    log_path = tmp_path / "ablation.jsonl"
    run_ablation_main(
        [
            "--synthetic",
            "--epochs",
            "1",
            "--max-batches",
            "1",
            "--eval-max-batches",
            "1",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--log-jsonl",
            str(log_path),
        ]
    )
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    final_modes = {record["mode"] for record in records if record["record_type"] == "final"}

    assert {"fp32_baseline", "strict_bitnet", "chimera"} <= final_modes


def test_resolve_device_accepts_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"
