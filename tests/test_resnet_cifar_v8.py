import argparse
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from chimera import entropy_from_ternary_ratios
from chimera.models.resnet_cifar import ChimeraResNet20
from chimera.optimizer import load_config
from experiments import train_resnet_cifar as trainer


def _loader(batch_size: int = 2) -> DataLoader:
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.arange(batch_size) % 10
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def _args(tmp_path: Path) -> argparse.Namespace:
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
        log_jsonl=tmp_path / "resnet_v8.jsonl",
        device="cpu",
        seed=21,
        download=False,
        max_batches=1,
        eval_max_batches=1,
        optimizer="chimera21",
        first_last_fp32=True,
        num_threads=1,
        controller_frequency="epoch",
        controller_affects="alpha_override",
        synthetic=True,
        save_checkpoint=False,
        checkpoint_dir=tmp_path / "ckpt",
        run_name="unit_resnet_v8",
    )


def _quant_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    return [module for module in model.modules() if hasattr(module, "forward_step")]


def test_eval_record_uses_evaluate_stats(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = load_config("configs/default.yaml")
    args = _args(tmp_path)
    sentinel_zero = 0.625
    sentinel_plus = 0.25
    sentinel_minus = 0.125
    sentinel_bits = entropy_from_ternary_ratios(
        sentinel_zero,
        sentinel_plus,
        sentinel_minus,
    )

    def fake_evaluate(*, model, loader, loss_fn, max_batches):
        stats = trainer.model_stats(model)
        stats.update(
            {
                "zero_ratio": sentinel_zero,
                "plus_ratio": sentinel_plus,
                "minus_ratio": sentinel_minus,
                "effective_bits": sentinel_bits,
                "mean_zero_ratio": sentinel_zero,
                "mean_plus_ratio": sentinel_plus,
                "mean_minus_ratio": sentinel_minus,
                "mean_effective_bits": sentinel_bits,
                "unweighted_mean_zero_ratio": sentinel_zero,
                "unweighted_mean_plus_ratio": sentinel_plus,
                "unweighted_mean_minus_ratio": sentinel_minus,
                "unweighted_mean_effective_bits": sentinel_bits,
                "global_zero_ratio": sentinel_zero,
                "global_plus_ratio": sentinel_plus,
                "global_minus_ratio": sentinel_minus,
                "global_effective_bits": sentinel_bits,
            }
        )
        return 1.2345, 0.5, stats

    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)

    trainer.run_training(
        args=args,
        config=config,
        train_loader=_loader(),
        test_loader=_loader(),
    )
    records = [
        json.loads(line)
        for line in args.log_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    epoch = next(record for record in records if record["record_type"] == "epoch")
    eval_record = next(record for record in records if record["record_type"] == "eval")

    assert eval_record["split"] == "test"
    assert eval_record["test_loss"] == pytest.approx(1.2345)
    assert eval_record["global_zero_ratio"] == pytest.approx(sentinel_zero)
    assert eval_record["global_effective_bits"] == pytest.approx(sentinel_bits)
    assert epoch["global_zero_ratio"] != pytest.approx(sentinel_zero)


def test_evaluate_does_not_mutate_forward_steps_after_train_epoch() -> None:
    config = load_config("configs/default.yaml")
    args = argparse.Namespace(
        max_batches=1,
        log_jsonl=None,
        mode="chimera",
        seed=31,
        lr=float(config["lr"]),
        target_bits=1.32,
        target_branch="sparse",
        target_zero_ratio=None,
        controller_frequency="epoch",
        controller_affects="alpha_override",
    )
    model = ChimeraResNet20(
        mode="chimera",
        quant_config=trainer.quant_config_from_config(config),
        first_last_fp32=True,
    )
    optimizer = trainer.build_optimizer(
        model.parameters(),
        optimizer_name="chimera21",
        lr=float(config["lr"]),
        config=config,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = _loader()

    trainer.train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        args=args,
        config=config,
        optimizer_name="chimera21",
        epoch=1,
        global_step=0,
        controller=None,
        echo=False,
    )
    modules = _quant_modules(model)
    steps_before = [int(module.forward_step.item()) for module in modules]
    scales_before = [module.scale_ema.clone() for module in modules]

    trainer.evaluate(
        model=model,
        loader=loader,
        loss_fn=loss_fn,
        max_batches=1,
    )

    assert [int(module.forward_step.item()) for module in modules] == steps_before
    for module, scale_before in zip(modules, scales_before):
        assert torch.equal(module.scale_ema, scale_before)
    assert model.training
