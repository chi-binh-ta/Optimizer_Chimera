import argparse
import json

import pytest

from chimera.logging_utils import (
    JSONL_FIELDS,
    make_benchmark_record,
    make_summary_record,
    write_jsonl_record,
)
from chimera.optimizer import load_config
from experiments.compare_quant_modes import (
    INIT_SEED_OFFSET,
    SEED_BASE,
    build_layer,
    build_optimizer,
    run_benchmark,
    run_mode,
)


def test_config_based_lr_usage() -> None:
    config = load_config("configs/default.yaml")
    lr = float(config["lr"])
    layer = build_layer(
        "chimera",
        config,
        in_features=2,
        out_features=2,
        steps=2,
    )
    optimizer = build_optimizer(layer, config, lr=lr)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(lr)


def test_jsonl_log_schema(tmp_path) -> None:
    config = load_config("configs/default.yaml")
    stats = {
        "gamma": 0.1,
        "alpha": 0.2,
        "zero_ratio": 0.3,
        "plus_ratio": 0.4,
        "minus_ratio": 0.3,
        "effective_bits": 1.57,
    }
    record = make_benchmark_record(
        run_id="test-run",
        script="test_script",
        mode="chimera",
        seed=1,
        init_seed=2,
        step=3,
        loss=0.5,
        stats=stats,
        lr=float(config["lr"]),
        config=config,
    )
    path = tmp_path / "records.jsonl"
    write_jsonl_record(path, record)
    loaded = json.loads(path.read_text(encoding="utf-8").strip())

    assert set(JSONL_FIELDS) <= set(loaded)
    assert loaded["run_id"] == "test-run"
    assert loaded["mode"] == "chimera"
    assert loaded["record_type"] == "step"
    assert loaded["alpha_used"] is True
    assert "effective_bits" in loaded
    assert loaded["lr"] == pytest.approx(float(config["lr"]))


def test_summary_record_schema() -> None:
    config = load_config("configs/default.yaml")
    record = make_summary_record(
        run_id="summary",
        script="compare_quant_modes",
        mode="strict_bitnet",
        lr=float(config["lr"]),
        config=config,
        mean_final_loss=1.0,
        std_final_loss=0.1,
        mean_zero_ratio=0.63,
        std_zero_ratio=0.01,
        mean_plus_ratio=0.185,
        std_plus_ratio=0.005,
        mean_minus_ratio=0.185,
        std_minus_ratio=0.005,
        mean_effective_bits=1.32,
        std_effective_bits=0.02,
    )

    assert set(JSONL_FIELDS) <= set(record)
    assert record["record_type"] == "summary"
    assert record["alpha_used"] is False
    assert record["plus_ratio"] is None
    assert record["minus_ratio"] is None
    assert record["mean_plus_ratio"] == pytest.approx(0.185)
    assert record["mean_minus_ratio"] == pytest.approx(0.185)
    assert record["mean_effective_bits"] == pytest.approx(1.32)


def test_compare_quant_modes_deterministic_for_fixed_seed() -> None:
    config = load_config("configs/default.yaml")
    kwargs = {
        "mode": "chimera",
        "seed": SEED_BASE,
        "init_seed": SEED_BASE + INIT_SEED_OFFSET,
        "config": config,
        "steps": 3,
        "batch_size": 4,
        "in_features": 3,
        "out_features": 2,
        "lr": float(config["lr"]),
        "echo": False,
    }

    first = run_mode(**kwargs)
    second = run_mode(**kwargs)

    assert first.final_loss == pytest.approx(second.final_loss)
    assert first.final_stats["zero_ratio"] == pytest.approx(second.final_stats["zero_ratio"])


def test_run_benchmark_uses_config_quant_modes() -> None:
    config = load_config("configs/default.yaml")
    args = argparse.Namespace(
        steps=2,
        seeds=1,
        batch_size=4,
        in_features=3,
        out_features=2,
        lr=float(config["lr"]),
        log_jsonl=None,
        device="cpu",
    )

    results = run_benchmark(config=config, args=args, echo=False)

    assert [result.mode for result in results] == list(config["quant_modes"])
