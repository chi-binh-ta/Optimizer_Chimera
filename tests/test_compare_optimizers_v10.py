import json

from experiments.compare_optimizers import main as compare_optimizers_main
from experiments.run_optimizer_stress_suite import main as stress_suite_main


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_sparse_relu_problem_creates_nonzero_zero_grad_ratio(tmp_path) -> None:
    path = tmp_path / "sparse_relu.jsonl"
    compare_optimizers_main(
        [
            "--steps",
            "5",
            "--problem",
            "sparse_relu",
            "--sparsity",
            "0.7",
            "--log-jsonl",
            str(path),
        ]
    )
    records = _records(path)
    chimera_records = [
        record
        for record in records
        if record["optimizer_name"] == "chimera21"
        and record["record_type"] == "step"
    ]

    assert max(record["zero_grad_ratio"] for record in chimera_records) > 0.0


def test_noisy_quadratic_crossgate_reduces_collision_or_kappa(tmp_path) -> None:
    path = tmp_path / "noisy_quadratic.jsonl"
    compare_optimizers_main(
        [
            "--steps",
            "8",
            "--problem",
            "noisy_quadratic",
            "--noise-scale",
            "2.0",
            "--log-jsonl",
            str(path),
        ]
    )
    records = _records(path)
    summaries = {
        record["optimizer_name"]: record
        for record in records
        if record["record_type"] == "summary"
    }
    steps = [
        record
        for record in records
        if record["record_type"] == "step"
        and record["optimizer_name"] in {"chimera21", "chimera21_crossgate"}
    ]
    mean_kappa = {}
    for name in ("chimera21", "chimera21_crossgate"):
        values = [record["mean_kappa"] for record in steps if record["optimizer_name"] == name]
        mean_kappa[name] = sum(values) / len(values)

    assert (
        summaries["chimera21_crossgate"]["mean_collision_score"]
        <= summaries["chimera21"]["mean_collision_score"]
        or mean_kappa["chimera21_crossgate"] <= mean_kappa["chimera21"]
    )


def test_zero_policy_variants_appear_in_summaries(tmp_path) -> None:
    path = tmp_path / "zero_variants.jsonl"
    compare_optimizers_main(
        [
            "--steps",
            "3",
            "--problem",
            "sparse_relu",
            "--sparsity",
            "0.7",
            "--log-jsonl",
            str(path),
        ]
    )
    summaries = [
        record
        for record in _records(path)
        if record["record_type"] == "summary"
    ]
    policies = {
        record["optimizer_name"]: record["zero_grad_policy"]
        for record in summaries
    }

    assert policies["chimera21_zero_freeze"] == "freeze"
    assert policies["chimera21_zero_decay"] == "decay"
    assert policies["chimera21_zero_inertia"] == "inertia"


def test_saddle_writes_summaries_with_chimera_collision_scores(tmp_path) -> None:
    path = tmp_path / "saddle.jsonl"
    compare_optimizers_main(
        [
            "--steps",
            "4",
            "--problem",
            "saddle",
            "--log-jsonl",
            str(path),
        ]
    )
    summaries = [
        record
        for record in _records(path)
        if record["record_type"] == "summary"
    ]
    names = {record["optimizer_name"] for record in summaries}

    assert {
        "adam",
        "chimera21",
        "chimera21_crossgate",
        "chimera21_zero_freeze",
        "chimera21_zero_decay",
        "chimera21_zero_inertia",
        "chimera21_psi_int8",
    } <= names
    for record in summaries:
        if record["optimizer_name"].startswith("chimera21"):
            assert record["mean_collision_score"] is not None


def test_step_time_ms_exists_and_is_positive(tmp_path) -> None:
    path = tmp_path / "timing.jsonl"
    compare_optimizers_main(
        [
            "--steps",
            "2",
            "--problem",
            "regression",
            "--timing-warmup-steps",
            "1",
            "--log-jsonl",
            str(path),
        ]
    )
    step = next(record for record in _records(path) if record["record_type"] == "step")
    summary = next(record for record in _records(path) if record["record_type"] == "summary")

    assert step["step_time_ms"] > 0.0
    assert summary["mean_step_time_ms"] > 0.0
    assert summary["mean_step_time_ms_after_warmup"] > 0.0
    assert summary["median_step_time_ms_after_warmup"] > 0.0
    assert summary["timing_warmup_steps"] == 1


def test_optimizer_stress_suite_writes_all_problem_summaries(tmp_path) -> None:
    path = tmp_path / "stress_suite.jsonl"
    stress_suite_main(
        [
            "--steps",
            "2",
            "--log-jsonl",
            str(path),
        ]
    )
    summaries = [
        record
        for record in _records(path)
        if record["record_type"] == "summary"
    ]
    problems = {record["problem"] for record in summaries}

    assert {"regression", "sparse_relu", "noisy_quadratic", "saddle"} <= problems
    assert all("problem" in record for record in _records(path))
