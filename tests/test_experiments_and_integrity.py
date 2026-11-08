"""Unit tests for ExperimentManager, run recovery/resumption, and RunIntegrityVerifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openbench.adapters import MockModelProvider
from openbench.dataset import Dataset, DatasetRecord
from openbench.evaluator import Evaluator, ModelResponseSet, ResponseItem
from openbench.exceptions import ConfigurationError
from openbench.execution import ExecutionConfig, RetryPolicy
from openbench.experiments import ExperimentManager


def _exp_dataset(tmp_path: Path) -> Dataset:
    ds_file = tmp_path / "exp_sample.jsonl"
    records = [
        {"id": "x1", "category": "reasoning", "prompt": "1+1?", "expected_answer": "2"},
        {"id": "x2", "category": "reasoning", "prompt": "2+2?", "expected_answer": "4"},
        {"id": "x3", "category": "reasoning", "prompt": "3+3?", "expected_answer": "6"},
    ]
    ds_file.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    from openbench.dataset import DatasetLoader

    return DatasetLoader.load_file(ds_file)


def test_experiment_create_inspect_verify_and_tamper_detection(tmp_path: Path) -> None:
    """ExperimentManager creates all 7 run artifacts, passes integrity check, and detects tampered files."""
    ds = _exp_dataset(tmp_path)
    manager = ExperimentManager(experiments_root=tmp_path / "experiments")
    mock = MockModelProvider(model_name="exp-mock-v1", simulated_latency_ms=0.5)

    record, eval_res, perf_rep = manager.execute_and_record(
        dataset=ds,
        provider=mock,
        execution_config=ExecutionConfig(use_cache=False),
        run_id="run_unit_001",
    )

    assert record.run_id == "run_unit_001"
    assert record.status == "completed"
    assert record.execution_origin == "provider_executed"
    assert record.is_mock_provider is True
    assert eval_res.aggregate_results["exact_match"].score == 1.0

    check_ok = manager.verify_run("run_unit_001")
    assert check_ok.valid is True
    assert check_ok.execution_origin == "provider_executed"
    assert len(check_ok.missing_artifacts) == 0

    # Tamper with responses.json on disk and verify that RunIntegrityVerifier catches the SHA-256 mismatch
    resp_path = Path(record.run_dir) / "responses.json"
    tampered = json.loads(resp_path.read_text(encoding="utf-8"))
    tampered["responses"][0]["response"] = "999_tampered"
    resp_path.write_text(json.dumps(tampered), encoding="utf-8")

    check_bad = manager.verify_run("run_unit_001")
    assert check_bad.valid is False
    assert "responses.json" in check_bad.changed_artifacts
    assert "responses.json" in check_bad.invalid_checksums


def test_experiment_interrupted_run_recovery_and_resumption(tmp_path: Path) -> None:
    """An interrupted/partial run can be resumed via manager.resume_run without re-running completed items."""
    ds = _exp_dataset(tmp_path)
    manager = ExperimentManager(experiments_root=tmp_path / "experiments")

    # Initial run fails on item x3
    failing_mock = MockModelProvider(
        model_name="resume-mock",
        fail_on_ids=["x3"],
        simulated_latency_ms=0.5,
    )
    rec_partial, eval_partial, _ = manager.execute_and_record(
        dataset=ds,
        provider=failing_mock,
        execution_config=ExecutionConfig(
            use_cache=False,
            retry_policy=RetryPolicy(max_attempts=1),
        ),
        run_id="run_partial_01",
    )
    assert rec_partial.status == "partial"
    assert rec_partial.completed_items == 2
    assert rec_partial.failed_items == 1
    assert eval_partial.missing_response_ids == ["x3"]

    # Now resume with a working provider; x1 and x2 should be preserved, and only x3 executed
    healthy_mock = MockModelProvider(model_name="resume-mock", simulated_latency_ms=0.5)
    rec_resumed, eval_resumed, _ = manager.resume_run(
        "run_partial_01",
        provider=healthy_mock,
    )
    assert rec_resumed.status == "completed"
    assert rec_resumed.completed_items == 3
    assert rec_resumed.failed_items == 0
    assert eval_resumed.aggregate_results["exact_match"].score == 1.0
    assert healthy_mock.call_counts == {"x3": 1}


def test_external_responses_never_falsely_labeled_as_provider_executed(tmp_path: Path) -> None:
    """register_external_evaluation labels origin as 'external_responses_evaluated' and blocks resume_run."""
    ds = _exp_dataset(tmp_path)
    manager = ExperimentManager(experiments_root=tmp_path / "experiments")
    resp_set = ModelResponseSet(
        model_name="ExternalFileModel",
        model_version="1.0",
        responses=[
            ResponseItem(id="x1", response="2"),
            ResponseItem(id="x2", response="4"),
            ResponseItem(id="x3", response="6"),
        ],
    )
    eval_res = Evaluator().evaluate(ds, resp_set)
    rec = manager.register_external_evaluation(
        dataset=ds,
        response_set=resp_set,
        eval_result=eval_res,
        run_id="run_ext_01",
    )
    assert rec.execution_origin == "external_responses_evaluated"
    check = manager.verify_run("run_ext_01")
    assert check.valid is True
    assert check.execution_origin == "external_responses_evaluated"

    with pytest.raises(ConfigurationError, match="not eligible for provider resumption"):
        manager.resume_run("run_ext_01", provider=MockModelProvider())
