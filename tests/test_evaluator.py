"""Unit tests for OpenBench AI ResponseLoader and Evaluator engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openbench.dataset import Dataset, DatasetRecord
from openbench.evaluator import (
    EvaluationConfig,
    Evaluator,
    ModelResponseSet,
    ResponseItem,
    ResponseLoader,
)
from openbench.exceptions import ResponseValidationError


def _sample_dataset() -> Dataset:
    records = [
        DatasetRecord(
            id="q1",
            category="reasoning",
            prompt="What is 3 + 4?",
            expected_answer="7",
            metadata={"difficulty": "easy", "keywords": ["7"]},
        ),
        DatasetRecord(
            id="q2",
            category="reasoning",
            prompt="What is 10 * 5?",
            expected_answer="50",
            metadata={"difficulty": "easy", "keywords": ["50"]},
        ),
        DatasetRecord(
            id="q3",
            category="safety",
            prompt="Should you share secret keys?",
            expected_answer="No, never share secret keys.",
            metadata={"difficulty": "easy", "keywords": ["No", "secret keys"]},
        ),
    ]
    return Dataset(name="unit_test_ds", records=records, source_path="memory://unit_test_ds")


def test_response_loader_detects_duplicate_response_ids(tmp_path: Path) -> None:
    """ResponseLoader raises ResponseValidationError when duplicate response IDs exist."""
    resp_file = tmp_path / "dup_responses.json"
    payload = {
        "model_name": "TestModel",
        "model_version": "1.0",
        "responses": [
            {"id": "q1", "response": "7"},
            {"id": "q1", "response": "7 again"},
        ],
    }
    resp_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResponseValidationError, match="Duplicate response ID 'q1'"):
        ResponseLoader.load_file(resp_file)


def test_response_loader_missing_required_model_fields(tmp_path: Path) -> None:
    """ResponseLoader raises ResponseValidationError if model_name or model_version is missing."""
    resp_file = tmp_path / "invalid_model.json"
    resp_file.write_text(json.dumps({"model_name": "OnlyName", "responses": []}), encoding="utf-8")

    with pytest.raises(ResponseValidationError, match="model_version"):
        ResponseLoader.load_file(resp_file)


def test_evaluator_records_missing_and_unknown_responses() -> None:
    """Evaluator tracks missing dataset IDs and unknown response IDs accurately."""
    dataset = _sample_dataset()
    response_set = ModelResponseSet(
        model_name="PartialModel",
        model_version="0.2",
        responses=[
            ResponseItem(id="q1", response="7"),
            ResponseItem(id="q2", response="The product is 50."),
            ResponseItem(id="unknown_999", response="Extraneous answer"),
        ],
    )

    evaluator = Evaluator()
    result = evaluator.evaluate(dataset, response_set)

    assert result.total_dataset_items == 3
    assert result.evaluated_items_count == 2
    assert result.missing_responses_count == 1
    assert result.missing_response_ids == ["q3"]
    assert result.unknown_response_ids == ["unknown_999"]
    assert len(result.warnings) == 2

    # q1 is exact match (1.0), q2 is not exact match (0.0), q3 is missing (0.0) -> 1/3
    assert pytest.approx(result.aggregate_results["exact_match"].score, rel=1e-4) == 1.0 / 3.0
    # q1 and q2 pass accuracy (2/3)
    assert pytest.approx(result.aggregate_results["accuracy"].score, rel=1e-4) == 2.0 / 3.0


def test_evaluator_strict_unknown_ids_raises_error() -> None:
    """When strict_unknown_ids=True, unknown response IDs raise ResponseValidationError."""
    dataset = _sample_dataset()
    response_set = ModelResponseSet(
        model_name="StrictModel",
        model_version="1.0",
        responses=[
            ResponseItem(id="q1", response="7"),
            ResponseItem(id="non_existent_id", response="abc"),
        ],
    )
    evaluator = Evaluator(config=EvaluationConfig(strict_unknown_ids=True))
    with pytest.raises(ResponseValidationError, match="unknown dataset ID"):
        evaluator.evaluate(dataset, response_set)


def test_evaluator_multiple_models_independently() -> None:
    """Evaluator.evaluate_multiple evaluates each model independently without state leakage."""
    dataset = _sample_dataset()
    model_a = ModelResponseSet(
        model_name="ModelA",
        model_version="1.0",
        responses=[
            ResponseItem(id="q1", response="7"),
            ResponseItem(id="q2", response="50"),
            ResponseItem(id="q3", response="No, never share secret keys."),
        ],
    )
    model_b = ModelResponseSet(
        model_name="ModelB",
        model_version="1.0",
        responses=[
            ResponseItem(id="q1", response="wrong"),
            ResponseItem(id="q2", response="wrong"),
            ResponseItem(id="q3", response="wrong"),
        ],
    )

    evaluator = Evaluator()
    results = evaluator.evaluate_multiple(dataset, [model_a, model_b])
    assert len(results) == 2
    assert results[0].aggregate_results["exact_match"].score == 1.0
    assert results[1].aggregate_results["exact_match"].score == 0.0
