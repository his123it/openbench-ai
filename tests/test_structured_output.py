"""Unit tests for StructuredOutputMetric (JSON parsing, required keys, field types, JSON Schema)."""

from __future__ import annotations

from openbench.dataset import DatasetRecord
from openbench.metrics.structured_output import StructuredOutputMetric


def _structured_record() -> DatasetRecord:
    return DatasetRecord(
        id="struct_test_01",
        category="structured_output",
        prompt="Return health check JSON.",
        expected_answer='{"service": "auth-api", "healthy": true, "latency_ms": 12}',
        metadata={
            "required_keys": ["service", "healthy", "latency_ms"],
            "field_types": {"service": "str", "healthy": "bool", "latency_ms": "int"},
            "expected_schema": {
                "type": "object",
                "required": ["service", "healthy", "latency_ms"],
                "properties": {
                    "service": {"type": "string"},
                    "healthy": {"type": "boolean"},
                    "latency_ms": {"type": "integer", "minimum": 0},
                },
            },
        },
    )


def test_structured_output_valid_structure_and_exact_values() -> None:
    """Valid JSON matching all required keys, types, and values passes both structural and value checks."""
    rec = _structured_record()
    metric = StructuredOutputMetric()

    res = metric.compute_item(
        rec,
        '{"service": "auth-api", "healthy": true, "latency_ms": 12}',
    )
    assert res.passed is True
    assert res.score == 1.0
    assert res.details["valid_json"] is True
    assert res.details["valid_structure"] is True
    assert res.details["value_match"] is True


def test_structured_output_distinguishes_valid_structure_from_different_values() -> None:
    """Response with valid schema/types but different values passes structure check while reporting value_match=False."""
    rec = _structured_record()
    metric_struct_only = StructuredOutputMetric(require_value_match=False)
    metric_strict_val = StructuredOutputMetric(require_value_match=True)

    candidate = '{"service": "billing-api", "healthy": false, "latency_ms": 99}'

    res_struct = metric_struct_only.compute_item(rec, candidate)
    assert res_struct.details["valid_structure"] is True
    assert res_struct.details["value_match"] is False
    assert res_struct.passed is True
    assert res_struct.score == 1.0

    res_strict = metric_strict_val.compute_item(rec, candidate)
    assert res_strict.details["valid_structure"] is True
    assert res_strict.details["value_match"] is False
    assert res_strict.passed is False
    assert res_strict.score == 0.5


def test_structured_output_detects_malformed_json_missing_keys_and_wrong_types() -> None:
    """Malformed JSON, missing required keys, and wrong field types are reported accurately."""
    rec = _structured_record()
    metric = StructuredOutputMetric()

    bad_json = metric.compute_item(rec, "{not valid json")
    assert bad_json.passed is False
    assert bad_json.score == 0.0
    assert bad_json.details["valid_json"] is False

    missing_and_wrong_type = metric.compute_item(
        rec,
        '{"service": "auth-api", "healthy": "yes"}',
    )
    assert missing_and_wrong_type.passed is False
    assert "latency_ms" in missing_and_wrong_type.details["missing_keys"]
    assert len(missing_and_wrong_type.details["type_errors"]) >= 1
