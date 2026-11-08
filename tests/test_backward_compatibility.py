"""Dedicated backward compatibility test suite verifying v0.1.0 contracts in v0.2.0."""

from __future__ import annotations

import json
from pathlib import Path

from openbench import (
    ComparisonResult,
    ConfigurationError,
    Dataset,
    DatasetLoader,
    DatasetRecord,
    DatasetValidationError,
    EmptyDatasetError,
    EvaluationConfig,
    EvaluationResult,
    Evaluator,
    ItemEvaluationResult,
    MetricError,
    ModelComparator,
    ModelResponseSet,
    OpenBenchError,
    ReportGenerator,
    ResponseItem,
    ResponseLoader,
    ResponseValidationError,
)
from openbench.cli import main
from openbench.metrics import (
    METRIC_REGISTRY,
    AccuracyMetric,
    AggregateMetricResult,
    BaseMetric,
    ExactMatchMetric,
    KeywordCoverageMetric,
    MetricResult,
    NormalizationConfig,
    create_metric,
    normalize_text,
    register_metric,
)


def test_v0_1_0_public_imports_and_symbols_preserved() -> None:
    """All v0.1.0 public classes, exceptions, and functions remain importable from their original paths."""
    symbols = [
        ComparisonResult,
        ConfigurationError,
        Dataset,
        DatasetLoader,
        DatasetRecord,
        DatasetValidationError,
        EmptyDatasetError,
        EvaluationConfig,
        EvaluationResult,
        Evaluator,
        ItemEvaluationResult,
        MetricError,
        ModelComparator,
        ModelResponseSet,
        OpenBenchError,
        ReportGenerator,
        ResponseItem,
        ResponseLoader,
        ResponseValidationError,
        METRIC_REGISTRY,
        AccuracyMetric,
        AggregateMetricResult,
        BaseMetric,
        ExactMatchMetric,
        KeywordCoverageMetric,
        MetricResult,
        NormalizationConfig,
        create_metric,
        normalize_text,
        register_metric,
    ]
    assert all(sym is not None for sym in symbols)


def test_original_48_examples_preserved_with_exact_ids() -> None:
    """All 48 original v0.1.0 evaluation items remain present with their exact IDs."""
    repo_root = Path(__file__).resolve().parent.parent
    expected_prefixes = [
        ("reasoning", "reasoning_"),
        ("coding", "coding_"),
        ("instruction_following", "instruction_"),
        ("safety", "safety_"),
    ]
    for category, prefix in expected_prefixes:
        ds = DatasetLoader.load_file(repo_root / "datasets" / category / "sample.jsonl")
        ids = {r.id for r in ds.records}
        for i in range(1, 13):
            expected_id = f"{prefix}{i:03d}"
            assert expected_id in ids, f"Missing original v0.1.0 record ID: {expected_id}"


def test_v0_1_0_json_report_deserialization_and_schema_compatibility(tmp_path: Path) -> None:
    """EvaluationResult.from_dict can still deserialize a v0.1.0 JSON report lacking v0.2.0 fields."""
    v0_1_0_report_payload = {
        "project_version": "0.1.0",
        "evaluation_timestamp": "2026-11-08T00:00:00+00:00",
        "dataset": {
            "name": "reasoning/sample",
            "version": "0.1.0",
            "source_path": "datasets/reasoning/sample.jsonl",
            "total_records": 1,
            "categories": {"reasoning": 1},
            "difficulties": {"easy": 1},
        },
        "model": {
            "model_name": "LegacyModel",
            "model_version": "1.0",
            "total_dataset_items": 1,
            "evaluated_items": 1,
            "missing_responses_count": 0,
            "unknown_responses_count": 0,
        },
        "configuration": {
            "metric_names": ["exact_match", "accuracy", "keyword_coverage"],
            "normalization": {
                "strip_whitespace": True,
                "case_sensitive": False,
                "collapse_whitespace": True,
                "strip_punctuation": False,
            },
            "strict_unknown_ids": False,
        },
        "metrics_used": [{"name": "exact_match", "version": "0.1.0"}],
        "aggregate_results": {
            "exact_match": {
                "metric_name": "exact_match",
                "metric_version": "0.1.0",
                "score": 1.0,
                "total_items": 1,
                "evaluated_items": 1,
                "passed_items": 1,
                "details": {},
            }
        },
        "individual_results": [
            {
                "id": "reasoning_001",
                "category": "reasoning",
                "difficulty": "easy",
                "prompt": "What is 15 multiplied by 12?",
                "expected_answer": "180",
                "model_response": "180",
                "status": "evaluated",
                "metrics": {
                    "exact_match": {
                        "metric_name": "exact_match",
                        "score": 1.0,
                        "passed": True,
                        "details": {},
                    }
                },
            }
        ],
        "missing_responses": [],
        "unknown_response_ids": [],
        "warnings": [],
        "limitations": [],
    }

    restored = EvaluationResult.from_dict(v0_1_0_report_payload)
    assert restored.model_name == "LegacyModel"
    assert restored.aggregate_results["exact_match"].score == 1.0

    # Also verify that newly generated v0.2.0 JSON reports contain all v0.1.0 top-level and card keys
    new_json_str = ReportGenerator.to_json(restored)
    new_payload = json.loads(new_json_str)
    for v1_key in (
        "project_version",
        "evaluation_timestamp",
        "dataset",
        "model",
        "configuration",
        "metrics_used",
        "aggregate_results",
        "individual_results",
        "missing_responses",
        "unknown_response_ids",
        "warnings",
        "limitations",
        "reproducibility_card",
    ):
        assert v1_key in new_payload


def test_v0_1_0_and_v0_2_0_cli_commands_coexist(tmp_path: Path) -> None:
    """All v0.1.0 CLI commands and new v0.2.0 CLI commands execute cleanly."""
    repo_root = Path(__file__).resolve().parent.parent
    ds_path = repo_root / "datasets" / "reasoning" / "sample.jsonl"
    resp_path = repo_root / "examples" / "sample_responses.json"
    cfg_path = repo_root / "configs" / "reasoning.yaml"
    out_dir = tmp_path / "cli_out"
    hist_file = tmp_path / "history.jsonl"

    # v0.1.0 validate-dataset
    assert main(["validate-dataset", str(ds_path)]) == 0

    # v0.2.0 list-datasets, inspect-dataset, dataset-stats
    assert main(["list-datasets", "--path", str(repo_root / "datasets"), "--category", "mathematics"]) == 0
    assert main(["inspect-dataset", str(ds_path), "--samples", "1"]) == 0
    assert main(["dataset-stats", str(ds_path)]) == 0

    # v0.2.0 evaluate with --config and --history-file
    assert (
        main(
            [
                "evaluate",
                "--config",
                str(cfg_path),
                "--responses",
                str(resp_path),
                "--history-file",
                str(hist_file),
                "--output",
                str(out_dir),
            ]
        )
        == 0
    )

    json_report = out_dir / "alphabaseline-7b.json"
    assert json_report.exists()
    assert (out_dir / "alphabaseline-7b.csv").exists()

    # v0.2.0 history list & show
    assert main(["history", "list", "--store", str(hist_file)]) == 0
    entries = [json.loads(line) for line in hist_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    eval_id = entries[0]["evaluation_id"]
    assert main(["history", "show", eval_id, "--store", str(hist_file)]) == 0

    # v0.2.0 export --format csv
    exported_csv = tmp_path / "exported_agg.csv"
    assert (
        main(
            [
                "export",
                "--input",
                str(json_report),
                "--format",
                "csv",
                "--mode",
                "aggregate",
                "--output",
                str(exported_csv),
            ]
        )
        == 0
    )
    assert exported_csv.exists()
