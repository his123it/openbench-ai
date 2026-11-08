# OpenBench AI v0.2.0 — Backward Compatibility Report

This report documents the verification status of every **OpenBench AI v0.1.0** component, public API, CLI command, schema, and behavior when executed under **OpenBench AI v0.2.0**.

---

## 1. Public API & Import Compatibility Matrix

| v0.1.0 Module / Symbol | Import Path | v0.2.0 Status | Verification Test |
| :--- | :--- | :--- | :--- |
| `Dataset`, `DatasetLoader`, `DatasetRecord` | `openbench` / `openbench.dataset` | **100% Preserved** (Additive properties & `filter()` added) | `test_dataset.py`, `test_backward_compatibility.py` |
| `Evaluator`, `EvaluationConfig`, `EvaluationResult`, `ItemEvaluationResult` | `openbench` / `openbench.evaluator` | **100% Preserved** (Additive breakdowns & optional telemetry added) | `test_evaluator.py`, `test_backward_compatibility.py` |
| `ResponseLoader`, `ModelResponseSet`, `ResponseItem` | `openbench` / `openbench.evaluator` | **100% Preserved** (Optional `latency_ms` & `token_usage` added) | `test_evaluator.py`, `test_integrations.py` |
| `ModelComparator`, `ComparisonResult`, `ModelComparisonEntry`, `MetricLeader` | `openbench` / `openbench.comparison` | **100% Preserved** (Additive category/difficulty & margin fields added) | `test_comparison.py`, `test_backward_compatibility.py` |
| `ReportGenerator` | `openbench` / `openbench.reporting` | **100% Preserved** (`to_json`, `save_json`, `to_markdown`, `save_markdown` preserved; `to_csv`, `save_csv` added) | `test_reporting.py`, `test_backward_compatibility.py` |
| `BaseMetric`, `MetricResult`, `AggregateMetricResult` | `openbench.metrics` / `openbench.metrics.base` | **100% Preserved** (`MetricMetadata` & `get_metadata()` added) | `test_metrics.py`, `test_advanced_metrics.py` |
| `NormalizationConfig`, `normalize_text` | `openbench.metrics` / `openbench.metrics.normalization` | **100% Identical** | `test_metrics.py` |
| `ExactMatchMetric`, `AccuracyMetric`, `KeywordCoverageMetric` | `openbench.metrics` | **100% Identical** (Exact same scoring & normalization rules) | `test_metrics.py` |
| `METRIC_REGISTRY`, `create_metric`, `register_metric` | `openbench.metrics` | **100% Preserved** (Including `"keyword_match"` alias) | `test_metrics.py`, `test_advanced_metrics.py` |
| `OpenBenchError`, `DatasetValidationError`, `EmptyDatasetError`, `ResponseValidationError`, `MetricError`, `ConfigurationError` | `openbench` / `openbench.exceptions` | **100% Identical** | `test_dataset.py`, `test_evaluator.py` |

---

## 2. CLI Command & Argument Compatibility Matrix

| v0.1.0 CLI Command | v0.2.0 Behavior | Exit Code Behavior | Verification Test |
| :--- | :--- | :--- | :--- |
| `openbench --version` | Outputs `openbench 0.2.0` | `0` | `test_cli.py::test_cli_version_flag` |
| `openbench` (no args) | Prints CLI help message | `2` | `test_cli.py::test_cli_no_arguments_returns_exit_code_2` |
| `openbench validate-dataset PATH` | Validates JSONL file or directory and prints summary | `0` on valid, `1` on invalid/missing | `test_cli.py::test_cli_validate_dataset_success_and_failure` |
| `openbench evaluate --dataset ... --responses ... --metrics ... --output ...` | Evaluates responses and writes `.json` and `.md` reports | `0` on success, `1` on error | `test_cli.py::test_cli_evaluate_and_compare_workflow` |
| `openbench compare --results ... --results ... --output ...` | Compares 2+ result JSON files and writes comparison reports | `0` on success, `2` if `< 2` files | `test_cli.py::test_cli_evaluate_and_compare_workflow`, `test_cli.py::test_cli_compare_fewer_than_two_files_fails` |

---

## 3. Dataset & Report Schema Compatibility Matrix

| Artifact / Schema | v0.1.0 Contract | v0.2.0 Status | Verification Test |
| :--- | :--- | :--- | :--- |
| **Original 48 Dataset Examples** | `reasoning_001`–`012`, `coding_001`–`012`, `instruction_001`–`012`, `safety_001`–`012` | **100% Preserved Verbatim** at lines 1–12 of their original files | `test_backward_compatibility.py::test_original_48_examples_preserved_with_exact_ids` |
| **JSONL Record Validation** | Requires `id`, `category`, `prompt`, `expected_answer`; optional `metadata` | **100% Preserved** (New metadata fields are optional) | `test_dataset.py` |
| **v0.1.0 JSON Report Deserialization** | `EvaluationResult.from_dict(v1_json)` | **100% Compatible** (Provides safe defaults for new v0.2.0 keys) | `test_backward_compatibility.py::test_v0_1_0_json_report_deserialization_and_schema_compatibility` |
| **Example Scripts** | `examples/basic_evaluation.py`, `examples/compare_models.py`, `scripts/validate_datasets.py`, `scripts/run_example.py` | **100% Functional** (Executed with exit code `0`) | Verified via `scripts/run_example.py` |
