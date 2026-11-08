# Backward Compatibility Contract (v0.1.0 -> v0.2.0)

OpenBench AI `v0.2.0` is a strictly additive upgrade over `v0.1.0`. Every public class, function signature, CLI command, dataset file path, JSON/JSONL schema, metric calculation, and report field from `v0.1.0` is preserved without breaking changes.

---

## 1. Preserved Public Python APIs

All `v0.1.0` import paths continue to work identically:

- `openbench`:
  - `__version__` (updated to `"0.2.0"`)
  - `Dataset`, `DatasetLoader`, `DatasetRecord`
  - `Evaluator`, `EvaluationConfig`, `EvaluationResult`, `ItemEvaluationResult`, `ModelResponseSet`, `ResponseItem`, `ResponseLoader`
  - `ModelComparator`, `ComparisonResult`
  - `ReportGenerator`
  - `OpenBenchError`, `DatasetValidationError`, `EmptyDatasetError`, `ResponseValidationError`, `MetricError`, `ConfigurationError`
- `openbench.metrics`:
  - `METRIC_REGISTRY`, `BaseMetric`, `MetricResult`, `AggregateMetricResult`
  - `NormalizationConfig`, `normalize_text`
  - `ExactMatchMetric`, `AccuracyMetric`, `KeywordCoverageMetric`
  - `create_metric`, `register_metric`

---

## 2. Preserved CLI Commands & Arguments

Every `v0.1.0` command works with its exact `v0.1.0` syntax and exit code semantics (`0` success, `1` validation/runtime error, `2` CLI usage error):

- `openbench --version`
- `openbench validate-dataset datasets/reasoning/sample.jsonl`
- `openbench evaluate --dataset datasets/reasoning/sample.jsonl --responses examples/sample_responses.json --metrics exact_match accuracy keyword_coverage --output results/`
- `openbench compare --results results/model_a.json --results results/model_b.json --output results/comparison.md`

---

## 3. Preserved Input Schemas & Built-in Datasets

- **Dataset JSONL Schema**: Only `id`, `category`, `prompt`, and `expected_answer` are required. All `v0.2.0` metadata fields (`language`, `source`, `version`, `rationale`, `required_keys`, `field_types`, `expected_schema`) are optional.
- **Original 48 Examples**: All 48 evaluation examples from `v0.1.0` (`reasoning_001`–`012`, `coding_001`–`012`, `instruction_001`–`012`, `safety_001`–`012`) are preserved verbatim in their original files.
- **Response JSON Schema**: `model_name`, `model_version`, and `responses: [{"id": ..., "response": ...}]` work unchanged. New telemetry fields (`latency_ms`, `token_usage`, `model_provider`, `inference_config`) are strictly optional.

---

## 4. Preserved Metric Behavior

- `exact_match`, `accuracy`, and `keyword_coverage` (including its `keyword_match` alias) retain their exact normalization defaults (`strip_whitespace=True`, `case_sensitive=False`, `collapse_whitespace=True`, `strip_punctuation=False`) and scoring formulas.
- Missing responses continue to be penalized as `0.0` in aggregate scores (`total_score_sum / total_dataset_items`) while recording `evaluated_only_score` in `details`.

---

## 5. Preserved Report Schemas

- `EvaluationResult.from_dict()` deserializes both `v0.1.0` JSON reports and `v0.2.0` JSON reports.
- All `v0.1.0` top-level JSON report keys (`project_version`, `evaluation_timestamp`, `dataset`, `model`, `configuration`, `metrics_used`, `aggregate_results`, `individual_results`, `missing_responses`, `unknown_response_ids`, `warnings`, `limitations`, `reproducibility_card`) remain intact.

---

## 6. Deprecations in v0.2.0

**None.** Zero `v0.1.0` APIs, CLI flags, or schema fields are deprecated or removed in `v0.2.0`.
