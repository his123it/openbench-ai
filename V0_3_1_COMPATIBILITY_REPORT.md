# OpenBench AI v0.3.1 — Compatibility Report

**Target Version**: `0.3.1`

**Project Author**: Lakshay Chhaba (`https://github.com/lakshaychhaba0001-sys/openbench-ai`)

---

## 1. Compatibility Overview Across Releases (`v0.1.0`, `v0.2.0`, `v0.3.0` -> `v0.3.1`)

OpenBench AI `v0.3.1` is a maintenance, security, and reliability release that also delivers the `v0.3.0` platform features. It preserves existing public Python APIs, CLI commands, dataset schemas, response schemas, provider adapters, and `v0.3.0` experiment run directories while introducing targeted behavioral refinements documented below.

---

## 2. Preserved Functionality Matrix

| Release / Subsystem | Preserved Interfaces & Behavior | v0.3.1 Verification Status |
| :--- | :--- | :--- |
| **v0.1.0 Public APIs & CLI** | `Dataset`, `DatasetLoader`, `DatasetRecord`, `Evaluator`, `EvaluationConfig`, `EvaluationResult`, `ItemEvaluationResult`, `ResponseLoader`, `ModelResponseSet`, `ResponseItem`, `ModelComparator`, `ComparisonResult`, `ReportGenerator`, `ExactMatchMetric`, `AccuracyMetric`, `KeywordCoverageMetric`, `NormalizationConfig`, `normalize_text`, `create_metric`, `register_metric`, and CLI commands (`validate-dataset`, `evaluate`, `compare`, `--version`). | **Preserved** (`tests/test_backward_compatibility.py`) |
| **v0.2.0 Public APIs & CLI** | `DatasetCatalog`, `DatasetCatalogEntry`, `EvaluationProfile`, `ProfileLoader`, `ResultHistoryStore`, `HistoryEntry`, `PrecisionMetric`, `RecallMetric`, `F1Metric`, `RougeMetric` (`rouge_1`, `rouge_l`), `BleuMetric`, `SemanticSimilarityMetric`, `StructuredOutputMetric`, `LocalCallableAdapter`, `DeterministicLookupAdapter`, `HuggingFaceAdapter`, CSV export, and CLI commands (`list-datasets`, `inspect-dataset`, `dataset-stats`, `history list\|show`, `export`). | **Preserved** (`tests/test_catalog.py`, `tests/test_advanced_metrics.py`, `tests/test_configuration.py`, `tests/test_history.py`) |
| **v0.3.0 Public APIs & CLI** | `ExecutionEngine`, `ExecutionConfig`, `RetryPolicy`, `ExecutionOutcome`, `ExperimentManager`, `ExperimentRecord`, `RunIntegrityVerifier`, `IntegrityCheckResult`, `PerformanceMonitor`, `PerformanceReport`, `PricingConfig`, `RequestTelemetry`, `BenchmarkAnalytics`, `ResponseCache`, `CacheRecord`, `MockModelProvider`, `OllamaAdapter`, `GenerationConfig`, `ProviderHealthStatus`, and CLI commands (`run`, `runs list\|show\|verify\|resume\|compare`, `providers list\|check`, `cache clear\|stats`). | **Preserved** (`tests/test_execution.py`, `tests/test_providers.py`, `tests/test_experiments_and_integrity.py`, `tests/test_performance_and_analytics.py`, `tests/test_v0_3_compatibility.py`) |
| **Built-in Datasets (200 Records)** | All 10 dataset files (`datasets/*/sample.jsonl`) and all 200 records (including all 48 original `v0.1.0` records and the 5 legacy keyword records `coding_003`, `coding_007`, `coding_010`, `reasoning_004`, `reasoning_012`) are preserved unmodified. | **Preserved** (`scripts/validate_datasets.py`) |
| **v0.3.0 Experiment Artifacts** | Existing `experiments/run_<id>/` directories containing `config.json`, `model_metadata.json`, `responses.json`, `evaluation.json`, `performance.json`, `reproducibility.json`, and `logs.jsonl` remain readable and verifiable in `v0.3.1`, with added resilience against corrupted or incomplete run directories during `list_runs()` and `inspect_run()`. | **Preserved & Hardened** (`tests/test_v0_3_1_maintenance.py`) |

---

## 3. Documented Behavioral Changes & Non-100%-Identical Refinements in v0.3.1

In accordance with strict engineering transparency, `v0.3.1` does **not** claim 100% behavioral identity where bug fixes or security hardening intentionally changed observable output:

1. **CLI `openbench evaluate` Warning Emission to `stderr`**:
   - *Previous (`v0.3.0`)*: Evaluation warnings (missing responses, unknown response IDs) were written inside the JSON/Markdown report files but not printed to `sys.stderr` during `openbench evaluate`.
   - *New (`v0.3.1`)*: `openbench evaluate` prints each evaluation warning (`Warning [<model_name>]: ...`) to `sys.stderr` while keeping `sys.stdout` unchanged and returning exit code `0`. Callers that merged `stderr` into `stdout` (`2>&1`) when parsing CLI output will now see warning lines unless `stderr` is separated.
2. **Markdown Table Pipe Escaping (`escape_markdown_cell`)**:
   - *Previous (`v0.3.0`)*: Unescaped `|` characters inside a model name, dataset name, or category name could break Markdown table column alignment.
   - *New (`v0.3.1`)*: Pipe characters inside Markdown table cells are escaped as `\|` and newlines are normalized to spaces.
3. **`semantic_similarity` Default Provider Classification (`lexical_similarity`)**:
   - *Previous (`v0.3.0`)*: `SemanticSimilarityMetric` with the default `LocalNgramEmbeddingProvider` did not include a `similarity_type` discriminator in `details`.
   - *New (`v0.3.1`)*: `LocalNgramEmbeddingProvider` is marked `is_semantic_embedding = False` and sets `details["similarity_type"] = "lexical_similarity"` on both item and aggregate results. The `"semantic_similarity"` metric name is retained as a legacy alias when used with the default provider, and `"lexical_similarity"` (`LexicalSimilarityMetric`) is registered as a dedicated identifier. Passing `require_semantic_provider=True` to `SemanticSimilarityMetric` raises `MetricError` if the provider is not a true embedding provider (`is_semantic_embedding = True`).
4. **Value-Level Secret Redaction in Serialized Configs, Reports, and Logs**:
   - *Previous (`v0.3.0`)*: `redact_sensitive_data()` redacted dictionary keys matching sensitive substrings (which also inadvertently matched `"key"` inside `"keyword_coverage"` if passed through `redact_sensitive_data()`) and only scrubbed `Bearer ...`, `sk-...`, and `user:pass@` in error strings.
   - *New (`v0.3.1`)*: Uses exact token boundaries + an explicit allowlist for legitimate metric/telemetry keys (`keyword_coverage`, `token_usage`, `cache_key`, etc.) while scanning all string values recursively for URL query secrets (`?token=...`, `&api_key=...`), `Authorization: Bearer ...` headers, inline `password=...` / `client_secret=...` assignments, and API token prefixes.
5. **SPDX License String in `pyproject.toml`**:
   - *Previous (`v0.3.0`)*: `license = { text = "MIT" }`.
   - *New (`v0.3.1`)*: `license = "MIT"` (PEP 639 SPDX string). Requires `setuptools>=61.0` (`pyproject.toml` specifies `setuptools>=65.0`).

---

## 4. Deprecations & Known Issues

- **Legacy Metric Alias**: Using `"semantic_similarity"` with the default `LocalNgramEmbeddingProvider` is documented as a legacy alias for `"lexical_similarity"`. It is not removed in `v0.3.1`.
- **Known Legacy Dataset Keyword Exceptions**: Five `v0.1.0` records (`coding_003`, `coding_007`, `coding_010`, `reasoning_004`, `reasoning_012`) include supplementary explanation keywords in `metadata.keywords` that do not appear verbatim in `expected_answer`. They are preserved unmodified and reported separately by `scripts/validate_datasets.py`. The seven `v0.2.0` records whose alternative-spelling keywords were corrected in `v0.2.0` are not exempt and must self-score `1.0`.
