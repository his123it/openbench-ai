# Changelog

All notable changes to **OpenBench AI** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-08

### Added
- **Dataset Expansion (200 Original Examples across 10 Categories)**:
  - Expanded existing `reasoning`, `coding`, `instruction_following`, and `safety` datasets from 12 to 20 examples each while preserving all 48 original `v0.1.0` records verbatim.
  - Added 6 new dataset categories (20 examples each): `mathematics`, `debugging`, `factual_knowledge` (with provenance/source metadata), `multilingual` (with ISO language metadata), `summarization`, and `structured_output` (with JSON key/type/schema metadata).
  - Added `openbench.catalog` (`DatasetCatalog` and `DatasetCatalogEntry`) for dataset discovery, metadata inspection, filtering by category/difficulty/language/tag, SHA-256 dataset checksums, and dataset statistics.
- **Advanced Evaluation Metrics (`openbench.metrics`)**:
  - `PrecisionMetric` (`precision`), `RecallMetric` (`recall`), and `F1Metric` (`f1`) with documented True Positive, False Positive, and False Negative derivation for binary classification decisions and element-set predictions.
  - `RougeMetric` (`rouge`, `rouge_1`, `rouge_l`) computing unigram and Longest Common Subsequence (LCS) F1, precision, and recall.
  - `BleuMetric` (`bleu`) computing smoothed sentence-level BLEU with clipped n-gram precision and exponential brevity penalty.
  - `SemanticSimilarityMetric` (`semantic_similarity`) with pluggable `EmbeddingProvider` architecture (`LocalNgramEmbeddingProvider`, `CallableEmbeddingProvider`, and optional `HuggingFaceEmbeddingProvider` with `local_files_only=True` default).
  - `StructuredOutputMetric` (`structured_output`) validating JSON parsing, required keys, field types, and optional `jsonschema` compliance while reporting structural validity separately from value correctness.
  - Added `MetricMetadata` and `list_registered_metrics()`.
- **Expanded Evaluation Engine & Telemetry (`openbench.evaluator`)**:
  - Added category-wise (`category_breakdown`), difficulty-wise (`difficulty_breakdown`), missing-response (`missing_breakdown`), and invalid-response (`invalid_response_breakdown`) breakdowns to `EvaluationResult`.
  - Added optional `latency_ms` and `token_usage` tracking on `ResponseItem` and `performance_summary` when supplied by the caller or adapter (never fabricated).
- **Enhanced Model Comparison (`openbench.comparison`)**:
  - Added category-wise comparisons, difficulty-wise comparisons, runner-up margins, score spread/deltas (`score_differences`), relative performance summaries, historical run comparison (`compare_history`), and optional explicit user-weighted scoring (`weights`).
- **Advanced Reporting & Local History (`openbench.reporting`, `openbench.history`)**:
  - Added CSV export (`ReportGenerator.to_csv`, `ReportGenerator.save_csv`) for individual items, aggregate metrics, and model comparisons.
  - Added `ResultHistoryStore` (`openbench.history`) for zero-server local JSONL run tracking.
  - Extended the **Benchmark Reproducibility Card** with `evaluation_id`, `model_provider`, `dataset_identifier`, `dataset_checksum` (SHA-256), `python_version`, `operating_system`, `inference_configuration` (with automatic secret redaction), and `metric_configurations`.
- **YAML Configuration Profiles (`openbench.configuration`, `configs/`)**:
  - Added `EvaluationProfile` and `ProfileLoader` with strict field validation and CLI override merging.
  - Added `configs/reasoning.yaml`, `configs/coding.yaml`, and `configs/instruction_following.yaml`.
- **Optional Model Adapters (`openbench.adapters`)**:
  - Added `BaseModelAdapter`, `LocalCallableAdapter`, `DeterministicLookupAdapter`, and experimental `HuggingFaceAdapter`.
- **New CLI Commands (`openbench.cli`)**:
  - Added `openbench list-datasets`, `openbench inspect-dataset`, `openbench dataset-stats`, `openbench history list`, `openbench history show`, `openbench export --format csv`, and `openbench evaluate --config`.
- **Documentation & Testing**:
  - Added 7 new documentation guides in `docs/` and 7 new test modules in `tests/` (bringing the total test suite to 59 tests across 13 test files).

### Changed
- Bumped package version from `0.1.0` to `0.2.0` in `pyproject.toml` and `src/openbench/__init__.py`.
- Added optional dependency groups (`metrics`, `semantic`, `config`, `huggingface`, `all`) in `pyproject.toml` while keeping core runtime dependencies empty (`dependencies = []`).

---

## [0.1.0] - 2026-11-08

### Added
- **Dataset System (`openbench.dataset`)**:
  - JSONL file and directory loader (`DatasetLoader`).
  - Strict schema validation (`id`, `category`, `prompt`, `expected_answer`, `metadata`).
  - Exact line-number reporting for malformed JSON, blank lines, missing fields, and duplicate IDs.
- **Sample Benchmark Datasets (`datasets/`)**:
  - 48 original evaluation items across 4 categories: `reasoning` (12), `coding` (12), `instruction_following` (12), and `safety` (12).
- **Evaluation Metrics (`openbench.metrics`)**:
  - Extensible `BaseMetric` interface returning structured `MetricResult` and `AggregateMetricResult` objects.
  - Configurable `NormalizationConfig` for whitespace stripping/collapsing, case sensitivity, and explicit punctuation handling.
  - `ExactMatchMetric`, `AccuracyMetric` (with bounded containment and exact-only rules), and `KeywordCoverageMetric`.
- **Evaluation Engine (`openbench.evaluator`)**:
  - `ResponseLoader` supporting single-model and multi-model JSON files with duplicate ID detection.
  - Central `Evaluator` matching model responses to dataset items, tracking missing/unknown IDs, and computing aggregate metrics.
- **Model Comparison (`openbench.comparison`)**:
  - `ModelComparator` comparing two or more models across metrics and identifying per-metric leaders without misleading composite rankings.
- **Reporting & Reproducibility (`openbench.reporting`)**:
  - Machine-readable JSON and human-readable Markdown report generation.
  - Built-in **Benchmark Reproducibility Card** capturing dataset/model/metric versions, timestamps, normalization settings, and limitations.
- **CLI (`openbench.cli`)**:
  - `openbench validate-dataset`, `openbench evaluate`, `openbench compare`, and `openbench --version`.
- **Quality & CI**:
  - Unit and integration tests using `pytest`.
  - `ruff` linter configuration and GitHub Actions workflows (`tests.yml`, `lint.yml`).
