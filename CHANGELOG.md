# Changelog

All notable changes to **OpenBench AI** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  - 40 unit and integration tests using `pytest`.
  - `ruff` linter configuration and GitHub Actions workflows (`tests.yml`, `lint.yml`).
