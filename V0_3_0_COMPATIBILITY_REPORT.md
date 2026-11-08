# OpenBench AI v0.3.0 — Backward Compatibility Report

This report documents the compatibility verification of all **v0.1.0** and **v0.2.0** interfaces, CLI commands, configurations, datasets, and report formats under **OpenBench AI v0.3.0**.

---

## 1. Public API & Module Compatibility Matrix

| Feature Area | Modules / Symbols | v0.1.0 / v0.2.0 Contract | v0.3.0 Status | Test Suite Coverage |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset Loader & Catalog** | `Dataset`, `DatasetLoader`, `DatasetRecord`, `DatasetCatalog` | Load/filter/validate JSONL files & directories; compute SHA-256 checksum & stats | **100% Preserved** | `test_dataset.py`, `test_catalog.py` |
| **Baseline & Advanced Metrics** | All 12 metrics in `openbench.metrics` (`exact_match`, `accuracy`, `keyword_coverage`, `precision`, `recall`, `f1`, `rouge`, `rouge_1`, `rouge_l`, `bleu`, `semantic_similarity`, `structured_output`) | Deterministic item & aggregate scoring; `create_metric`, `register_metric`, `list_registered_metrics` | **100% Identical** | `test_metrics.py`, `test_advanced_metrics.py`, `test_structured_output.py` |
| **Evaluation Engine** | `Evaluator`, `EvaluationConfig`, `EvaluationResult`, `ResponseLoader`, `ModelResponseSet`, `ResponseItem` | Evaluate static JSON response files without needing any model provider | **100% Preserved** | `test_evaluator.py`, `test_backward_compatibility.py` |
| **Model Comparison** | `ModelComparator`, `ComparisonResult`, `ModelComparisonEntry`, `MetricLeader` | Multi-model comparison, per-metric leaders, category/difficulty breakdown, historical comparison | **100% Preserved** | `test_comparison.py`, `test_history.py` |
| **Reporting & History** | `ReportGenerator` (`to_json`, `to_markdown`, `to_csv`), `ResultHistoryStore` | JSON, Markdown, CSV reports, Reproducibility Cards, and local `history.jsonl` | **100% Preserved** | `test_reporting.py`, `test_history.py` |
| **YAML Profiles** | `EvaluationProfile`, `ProfileLoader` (`configs/reasoning.yaml`, `coding.yaml`, `instruction_following.yaml`) | Load v0.2.0 YAML profiles without requiring `provider` or `execution` sections | **100% Preserved** (Additive `provider`, `generation`, `execution`, `performance`, `privacy` sections supported) | `test_configuration.py`, `test_v0_3_compatibility.py` |
| **Model Adapters** | `BaseModelAdapter`, `LocalCallableAdapter`, `DeterministicLookupAdapter`, `HuggingFaceAdapter` | `generate_one`, `generate_responses`, `reproducibility_metadata` | **100% Preserved** (Added `OllamaAdapter`, `MockModelProvider`, `check_connection`, and registry) | `test_integrations.py`, `test_providers.py` |

---

## 2. CLI Command Compatibility Matrix

| Command Group | CLI Invocation | Origin Version | v0.3.0 Status |
| :--- | :--- | :--- | :--- |
| **Version** | `openbench --version` | `v0.1.0` | **Preserved** (`openbench 0.3.0`, exit code `0`) |
| **Dataset Validation** | `openbench validate-dataset PATH` | `v0.1.0` | **Preserved** |
| **Static Evaluation** | `openbench evaluate --dataset ... --responses ... --output ...` | `v0.1.0` | **Preserved** |
| **Profile Evaluation** | `openbench evaluate --config ... --responses ... --output ...` | `v0.2.0` | **Preserved** |
| **Model Comparison** | `openbench compare --results ... --results ... --output ...` | `v0.1.0` | **Preserved** |
| **Dataset Catalog** | `openbench list-datasets`, `inspect-dataset`, `dataset-stats` | `v0.2.0` | **Preserved** |
| **Result History** | `openbench history list`, `openbench history show ID` | `v0.2.0` | **Preserved** |
| **Report Export** | `openbench export --input ... --format csv --output ...` | `v0.2.0` | **Preserved** |
| **Model Execution** | `openbench run ...` | `v0.3.0` | **New (Verified)** |
| **Experiment Management** | `openbench runs list\|show\|verify\|resume\|compare` | `v0.3.0` | **New (Verified)** |
| **Provider Inspection** | `openbench providers list\|check` | `v0.3.0` | **New (Verified)** |
| **Cache Management** | `openbench cache clear\|stats` | `v0.3.0` | **New (Verified)** |
