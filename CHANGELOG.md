# Changelog

All notable changes to **OpenBench AI** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-11-08

### Added
- **Like-for-Like Subset Evaluation (`--ids-file`)**:
  - Added `load_ids_file()` and `IdFilterSpec` in `openbench.dataset` to load, validate, and deduplicate dataset ID lists from plain-text or JSON array files.
  - Added `--ids-file PATH` to `openbench evaluate` and `ids_file` / `selected_ids` to `EvaluationConfig` and `EvaluationProfile`.
  - Intentionally excluded dataset records are tracked in `id_filter.excluded_ids` and are never counted as missing/failed responses.
  - Included ID filter counts and `selected_ids_checksum` (SHA-256) in JSON reports, Markdown reports, and the Benchmark Reproducibility Card.
- **Optional Fenced JSON Support (`strip_markdown_fence`)**:
  - Added `strip_markdown_fence: bool = False` (default `False`) and `extract_fenced_json()` to `StructuredOutputMetric` (`openbench.metrics.structured_output`) to strip a single valid surrounding Markdown code fence (` ```json ... ``` ` or ` ``` ... ``` `) when enabled, without stripping surrounding prose or repairing malformed JSON.
- **Explicit `lexical_similarity` Metric Identifier**:
  - Added `LexicalSimilarityMetric` (`lexical_similarity`) alongside `SemanticSimilarityMetric` (`semantic_similarity`).
- **Dataset Self-Consistency Validation**:
  - Extended `scripts/validate_datasets.py` with automated self-consistency verification checking that every reference answer scores `1.0` against itself on applicable deterministic metrics while reporting the 5 known `v0.1.0` legacy keyword exceptions (`coding_003`, `coding_007`, `coding_010`, `reasoning_004`, `reasoning_012`) separately. The records corrected in `v0.2.0` are not exempt.
- **Typing & Citation Metadata**:
  - Added `CITATION.cff` and PEP 561 `src/openbench/py.typed` marker.

### Fixed
- **Value-Level Secret Redaction (`openbench.security`)**:
  - Extended `redact_sensitive_data()` and `sanitize_error_message()` to redact credentials embedded inside string values even when the dictionary key is benign (e.g., `{"endpoint": "https://example.com/api?token=abc123"}`), covering URL query tokens, `Authorization: Bearer ...` headers, inline `password=...` / `client_secret=...` assignments, and token prefixes while leaving live runtime request configs unmutated.
- **Semantic vs. Lexical Similarity Labeling (`openbench.metrics.semantic_similarity`)**:
  - Classified `LocalNgramEmbeddingProvider` with `is_semantic_embedding = False` (`similarity_kind = "lexical_similarity"`). Results produced with the default provider are now explicitly labeled `similarity_type = "lexical_similarity"`, and `require_semantic_provider=True` enforces a genuine embedding provider.
- **CLI Evaluation Warnings on `stderr` (`openbench.cli`)**:
  - `openbench evaluate` now emits all evaluation warnings (missing responses, unknown IDs, excluded IDs, invalid responses) to `sys.stderr` while preserving clean `sys.stdout` output and exit code `0`.
- **Markdown Table Pipe Escaping (`openbench.reporting`)**:
  - Added `escape_markdown_cell()` to escape `|` (`\|`) and newlines across model names, dataset names, provider names, categories, difficulties, and metric labels in Markdown tables.
- **SPDX License Metadata & Repository URLs**:
  - Updated `pyproject.toml` from `license = { text = "MIT" }` to `license = "MIT"`.
  - Replaced placeholder `<your-username>` clone URLs with `https://github.com/lakshaychhaba0001-sys/openbench-ai.git`.
- **Prose no longer mistaken for credentials**: a bare `Bearer`/`Basic` word is only redacted when followed by a credential-like value, so text such as "bearer tokens" or "basic refusal" in prompts, answers, and reports is preserved. `Authorization:` headers are always redacted.
- **Test isolation**: experiment tests no longer share the working-directory `.openbench_cache/`, which made a second test run fail.
- **Lint**: resolved unused imports, import ordering, an unnecessary f-string, and `typing.Callable` imports.

### Improved
- **Experiment Directory Resilience (`openbench.experiments`)**:
  - Hardened `ExperimentManager.list_runs()` and `inspect_run()` to handle incomplete or corrupted JSON artifacts gracefully without crashing.
- **CI Workflow Matrix (`.github/workflows/tests.yml`)**:
  - Added Python `3.13` to the test matrix (`3.10`, `3.11`, `3.12`, `3.13`), added `-W error::SyntaxWarning` to `pytest`, and added a dedicated `test-core-only` CI job (`pip install -e .` with no optional extras).

### Changed
- Removed the unused `ollama` (`httpx`) and `performance` (`tabulate`) optional dependency groups; neither package is imported. The Ollama adapter uses the standard library HTTP client.
- `redact_secrets()` in `openbench.evaluator` is kept as a backward-compatible alias of `openbench.security.redact_sensitive_data()`.

### Compatibility Notes
- Existing `v0.1.0` and `v0.2.0` public Python APIs, CLI commands, and dataset/response/report schemas remain supported. Experiment run directories use artifact schema version `0.3.0`.
- Minor observable behavioral refinements: `openbench evaluate` now writes report warnings to `stderr` (previously only stored inside the report files), Markdown tables escape `|` as `\|`, and `semantic_similarity` with the default `local_ngram_hash` provider records `details["similarity_type"] = "lexical_similarity"`.

---

## [0.3.0] - 2026-11-08

_Not published as a separate release; these changes ship in 0.3.1._

### Added
- **Model Execution Engine (`openbench.execution`)**:
  - Provider-independent `ExecutionEngine` decoupled from `Evaluator`.
  - Configurable `ExecutionConfig` and `RetryPolicy` supporting per-request timeouts, thread-pool concurrency, exponential backoff, error classification (`timeout`, `connection_error`, `rate_limit`, `validation_error`, `provider_error`), duplicate in-flight request protection, cooperative cancellation, and interrupted run resumption.
- **Model Provider & Adapter Framework (`openbench.adapters`)**:
  - Added `PROVIDER_REGISTRY`, `register_provider`, `create_provider`, `list_registered_providers`, `GenerationConfig`, and `ProviderHealthStatus`.
  - Added `OllamaAdapter` (`openbench.adapters.ollama`) supporting `/api/generate` (streaming and non-streaming NDJSON), `/api/tags` connection checks, token extraction, and endpoint validation.
  - Added `MockModelProvider` (`openbench.adapters.mock`) with `is_mock=True` for deterministic offline testing, CI smoke tests, and retry/timeout simulation.
  - Enhanced `HuggingFaceAdapter` with `check_connection()`, `MemoryError` handling, and `huggingface`/`hf` aliases.
- **Structured Experiment Management (`openbench.experiments`)**:
  - `ExperimentManager` managing `experiments/run_<id>/` directories with 7 standardized artifacts (`config.json`, `model_metadata.json`, `responses.json`, `evaluation.json`, `performance.json`, `reproducibility.json`, `logs.jsonl`).
  - Supports creating, listing, inspecting, resuming partial/interrupted runs, and comparing runs.
- **Benchmark Integrity & Provenance (`openbench.integrity`)**:
  - `RunIntegrityVerifier` verifying SHA-256 checksums across dataset, configurations, and on-disk artifacts, detecting tampered or missing files, and enforcing provenance labels (`provider_executed`, `external_responses_evaluated`, `manually_imported`).
- **Performance Monitoring (`openbench.performance`)**:
  - `PerformanceMonitor` and `PerformanceReport` tracking wall-clock duration, P50/P95/P99 latency percentiles, category latency breakdowns, throughput, retries, timeouts, token provenance (`provider_reported`, `locally_estimated`, `unavailable`), and optional user-supplied `PricingConfig`.
- **Deterministic Response Caching (`openbench.cache`)**:
  - `ResponseCache` computing composite SHA-256 keys over provider, model, revision, dataset, prompt, and generation hyperparameters to prevent cross-model or cross-config cache contamination.
- **Advanced Analytics (`openbench.analytics`)**:
  - `BenchmarkAnalytics` providing failure/error categorization, repeated-run sample variance ($n-1$ denominator with explicit sample size $n$), output consistency rates, and per-example model disagreement analysis.
- **Security & Privacy Controls (`openbench.security`)**:
  - Recursive credential redaction, error message token scrubbing, configurable `privacy_mode` prompt hashing, endpoint URL validation, and workspace output path validation.
- **CLI Expansion (`openbench.cli`)**:
  - Added `openbench run`, `openbench runs list|show|verify|resume|compare`, `openbench providers list|check`, and `openbench cache clear|stats`.
- **Configuration, Documentation & Tests**:
  - Added 4 new YAML profiles (`configs/local_ollama.yaml`, `configs/huggingface_local.yaml`, `configs/mock_experiment.yaml`, `configs/repeated_benchmark.yaml`).
  - Added 11 new documentation guides in `docs/` and 6 new test suites in `tests/`.

### Changed
- Bumped package version from `0.2.0` to `0.3.0`.

---

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
  - Added 7 new documentation guides in `docs/` and 7 new test modules in `tests/` (bringing the total test suite to 61 tests across 13 test files).

### Changed
- Bumped package version from `0.1.0` to `0.2.0` in `pyproject.toml` and `src/openbench/__init__.py`.
- Added optional dependency groups (`metrics`, `semantic`, `config`, `huggingface`, `all`) in `pyproject.toml` while keeping core runtime dependencies empty (`dependencies = []`).

### Fixed
- JSON reports no longer write credential-like `inference_config` keys (e.g. `api_key`, `auth_token`) in plain text under `model.inference_config`; they are redacted as in the reproducibility card.
- `precision`, `recall`, `f1`, `rouge`, and `bleu` now handle symbol-only answers (e.g. `$?`): identical answers score `1.0` and different symbols score `0.0`, instead of `0.0` and `1.0` respectively.
- LaTeX formulas in metric docstrings no longer contain invalid escape sequences.
- Corrected optional-dependency install hints in error messages.
- Dataset fixes: removed contradictory alternative keywords from 5 new records, aligned 2 summarization references with their keywords, disambiguated `debugging_018` and `multilingual_004`, and set new records' `version` metadata to `0.2.0`.

---

## [0.1.0] - 2026-11-08

### Added
- Initial baseline release with JSONL `DatasetLoader`, `Evaluator`, `ExactMatchMetric`, `AccuracyMetric`, `KeywordCoverageMetric`, `ModelComparator`, JSON/Markdown `ReportGenerator` with Benchmark Reproducibility Card, CLI, 48 examples across 4 categories, tests, and CI.
