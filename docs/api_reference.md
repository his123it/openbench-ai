# Python API reference (v0.4.0)

Generated from `openbench.__all__` and `openbench.stats.__all__` by `scripts/generate_reference_docs.py`; do not edit by hand.
The first 70 names are the unchanged v0.3.1 surface; v0.4.0 only appends names and appends optional parameters.

## `openbench`

| Name | Kind | Module | Summary |
| :--- | :--- | :--- | :--- |
| `__version__` | str | `` |  |
| `BaseModelAdapter` | class | `openbench.adapters.base` | Abstract interface for connecting local or external language models to OpenBench AI. |
| `BenchmarkAnalytics` | class | `openbench.analytics` | Provides statistical and diagnostic analysis across evaluations and repeated runs. |
| `CacheRecord` | class | `openbench.cache` | Auditable cache entry storing response data and full request provenance. |
| `ComparisonResult` | class | `openbench.comparison` | Structured result comparing two or more model EvaluationResults. |
| `ConfigurationError` | class | `openbench.exceptions` | Raised when evaluation or comparison configuration is invalid. |
| `Dataset` | class | `openbench.dataset` | Represents a validated collection of DatasetRecord items. |
| `DatasetCatalog` | class | `openbench.catalog` | Manages discovery, inspection, filtering, and statistics across OpenBench datasets. |
| `DatasetCatalogEntry` | class | `openbench.catalog` | Metadata summary for a single discovered dataset file in the catalog. |
| `DatasetLoader` | class | `openbench.dataset` | Loads and validates JSONL evaluation datasets from files or directories. |
| `DatasetRecord` | class | `openbench.dataset` | Represents a single standardized evaluation item. |
| `DatasetValidationError` | class | `openbench.exceptions` | Raised when a dataset file or record fails validation. |
| `DeterministicLookupAdapter` | class | `openbench.adapters.local` | Deterministic local adapter backed by an in-memory lookup dictionary for testing. |
| `EmptyDatasetError` | class | `openbench.exceptions` | Raised when a dataset file or directory contains no valid evaluation records. |
| `EvaluationConfig` | class | `openbench.evaluator` | Configuration for an evaluation run. |
| `EvaluationProfile` | class | `openbench.configuration` | Represents a validated YAML evaluation and/or model execution configuration profile. |
| `EvaluationResult` | class | `openbench.evaluator` | Complete structured result of evaluating one model on one dataset. |
| `Evaluator` | class | `openbench.evaluator` | Central evaluation engine for matching model responses to datasets and computing metrics. |
| `ExecutionConfig` | class | `openbench.execution` | Configuration for the provider-independent ExecutionEngine. |
| `ExecutionEngine` | class | `openbench.execution` | Provider-independent execution engine for running datasets against model adapters. |
| `ExecutionOutcome` | class | `openbench.execution` | Result of executing a dataset against a model provider. |
| `ExperimentManager` | class | `openbench.experiments` | Manages structured experiment execution, storage, resumption, comparison, and integrity verification. |
| `ExperimentRecord` | class | `openbench.experiments` | Complete metadata record for a managed OpenBench AI experiment run. |
| `GenerationConfig` | class | `openbench.adapters.base` | Standardized generation parameters passed to model providers. |
| `HistoryEntry` | class | `openbench.history` | Lightweight summary record stored in the local JSONL evaluation history. |
| `HuggingFaceAdapter` | class | `openbench.adapters.huggingface` | Optional adapter for local Hugging Face `transformers` text-generation pipelines. |
| `IdFilterSpec` | class | `openbench.dataset` | Validated specification of an ID filter loaded from an `--ids-file` or ID list. |
| `IntegrityCheckResult` | class | `openbench.integrity` | Result of verifying the integrity and provenance of an experiment run directory. |
| `ItemEvaluationResult` | class | `openbench.evaluator` | Evaluation outcome for a single dataset item. |
| `LexicalSimilarityMetric` | class | `openbench.metrics.semantic_similarity` | Explicitly named `lexical_similarity` metric using `LocalNgramEmbeddingProvider`. |
| `LocalCallableAdapter` | class | `openbench.adapters.local` | Adapter that wraps any local Python function `fn(prompt: str) -> str \| dict`. |
| `MetricError` | class | `openbench.exceptions` | Raised when a metric configuration or computation fails. |
| `MetricLeader` | class | `openbench.comparison` | Identifies the highest score and leading model(s) for a specific metric. |
| `MockModelProvider` | class | `openbench.adapters.mock` | Deterministic mock provider for tests, examples, and retry/timeout simulation. |
| `ModelComparator` | class | `openbench.comparison` | Compares two or more EvaluationResult objects across individual metrics, categories, and difficulties. |
| `ModelComparisonEntry` | class | `openbench.comparison` | Comparison summary row for a single evaluated model. |
| `ModelResponseSet` | class | `openbench.evaluator` | Represents a validated set of responses produced by a single language model. |
| `OllamaAdapter` | class | `openbench.adapters.ollama` | Adapter for local Ollama-compatible `/api/generate` and `/api/tags` endpoints. |
| `OllamaConnectionError` | class | `openbench.adapters.ollama` | Raised when the local or configured Ollama server cannot be reached. |
| `OpenBenchError` | class | `openbench.exceptions` | Base exception for all OpenBench AI errors. |
| `PROVIDER_REGISTRY` | dict | `` |  |
| `PerformanceMonitor` | class | `openbench.performance` | Aggregates request telemetry into a transparent, non-fabricated PerformanceReport. |
| `PerformanceReport` | class | `openbench.performance` | Structured performance report for a model execution or evaluation run. |
| `PricingConfig` | class | `openbench.performance` | Optional user-supplied pricing configuration for estimated token cost calculation. |
| `ProfileLoader` | class | `openbench.configuration` | Loads and validates YAML configuration profiles. |
| `ProviderHealthStatus` | class | `openbench.adapters.base` | Result of checking a model provider's availability and configuration. |
| `ReportGenerator` | class | `openbench.reporting` | Generates machine-readable JSON, CSV, and human-readable Markdown evaluation reports. |
| `RequestTelemetry` | class | `openbench.performance` | Telemetry record for a single prompt execution attempt/item. |
| `ResponseCache` | class | `openbench.cache` | File-backed deterministic response cache preventing cross-model/cross-config collisions. |
| `ResponseItem` | class | `openbench.evaluator` | Represents a single model response entry with optional telemetry measurements. |
| `ResponseLoader` | class | `openbench.evaluator` | Loads and validates model response JSON files. |
| `ResponseValidationError` | class | `openbench.exceptions` | Raised when a model response file or entry fails schema validation. |
| `ResultHistoryStore` | class | `openbench.history` | File-backed JSONL evaluation history store requiring zero database servers. |
| `RetryPolicy` | class | `openbench.execution` | Configurable retry and exponential backoff policy for model execution. |
| `RunIntegrityVerifier` | class | `openbench.integrity` | Verifies artifact completeness, SHA-256 checksums, and provenance consistency of experiment runs. |
| `classify_execution_error` | function | `openbench.execution` | Classify an exception raised during model generation into a standard failure category. |
| `compute_dict_sha256` | function | `openbench.integrity` | Compute deterministic SHA-256 hex digest of a JSON-serializable dictionary. |
| `compute_file_sha256` | function | `openbench.integrity` | Compute the SHA-256 hex digest of a file on disk. |
| `compute_percentile` | function | `openbench.performance` | Compute a percentile (0..100) from a non-empty sorted list of floats using linear interpolation. |
| `create_provider` | function | `openbench.adapters.base` | Instantiate a registered provider adapter by name. |
| `escape_markdown_cell` | function | `openbench.reporting` | Escape pipe characters (`\|`) and newlines so Markdown table cells remain valid. |
| `list_registered_providers` | function | `openbench.adapters.base` | Return metadata dictionaries for all unique registered model providers. |
| `load_ids_file` | function | `openbench.dataset` | Load and validate an ID filter file (plain text with one ID per line, or a JSON array of IDs). |
| `redact_prompt_if_private` | function | `openbench.security` | Mask prompt or response text in logs when configurable `privacy_mode=True` is active. |
| `redact_sensitive_data` | function | `openbench.security` | Recursively redact sensitive keys and string-embedded credentials across dicts, lists, and strings. |
| `register_provider` | function | `openbench.adapters.base` | Register a BaseModelAdapter subclass in `PROVIDER_REGISTRY`. |
| `resolve_env_credential` | function | `openbench.security` | Read an optional credential from an environment variable name without storing it in configs. |
| `sanitize_error_message` | function | `openbench.security` | Scrub embedded credentials, query tokens, headers, assignments, and API keys from a string. |
| `validate_endpoint_url` | function | `openbench.security` | Validate an external or local HTTP/HTTPS endpoint URL safely. |
| `validate_safe_output_path` | function | `openbench.security` | Validate a filesystem output path to prevent null-byte injection or unauthorized traversal. |
| `ComparisonArm` *(v0.4.0)* | class | `openbench.paired` | One side of a comparison: a single run, a repeat group, or an evaluation JSON file. |
| `ComparisonPolicy` *(v0.4.0)* | class | `openbench.paired` | Analysis parameters (all recorded in the output). |
| `CompatibilityReport` *(v0.4.0)* | class | `openbench.paired` | CompatibilityReport(hard: 'list[dict[str, Any]]' = <factory>, warnings: 'list[dict[str, Any]]' = <factory>, info: 'list[dict[str, Any]]' = <factory>) |
| `IncompatibleComparisonError` *(v0.4.0)* | class | `openbench.exceptions` | Raised when a paired comparison has hard incompatibilities (v0.4.0). |
| `PairedComparisonEngine` *(v0.4.0)* | class | `openbench.paired` | Compares stored arms on aligned record IDs. Never calls a model provider. |
| `PairedComparisonResult` *(v0.4.0)* | class | `openbench.paired` | Full output of :meth:`PairedComparisonEngine.compare` (JSON via :meth:`to_dict`). |
| `RepeatGroupResult` *(v0.4.0)* | class | `openbench.experiments` | Result of executing (or resuming) a repeat group of runs (v0.4.0). |
| `RunLineage` *(v0.4.0)* | class | `openbench.lineage` | Identity of one run within an experiment (v0.4.0). |
| `AnnotationImporter` *(v0.4.0)* | class | `openbench.preferences` | Parse and validate annotation files (JSONL or CSV) against a task key. |
| `PreferenceAnalyzer` *(v0.4.0)* | class | `openbench.preferences` | Analyse validated annotations joined with one or more task keys. |
| `PreferenceAnnotation` *(v0.4.0)* | class | `openbench.preferences` | PreferenceAnnotation(annotation_id: 'str', task_id: 'str', annotator_id: 'str', outcome: 'str', rubric_scores: 'dict[str, float]' = <factory>, rationale: 'str \ |
| `PreferenceTaskBuilder` *(v0.4.0)* | class | `openbench.preferences` | Build blind A/B tasks and a separate identity key from two sources. |
| `AuditFinding` *(v0.4.0)* | class | `openbench.quality` | AuditFinding(check_id: 'str', severity: 'str', message: 'str', record_ids: 'tuple[str, ...]' = (), run_ids: 'tuple[str, ...]' = (), remediation: 'str' = '', det |
| `AuditReport` *(v0.4.0)* | class | `openbench.quality` | AuditReport(kind: 'str', subject: 'dict[str, Any]', parameters: 'dict[str, Any]', findings: 'list[AuditFinding]', extra_summary: 'dict[str, Any]' = <factory>) |
| `DatasetAuditor` *(v0.4.0)* | class | `openbench.quality` | Audit JSONL dataset files or a directory of them (read-only). |
| `ExperimentAuditor` *(v0.4.0)* | class | `openbench.quality` | Read-only quality checks for stored runs and repeat groups (EQ checks). |

## `openbench.stats`

| Name | Kind | Module | Summary |
| :--- | :--- | :--- | :--- |
| `BINARY_EXACT_MAX_N` | int | `` |  |
| `AdjustedPValues` | class | `openbench.stats.results` | Raw and multiplicity-adjusted p-values for one family of tests. |
| `BinaryPairedCounts` | class | `openbench.stats.effects` | Discordance table for a binary metric with one repetition per arm. |
| `ConfidenceInterval` | class | `openbench.stats.results` | A bootstrap confidence interval for a mean or a mean paired difference. |
| `EquivalenceInterval` | class | `openbench.stats.results` | The per-comparison (1 - 2 alpha) interval dual to a TOST. |
| `EquivalenceResult` | class | `openbench.stats.results` | Result of a two one-sided tests (TOST) equivalence analysis for one comparison. |
| `PairedEffects` | class | `openbench.stats.effects` | Descriptive effect sizes for paired continuous differences d = y_B - y_A. |
| `StatWarning` | class | `openbench.stats.results` | A machine-readable warning attached to a statistical result. |
| `TestResult` | class | `openbench.stats.results` | Result of a hypothesis test of no difference between two paired arms. |
| `adjust_p_values` | function | `openbench.stats.tests` | Adjust a family of p-values: ``"holm"`` (default), ``"bh"`` or ``"none"``. |
| `binary_exact_interval` | function | `openbench.stats.equivalence` | (1 - 2 alpha) interval by inverting the conservative one-sided exact tests. |
| `binary_exact_p_lower` | function | `openbench.stats.equivalence` | Conservative exact p-value for H: Delta <= delta0 (by symmetry: swap b, c; negate). |
| `binary_exact_p_upper` | function | `openbench.stats.equivalence` | Conservative exact unconditional p-value for H: Delta >= delta0 (reject for small Z). |
| `binary_paired_counts` | function | `openbench.stats.effects` | Count the 2 x 2 paired table from aligned binary scores (0.0 or 1.0). |
| `bootstrap_mean_ci` | function | `openbench.stats.bootstrap` | Percentile bootstrap CI for the mean of ``values``. |
| `classify_practical_importance` | function | `openbench.stats.effects` | Classify a CI [lower, upper] of a difference against the practical threshold dp. |
| `decompose_variability` | function | `openbench.stats.variability` | Decompose variability across the repetitions of one configuration. |
| `derive_seed` | function | `openbench.stats.rng` | Return the 64-bit seed for one random stream. |
| `derive_stream` | function | `openbench.stats.rng` | Return a fresh :class:`random.Random` seeded by :func:`derive_seed`. |
| `equivalence_family_holm` | function | `openbench.stats.equivalence` | Holm-adjust the p_TOST values of one equivalence family. |
| `mcnemar_exact` | function | `openbench.stats.tests` | Exact McNemar test. |
| `paired_bootstrap_ci` | function | `openbench.stats.bootstrap` | Percentile bootstrap CI for the mean paired difference (alias of :func:`bootstrap_mean_ci` applied to aligned differences y_B - y_A). |
| `paired_effects` | function | `openbench.stats.effects` | Mean difference, SD, d_z, probability of superiority and wins/losses/ties. |
| `score_statistic` | function | `openbench.stats.equivalence` | Return ``(z, q10_tilde, variance)`` of Tango's score statistic at ``delta0``. |
| `select_difference_test` | function | `openbench.stats.tests` | Return the difference test for a metric kind: ``"mcnemar_exact"``, ``"sign_flip"`` or ``None`` (unsupported; CI only). |
| `sign_flip_test` | function | `openbench.stats.tests` | Two-sided paired sign-flip permutation test of no difference. |
| `studentized_mean` | function | `openbench.stats.tests` | t = mean / (sd / sqrt(n)); ``None`` when n < 2, +/-inf when the SD is zero. |
| `tost_bootstrap_t` | function | `openbench.stats.equivalence` | Null-imposed bootstrap-t TOST for paired continuous differences (section 1.9.1.4). |
| `tost_paired_binary_exact` | function | `openbench.stats.equivalence` | Exact unconditional score TOST for a paired binary metric (one repetition per arm). |

## Other v0.4.0 modules

- `openbench.lineage`: `RunLineage`, ID validation, seed status, record content hashes.
- `openbench.paired`: `ComparisonArm`, `ComparisonPolicy`, `CompatibilityReport`, `PairedComparisonEngine`, `check_compatibility`.
- `openbench.preferences`: task builder, key loader, annotation importer, analyzer, `cohens_kappa`, `fleiss_kappa`, `csv_safe`.
- `openbench.quality`: `DatasetAuditor`, `ExperimentAuditor`, `AuditFinding`, `AuditReport`, normalisation and shingling helpers.
- `openbench.evalstats`: `compute_evaluation_statistics` (the `evaluate --confidence-intervals` block).
- `openbench.reporting_v040`: JSON, Markdown and CSV renderers; `openbench.provenance`: provenance blocks.
