# OpenBench AI (v0.3.1)

**A modular, open-source LLM experiment execution, evaluation, and benchmarking platform.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](pyproject.toml)
[![Version: 0.3.1](https://img.shields.io/badge/Version-0.3.1-2ea44f.svg)](CHANGELOG.md)

---

## Project Overview

**OpenBench AI** is a modular Python toolkit and experiment platform designed to help developers and researchers run reproducible language model benchmarks, evaluate model responses against standardized JSONL datasets, monitor latency and token performance, and verify benchmark artifact integrity—without requiring paid commercial API keys.

> **Note**: OpenBench AI is maintained by its project owner, and its initial development was AI-assisted. Contributions, bug reports, and new datasets are welcome!

---

## Highlights in v0.3.1 (Maintenance, Security & Reliability Release)

- **Like-for-Like Subset Evaluation (`--ids-file`)**: Evaluate response files against an explicit subset of dataset IDs (`openbench evaluate --ids-file legacy_ids.txt ...`) so intentionally excluded dataset records are never penalized as missing responses.
- **Stronger Value-Level Secret Redaction**: Redacts URL query tokens (`?token=...`, `&api_key=...`), `Authorization: Bearer ...` headers, inline `password=...` / `client_secret=...` assignments, and API token prefixes embedded inside string values while preserving live runtime credentials during model execution.
- **Optional Fenced JSON Stripping (`strip_markdown_fence`)**: `StructuredOutputMetric` can optionally strip a single valid surrounding Markdown code fence (```` ```json ... ``` ````) when `strip_markdown_fence=True` (default `False`).
- **Accurate Lexical vs. Semantic Similarity Labeling**: Distinguishes surface n-gram feature hashing (`lexical_similarity` via `LocalNgramEmbeddingProvider`) from neural embedding similarity (`semantic_similarity` via `HuggingFaceEmbeddingProvider` or verified embedding providers).
- **Stderr Warning Emission & Markdown Table Escaping**: `openbench evaluate` prints evaluation warnings to `stderr` while keeping `stdout` clean, and Markdown tables escape `|` characters in model, dataset, and category names.
- **Core v0.3.0 Platform Preserved**: Full support for `ExecutionEngine`, `OllamaAdapter`, `HuggingFaceAdapter` (experimental), `MockModelProvider`, `ExperimentManager` (`experiments/run_<id>/`), `RunIntegrityVerifier`, `PerformanceMonitor`, and `BenchmarkAnalytics`.

---

## Installation

```bash
git clone https://github.com/lakshaychhaba0001-sys/openbench-ai.git
cd openbench-ai

# Core installation (zero external runtime dependencies)
python -m pip install -e .

# Development + YAML config + JSON Schema validation
python -m pip install -e ".[dev]"

# Optional integration groups
python -m pip install -e ".[huggingface]"  # transformers + torch for local HF models
python -m pip install -e ".[semantic]"     # sentence-transformers for neural embeddings
python -m pip install -e ".[all]"          # All optional dependencies
```

The Ollama adapter needs no extra packages: it uses the Python standard library HTTP client.

---

## Quick Start & CLI Reference

### 1. Evaluate Static Response Files (Full Dataset or `--ids-file` Subset)

```bash
# Full dataset evaluation
openbench evaluate   --dataset datasets/reasoning/sample.jsonl   --responses examples/sample_responses.json   --metrics exact_match accuracy keyword_coverage   --output results/

# Like-for-like subset evaluation using --ids-file
openbench evaluate   --dataset datasets/reasoning/sample.jsonl   --responses examples/sample_responses.json   --ids-file legacy_ids.txt   --output results/
```

### 2. Execute a Model Experiment & Manage Runs

```bash
openbench providers list
openbench providers check mock

openbench run --config configs/mock_experiment.yaml

openbench runs list
openbench runs show <RUN_ID>
openbench runs verify <RUN_ID>
openbench runs resume <RUN_ID>
openbench runs compare <RUN_ID_1> <RUN_ID_2> --output results/run_comparison.md

openbench cache stats
openbench cache clear
```

### 3. Explore & Validate the 200-Item Dataset Catalog

```bash
openbench list-datasets
openbench inspect-dataset datasets/mathematics/sample.jsonl --samples 2
openbench dataset-stats datasets/
openbench validate-dataset datasets/
python scripts/validate_datasets.py
```

---

## Built-in Dataset Catalog (200 Examples)

| Category | Path | Examples | Highlights |
| :--- | :--- | ---: | :--- |
| `reasoning` | `datasets/reasoning/sample.jsonl` | 20 | Deductive logic, scheduling, combinatorics, rate problems |
| `mathematics` | `datasets/mathematics/sample.jsonl` | 20 | Calculus, linear algebra, number theory, geometry |
| `coding` | `datasets/coding/sample.jsonl` | 20 | Python stdlib, algorithms, Big-O, SQL, HTTP, Git |
| `debugging` | `datasets/debugging/sample.jsonl` | 20 | Root-cause analysis across Python, SQL, C, JS, and Git |
| `instruction_following` | `datasets/instruction_following/sample.jsonl` | 20 | Strict formatting, delimiters, casing, sorting |
| `factual_knowledge` | `datasets/factual_knowledge/sample.jsonl` | 20 | Verifiable science & CS facts with `source` metadata |
| `safety` | `datasets/safety/sample.jsonl` | 20 | Harm refusal, secure coding, medical boundaries with `rationale` |
| `multilingual` | `datasets/multilingual/sample.jsonl` | 20 | 11 languages (`es`, `fr`, `de`, `hi`, `pt`, `it`, `ja`, `zh`, `ru`, `nl`, `la`) |
| `summarization` | `datasets/summarization/sample.jsonl` | 20 | Single-sentence technical and scientific summarization |
| `structured_output` | `datasets/structured_output/sample.jsonl` | 20 | JSON generation with `required_keys`, `field_types`, `expected_schema` |

---

## Metrics Reference

| Metric Identifier | Version | Description | Key Methodological Limitation |
| :--- | :--- | :--- | :--- |
| `exact_match` | `0.1.0` | Normalized string equality (`1.0` or `0.0`). | Penalizes conversational framing. |
| `accuracy` | `0.1.0` | Deterministic correctness via exact or word-bounded answer containment. | Does not parse negation or complex paraphrases. |
| `keyword_coverage` | `0.1.0` | Proportion of expected keywords present in the response. | **Not equivalent to semantic correctness.** |
| `precision` | `0.2.0` | $\text{TP} / (\text{TP} + \text{FP})$ for binary decisions or element sets. | Applies to discrete decisions/sets, not free-form prose truth. |
| `recall` | `0.2.0` | $\text{TP} / (\text{TP} + \text{FN})$ for binary decisions or element sets. | Does not penalize extra false-positive elements alone. |
| `f1` | `0.2.0` | Harmonic mean of `precision` and `recall`. | Ignores word order outside discrete elements. |
| `rouge` / `rouge_1` / `rouge_l` | `0.2.0` | Unigram (`rouge_1`) and LCS (`rouge_l`) overlap F1. | Cannot detect factual hallucinations or antonym flips. |
| `bleu` | `0.2.0` | Smoothed clipped $n$-gram precision with brevity penalty. | Penalizes valid synonyms and paraphrases. |
| `semantic_similarity` | `0.2.0` | Embedding cosine similarity via configurable `EmbeddingProvider`; with the default `local_ngram_hash` provider it is a legacy alias for `lexical_similarity`. Pass `require_semantic_provider=True` to require a neural embedding provider. | **Not factual correctness**; opposite statements can have high similarity. |
| `lexical_similarity` | `0.3.1` | Cosine similarity of hashed word and character 3-gram features. | Surface wording only; paraphrases score low and negations score high. |
| `structured_output` | `0.2.0` | Validates JSON syntax, required keys, field types, and JSON Schema. | Structural validity is separate from factual value accuracy. |

---

## Documentation Index

- **Execution, Providers & Integrity**:
  - [Model Execution Engine](docs/model_execution.md) | [Providers Overview](docs/providers.md) | [Local Ollama Guide](docs/ollama.md) | [Hugging Face Guide (Experimental)](docs/huggingface.md) | [Custom Provider Guide](docs/custom_provider.md)
  - [Experiment Tracking](docs/experiment_tracking.md) | [Performance Monitoring](docs/performance_monitoring.md) | [Reliability & Caching](docs/reliability.md) | [Benchmark Integrity](docs/benchmark_integrity.md) | [Security & Privacy](docs/security_and_privacy.md)
- **Datasets, Metrics & Compatibility**:
  - [Installation Guide](docs/installation.md) | [Usage Reference](docs/usage.md) | [Architecture](docs/architecture.md)
  - [Dataset Catalog](docs/dataset_catalog.md) | [Core Metrics](docs/metrics.md) | [Advanced Metrics (v0.3.1)](docs/advanced_metrics.md)
  - [YAML Configuration Profiles](docs/configuration.md) | [Historical Tracking](docs/historical_tracking.md) | [Compatibility Guide](docs/v0_3_compatibility.md)
- **v0.3.1 Release Deliverables**:
  - [v0.3.1 Compatibility Report](V0_3_1_COMPATIBILITY_REPORT.md) | [v0.3.1 Validation Report](V0_3_1_VALIDATION_REPORT.md) | [v0.3.1 Integration Guide](V0_3_1_INTEGRATION_GUIDE.md) | [v0.3.1 Change Manifest](V0_3_1_CHANGE_MANIFEST.json)

---

## License

Released under the [MIT License](LICENSE).
