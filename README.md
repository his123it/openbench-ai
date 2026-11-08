# OpenBench AI (v0.2.0)

**A modular, open-source LLM evaluation and benchmarking toolkit for reproducible, offline-first model comparison.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](pyproject.toml)
[![Version: 0.2.0](https://img.shields.io/badge/Version-0.2.0-2ea44f.svg)](CHANGELOG.md)

---

## Project Overview

**OpenBench AI** is an open-source Python toolkit designed to help developers, students, and AI evaluation researchers evaluate and compare language model outputs against standardized JSONL datasets using transparent, reproducible evaluation methods.

> **Note**: OpenBench AI is maintained by its project owner, and its initial development was AI-assisted. Contributions, bug reports, and new datasets are welcome!

---

## What's New in v0.2.0

OpenBench AI `v0.2.0` expands the `v0.1.0` baseline while maintaining **100% backward compatibility** with all `v0.1.0` APIs, CLI commands, datasets, and schemas:

1. **200 Original Benchmark Examples across 10 Categories**:
   - Expanded `reasoning`, `coding`, `instruction_following`, and `safety` to 20 items each (preserving all 48 original `v0.1.0` items verbatim).
   - Added 6 new 20-item categories: `mathematics`, `debugging`, `factual_knowledge`, `multilingual`, `summarization`, and `structured_output`.
2. **Dataset Catalog & Discovery (`openbench.catalog`)**:
   - Inspect, filter, validate, and compute statistics across datasets via `list-datasets`, `inspect-dataset`, and `dataset-stats`.
3. **12 Registered Evaluation Metrics (`openbench.metrics`)**:
   - Preserved `exact_match`, `accuracy`, and `keyword_coverage` (`keyword_match`).
   - Added `precision`, `recall`, `f1`, `rouge` (`rouge_1`, `rouge_l`), `bleu`, `semantic_similarity` (with pluggable embedding providers), and `structured_output` (JSON syntax, required keys, field types, and optional JSON Schema validation).
4. **Deeper Evaluation & Comparison Analysis**:
   - Category-wise and difficulty-wise score breakdowns, missing and invalid response breakdowns, optional latency and token usage summaries, runner-up margins, score spread analysis, and optional explicit user-weighted scoring.
5. **CSV Export, Local Run History & Enhanced Reproducibility Cards**:
   - Export individual item results, aggregate scores, or model comparisons to `.csv`.
   - Track longitudinal evaluation runs in a local JSONL store (`openbench history list` / `show`).
   - Extended **Benchmark Reproducibility Card** with SHA-256 dataset checksums, Python/OS environment details, model provider metadata, and automatic secret redaction.
6. **Declarative YAML Configuration Profiles & Optional Model Adapters**:
   - Run repeatable evaluations with `openbench evaluate --config configs/reasoning.yaml ...`.
   - Connect local Python functions (`LocalCallableAdapter`) or optional local Hugging Face pipelines (`HuggingFaceAdapter`).

---

## Installation

### Core Installation (Zero External Runtime Dependencies)

```bash
git clone https://github.com/<your-username>/openbench-ai.git
cd openbench-ai
python -m pip install -e .
```

### Development & Optional Feature Groups

```bash
# Recommended for development, testing, YAML configs, and JSON Schema validation
python -m pip install -e ".[dev]"

# Install specific optional feature groups
python -m pip install -e ".[config]"       # PyYAML for YAML configuration profiles
python -m pip install -e ".[metrics]"      # jsonschema for structured_output schema validation
python -m pip install -e ".[semantic]"     # sentence-transformers for neural embeddings
python -m pip install -e ".[huggingface]"  # transformers + torch for HuggingFaceAdapter
python -m pip install -e ".[all]"          # All optional dependencies
```

---

## Quick Start & CLI Reference

### 1. Explore & Validate the Dataset Catalog

```bash
# List all 10 built-in datasets (200 total examples)
openbench list-datasets

# Inspect metadata, SHA-256 checksum, and sample items
openbench inspect-dataset datasets/mathematics/sample.jsonl --samples 2

# Compute dataset statistics across the entire repository
openbench dataset-stats datasets/

# Validate a single dataset or directory tree (v0.1.0 command preserved)
openbench validate-dataset datasets/
```

### 2. Evaluate Model Responses (CLI Flags or YAML Profile)

```bash
# Standard CLI evaluation (v0.1.0 syntax preserved)
openbench evaluate   --dataset datasets/reasoning/sample.jsonl   --responses examples/sample_responses.json   --metrics exact_match accuracy keyword_coverage   --output results/

# Profile-driven evaluation (v0.2.0)
openbench evaluate   --config configs/reasoning.yaml   --responses examples/sample_responses.json   --output results/
```

### 3. Compare Multiple Models & Export to CSV

```bash
python examples/compare_models.py

openbench compare   --results results/model_a.json   --results results/model_b.json   --output results/comparison.md

# Export individual or aggregate evaluation results to CSV
openbench export   --input results/model_a.json   --format csv   --mode individual   --output results/model_a_items.csv
```

### 4. Track & Inspect Historical Evaluation Runs

```bash
openbench history list --store results/history.jsonl
openbench history show <EVALUATION_ID> --store results/history.jsonl
```

---

## Built-in Dataset Catalog (200 Original Examples)

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
| `semantic_similarity` | `0.2.0` | Embedding cosine similarity via configurable `EmbeddingProvider`. | **Not factual correctness**; opposite statements can have high similarity. |
| `structured_output` | `0.2.0` | Validates JSON syntax, required keys, field types, and JSON Schema. | Structural validity is separate from factual value accuracy. |

---

## Documentation Index

- [Installation Guide](docs/installation.md)
- [Usage & CLI Reference](docs/usage.md)
- [Architecture Overview](docs/architecture.md)
- [Backward Compatibility Contract](docs/backward_compatibility.md)
- [Dataset Catalog & Schema Guide](docs/dataset_catalog.md)
- [Core Metrics Reference](docs/metrics.md)
- [Advanced Metrics Reference (v0.2.0)](docs/advanced_metrics.md)
- [Evaluation Methodology & Reproducibility](docs/evaluation_methodology.md)
- [YAML Configuration Profiles](docs/configuration.md)
- [Optional Model Integrations & Adapters](docs/integrations.md)
- [Local Historical Tracking](docs/historical_tracking.md)
- [Future Dashboard Architecture](docs/dashboard_architecture.md)
- [Contributing Guide](CONTRIBUTING.md)
- [v0.2.0 Migration Guide](V0_2_0_MIGRATION_GUIDE.md)
- [v0.2.0 Build Instructions](V0_2_0_BUILD_INSTRUCTIONS.md)
- [v0.2.0 Validation Checklist](V0_2_0_VALIDATION_CHECKLIST.md)
- [Backward Compatibility Report](BACKWARD_COMPATIBILITY_REPORT.md)

---

## Roadmap

- **v0.1.0 (Completed)**: JSONL dataset loader, deterministic baseline metrics (`exact_match`, `accuracy`, `keyword_coverage`), model comparison, JSON/Markdown reports, Reproducibility Card, CLI, tests, and CI.
- **v0.2.0 (Current Release)**: 200 examples across 10 categories, dataset catalog/stats, advanced metrics (`precision`, `recall`, `f1`, `rouge`, `bleu`, `semantic_similarity`, `structured_output`), category/difficulty breakdowns, CSV export, local run history, YAML profiles, and model adapters.
- **v0.3.0**: Native local inference servers (llama.cpp / vLLM / Ollama runner integration) and execution-based code sandbox evaluation.
- **v0.5.0**: Structured human evaluation workflows and pairwise Bradley-Terry/Elo ranking.
- **v1.0.0**: Interactive web dashboard consuming OpenBench JSON/JSONL contracts and verified community benchmark leaderboards.

---

## License

Released under the [MIT License](LICENSE).
