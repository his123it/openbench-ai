# OpenBench AI (v0.1.0)

**A modular, open-source LLM evaluation and benchmarking toolkit for reproducible, offline-first model comparison.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](pyproject.toml)

---

## Project Overview & College Project Transparency

**OpenBench AI** is an open-source Python toolkit designed to help developers, students, and AI evaluation researchers test and compare language model outputs against standardized JSONL datasets using transparent, reproducible deterministic metrics.

> **Educational & Maintenance Transparency Note**:
> OpenBench AI is developed as an open-source college project. Initial architecture, implementation, and maintenance are carried out by the project owner. This repository contains no fabricated contributors, synthetic benchmark endorsements, or unverified claims. Genuine contributions, bug reports, and dataset expansions from the community are warmly welcomed!

---

## Why OpenBench AI Exists

Evaluating Large Language Models (LLMs) often involves heavyweight frameworks that require paid commercial API keys, complex database servers, or opaque LLM-as-a-judge prompts before a developer can run a single test.

OpenBench AI `v0.1.0` solves this by providing a **zero-API-key, dependency-light baseline**:
1. **Offline & Free to Run**: Evaluate pre-collected model responses stored in clean JSON files without needing paid API credentials.
2. **Strict Data Hygiene**: Reports exact line numbers for malformed JSON, missing fields, blank lines, and duplicate IDs instead of silently skipping bad records.
3. **Methodological Honesty**: Separates strict string matching (`exact_match`), bounded answer containment (`accuracy`), and lexical overlap (`keyword_coverage`), and refuses to blend incompatible metrics into a misleading single "overall score."
4. **Built-in Reproducibility Cards**: Every evaluation automatically generates a **Benchmark Reproducibility Card** recording dataset versions, model versions, normalization rules, timestamps, and explicit metric limitations.

---

## Key Features (v0.1.0)

- **Robust JSONL Dataset Loader**: Load individual `.jsonl` files or entire dataset directories with cross-file duplicate ID detection and line-level validation errors.
- **4 Original Built-in Datasets (48 Examples)**:
  - `datasets/reasoning/sample.jsonl` (12 examples)
  - `datasets/coding/sample.jsonl` (12 examples)
  - `datasets/instruction_following/sample.jsonl` (12 examples)
  - `datasets/safety/sample.jsonl` (12 examples)
- **Configurable Deterministic Metrics**:
  - `exact_match`: Normalized string equality with configurable whitespace, case, and punctuation rules.
  - `accuracy`: Deterministic correctness checking using exact or word-bounded answer containment.
  - `keyword_coverage`: Lexical keyword tracking returning matched keywords, missing keywords, and coverage percentage.
- **Multi-Model Comparison Engine**: Compare two or more models side-by-side and identify per-metric leaders.
- **Dual Reporting Formats**: Export machine-readable `.json` reports and human-readable `.md` reports with embedded Benchmark Reproducibility Cards.
- **Command-Line Interface (`openbench`)**: Validate datasets, run evaluations, and compare models directly from your terminal.

---

## Installation

Clone the repository and install in editable mode with development dependencies (`pytest` and `ruff`):

```bash
git clone https://github.com/<your-username>/openbench-ai.git
cd openbench-ai
python -m pip install -e ".[dev]"
```

Verify the installation:

```bash
openbench --version
# Output: openbench 0.1.0
```

---

## Quick Start & CLI Usage

### 1. Validate a Dataset

```bash
openbench validate-dataset datasets/reasoning/sample.jsonl
```

Output:
```text
Dataset valid: 'reasoning/sample' (12 records; categories: reasoning: 12)
```

You can also validate all datasets in the directory at once:

```bash
openbench validate-dataset datasets/
```

### 2. Evaluate a Model

```bash
openbench evaluate \
  --dataset datasets/reasoning/sample.jsonl \
  --responses examples/sample_responses.json \
  --metrics exact_match accuracy keyword_coverage \
  --output results/
```

This generates both `results/alphabaseline-7b.json` and `results/alphabaseline-7b.md`.

### 3. Compare Multiple Models

After generating evaluation results (or by running `python examples/compare_models.py` which produces `results/model_a.json` and `results/model_b.json`):

```bash
openbench compare \
  --results results/model_a.json \
  --results results/model_b.json \
  --output results/comparison.md
```

### 4. Show Version

```bash
openbench --version
```

---

## Dataset Format (JSONL)

Each line in a `.jsonl` dataset file must be a valid JSON object matching this schema:

```json
{
  "id": "reasoning_001",
  "category": "reasoning",
  "prompt": "What is 15 multiplied by 12?",
  "expected_answer": "180",
  "metadata": {
    "difficulty": "easy",
    "tags": ["arithmetic", "multiplication"],
    "keywords": ["180"]
  }
}
```

- **Required fields**: `id`, `category`, `prompt`, `expected_answer` (all non-empty strings).
- **Optional `metadata`**: `difficulty` (`"easy"`, `"medium"`, `"hard"`), `tags` (list of strings), `keywords` (list of strings for `keyword_coverage`), and `disallow_substring` (boolean for strict `accuracy`).

---

## Response Format (JSON)

Model responses are supplied in JSON format:

```json
{
  "model_name": "AlphaBaseline-7B",
  "model_version": "1.0",
  "responses": [
    {
      "id": "reasoning_001",
      "response": "180"
    },
    {
      "id": "reasoning_002",
      "response": "12"
    }
  ]
}
```

---

## Metrics Explained

| Metric | Identifier | What It Measures | Key Limitation |
| :--- | :--- | :--- | :--- |
| **Exact Match** | `exact_match` | Checks if the normalized response equals the normalized expected answer (`1.0` or `0.0`). | Penalizes valid conversational framing (e.g., `"The answer is 180."` scores `0.0` when expected answer is `"180"`). |
| **Accuracy** | `accuracy` | Checks deterministic correctness via exact match or word-bounded containment of the expected answer (`1.0` or `0.0`). | Does not parse complex negation or semantic paraphrases. |
| **Keyword Coverage** | `keyword_coverage` | Proportion (`0.0` to `1.0`) of expected keywords present in the normalized response, plus lists of matched and missing keywords. | **Not equivalent to semantic correctness**—a response may include all keywords while negating them. |

*Note on Missing Responses*: If a dataset contains 12 items and a model only provides 10 responses, the 2 missing items are recorded under `missing_responses` and counted as `0.0` in aggregate metric scores so models cannot artificially boost scores by skipping hard prompts.

---

## Example Results (`examples/compare_models.py`)

Running `python examples/compare_models.py` compares `AlphaBaseline-7B (v1.0)` against `BetaChat-3B (v0.9)` on the 12-item reasoning dataset:

| Model | Version | Evaluated / Total | Missing | Exact Match | Accuracy | Keyword Coverage |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `AlphaBaseline-7B` | `1.0` | 12 / 12 | 0 | 91.67% (0.9167) | 100.00% (1.0000) | 91.67% (0.9167) |
| `BetaChat-3B` | `0.9` | 11 / 12 | 1 | 41.67% (0.4167) | 75.00% (0.7500) | 75.00% (0.7500) |

---

## Methodological Limitations

1. **Not a Measure of General Intelligence**: OpenBench AI `v0.1.0` uses deterministic reference-based string and lexical metrics. High scores on small sample datasets do not imply broad reasoning capability or scientific validation.
2. **Keyword Coverage != Semantic Correctness**: Lexical overlap only checks whether expected terms appear in the output text.
3. **Safety Dataset Scope**: The built-in `datasets/safety/sample.jsonl` evaluates basic refusal and defensive guidance patterns using reference checks. It **does not provide comprehensive safety certification** or adversarial red-teaming assurance.

---

## Roadmap

- **v0.1.0 (Current Baseline)**:
  - JSONL dataset loader with line-accurate validation
  - Central evaluation engine & response matcher
  - Deterministic metrics (`exact_match`, `accuracy`, `keyword_coverage`)
  - Multi-model comparison & per-metric leader identification
  - JSON and Markdown reports with Benchmark Reproducibility Cards
  - CLI (`validate-dataset`, `evaluate`, `compare`, `--version`), unit tests, and GitHub Actions CI
- **v0.2.0**:
  - Dynamic custom metric plugin discovery (`entry_points`)
  - Expanded domain datasets
  - Metric score visualization charts
  - Configurable YAML/TOML evaluation profiles
- **v0.3.0**:
  - Local model inference runners (llama.cpp / Ollama / vLLM adapters)
  - Hugging Face Hub dataset and model integration
  - Additional deterministic and execution-based evaluation methods
- **v0.5.0**:
  - Structured human evaluation workflows
  - Pairwise model output comparison & Elo/Bradley-Terry tracking
  - Evaluation configuration schema versioning
- **v1.0.0**:
  - Reproducible public leaderboard generator
  - Interactive web dashboard
  - Curated community dataset registry
  - Historical regression tracking & verified reproducibility workflows

---

## Documentation Index

- [Installation Guide](docs/installation.md)
- [Usage & CLI Reference](docs/usage.md)
- [Architecture Overview](docs/architecture.md)
- [Metrics Reference & Limitations](docs/metrics.md)
- [Evaluation Methodology](docs/evaluation_methodology.md)
- [Contributing Guide](docs/contributing.md)
- [Build & Verification Instructions](BUILD_INSTRUCTIONS.md)
- [Pre-Release Validation Checklist](VALIDATION_CHECKLIST.md)

---

## Contributing & License

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md) before submitting a pull request.

Released under the [MIT License](LICENSE).
