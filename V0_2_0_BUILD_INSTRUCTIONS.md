# OpenBench AI v0.2.0 — Build, Test & Verification Instructions

This document details how to install, validate, test, lint, and execute all **OpenBench AI v0.2.0** capabilities.

---

## 1. Python & Dependency Requirements

- **Python**: `>=3.10` (tested on Python 3.10, 3.11, and 3.12)
- **Core Runtime Dependencies**: None (`dependencies = []`)
- **Optional Dependency Groups**:
  - `dev`: `pytest`, `ruff`, `PyYAML`, `jsonschema`
  - `config`: `PyYAML`
  - `metrics`: `jsonschema`
  - `semantic`: `sentence-transformers`
  - `huggingface`: `transformers`, `torch`
  - `all`: All optional packages above

---

## 2. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

*(In an offline environment where packages are pre-installed in the system environment, install with `python3 -m pip install --no-build-isolation --no-deps -e .`.)*

---

## 3. Dataset Validation (200 Examples across 10 Categories)

```bash
python scripts/validate_datasets.py
openbench validate-dataset datasets/
openbench list-datasets
openbench dataset-stats datasets/
```

---

## 4. Running Tests & Linters

```bash
# Run full pytest test suite (13 test files, 59 test functions)
pytest -v

# Run Ruff linter
ruff check src tests examples scripts
```

---

## 5. Testing Old (v0.1.0) and New (v0.2.0) CLI Commands

```bash
# Version check
openbench --version

# v0.1.0 evaluate command
openbench evaluate   --dataset datasets/reasoning/sample.jsonl   --responses examples/sample_responses.json   --metrics exact_match accuracy keyword_coverage   --output results/

# v0.2.0 profile-based evaluate command with history recording
openbench evaluate   --config configs/reasoning.yaml   --responses examples/sample_responses.json   --history-file results/history.jsonl   --output results/

# v0.2.0 dataset inspection & stats
openbench inspect-dataset datasets/structured_output/sample.jsonl --samples 2
openbench dataset-stats datasets/mathematics/sample.jsonl

# v0.2.0 history inspection
openbench history list --store results/history.jsonl

# v0.2.0 CSV export
openbench export   --input results/alphabaseline-7b.json   --format csv   --mode individual   --output results/alphabaseline-7b-items.csv

# Multi-model comparison
python examples/compare_models.py
openbench compare   --results results/model_a.json   --results results/model_b.json   --output results/comparison.md
```
