# Build, Installation, and Verification Instructions

This document provides step-by-step instructions to build, install, test, lint, and run **OpenBench AI v0.1.0** locally.

---

## 1. Python Requirements

- **Python Version**: Python `3.10`, `3.11`, or `3.12`
- **Core Runtime Dependencies**: None (Python standard library only)
- **Development Dependencies**: `pytest>=7.4.0`, `ruff>=0.4.0`

---

## 2. Expected Directory Layout

```text
openbench-ai/
├── src/
│   └── openbench/
│       ├── __init__.py
│       ├── cli.py
│       ├── dataset.py
│       ├── evaluator.py
│       ├── comparison.py
│       ├── reporting.py
│       ├── exceptions.py
│       └── metrics/
│           ├── __init__.py
│           ├── base.py
│           ├── normalization.py
│           ├── exact_match.py
│           ├── accuracy.py
│           └── keyword_match.py
├── datasets/
│   ├── reasoning/
│   │   └── sample.jsonl
│   ├── coding/
│   │   └── sample.jsonl
│   ├── instruction_following/
│   │   └── sample.jsonl
│   └── safety/
│       └── sample.jsonl
├── examples/
│   ├── sample_responses.json
│   ├── basic_evaluation.py
│   └── compare_models.py
├── tests/
│   ├── test_dataset.py
│   ├── test_evaluator.py
│   ├── test_metrics.py
│   ├── test_comparison.py
│   ├── test_reporting.py
│   └── test_cli.py
├── docs/
│   ├── installation.md
│   ├── usage.md
│   ├── architecture.md
│   ├── metrics.md
│   ├── evaluation_methodology.md
│   └── contributing.md
├── scripts/
│   ├── validate_datasets.py
│   └── run_example.py
├── .github/
│   ├── workflows/
│   │   ├── tests.yml
│   │   └── lint.yml
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── pull_request_template.md
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── BUILD_INSTRUCTIONS.md
├── VALIDATION_CHECKLIST.md
└── FILE_MANIFEST.json
```

---

## 3. Installation Commands

From the `openbench-ai/` project root:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

---

## 4. Test Commands

Run the full test suite with `pytest`:

```bash
pytest -v
```

Validate all built-in JSONL datasets:

```bash
python scripts/validate_datasets.py
```

Run the end-to-end example workflow:

```bash
python scripts/run_example.py
```

---

## 5. Lint Commands

Run Ruff linter across all source, test, example, and script directories:

```bash
ruff check src tests examples scripts
```

---

## 6. CLI Examples

```bash
# Show version
openbench --version

# Validate a single dataset file
openbench validate-dataset datasets/reasoning/sample.jsonl

# Validate all datasets in the directory tree
openbench validate-dataset datasets/

# Evaluate a model on the reasoning dataset
openbench evaluate \
  --dataset datasets/reasoning/sample.jsonl \
  --responses examples/sample_responses.json \
  --metrics exact_match accuracy keyword_coverage \
  --output results/

# Generate two model results and compare them
python examples/compare_models.py
openbench compare \
  --results results/model_a.json \
  --results results/model_b.json \
  --output results/comparison.md
```
