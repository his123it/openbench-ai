# Contributing to OpenBench AI

Thank you for your interest in contributing to **OpenBench AI**!

## Project Maintenance & Governance

OpenBench AI is maintained by its project owner, and its initial development was AI-assisted. We welcome bug fixes, dataset expansions, evaluation metrics, provider adapters, and documentation improvements from the open-source community.

---

## 1. Setting Up Your Development Environment

### Prerequisites
- Python **3.10+** (supported on Python 3.10, 3.11, 3.12, and 3.13)
- Git

### Installation
Clone the repository and install the package in editable mode with development dependencies:

```bash
git clone https://github.com/lakshaychhaba0001-sys/openbench-ai.git
cd openbench-ai
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify that the CLI is installed properly:

```bash
openbench --version
```

---

## 2. Running Tests and Linters

Before submitting a pull request, run the full validation suite locally:

```bash
# 1. Run all unit and CLI tests with SyntaxWarning treated as error
pytest -W error::SyntaxWarning -v

# 2. Check code formatting and linting with Ruff
ruff check src tests examples scripts

# 3. Validate all JSONL datasets and reference-answer self-consistency
python scripts/validate_datasets.py

# 4. Optional: statistical reference cross-checks (NumPy/SciPy/statsmodels, tests only)
python -m pip install -e ".[dev,stats-ref]"
OPENBENCH_RUN_SLOW=1 pytest tests/reference tests/test_stats_*.py

# 5. Regenerate the CLI and API reference after changing commands or public names
python scripts/generate_reference_docs.py
```

Statistical changes must keep the documented assumptions and exact or approximate labels in
`docs/statistical_methodology.md` accurate. Never describe a non-significant difference as
equivalence. Dataset records must not be changed silently: the quality audit and the
dataset digest tests will fail, and any correction needs its own documented review.

---

## 3. How to Add a Dataset

Datasets live under `datasets/<category>/<name>.jsonl`. Every line must be a valid JSON object matching this schema:

```json
{"id": "reasoning_021", "category": "reasoning", "prompt": "Your original evaluation prompt?", "expected_answer": "Unambiguous reference answer", "metadata": {"difficulty": "medium", "tags": ["logic"], "keywords": ["reference", "answer"], "version": "0.3.0"}}
```

### Dataset Requirements
1. **Originality**: Do not copy copyrighted or verbatim items from existing benchmarks without proper licensing. Write original, unambiguous test cases.
2. **Unique IDs**: Every record ID must be unique both within its file and across the entire `datasets/` directory.
3. **Self-Consistency**: Every new reference answer must score `1.0` against itself on all applicable deterministic metrics (`exact_match`, `accuracy`, `keyword_coverage`, `precision`, `recall`, `f1`, `rouge_1`, `rouge_l`, `bleu`, `lexical_similarity`), and all keywords in `metadata.keywords` must appear in `expected_answer`.
4. **Validation**: Always run `openbench validate-dataset datasets/<category>/<name>.jsonl` and `python scripts/validate_datasets.py` before committing.

---

## 4. How to Implement a Custom Metric

1. Subclass `BaseMetric` from `openbench.metrics.base` and return structured `MetricResult` and `AggregateMetricResult` objects.
2. Register your metric in `src/openbench/metrics/__init__.py` inside `METRIC_REGISTRY` (or dynamically via `register_metric()`).
3. Add comprehensive unit tests and document the metric's formula and limitations.

---

## 5. Pull Request Expectations & Code Quality Standards

- Follow **PEP 8** and keep line lengths within 100 characters (`pyproject.toml` configures Ruff with `E, F, I, B, UP, N`).
- Use raw docstrings (`r"""..."""`) whenever a docstring contains LaTeX formulas or backslashes so `pytest -W error::SyntaxWarning` passes cleanly.
- Document known methodological limitations of any new metric or dataset clearly.
