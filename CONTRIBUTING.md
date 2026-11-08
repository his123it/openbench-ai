# Contributing to OpenBench AI

Thank you for your interest in contributing to **OpenBench AI**!

## Project Context & Transparency

OpenBench AI was initiated as a college software engineering and AI evaluation project. The initial architecture, baseline datasets, and maintenance are carried out by the project owner. We warmly welcome genuine contributions, bug fixes, new datasets, and additional evaluation metrics from students, researchers, and open-source developers.

---

## 1. Setting Up Your Development Environment

### Prerequisites
- Python **3.10+** (tested on Python 3.10, 3.11, and 3.12)
- Git

### Installation
Clone the repository and install the package in editable mode with development dependencies (`pytest` and `ruff`):

```bash
git clone https://github.com/<your-username>/openbench-ai.git
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
# 1. Run all unit and CLI tests
pytest -v

# 2. Check code formatting and linting with Ruff
ruff check src tests examples scripts

# 3. Validate all JSONL datasets
python scripts/validate_datasets.py
```

---

## 3. How to Add a Dataset

Datasets live under `datasets/<category>/<name>.jsonl`. Every line must be a valid JSON object matching this schema:

```json
{"id": "reasoning_013", "category": "reasoning", "prompt": "Your original evaluation prompt?", "expected_answer": "Unambiguous reference answer", "metadata": {"difficulty": "medium", "tags": ["logic"], "keywords": ["reference", "answer"]}}
```

### Dataset Requirements
1. **Originality**: Do not copy copyrighted or verbatim items from existing benchmarks (such as MMLU or HumanEval) without proper licensing. Write original, unambiguous test cases.
2. **Unique IDs**: Every record ID must be unique both within its file and across the entire `datasets/` directory.
3. **Allowed Difficulties**: `metadata.difficulty` must be `"easy"`, `"medium"`, or `"hard"`.
4. **Validation**: Always run `openbench validate-dataset datasets/<category>/<name>.jsonl` and `python scripts/validate_datasets.py` before committing.

---

## 4. How to Implement a Custom Metric

1. Create a new module in `src/openbench/metrics/` or subclass `BaseMetric` from `openbench.metrics.base`:

```python
from openbench.dataset import DatasetRecord
from openbench.metrics.base import BaseMetric, MetricResult
from openbench.metrics.normalization import NormalizationConfig


class PrefixMatchMetric(BaseMetric):
    name = "prefix_match"
    version = "0.1.0"
    description = "Checks if normalized response starts with normalized expected answer."
    limitation_note = "Only checks the leading prefix of the response."

    def __init__(self, normalization: NormalizationConfig | None = None) -> None:
        self.normalization = normalization or NormalizationConfig()

    def compute_item(self, record: DatasetRecord, response_text: str) -> MetricResult:
        passed = response_text.strip().lower().startswith(record.expected_answer.strip().lower())
        return MetricResult(
            metric_name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details={},
        )
```

2. Register your metric in `src/openbench/metrics/__init__.py` inside `METRIC_REGISTRY` (or dynamically via `register_metric(PrefixMatchMetric)`).
3. Add comprehensive unit tests in `tests/test_metrics.py` and document the metric's formula and limitations in `docs/metrics.md`.

---

## 5. Pull Request Expectations & Code Quality Standards

- Follow **PEP 8** and keep line lengths within 100 characters (`pyproject.toml` configures Ruff automatically).
- Include type hints on all public functions, methods, and dataclasses.
- Document known methodological limitations of any new metric or dataset clearly—never claim a heuristic metric measures full semantic comprehension or general intelligence.
- Ensure all GitHub Actions checks (`Tests` and `Lint`) pass.
