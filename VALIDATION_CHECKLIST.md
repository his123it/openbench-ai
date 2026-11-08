# Pre-Publication Validation Checklist (v0.1.0)

Use this checklist to verify every aspect of **OpenBench AI v0.1.0** before publishing or tagging a release on GitHub.

## 1. Package & Installation Verification
- [x] `pyproject.toml` defines package metadata, Python `>=3.10` requirement, MIT license, `[dev]` extras (`pytest`, `ruff`), and CLI script `openbench = "openbench.cli:main"`.
- [x] Package uses standard `src/openbench/__init__.py` and `src/openbench/metrics/__init__.py` package structure.
- [x] `python -m pip install -e ".[dev]"` (or `--no-build-isolation --no-deps -e .` in offline environments) installs cleanly.
- [x] `openbench --version` prints `openbench 0.1.0` and exits with code `0`.

## 2. Dataset System & Built-in Datasets
- [x] `datasets/reasoning/sample.jsonl` contains 12 original, unambiguous test cases.
- [x] `datasets/coding/sample.jsonl` contains 12 original, unambiguous test cases.
- [x] `datasets/instruction_following/sample.jsonl` contains 12 original, unambiguous test cases.
- [x] `datasets/safety/sample.jsonl` contains 12 original safe-response test cases with explicit documentation of reference-check limitations.
- [x] Total built-in evaluation examples: **48 original items** (exceeds minimum requirement of 40).
- [x] `DatasetLoader` detects missing files, empty files, blank lines, malformed JSON (reporting exact line numbers), missing required fields, and duplicate IDs within and across files.
- [x] `python scripts/validate_datasets.py` passes with 0 errors.

## 3. Metrics, Evaluator, Comparison & Reporting
- [x] `ExactMatchMetric` supports configurable whitespace, case, and punctuation normalization.
- [x] `AccuracyMetric` implements deterministic correctness rules with word-boundary protection against false prefix matches (e.g., `"18"` vs `"180"`).
- [x] `KeywordCoverageMetric` reports matched keywords, missing keywords, and coverage percentage, and documents that lexical coverage is not semantic correctness.
- [x] `Evaluator` matches responses by ID, detects duplicate/unknown/missing response IDs, and remains completely decoupled from the CLI.
- [x] `ModelComparator` compares 2+ models and identifies per-metric leaders without creating a misleading overall composite rank.
- [x] `ReportGenerator` produces both `.json` and `.md` reports with a complete **Benchmark Reproducibility Card**.

## 4. Testing, Code Quality & CI/CD
- [x] 35+ meaningful unit and integration tests implemented across `tests/test_dataset.py`, `tests/test_metrics.py`, `tests/test_evaluator.py`, `tests/test_comparison.py`, `tests/test_reporting.py`, and `tests/test_cli.py`.
- [x] `.github/workflows/tests.yml` and `.github/workflows/lint.yml` configured with minimal `contents: read` permissions and pinned major action versions (`@v4`, `@v5`).
- [x] Issue templates (`bug_report.md`, `feature_request.md`) and `pull_request_template.md` included under `.github/`.

## 5. Documentation & Transparency
- [x] `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`, `LICENSE`, `BUILD_INSTRUCTIONS.md`, `VALIDATION_CHECKLIST.md`, `FILE_MANIFEST.json`, and all 6 files in `docs/` are complete and consistent.
- [x] Repository states that OpenBench AI is maintained by the project owner and that initial development was AI-assisted, with no fabricated claims or endorsements.
