# Compatibility Guide (v0.1.0, v0.2.0 & v0.3.0 -> v0.3.1)

OpenBench AI `v0.3.1` is a maintenance, security, and reliability release built directly on the `v0.3.0` codebase.

---

## 1. Preserved Functionality

- **Public Python APIs (`v0.1.0` – `v0.3.0`)**: All existing classes, functions, exceptions, metrics, and adapters remain importable from their established module paths.
- **CLI Commands & Flags**: All `v0.1.0`, `v0.2.0`, and `v0.3.0` CLI subcommands and options remain intact; `--ids-file` is added as an optional flag on `openbench evaluate`.
- **Datasets**: All 10 dataset files and all 200 records (including all 48 original `v0.1.0` records) are preserved with identical IDs and content.
- **Experiment Artifacts**: `v0.3.1` reads and verifies existing `v0.3.0` `experiments/run_<id>/` directories without requiring regeneration.

---

## 2. Behavioral Refinements & Compatibility Notes in v0.3.1

1. **`openbench evaluate` Warnings on `stderr`**: Evaluation warnings (such as missing responses, unknown response IDs, or active ID filter exclusions) are now emitted to `sys.stderr` during CLI execution while keeping `sys.stdout` clean and preserving exit code `0`. Scripts that combined `2>&1` into `stdout` should read `stdout` and `stderr` separately.
2. **Markdown Table Pipe Escaping**: Model names, dataset names, categories, and provider names containing `|` now render as `\|` inside Markdown table cells so tables remain valid Markdown.
3. **`semantic_similarity` Default Provider Labeling**: `SemanticSimilarityMetric` retains its class name and `"semantic_similarity"` metric key, but explicitly labels results from the default `LocalNgramEmbeddingProvider` with `details["similarity_type"] = "lexical_similarity"` (and adds `"lexical_similarity"` as a first-class metric identifier).
4. **SPDX License String in `pyproject.toml`**: Updated from `license = { text = "MIT" }` to `license = "MIT"` (PEP 639 SPDX format). Very old `setuptools` versions (`<61.0`) may not recognize SPDX string format; `pyproject.toml` specifies `setuptools>=65.0`.
