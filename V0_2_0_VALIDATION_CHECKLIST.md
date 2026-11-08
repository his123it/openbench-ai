# OpenBench AI v0.2.0 — Validation Checklist

Every item below was verified before packaging the `v0.2.0` release archives:

## 1. Backward Compatibility (`v0.1.0` Preservation)
- [x] All `v0.1.0` module import paths and public symbol names preserved.
- [x] All 48 original `v0.1.0` evaluation examples preserved with identical IDs (`reasoning_001`–`012`, `coding_001`–`012`, `instruction_001`–`012`, `safety_001`–`012`).
- [x] Existing `exact_match`, `accuracy`, and `keyword_coverage` metrics produce identical scores on `v0.1.0` inputs.
- [x] `EvaluationResult.from_dict()` cleanly loads legacy `v0.1.0` JSON reports.
- [x] All `v0.1.0` CLI commands (`validate-dataset`, `evaluate`, `compare`, `--version`) and example scripts (`examples/basic_evaluation.py`, `examples/compare_models.py`) work without modification.
- [x] Dedicated `docs/backward_compatibility.md`, `BACKWARD_COMPATIBILITY_REPORT.md`, and `tests/test_backward_compatibility.py` created and verified.

## 2. Dataset Expansion & Catalog (`200` Examples across `10` Categories)
- [x] 10 categories with 20 validated original examples each (200 total examples):
  - `reasoning` (20), `mathematics` (20), `coding` (20), `debugging` (20), `instruction_following` (20), `factual_knowledge` (20), `safety` (20), `multilingual` (20), `summarization` (20), `structured_output` (20).
- [x] `factual_knowledge` items include verifiable `source` provenance metadata.
- [x] `safety` items include documented evaluation `rationale` metadata and clear disclaimers.
- [x] `DatasetCatalog` supports listing, inspection, SHA-256 checksums, category/difficulty/language/tag filtering, and comprehensive dataset statistics.

## 3. Advanced Metrics (`src/openbench/metrics/`)
- [x] `precision`, `recall`, and `f1` implemented with documented TP, FP, FN, and TN derivation for binary classification and element-set prediction tasks.
- [x] `rouge` (`rouge_1`, `rouge_l`) and `bleu` implemented with documented tokenization, LCS, clipped n-gram precision, and brevity penalty.
- [x] `semantic_similarity` implemented with pluggable `EmbeddingProvider` (`LocalNgramEmbeddingProvider`, `CallableEmbeddingProvider`, and optional offline-default `HuggingFaceEmbeddingProvider`) and explicit warnings that similarity is not factual correctness.
- [x] `structured_output` implemented with JSON parsing, required keys, field types, and optional `jsonschema` validation, reporting structural validity separately from value match.

## 4. Evaluator, Comparison, Reporting, History, Configs & Adapters
- [x] Category-wise and difficulty-wise breakdowns added to `Evaluator` and `ModelComparator`.
- [x] Optional `latency_ms` and `token_usage` recorded when provided and never fabricated when absent.
- [x] CSV export (`to_csv`, `save_csv`, `openbench export --format csv`) implemented for individual items, aggregate metrics, and model comparisons.
- [x] Local JSONL result history store (`ResultHistoryStore`) and CLI (`openbench history list` / `show`) implemented.
- [x] Reproducibility Card extended with SHA-256 dataset checksum, Python/OS info, model provider, inference config, and secret redaction.
- [x] YAML configuration profiles (`configs/reasoning.yaml`, `configs/coding.yaml`, `configs/instruction_following.yaml`) validated with strict unknown-field rejection.
- [x] Local and optional Hugging Face model adapters implemented in `src/openbench/adapters/`.
