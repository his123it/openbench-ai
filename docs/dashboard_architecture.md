# Future Dashboard Architecture & Data Contracts (v0.2.0 Preparation)

OpenBench AI `v0.2.0` deliberately avoids bundling a heavyweight web framework inside the core Python package. Instead, all evaluation, comparison, catalog, and history outputs adhere to strict, versioned JSON data contracts so a future `v1.0.0` web dashboard (or static site generator) can consume them directly.

---

## 1. Core Data Contracts

1. **Single-Model Evaluation Contract (`EvaluationResult.to_dict()`)**:
   - Contains `evaluation_id`, `project_version`, `evaluation_timestamp`, `dataset` (with SHA-256 `checksum`), `model`, `aggregate_results`, `category_breakdown`, `difficulty_breakdown`, `missing_breakdown`, `invalid_response_breakdown`, `performance_summary`, `individual_results`, and `reproducibility_card`.
2. **Multi-Model Comparison Contract (`ComparisonResult.to_dict()`)**:
   - Contains `comparison_timestamp`, `dataset_name`, `metrics_compared`, `models`, `highest_by_metric`, `category_comparison`, `difficulty_comparison`, `score_differences`, and `relative_performance`.
3. **Longitudinal History Contract (`results/history.jsonl`)**:
   - Append-only JSONL stream of `HistoryEntry` objects indexed by `evaluation_id`.

---

## 2. Recommended Future API Boundaries (`v1.0.0`)

- **Read-Only Artifact Layer**: The future dashboard should treat `results/*.json` and `results/history.jsonl` (or an optional read-only SQLite replica) as its data source without coupling UI rendering to `openbench.evaluator`.
- **Security Considerations**:
  - `ReportGenerator.build_reproducibility_card()` automatically redacts any configuration key containing `"key"`, `"token"`, `"secret"`, `"password"`, `"credential"`, or `"auth"`.
  - When rendering `model_response` or `prompt` strings in a browser dashboard, all text must be HTML-escaped to prevent Cross-Site Scripting (XSS) from untrusted model outputs.
