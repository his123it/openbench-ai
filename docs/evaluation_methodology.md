# Evaluation Methodology & Reproducibility

OpenBench AI is designed around transparent, reproducible evaluation practices suitable for academic coursework, open-source benchmarking, and regression testing.

---

## 1. Handling Missing and Unknown Responses

A common pitfall in ad-hoc LLM evaluation scripts is computing average accuracy only over the prompts a model successfully answered. That allows a model (or broken inference pipeline) to skip difficult questions and report an artificially inflated score.

In OpenBench AI:
- Every dataset item missing from a model's response file is recorded in `missing_responses` and assigned an item status of `"missing"`.
- Aggregate metric scores (`AggregateMetricResult.score`) divide the sum of item scores by **`total_dataset_items`** (treating missing responses as `0.0`).
- For diagnostic transparency, the mean across answered items only is also recorded in `details["evaluated_only_score"]`.
- Extra response IDs that do not exist in the dataset are recorded in `unknown_response_ids` and flagged with a warning (or rejected with an error if `strict_unknown_ids=True`).

---

## 2. Why OpenBench AI Avoids a Single Composite "Overall Rank"

Different metrics measure fundamentally different properties:
- `exact_match` tests strict output formatting compliance.
- `accuracy` tests deterministic answer presence.
- `keyword_coverage` tests lexical recall of key concepts.

Averaging these three numbers into a single "OpenBench Score" would be methodologically unsound—a verbose model and a terse model have different trade-offs depending on the downstream application. Therefore, `ModelComparator` reports **per-metric leaders** and preserves the full metric breakdown.

---

## 3. The Benchmark Reproducibility Card

Every evaluation report (`.json` and `.md`) includes a **Benchmark Reproducibility Card** containing:
- Model name and version
- Dataset name, version, and source path
- Metric names and versions
- Evaluation UTC timestamp
- OpenBench AI toolkit version (`v0.1.0`)
- Active normalization configuration
- Total test cases, evaluated test cases, and missing response count
- Aggregate results per metric
- Explicit known limitations

---

## 4. Safety Evaluation Disclaimer

The `datasets/safety/sample.jsonl` benchmark contains 12 prompts covering phishing refusal, credential hygiene, SQL injection refusal, medical emergency escalation, academic integrity, password hashing, DDoS refusal, PHI de-identification, chemical hazard warnings, licensing compliance, indirect prompt injection, and safety evaluation limits.

In `v0.1.0`, these items are evaluated using reference-based string and keyword checks. **This does not constitute comprehensive safety certification** and should never be used as the sole gate for production deployment.
