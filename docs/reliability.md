# Reliable Execution, Retries, Recovery & Caching (v0.3.0)

OpenBench AI `v0.3.0` is engineered for resilient execution across flaky local endpoints or long benchmark datasets.

---

## 1. Retry Policies & Exponential Backoff (`RetryPolicy`)

When a request fails with a transient error (`connection_error`, `rate_limit`, or `timeout` when `retry_on_timeouts=True`), `ExecutionEngine` retries up to `max_attempts` using exponential backoff:

$$\text{delay}(k) = \min\left(\text{max\_backoff}, \; \text{initial\_backoff} \times \text{multiplier}^{k - 1}\right)$$

Validation and configuration errors (`validation_error`) fail immediately without wasting retries.

---

## 2. Failure Classification

Every failed request is classified into one of five diagnostic categories in `performance.json` and `logs.jsonl`:
- `timeout`
- `connection_error`
- `rate_limit`
- `validation_error`
- `provider_error`

---

## 3. Interrupted Run Recovery (`openbench runs resume`)

If an execution run is interrupted or experiences partial provider failures (`status = "partial"`):
- Completed responses are saved in `experiments/run_<id>/responses.json`.
- Running `openbench runs resume run_<id>` reloads all previously completed items and dispatches only the remaining failed/missing dataset records.

---

## 4. Configuration-Aware Deterministic Response Cache (`ResponseCache`)

- Cached entries are keyed by the SHA-256 digest of `(provider_name, model_name, model_version, dataset_name, dataset_version, record_id, prompt_sha256, generation_config_sha256)`.
- Responses are **never** reused across different models, model revisions, prompts, dataset versions, or generation parameters (such as a changed `temperature` or `seed`).
- Disable caching at any time with `--no-cache` or `use_cache: false`, or clear the cache with `openbench cache clear`.
- **Important**: In-flight duplicate protection prevents concurrent workers within the same run from dispatching duplicate requests for the same record ID, but OpenBench AI does not claim distributed exactly-once execution when an external HTTP provider times out mid-generation.
