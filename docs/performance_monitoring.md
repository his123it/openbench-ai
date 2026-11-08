# Performance Monitoring & Token Accounting (v0.3.0)

`PerformanceMonitor` (`openbench.performance`) captures execution timing, reliability counters, token usage, and optional cost estimates in `performance.json`.

---

## 1. Captured Timing & Reliability Metrics

- **Total Execution Duration**: Wall-clock duration (`total_duration_ms`).
- **Latency Distribution**: `min_ms`, `max_ms`, `average_ms`, `median_ms` (P50), `p95_ms`, and `p99_ms`, plus per-category latency breakdowns (`category_latency`).
- **Throughput**: Completed requests per second (`throughput_requests_per_sec`).
- **Reliability Counters**: `completed_requests`, `cached_requests`, `failed_requests`, `timeout_count`, and `retry_count`.

---

## 2. Strict Token Usage Provenance

OpenBench AI strictly distinguishes three token usage states in `token_usage_summary["source"]`:
1. **`"provider_reported"`**: Exact token counts returned directly by the model provider (e.g., Ollama's `prompt_eval_count` and `eval_count`).
2. **`"locally_estimated"`**: Heuristic word-based estimates (`~1.3 tokens/word`), computed **only** when `estimate_missing_tokens: true` is explicitly enabled in configuration, and accompanied by an explicit `estimation_disclaimer`.
3. **`"unavailable"`**: Reported as `null` when the provider does not return token counts and estimation is disabled. OpenBench AI never fabricates missing token numbers.

---

## 3. Optional User-Supplied Cost Estimation (`PricingConfig`)

Because commercial and cloud hosting rates change frequently, OpenBench AI **never hardcodes provider prices**. You can supply optional pricing rates in your YAML profile to estimate run cost:

```yaml
performance:
  estimate_missing_tokens: false
  pricing:
    prompt_cost_per_1k_tokens: 0.0015
    completion_cost_per_1k_tokens: 0.0020
    currency: USD
```
