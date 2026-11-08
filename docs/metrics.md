# Metrics Reference & Limitations

OpenBench AI `v0.1.0` provides three deterministic, reference-based metrics. Every metric inherits from `openbench.metrics.base.BaseMetric` and returns structured `MetricResult` and `AggregateMetricResult` objects.

---

## Text Normalization (`NormalizationConfig`)

Before comparing model outputs against reference answers, text is normalized using `NormalizationConfig`:

- `strip_whitespace` (`True` by default): Removes leading and trailing whitespace.
- `case_sensitive` (`False` by default): Converts text to lowercase unless set to `True`.
- `collapse_whitespace` (`True` by default): Replaces consecutive spaces, tabs, and newlines with a single space.
- `strip_punctuation` (`False` by default): Punctuation is **preserved by default** so significant symbols in code (`[1, 9]`, `__repr__`), fractions (`1/2`), or JSON are never silently stripped.

---

## 1. Exact Match (`exact_match`)

### Definition
Returns `1.0` (`passed=True`) if and only if:

```text
normalize_text(model_response) == normalize_text(expected_answer)
```

Otherwise returns `0.0` (`passed=False`).

### When to Use
- Short factual outputs, strict formatting tasks (`instruction_following`), or single-token answers where extra conversational text is undesirable.

### Limitations
- Any conversational wrapper (e.g., `"The answer is 180."` when `expected_answer` is `"180"`) scores `0.0`.

---

## 2. Accuracy (`accuracy`)

### Definition
Calculates the proportion of evaluation items answered correctly according to a configurable deterministic rule (`rule` parameter):

1. **`"exact_or_substring"` (default)**:
   - Marks an item correct (`1.0`) if `normalize_text(response) == normalize_text(expected_answer)`, OR
   - If `record.metadata.get("disallow_substring")` is not `True`, checks whether the normalized expected answer appears inside the normalized response respecting word boundaries (`\b`) on alphanumeric edges.
   - Word boundaries prevent false positive substring matches (for example, expected answer `"18"` will match `"The answer is 18."` but will **not** match `"The answer is 180."`).
2. **`"exact_only"`**:
   - Identical to `exact_match`.
3. **`"word_boundary"`**:
   - Checks word-bounded containment directly.

### Limitations
- Does not parse semantic negation (e.g., `"The answer is definitely not 180"` still contains the bounded token `"180"`).

---

## 3. Keyword Coverage (`keyword_coverage`)

### Definition
Evaluates the fraction of expected keywords present in the normalized model response:

- Expected keywords are read from `record.metadata["keywords"]` if provided, or fall back to `[record.expected_answer]`.
- For each keyword $k_i$, checks whether `normalize_text(k_i)` is a substring of `normalize_text(response)`.
- Item score:
  $$\text{score} = \frac{|\text{matched\_keywords}|}{|\text{expected\_keywords}|}$$
- Returns `matched_keywords`, `missing_keywords`, `total_keywords`, and `coverage_percentage` (`0.0` to `100.0`).

### Explicit Methodological Warning
> **Keyword coverage is NOT equivalent to semantic correctness.**
> A model response can include 100% of the target keywords while stating the exact opposite of the intended meaning or using the terms in an unsafe context. Always inspect `exact_match`, `accuracy`, and individual item outputs alongside `keyword_coverage`.
