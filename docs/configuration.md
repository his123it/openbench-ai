# YAML Evaluation Configuration Profiles (v0.2.0)

OpenBench AI `v0.2.0` supports declarative YAML evaluation profiles via `openbench.configuration.ProfileLoader` (requires optional `PyYAML`, included in `pip install -e ".[config]"` or `".[dev]"`).

---

## 1. Profile Schema

```yaml
name: reasoning-profile
description: Comprehensive evaluation profile for mathematical and logical reasoning datasets.

dataset:
  path: datasets/reasoning/sample.jsonl
  category: reasoning

metrics:
  - exact_match
  - accuracy
  - keyword_coverage
  - semantic_similarity

normalization:
  strip_whitespace: true
  case_sensitive: false
  collapse_whitespace: true
  strip_punctuation: false

options:
  difficulty:
    - easy
    - medium
    - hard
  strict_unknown_ids: false
  record_history: true

output_format:
  - json
  - markdown
  - csv
```

---

## 2. Validation Rules

- **Strict Field Checking**: Unknown top-level keys or unknown nested keys inside `dataset`, `normalization`, or `options` immediately raise `ConfigurationError`—invalid fields are never silently ignored.
- **Metric Validation**: Every metric listed in `metrics:` is verified against `METRIC_REGISTRY` before evaluation begins.

---

## 3. Running with CLI Overrides

```bash
# Run with profile defaults
openbench evaluate   --config configs/reasoning.yaml   --responses examples/sample_responses.json   --output results/

# Override dataset difficulty or metrics from the command line
openbench evaluate   --config configs/reasoning.yaml   --responses examples/sample_responses.json   --difficulty hard   --metrics exact_match accuracy   --output results/
```

Built-in profiles are provided in `configs/reasoning.yaml`, `configs/coding.yaml`, and `configs/instruction_following.yaml`.
