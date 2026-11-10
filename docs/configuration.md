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

## Repetitions and seeds in profiles (v0.4.0)

- `options.repeat_count: N` with N ≥ 2 creates a **repeat group** of N runs. `openbench run --repetitions` overrides it. N = 1 is a single run.
- `generation.seed` is the **generation base seed**: repetition *i* uses `seed + i` (`openbench run --seed` overrides it). v0.3.1 used the same seed for every repetition.
- Each repetition has its own cache context, so repetitions never share cached responses. `execution.use_cache: false` and `--no-cache` disable the cache as before.
- Only adapters that place the seed in the provider request (currently Ollama) report `seed_status: passed_to_provider`. A seed does not guarantee deterministic output.

Example (`configs/repeated_seeded_mock.yaml`):

```yaml
name: repeated-seeded-mock
dataset:
  path: datasets/reasoning/sample.jsonl
provider:
  name: mock
  model: mock-repeated-seeded
generation:
  temperature: 0.7
  seed: 100
metrics: [exact_match, keyword_coverage]
options:
  repeat_count: 3
```

See [repeated_runs.md](repeated_runs.md).
