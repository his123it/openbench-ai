# OpenBench AI v0.4.0

**Released 2026-11-10.**

OpenBench AI is a Python toolkit for reproducible language model benchmarks. It runs models against JSONL datasets, scores them with versioned metrics and records every run with its configuration, checksums and provenance.

v0.4.0 answers the question benchmarks are usually run for: **is the difference between two models real, and how much should you trust it?** It adds:
- repeated experiments;
- paired statistical comparison;
- blinded human preference evaluation;
- read-only quality audits.

The core installation still has no runtime dependencies. Every analysis runs locally on stored results and never calls a model provider.

## Highlights

### Compare models with appropriate statistics

A one-point gain on 20 questions is often noise. `openbench analyze` aligns two or more models item by item, using stable record IDs, and reports:

- the mean difference, with a percentile bootstrap confidence interval;
- an exact McNemar test for binary metrics, or a paired sign-flip permutation test otherwise;
- Holm-adjusted p-values (or Benjamini–Hochberg) within clearly defined families;
- effect sizes; wins, losses and ties; coverage and failure rates; and category and difficulty breakdowns;
- an optional equivalence test (TOST) against a margin you choose in advance, for binary metrics.

A non-significant result is never reported as "equivalent". Inputs that cannot be compared fairly are refused: different record content, metric versions, scoring options or normalisation. The exception is when you request an exploratory view, which is labelled as such in every output format.

### Measure run-to-run variability

`openbench run --repetitions N --seed S` creates a *repeat group*. Each repetition is a separate run that records:
- its group, index and seed (`S + i`);
- the exact record IDs and a content hash for each record;
- whether the seed reached the provider.

The analysis reports three things separately: generation noise within items, item-to-item spread, and provider failures. Interrupted groups resume without repeating completed work.

### Collect blinded human preferences

`openbench preferences create` writes blind A/B task files that contain no model names, and keeps the identity key in a separate file. `preferences validate` checks the imported annotations (JSONL or CSV) as untrusted input. `preferences analyze` reports:
- task-weighted win rates with intervals that respect clustering by task;
- tie and "neither" outcomes, and position bias;
- Cohen's and Fleiss' κ;
- an optional Elo rating, with its limitations stated.

### Audit quality before trusting a number

`validate-dataset --quality-checks` runs 14 dataset checks, and `runs verify --quality-checks` runs 12 run checks. They look for:
- duplicate IDs, and exact and near-duplicate prompts;
- empty, placeholder or inconsistent references, and invalid metric configurations;
- missing outputs, incomplete runs and groups, integrity failures, and sample sizes too small for conclusions.

Audits never modify files.

### Reproducible, checkable reports

Every new analysis renders to JSON, Markdown and CSV from one payload, so the numbers agree across formats. Each report includes a provenance block: the software and environment, the SHA-256 of every input, seeds, parameters, and the run lineage of each compared model. JSON output is strict, with no NaN or infinity. CSV cells are protected against spreadsheet formula injection.

## Important correctness fix

In v0.3.1, repeated runs reused cached responses across repetitions, so they looked more consistent than they really were. In v0.4.0, repetitions never share cache entries. Repeated runs therefore make more provider calls than before. Single runs keep the v0.3.1 cache key and behaviour.

## Quick tour

```bash
git clone --branch v0.4.0 https://github.com/lakshaychhaba0001-sys/openbench-ai.git
cd openbench-ai && python -m pip install -e .

# 1. Audit the data first (read-only)
openbench validate-dataset datasets/ --quality-checks

# 2. Run one model three times and another once (the mock provider needs no network)
openbench run --dataset datasets/reasoning/sample.jsonl --provider mock --model model-a --repetitions 3 --seed 7
openbench run --dataset datasets/reasoning/sample.jsonl --provider mock --model model-b --seed 7
openbench runs list
openbench runs show grp_<group-id>

# 3. Compare the repeat group with the single run, item by item
openbench analyze --runs grp_<group-id> run_<model-b-run> --baseline run_<model-b-run> \
  --format markdown --output comparison.md

# 4. Variability across the repetitions of one group
openbench analyze --runs grp_<group-id> --format markdown --output variability.md

# 5. Blinded preferences: give prefs/tasks.jsonl to annotators, keep the key private
openbench preferences create --runs run_<model-a-run> run_<model-b-run> --output prefs --key-output private/key.json
openbench preferences validate --key private/key.json --annotations annotations.jsonl
openbench preferences analyze --key private/key.json --annotations annotations.jsonl --format markdown --output preferences.md

# 6. Opt-in confidence intervals for a single evaluation
openbench evaluate --dataset datasets/reasoning/sample.jsonl --responses examples/sample_responses.json \
  --metrics exact_match keyword_coverage --confidence-intervals --output results/
```

`examples/v040_workflow.py` runs the whole workflow end to end with the mock provider.

## Compatibility and migration

- **Preserved:**
  - the 19 v0.3.1 commands with their flags, defaults and choices;
  - the 70 public Python names;
  - the 200 dataset records;
  - metric names, versions and scores;
  - the `evaluate`, `compare` and `export` output formats.

  These are checked by tests against frozen v0.3.1 inventories and golden outputs.
- **Older artifacts:** v0.3.x run directories can be listed, shown, verified, compared and analysed without any file being modified.
- **Documented behaviour changes:**
  - repetition seeds are `seed + i`;
  - `runs resume` restores ID filters, normalisation and metric options;
  - CSV formula-injection protection;
  - credential-scrubbed CLI errors;
  - run provenance schema `0.4.0` (the v0.3.1 `runs verify` reports this schema as unsupported).
- **Additions to the CLI and API:** two new commands, `analyze` and `preferences create|validate|analyze`, plus new optional flags on existing commands. The public API grows from 70 to 86 names.

See the [migration guide](https://github.com/lakshaychhaba0001-sys/openbench-ai/blob/v0.4.0/V0_4_0_MIGRATION_GUIDE.md) for the full list.

## Validation

Before release, the full validation ran twice on Linux, each time from a fresh clone, with CPython 3.10, 3.11, 3.12 and 3.13:

- **Full test suite:** 500 tests on each Python version, 0 failures.
- **Statistical tests:** 381 tests, cross-checked against NumPy, SciPy and statsmodels, 0 failures.
- **Calibration:** the continuous bootstrap-t calibration re-run passed.
- **v0.3.1 tests:** the 92 tests from v0.3.1, unmodified apart from two version strings, passed.
- **v0.3.1 output:** `evaluate` on all 200 records with 11 metrics gave the same JSON, Markdown and CSV as v0.3.1, apart from version and timestamp fields.
- **Built-in dataset audit:** 0 errors, 6 warnings and 8 informational notes, all documented.
- **Determinism:** `analyze` gave bit-identical output on all four Python versions.

The release candidate also had an independent review of the code, the statistics, compatibility and the documented design decisions. It found no defects.

The repository's GitHub Actions workflow runs on every pull request and every push to `main`. Its jobs are: lint, a core-only install, the Python 3.10–3.13 test matrix, a repeated full run, statistical reference tests and a wheel build. The results for the release commit are on the repository's Actions page.

## Limitations

- **No real model providers in testing.** All end-to-end runs used the built-in mock provider. The Ollama and Hugging Face adapters were tested only with fake transports.
- **Platforms and extras.** Tested on Linux only. The `huggingface` and `semantic` extras were not install-tested for this release.
- **Continuous-metric equivalence** is reported descriptively only. Its calibration did not meet the pre-specified size bound, so no equivalence decision is made.
- **Exact binary equivalence** uses a documented floating-point error allowance. It is an engineering bound, not a proof. It is limited to 500 pairs.
- **Near-duplicate detection** is lexical, not semantic.
- **Seeds** are recorded and forwarded where an adapter supports them (Ollama), but they do not guarantee deterministic provider output.
- **Statistical power.** The built-in datasets have 20 items per category, so single-category comparisons have low power.
- **Cache concurrency.** Several processes writing to one cache directory at the same time are not supported.

## Documentation

- [README](https://github.com/lakshaychhaba0001-sys/openbench-ai/blob/v0.4.0/README.md)
- [Migration guide](https://github.com/lakshaychhaba0001-sys/openbench-ai/blob/v0.4.0/V0_4_0_MIGRATION_GUIDE.md)
- [Changelog](https://github.com/lakshaychhaba0001-sys/openbench-ai/blob/v0.4.0/CHANGELOG.md)
- [Repeated runs](https://github.com/lakshaychhaba0001-sys/openbench-ai/blob/v0.4.0/docs/repeated_runs.md)
- [Model comparison](https://github.com/lakshaychhaba0001-sys/openbench-ai/blob/v0.4.0/docs/model_comparison.md)
- [Human preferences](https://github.com/lakshaychhaba0001-sys/openbench-ai/blob/v0.4.0/docs/human_preferences.md)
- [Quality audits](https://github.com/lakshaychhaba0001-sys/openbench-ai/blob/v0.4.0/docs/quality_audits.md)
- [Reproducibility](https://github.com/lakshaychhaba0001-sys/openbench-ai/blob/v0.4.0/docs/reproducibility.md)
- [Statistical methodology](https://github.com/lakshaychhaba0001-sys/openbench-ai/blob/v0.4.0/docs/statistical_methodology.md)
- [CLI reference](https://github.com/lakshaychhaba0001-sys/openbench-ai/blob/v0.4.0/docs/cli_reference.md)
- [Python API reference](https://github.com/lakshaychhaba0001-sys/openbench-ai/blob/v0.4.0/docs/api_reference.md)

OpenBench AI is maintained by its project owner, and its initial development was AI-assisted.
