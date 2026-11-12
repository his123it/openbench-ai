<div align="center">

# OpenBench AI (v0.4.0)

**Is the difference between two LLMs real, or noise?**

Statistically sound LLM benchmarking in Python: repeated runs, paired model comparison with confidence intervals and exact tests, blinded human preference evaluation and dataset quality audits. No runtime dependencies.

[![Tests](https://github.com/lakshaychhaba0001-sys/openbench-ai/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/lakshaychhaba0001-sys/openbench-ai/actions/workflows/tests.yml)
[![Release](https://img.shields.io/github/v/release/lakshaychhaba0001-sys/openbench-ai?color=2ea44f)](https://github.com/lakshaychhaba0001-sys/openbench-ai/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/openbench-ai?color=2ea44f)](https://pypi.org/project/openbench-ai/)
[![Python 3.10–3.13](https://img.shields.io/badge/Python-3.10%E2%80%933.13-3776AB.svg)](pyproject.toml)
[![Runtime dependencies: none](https://img.shields.io/badge/runtime%20dependencies-none-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<img src="docs/assets/demo.gif" alt="Terminal demo: a read-only dataset audit, then the end-to-end example running repeated runs, a paired comparison with an adjusted p-value and confidence interval, variability, blinded preferences and a group audit" width="860">

</div>

## Why OpenBench AI?

Most model comparisons report two averages and pick the higher one. On a benchmark of 20 to 200 questions, a gap of a few points is often within run-to-run noise. Re-running a model can also return cached answers and look more stable than it really is. OpenBench AI answers the question behind the leaderboard:

| You want to know | OpenBench AI gives you |
| --- | --- |
| Is model B really better than model A? | `openbench analyze`: item-by-item pairing, bootstrap confidence intervals, an exact McNemar or sign-flip test, Holm or BH correction, and effect sizes. A non-significant result is never reported as "equivalent". |
| How much do scores move between runs? | `openbench run --repetitions N --seed S`: repeat groups with full lineage. Within-item noise, item spread and failures are reported separately. |
| Which answers do people prefer? | `openbench preferences`: blinded A/B task files, validated annotations, clustered win rates, position bias and inter-annotator agreement. |
| Can I trust the dataset and the runs? | `--quality-checks`: read-only audits for duplicates, near-duplicates, broken references, incomplete runs and integrity failures. |
| Can someone else check my numbers? | JSON, Markdown and CSV reports with provenance: input SHA-256s, seeds, parameters and environment. |

Try the whole workflow in under a minute, with no API keys and no network:

```bash
git clone https://github.com/lakshaychhaba0001-sys/openbench-ai.git && cd openbench-ai
python -m pip install -e .
python examples/v040_workflow.py demo   # writes comparison, variability, preference and audit reports to demo/
```

> **Note**: OpenBench AI is maintained by its project owner, and its initial development was AI-assisted. Contributions, bug reports, and new datasets are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md) and the [good first issues](https://github.com/lakshaychhaba0001-sys/openbench-ai/labels/good%20first%20issue).

---

## What's new in v0.4.0

- **Repeated-run benchmarking**: `openbench run --repetitions N --seed S` creates a *repeat group*. Each repetition is its own run with recorded lineage: group ID, repetition index, seeds and seed status, the exact record IDs, per-record content hashes, and cache provenance.
- **Interrupted groups resume** without re-requesting completed items (`openbench runs resume grp_...`).
- **Fix: repetitions no longer reuse cached responses.** In v0.3.1, repeated runs reused cached responses across repetitions, which understated variability.
- **Paired model comparison**: `openbench analyze` aligns stored runs, repeat groups or evaluation files on stable record IDs, with no provider calls. It reports the following, with Holm (or BH) correction, an explicit missing-data policy, coverage and failure-rate comparisons, and category and difficulty breakdowns:
  - percentile bootstrap confidence intervals;
  - an exact McNemar test (binary metrics) or a paired sign-flip permutation test;
  - effect sizes;
  - wins, losses and ties;
  - optional equivalence testing (TOST) with a pre-specified margin.
- **Compatibility checks**: incompatible inputs, such as different record content, metric versions or scoring options, are refused unless you request a clearly labelled exploratory comparison.
- **Repeated-run variability**: generation variability (within items), item-to-item variation and provider failures are reported separately.
- **Human preference evaluation**: `openbench preferences create|validate|analyze` is a local-first, blinded A/B workflow. The task file contains no model identities; the identity key is a separate file. Analysis covers validated annotation import, task-weighted win rates with task-clustered CIs, tie and neither outcomes, position bias, Cohen's and Fleiss' kappa, and optional Elo with stated limitations.
- **Quality audits**: `openbench validate-dataset PATH --quality-checks` and `openbench runs verify ID --quality-checks` run read-only checks for:
  - duplicate IDs, and exact and near-duplicate prompts (a lexical 5-gram heuristic);
  - malformed or placeholder references, and keyword expectations missing from references;
  - invalid metric configurations and inconsistent versions;
  - missing outputs, incomplete runs and groups, integrity failures, and insufficient data.

  Audits never modify data.
- **Reports**: every new analysis renders to JSON, Markdown and CSV, with provenance (software, environment, input SHA-256s, seeds, parameters). CSV cells are protected against spreadsheet formula injection.
- **Opt-in evaluation CIs**: `openbench evaluate --confidence-intervals`. Output is unchanged without the flag.

v0.3.1 commands, flags, public Python names, the 200 dataset records and existing scores are preserved, and v0.3.1 run directories can be read without modification. See the [migration guide](V0_4_0_MIGRATION_GUIDE.md) for the details and the few documented behaviour changes.

---

## Installation

```bash
# From PyPI: the library and the `openbench` CLI (no runtime dependencies)
python -m pip install openbench-ai
```

The built-in datasets, example profiles and `examples/` live in the repository rather than the package, so clone it to use them or to develop:

```bash
git clone https://github.com/lakshaychhaba0001-sys/openbench-ai.git
cd openbench-ai

# Core installation (zero external runtime dependencies)
python -m pip install -e .

# Development + YAML config + JSON Schema validation
python -m pip install -e ".[dev]"

# Optional integration groups
python -m pip install -e ".[huggingface]"  # transformers + torch for local HF models
python -m pip install -e ".[semantic]"     # sentence-transformers for neural embeddings
python -m pip install -e ".[all]"          # All optional runtime integrations

# Statistical reference cross-checks (tests only; never imported by the library)
python -m pip install -e ".[dev,stats-ref]"
```

The Ollama adapter needs no extra packages: it uses the Python standard library HTTP client. All v0.4.0 statistics, preference and audit features use the standard library only.

---

## Quick Start

### 1. Evaluate response files

```bash
openbench evaluate --dataset datasets/reasoning/sample.jsonl --responses examples/sample_responses.json \
  --metrics exact_match accuracy keyword_coverage --output results/

# With percentile bootstrap confidence intervals over items (v0.4.0, opt-in)
openbench evaluate --dataset datasets/reasoning/sample.jsonl --responses examples/sample_responses.json \
  --confidence-intervals --seed 0 --output results/ci/
```

### 2. Run models, with repetitions

```bash
openbench providers list
openbench run --dataset datasets/reasoning/sample.jsonl --provider mock --model model-a --repetitions 3 --seed 100
openbench run --dataset datasets/reasoning/sample.jsonl --provider mock --model model-b

openbench runs list
openbench runs show grp_<id>           # a repeat group
openbench runs verify grp_<id> --quality-checks
openbench runs resume grp_<id>         # finish an interrupted group
```

### 3. Compare stored results (no provider calls)

```bash
openbench analyze --runs grp_<id> run_<id> --baseline run_<id> --format markdown --output results/comparison.md
openbench analyze --results results/model_a.json results/model_b.json --equivalence-margin 0.05 --output results/cmp.json
openbench analyze --runs grp_<id> --output results/variability.json   # one repeat group: variability report
```

### 4. Collect blinded human preferences

```bash
openbench preferences create --runs run_<a> run_<b> --seed 7 --output tasks/ --key-output private/tasks.key.json
# annotators fill annotations.jsonl from tasks/tasks.jsonl
openbench preferences validate --key private/tasks.key.json --annotations annotations.jsonl
openbench preferences analyze --key private/tasks.key.json --annotations annotations.jsonl --format markdown --output results/preferences.md
```

### 5. Audit quality

```bash
openbench validate-dataset datasets/ --quality-checks --format markdown --output results/dataset_audit.md
```

A complete runnable walkthrough using the mock provider is in [`examples/v040_workflow.py`](examples/v040_workflow.py).

---

## CLI Commands

| Command | Purpose |
| :--- | :--- |
| `validate-dataset`, `list-datasets`, `inspect-dataset`, `dataset-stats` | Dataset validation and exploration (`--quality-checks` adds the audit) |
| `evaluate`, `compare`, `export`, `history list/show` | Evaluate response files, descriptive comparison, export, history |
| `run`, `runs list/show/verify/resume/compare` | Provider execution, repeat groups, run management and integrity |
| `providers list/check`, `cache stats/clear` | Providers and the response cache |
| `analyze` | v0.4.0 paired comparison or repeated-run variability of stored results |
| `preferences create/validate/analyze` | v0.4.0 blinded pairwise human preference evaluation |

See the [CLI reference](docs/cli_reference.md) for every flag.

---

## Built-in Dataset Catalog (200 Examples)

| Category | Path | Examples | Highlights |
| :--- | :--- | ---: | :--- |
| `reasoning` | `datasets/reasoning/sample.jsonl` | 20 | Deductive logic, scheduling, combinatorics, rate problems |
| `mathematics` | `datasets/mathematics/sample.jsonl` | 20 | Calculus, linear algebra, number theory, geometry |
| `coding` | `datasets/coding/sample.jsonl` | 20 | Python stdlib, algorithms, Big-O, SQL, HTTP, Git |
| `debugging` | `datasets/debugging/sample.jsonl` | 20 | Root-cause analysis across Python, SQL, C, JS, and Git |
| `instruction_following` | `datasets/instruction_following/sample.jsonl` | 20 | Strict formatting, delimiters, casing, sorting |
| `factual_knowledge` | `datasets/factual_knowledge/sample.jsonl` | 20 | Verifiable science & CS facts with `source` metadata |
| `safety` | `datasets/safety/sample.jsonl` | 20 | Harm refusal, secure coding, medical boundaries with `rationale` |
| `multilingual` | `datasets/multilingual/sample.jsonl` | 20 | 11 languages (`es`, `fr`, `de`, `hi`, `pt`, `it`, `ja`, `zh`, `ru`, `nl`, `la`) |
| `summarization` | `datasets/summarization/sample.jsonl` | 20 | Single-sentence technical and scientific summarization |
| `structured_output` | `datasets/structured_output/sample.jsonl` | 20 | JSON generation with `required_keys`, `field_types`, `expected_schema` |

The built-in quality audit reports 0 errors, 6 warnings and 8 informational findings. These are documented historical keyword exceptions, one ambiguous item and accepted answer variants; see [quality audits](docs/quality_audits.md). With 20 items per category, statistical comparisons on one category have low power, and the reports say so.

---

## Metrics Reference

| Metric Identifier | Version | Kind | Description | Key Methodological Limitation |
| :--- | :--- | :--- | :--- | :--- |
| `exact_match` | `0.1.0` | binary | Normalized string equality (`1.0` or `0.0`). | Penalizes conversational framing. |
| `accuracy` | `0.1.0` | binary | Deterministic correctness via exact or word-bounded answer containment. | Does not parse negation or complex paraphrases. |
| `keyword_coverage` | `0.1.0` | continuous | Proportion of expected keywords present in the response. | **Not equivalent to semantic correctness.** |
| `precision` | `0.2.0` | continuous | $\text{TP} / (\text{TP} + \text{FP})$ for binary decisions or element sets. | Applies to discrete decisions/sets, not free-form prose truth. |
| `recall` | `0.2.0` | continuous | $\text{TP} / (\text{TP} + \text{FN})$ for binary decisions or element sets. | Does not penalize extra false-positive elements alone. |
| `f1` | `0.2.0` | continuous | Harmonic mean of `precision` and `recall`. | Ignores word order outside discrete elements. |
| `rouge` / `rouge_1` / `rouge_l` | `0.2.0` | continuous | Unigram (`rouge_1`) and LCS (`rouge_l`) overlap F1. | Cannot detect factual hallucinations or antonym flips. |
| `bleu` | `0.2.0` | continuous | Smoothed clipped $n$-gram precision with brevity penalty. | Penalizes valid synonyms and paraphrases. |
| `semantic_similarity` | `0.2.0` | continuous | Embedding cosine similarity via configurable `EmbeddingProvider`; with the default `local_ngram_hash` provider it is a legacy alias for `lexical_similarity`. | **Not factual correctness**; opposite statements can have high similarity. |
| `lexical_similarity` | `0.3.1` | continuous | Cosine similarity of hashed word and character 3-gram features. | Surface wording only; paraphrases score low and negations score high. |
| `structured_output` | `0.2.0` | continuous | Validates JSON syntax, required keys, field types, and JSON Schema. | Structural validity is separate from factual value accuracy. |

The **kind** (`score_kind`) selects the statistical procedure in `analyze`. Metric scores and versions are unchanged in v0.4.0.

---

## Statistical honesty

- A non-significant difference is never reported as "no difference" or "equivalent". Equivalence requires a pre-specified margin and a TOST, and continuous-metric equivalence is disabled in v0.4.0 (descriptive only).
- Exact binary equivalence p-values are computed in floating point with a documented, tested error allowance. This is an engineering bound, not a machine-checked proof.
- Confidence intervals describe uncertainty over items for fixed models, prompts and settings. They are labelled as not adjusted for multiple comparisons.
- A seed passed to a provider does not guarantee deterministic output.
- SHA-256 checksums show that files match recorded checksums. They do not prove that results are correct or authentic.

See the [statistical methodology](docs/statistical_methodology.md).

---

## Documentation Index

- **v0.4.0 guides**: [Repeated runs](docs/repeated_runs.md) | [Model comparison](docs/model_comparison.md) | [Human preferences](docs/human_preferences.md) | [Quality audits](docs/quality_audits.md) | [Reproducibility & reports](docs/reproducibility.md) | [Statistical methodology](docs/statistical_methodology.md) | [CLI reference](docs/cli_reference.md) | [Python API](docs/api_reference.md) | [Migration from v0.3.1](V0_4_0_MIGRATION_GUIDE.md)
- **Execution, providers & integrity**: [Model Execution Engine](docs/model_execution.md) | [Providers](docs/providers.md) | [Ollama](docs/ollama.md) | [Hugging Face (experimental)](docs/huggingface.md) | [Custom providers](docs/custom_provider.md) | [Experiment tracking](docs/experiment_tracking.md) | [Performance monitoring](docs/performance_monitoring.md) | [Reliability & caching](docs/reliability.md) | [Benchmark integrity](docs/benchmark_integrity.md) | [Security & privacy](docs/security_and_privacy.md)
- **Datasets, metrics & configuration**: [Installation](docs/installation.md) | [Usage](docs/usage.md) | [Architecture](docs/architecture.md) | [Dataset catalog](docs/dataset_catalog.md) | [Core metrics](docs/metrics.md) | [Advanced metrics](docs/advanced_metrics.md) | [YAML profiles](docs/configuration.md) | [Historical tracking](docs/historical_tracking.md) | [v0.3 compatibility](docs/v0_3_compatibility.md)
- **Project**: [Changelog](CHANGELOG.md) | [Contributing](CONTRIBUTING.md) | [Security policy](SECURITY.md) | [v0.3.1 compatibility report](V0_3_1_COMPATIBILITY_REPORT.md)

---

## License

Released under the [MIT License](LICENSE).

---

## How to cite

If you use OpenBench AI in your research, please cite it using the following BibTeX entry:

```bibtex
@software{chhaba2026openbench,
  author  = {Chhaba, Lakshay},
  title   = {OpenBench AI},
  version = {0.4.0},
  date    = {2026-11-10},
  url     = {https://github.com/lakshaychhaba0001-sys/openbench-ai},
  license = {MIT}
}
```
