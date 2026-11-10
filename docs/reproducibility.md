# Reproducibility and reporting (v0.4.0)

## What every v0.4.0 report records

Every analysis, preference and audit report carries a `provenance` block with:

- the OpenBench version, Python version, platform and generation time;
- the analysis seed and a SHA-256 of the analysis parameters;
- the input files with their SHA-256 values, plus, for comparison arms, the dataset name, version and checksum, the evaluation-configuration hash, the metric versions, and run lineage (group, index, seeds, seed status, selected-ID hash, status, timestamps);
- `provider_calls: 0` (reports never call providers);
- the integrity disclaimer below.

Statistical outputs record:

- the method name, test statistic, confidence level, resamples and permutations;
- the missing-data policy, exclusions and sample counts;
- raw and adjusted p-values with the correction and family;
- effect sizes, assumptions and validity labels, practical thresholds, small-sample warnings and the stated limitations.

An unavailable statistic is `null` with a reason, never 0.

## Seeds

| Seed | Set by | Default | Scope |
| --- | --- | --- | --- |
| `analysis_seed` | `analyze --seed`, `evaluate --confidence-intervals --seed`, `preferences analyze --seed` | 0 | Bootstrap, permutation and equivalence resampling |
| `generation_base_seed` | `run --seed`, or profile `generation.seed` | none | A run or group |
| `repetition_seed` | Derived as base + index | none | Each repetition |
| `preference_order_seed` | `preferences create --seed` | 0 | Display order of preference tasks |

- Each random procedure uses its own stream, derived from SHA-256 of the analysis seed, purpose, metric and arms. Results therefore do not depend on the order in which analyses run.
- Items are sorted by ID before resampling, and sums use `math.fsum`.
- Identical inputs, parameters and seed give identical results, including across Python versions: sums use `math.fsum` and standard deviations are computed exactly and rounded once, because `statistics.stdev` and the built-in `sum()` of floats changed in Python 3.12. This was checked on Python 3.10, 3.11, 3.12 and 3.13 on Linux.
- Outputs computed by v0.3.1 code paths (`evaluate` metric scores and aggregates, `compare`, history analytics) are unchanged from v0.3.1 and may still differ in the last bit between Python 3.11 and 3.12 for some inputs.

## Formats

| Report | JSON | Markdown | CSV |
| --- | --- | --- | --- |
| Paired comparison (`analyze`) | full payload | summary tables and per-comparison detail | one row per comparison |
| Repeated-run variability | full payload | per-metric summary | one row per item and metric |
| Human preferences | full payload | win rates, agreement, annotators, Elo | one row per pair and breakdown |
| Quality audit | full report | findings table | one row per finding |
| Evaluation (`evaluate`) | unchanged; `statistics` block with `--confidence-intervals` | unchanged; CI section with the flag | unchanged |

- **JSON is strict**: no NaN or infinity.
- **Markdown** is rendered from the same payload, so its numbers agree with the JSON.
- **Exploratory comparisons** carry a banner in every format.
- **CSV cells** that start with `=`, `+`, `-`, `@`, a tab or a carriage return are prefixed with `'` to stop spreadsheet formula injection. Plain numbers are left as they are. This also applies to the existing evaluation and comparison CSV; for the built-in data the output is unchanged.

## Integrity

`runs verify` checks the artifact checksums recorded in `reproducibility.json` and, for v0.4.0 runs, that lineage agrees between `config.json`, `reproducibility.json` and `model_metadata.json`. `runs verify grp_<id>` verifies every member and the group's lineage consistency.

> SHA-256 checksums show that files match recorded checksums. They do not prove that results are factually correct, that a model produced them, or who produced them.

## Reading older artifacts

- **v0.3.x run directories** (schema 0.3.0) are read without modification. They appear as single runs with `seed_status: unknown`.
- **Comparisons** that involve them fall back to the dataset checksum for content compatibility.
- **Never rewritten:** v0.4.0 never migrates, re-hashes or rewrites old runs.
- **Analyses** are never written into run directories. The only file written into a run directory after it is created is the legacy `repeated_runs_analysis.json`, at group creation.
