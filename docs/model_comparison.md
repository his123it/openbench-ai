# Paired model comparison (v0.4.0)

`openbench analyze` compares stored results. It never constructs or calls a model provider.

```bash
# Stored runs or repeat groups (experiments directory)
openbench analyze --runs grp_<a> run_<b> --baseline run_<b> --format markdown --output cmp.md

# Evaluation JSON files written by `openbench evaluate`
openbench analyze --results results/model_a.json results/model_b.json --equivalence-margin 0.05
```

Python:

```python
from openbench import ComparisonArm, ComparisonPolicy, ExperimentManager, PairedComparisonEngine

manager = ExperimentManager("experiments")
arms = [ComparisonArm.from_experiments(manager, "grp_..."), ComparisonArm.from_experiments(manager, "run_...")]
result = PairedComparisonEngine().compare(arms, ["exact_match", "f1"], ComparisonPolicy(analysis_seed=0))
```

## Alignment and missing data

- Items are aligned on **stable record IDs** after sorting. File order and response order never change a result. The same holds for arm order: without `--baseline`, pairs are formed in sorted arm-ID order.
- An arm's value for an item is the mean of its valid repetitions. A single run has one repetition.
- **Missing-data policy** (`--missing-policy`, recorded in the output):
  - `complete_case` (default): only items with a valid value in both arms are used. Missing outputs are **never counted as wrong answers**.
  - `missing_as_failure`: missing or invalid outputs score 0.0, as headline scores do.
- Exclusions are counted per arm and reason: missing, invalid, metric unavailable, not in all arms, or filtered out by `--ids-file`.

## Methods

| Metric kind | Interval | Test of no difference | Effect sizes |
| --- | --- | --- | --- |
| `binary`, one repetition per arm | Paired percentile bootstrap | Exact McNemar | Discordant counts, risk difference, wins/losses/ties |
| Continuous, or binary with repetitions | Paired percentile bootstrap | Paired sign-flip permutation test (exact enumeration up to 20 non-zero differences, otherwise Monte Carlo) | Mean difference, d_z (descriptive), probability of superiority |
| `unknown` (custom metrics) | Paired percentile bootstrap | None (the reason is recorded) | Mean difference |

Every test of no difference across all pairs and metrics forms one family. Raw and adjusted p-values are reported; `--correction` selects Holm (default), BH (controls the false discovery rate, flagged) or none (flagged). Category and difficulty breakdowns are descriptive unless `--stratum-tests` adds them to the family.

### What `descriptive_only` means on breakdowns

Each category and difficulty breakdown entry carries `"descriptive_only": true` unless `--stratum-tests` is used. That means no multiplicity-adjusted hypothesis test was run for that stratum — it does **not** mean there is no interval: the entry still includes a bootstrap confidence interval. Use `--stratum-tests` to add the breakdowns to the test family (with multiplicity adjustment) instead of leaving them descriptive-only.

When `--stratum-tests` is enabled, a stratum with only one paired item is still reported as `"descriptive_only": true`, because the `policy.stratum_tests and n >= 2` check in `PairedComparisonEngine._breakdown` requires at least two paired items before a category or difficulty group joins the test family.

The **failure-rate comparison** tests paired failure indicators (missing or invalid) per arm pair. It forms its own Holm family.

**Practical importance** (`--min-practical-difference`) classifies the CI against a threshold. It is descriptive and never an equivalence claim.

**Equivalence** (`--equivalence-margin`, which must be pre-specified) runs a TOST:

- **Binary metrics with one repetition** use the exact unconditional score test, and are decided by the Holm-adjusted p_TOST within a separate equivalence family.
- **Continuous metrics** are disabled for equivalence decisions in v0.4.0. Values are reported as descriptive only, with the reason `method_not_validated`.
- **Intervals** are labelled as not adjusted for multiple comparisons.

Wording rules: a non-significant difference is reported as "No statistically significant difference was detected (adjusted p = ...). The CI [L, U] is compatible with differences in this range." It is never reported as "equivalent" or "the same".

See [statistical_methodology.md](statistical_methodology.md) for assumptions and exact or approximate status.

## Compatibility checks

| Condition | Result |
| --- | --- |
| Analysed records differ in content (per-record hashes from v0.4.0 runs; otherwise the whole-dataset checksum) | **Hard incompatibility** |
| A metric's version differs, its options differ, or normalization differs | **Hard incompatibility** |
| A metric is missing from an arm, or a repeat group's members disagree internally | **Hard incompatibility** |
| Dataset version label differs but content is verified identical | Warning |
| Record ID sets differ | Warning (the intersection is used) |
| Generation configuration differs | Warning listing every differing key (values redacted) |
| Mock provider, incomplete runs or groups, different repetition counts | Warning |
| Provider differs, or same model identity | Info |

With any hard incompatibility, `analyze` exits 1 and lists every incompatibility. `--exploratory --exploratory-reason "<why>"` produces a **descriptive-only** report instead. It carries an `EXPLORATORY` banner, includes the stated reason, sets every inferential field to null with `exploratory_incompatible_inputs`, and exits 0. There is no flag that silently allows incompatible inputs.

Evaluation files do not carry per-record hashes. Two evaluation files of the same dataset filtered to different IDs therefore have different dataset checksums and are treated as hard-incompatible. Comparing v0.4.0 runs avoids this.

## Output

The JSON output contains:

- arms;
- per-pair compatibility;
- per pair and metric: n, exclusions, arm means and arm CIs, the difference and its CI, the test and its validity labels, raw and adjusted p, effect sizes, wins/losses/ties with IDs, disagreements sorted by size, coverage, breakdowns, and practical and equivalence results;
- failure-rate comparisons;
- the multiplicity families;
- limitations and provenance.

Markdown and CSV are rendered from the same payload. When an arm is a repeat group, its variability report is included.
