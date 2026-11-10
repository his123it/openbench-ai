# OpenBench AI v0.4.0: release notes (draft)

> **Draft for review.** This is a release candidate. It has not been published, tagged or released.

OpenBench AI v0.4.0 adds statistical benchmarking, repeated experiments, blinded human preference evaluation and dataset and run quality audits. The core installation still has no runtime dependencies.

## Highlights

- **Compare models with appropriate statistics.** `openbench analyze` aligns stored results on record IDs and reports:
  - bootstrap confidence intervals;
  - exact McNemar or paired sign-flip tests;
  - multiplicity-adjusted p-values;
  - effect sizes, and wins, losses and ties;
  - coverage and failure rates;
  - optional pre-specified equivalence tests.

  Incompatible inputs are refused unless you ask for a clearly labelled exploratory view.
- **Repeat experiments and see their variability.** `openbench run --repetitions N --seed S` creates a repeat group with full lineage. Interrupted groups resume without repeating work. Variability within items is reported separately from variation between items and from provider failures.
- **Collect blinded human preferences locally.** Annotators see blind task files with no model identities; the key is kept separately. Imported annotations are validated. Analysis reports task-weighted win rates with clustered intervals, agreement (Cohen's and Fleiss' κ), position bias and optional Elo.
- **Audit quality before you trust a number.** Read-only dataset and run checks cover duplicates, lexical near-duplicates, malformed or placeholder references, keyword expectations, version consistency, missing outputs, incomplete runs and integrity failures.
- **Reproducible reports.** JSON, Markdown and CSV reports carry provenance: software, environment, input checksums, seeds and parameters.

## Important fix

Repeated runs in v0.3.1 reused cached responses across repetitions, so they looked more consistent than they were. In v0.4.0 repetitions never share cache entries. Repeated runs therefore make more provider calls than before; single runs are unchanged.

## Compatibility

- **Unchanged:** all v0.3.1 commands and flags, public Python names, the 200 dataset records and scores.
- **v0.3.1 run directories** remain readable without modification.
- **Documented behaviour changes:**
  - per-repetition seeds;
  - restored settings on resume;
  - CSV formula-injection protection;
  - scrubbed error messages;
  - run schema 0.4.0.

See the [migration guide](V0_4_0_MIGRATION_GUIDE.md) and the [changelog](CHANGELOG.md).

## Known limitations

- **Continuous-metric equivalence** is disabled for decisions in v0.4.0 (descriptive values only).
- **Exact binary equivalence** uses a documented floating-point error allowance. It is an engineering bound, not a machine-checked proof.
- **Near-duplicate detection** is lexical, not semantic.
- **Seeds** passed to providers do not guarantee deterministic output.
- **Small samples:** the built-in datasets have 20 items per category, so statistical comparisons on one category have low power.
- **Testing scope:** tested on Linux with Python 3.10–3.13. GitHub Actions CI has not yet run for this release candidate.
