# Architecture Overview

OpenBench AI follows a modular, layered architecture under `src/openbench/` with strict separation of concerns between data loading, metric computation, orchestration, comparison, reporting, and CLI presentation.

## Component Diagram

```text
+-------------------------------------------------------------------+
|                        CLI Layer (cli.py)                         |
|   Parses arguments, handles exit codes, delegates to core APIs    |
+-----------------+-----------------------+-------------------------+
                  |                       |
                  v                       v
+-------------------------+   +-------------------------------------+
|  DatasetLoader          |   |  Evaluator (evaluator.py)           |
|  (dataset.py)           |-->|  Matches IDs, tracks missing items, |
|  Validates JSONL schema |   |  applies BaseMetric instances       |
+-------------------------+   +------------------+------------------+
                                                 |
                  +------------------------------+------------------+
                  |                                                 |
                  v                                                 v
+-------------------------------------+   +------------------------------------+
|  Metrics Package (metrics/)         |   |  Comparison & Reporting            |
|  - BaseMetric (base.py)             |   |  - ModelComparator (comparison.py) |
|  - Normalization (normalization.py) |   |  - ReportGenerator (reporting.py)  |
|  - ExactMatch (exact_match.py)      |   |  - Reproducibility Cards           |
|  - Accuracy (accuracy.py)           |   +------------------------------------+
|  - KeywordCoverage (keyword_match.py)|
+-------------------------------------+
```

## Module Responsibilities

1. **`openbench.exceptions`**: Defines a clean hierarchy rooted at `OpenBenchError`, including `DatasetValidationError` (with `file_path`, `line_number`, and `record_id` attributes), `EmptyDatasetError`, `ResponseValidationError`, `MetricError`, and `ConfigurationError`.
2. **`openbench.dataset`**: Implements `DatasetRecord`, `Dataset`, and `DatasetLoader`. Guarantees that no invalid or duplicate record is ever silently ignored.
3. **`openbench.metrics`**: Provides `BaseMetric`, `MetricResult`, `AggregateMetricResult`, `NormalizationConfig`, and concrete metric implementations registered in `METRIC_REGISTRY`.
4. **`openbench.evaluator`**: Implements `ResponseLoader` and `Evaluator`. Completely decoupled from `argparse`, `sys.exit`, or terminal printing so it can be embedded in notebooks, CI scripts, or future web services.
5. **`openbench.comparison`**: Implements `ModelComparator` and `ComparisonResult`, comparing models metric-by-metric and identifying ties and leaders.
6. **`openbench.reporting`**: Serializes `EvaluationResult` and `ComparisonResult` into JSON and Markdown reports, including the **Benchmark Reproducibility Card**.
7. **`openbench.cli`**: Thin command-line adapter exposing `validate-dataset`, `evaluate`, `compare`, and `--version`.

## v0.4.0 modules

| Module | Responsibility |
| --- | --- |
| `openbench/stats/` | Standard-library statistical primitives (bootstrap, McNemar, sign-flip, Holm/BH, effects, TOST, RNG streams) and `variability.py` (repeated-run decomposition). |
| `openbench/lineage.py` | `RunLineage`, ID validation, seed status, record content hashes. |
| `openbench/itemscores.py` | Per-item, per-repetition score extraction keyed by record ID (shared by variability and paired analysis). |
| `openbench/paired.py` | Comparison arms, policy, compatibility checks, `PairedComparisonEngine`. Never calls providers. |
| `openbench/preferences.py` | Blind task builder and key, annotation import and validation, preference analysis, κ, Elo. |
| `openbench/quality.py` + `quality_notes.json` | Read-only dataset (DQ) and experiment (EQ) audits; content-hash-keyed review notes. |
| `openbench/evalstats.py` | Opt-in evaluation CIs (`evaluate --confidence-intervals`). |
| `openbench/reporting_v040.py`, `openbench/provenance.py` | JSON, Markdown and CSV renderers, and provenance blocks. |

Changes to existing modules:

- `experiments.py`: repeat groups, lineage, group index, resume and verification.
- `cache.py` and `execution.py`: repetition-aware cache keys.
- `integrity.py`: schema 0.4.0 lineage checks.
- `adapters`: the seed capability attribute.
- `security.py`: `csv_safe`.
- `cli.py`: thin handlers for `analyze`, `preferences` and the new flags.

