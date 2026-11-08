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
