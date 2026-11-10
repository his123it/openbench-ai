# Migrating from OpenBench AI v0.3.1 to v0.4.0

v0.4.0 is additive. The following carry over from v0.3.1:

- the 19 v0.3.1 CLI commands, with all their flags, defaults and choices;
- the 70 public Python names, whose signatures were only extended with optional parameters;
- the 200 dataset records;
- metric names, versions and scores;
- the `evaluate`, `compare` and `export` output formats.

These were checked by tests against frozen v0.3.1 inventories and outputs. Untested edge cases may still exist, so please report anything that behaves differently. Below are the new structures, the few intentional behaviour changes, and what is not supported.

## Install

```bash
python -m pip install -e .                 # unchanged: no runtime dependencies
python -m pip install -e ".[dev,stats-ref]" # optional: NumPy/SciPy/statsmodels for the reference tests only
```

## Behaviour changes

| Area | v0.3.1 | v0.4.0 | Why |
| --- | --- | --- | --- |
| Repeated runs and the cache | Repetitions reused cached responses, so repeated runs returned identical answers. | Repetitions in a group never share cache entries, and independent groups never share entries. | Correctness fix: v0.3.1 understated variability. Repeated runs now make more provider calls. Single runs keep the v0.3.1 key and sharing. `--no-cache` is unchanged. |
| Repeated runs and seeds | Every repetition used the same `generation.seed`. | Repetition *i* uses `seed + i`, recorded in lineage. | Independent repetitions (approved decision D9). |
| `runs resume` | Rebuilt only metric names and category, difficulty, tag and language filters. ID filters, normalization, metric options and the seed were dropped, so a resumed `--ids-file` run could evaluate more items. | Restores all of them, using the exact record IDs from lineage for v0.4.0 runs. A v0.3.x run whose `--ids-file` no longer exists is refused instead of widened. v0.4.0 runs honour their recorded cache setting. | Correctness fix. |
| New run directories | `reproducibility.json` `schema_version` `0.3.0`. | `0.4.0` for all new runs, with a `lineage` block in `config.json` and `reproducibility.json` and repeat-group fields in `model_metadata.json`. | Lineage. v0.3.1 `runs verify` will report "Unsupported provenance schema_version" for v0.4.0 runs. |
| Repeat-group index | None. | `experiments/grp_<id>.json`, a file at the root, not a directory. | Rebuildable group summary. |
| CSV exports | Cells written raw. | Text cells starting with `=`, `+`, `-`, `@`, tab or CR are prefixed with `'`. Numbers are unchanged. | Security: spreadsheet formula injection from model output. Output is unchanged for the built-in data. |
| CLI error messages | Exception text printed as is. | Scrubbed of embedded credentials. | Security. |
| Cache file errors | A malformed entry could raise a raw `KeyError`, and a failed load left a half-loaded cache. | A clear `OpenBenchError` naming the line. Loading is atomic, and `cache clear` works on corrupted files. | Reliability fix. |
| `binary_paired_counts`, `studentized_mean`, `score_statistic` and other `openbench.stats` helpers (unreleased Stage 1 API) | n/a | Direct calls validate their inputs (booleans and non-finite values are rejected). | Hardening. |
| `__version__`, `--version`, `project_version` | `0.3.1` | `0.4.0` | Release. |

## New on-disk structures

- **Run lineage**: see [docs/repeated_runs.md](docs/repeated_runs.md). Readers treat runs without lineage as single runs with `seed_status: "unknown"`.
- **Group index**: `experiments/grp_<id>.json`, rebuildable from lineage.
- **Cache entries** for group repetitions add `repeat_group_id` and `repetition_index`. v0.3.1 readers ignore unknown keys, and entries without them are byte-identical to v0.3.1.
- **Evaluation JSON** has an optional `statistics` block, present only with `--confidence-intervals`.
- **New report types**, each with `schema_version` `0.4.0` and a `provenance` block: paired comparison, variability, preference tasks (`tasks.jsonl`), preference key (`tasks.key.json`), preference analysis and audit.

## New commands and flags

New commands:

- `analyze`
- `preferences create`, `preferences validate`, `preferences analyze`

New flags on existing commands:

- `run`: `--repetitions`, `--seed`, `--experiment-id`, `--parent-experiment-id`
- `runs list`: `--group`
- `runs show`, `runs verify` and `runs resume` accept a `grp_...` ID
- `runs verify`: `--quality-checks`, `--audit-output`
- `validate-dataset`: `--quality-checks`, `--output`, `--format`, `--near-duplicate-threshold`, `--metrics`
- `evaluate`: `--confidence-intervals`, `--confidence-level`, `--resamples`, `--seed`

See [docs/cli_reference.md](docs/cli_reference.md).

## New Python API

New names appended to `openbench.__all__`:

- comparison: `ComparisonArm`, `ComparisonPolicy`, `CompatibilityReport`, `PairedComparisonEngine`, `PairedComparisonResult`, `IncompatibleComparisonError`;
- repeated runs: `RepeatGroupResult`, `RunLineage`;
- preferences: `AnnotationImporter`, `PreferenceAnalyzer`, `PreferenceAnnotation`, `PreferenceTaskBuilder`;
- audits: `AuditFinding`, `AuditReport`, `DatasetAuditor`, `ExperimentAuditor`.

`openbench.stats` gains `decompose_variability`.

New optional parameters on existing APIs (all keyword-safe, with defaults):

- `ExperimentManager.execute_and_record(..., lineage=None)`
- `ExperimentManager.list_runs(..., repeat_group_id=None)`
- `ExecutionEngine(..., repetition_context=None)`
- `ResponseCache.compute_cache_key/get/put(..., repetition_context=None)`
- `MockModelProvider(..., stochastic_error_rate=0.0, stochastic_seed=0)`
- `EvaluationResult(..., statistics=None)`
- `ExperimentRecord(..., lineage={})`
- `CacheRecord(..., repeat_group_id=None, repetition_index=None)`

New methods:

- `ExperimentManager`: `execute_repeated`, `resume_group`, `rebuild_group_index`, `load_group`, `list_groups`, `load_group_evaluations`, `verify_group`, `load_evaluation`, `load_run_record`
- `BaseModelAdapter.with_generation_seed` and the class attribute `BaseModelAdapter.supports_generation_seed`
- the class attribute `BaseMetric.higher_is_better`

See [docs/api_reference.md](docs/api_reference.md).

## Old artifacts

- v0.3.x run directories can be listed, shown, verified (including `--quality-checks`), compared, loaded, analysed with `analyze`, and used for `preferences create`, without any file being modified. This is tested on a fixture written by v0.3.1.
- Old runs are never grouped automatically. Grouping them would be guesswork.
- v0.3.1 evaluation and comparison JSON files remain readable by `compare`, `export` and `analyze --results`.

## Not supported

- Grouping existing v0.3.x runs into a repeat group.
- Continuous-metric equivalence decisions. TOST values are reported as descriptive only in v0.4.0.
- Using v0.3.1 to verify v0.4.0 runs: it reports the newer schema version as unsupported.
- Concurrent writes to one cache directory from several processes.
