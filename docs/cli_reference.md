# CLI reference (v0.4.0)

Generated from the argument parser by `scripts/generate_reference_docs.py`; do not edit by hand.
Flags marked *(v0.4.0)* are new; every v0.3.1 command, flag, default and choice is unchanged.

## `openbench validate-dataset`

Validate a JSONL dataset file or directory of JSONL files.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `dataset_path` | yes |  |  | Path to a JSONL dataset file or directory. |
| `--quality-checks` |  |  |  | *(v0.4.0)* also run the read-only dataset quality audit (exit 1 only for error findings). |
| `--output` |  |  |  | *(v0.4.0)* write the audit report to this file. |
| `--format` |  | `json` | `json`, `markdown`, `csv` | *(v0.4.0)* audit report format. |
| `--near-duplicate-threshold` |  | `0.85` |  | *(v0.4.0)* DQ003 Jaccard threshold. |
| `--metrics` |  |  |  | *(v0.4.0)* metric names to check (DQ008). |

## `openbench evaluate`

Evaluate one or more pre-generated model response files against a dataset or YAML profile.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `--dataset` |  |  |  | Path to a JSONL dataset file or directory (required unless --config specifies dataset.path). |
| `--config` |  |  |  | Path to a YAML evaluation configuration profile. |
| `--responses` | yes |  |  | Path to a JSON file containing model responses. |
| `--metrics` |  |  |  | Space-separated list of metrics to compute. |
| `--category` |  |  |  | Optional category filter(s) to apply to the dataset. |
| `--difficulty` |  |  |  | Optional difficulty filter(s) (easy, medium, hard). |
| `--ids-file` |  |  |  | Optional path to a file listing dataset record IDs to evaluate (like-for-like subset evaluation). |
| `--history-file` |  |  |  | Optional path to local JSONL history file to record evaluation summary. |
| `--output` | yes |  |  | Output directory or file path (.json, .md, or .csv) for evaluation reports. |
| `--confidence-intervals` |  |  |  | *(v0.4.0)* add a 'statistics' block with percentile bootstrap CIs over items (output is unchanged without this flag). |
| `--confidence-level` |  | `0.95` |  | *(v0.4.0)* CI level for --confidence-intervals (default 0.95). |
| `--resamples` |  | `10000` |  | *(v0.4.0)* bootstrap resamples for --confidence-intervals (default 10000). |
| `--seed` |  | `0` |  | *(v0.4.0)* statistical analysis seed for --confidence-intervals (default 0). |

## `openbench compare`

Compare two or more JSON evaluation result files.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `--results` | yes |  |  | Paths to evaluation result JSON files (provide at least two files). |
| `--output` | yes |  |  | Output file path (.md, .json, or .csv) or directory for the comparison report. |

## `openbench list-datasets`

List all built-in or directory JSONL datasets in the catalog.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `--path` |  |  |  |  |
| `--category` |  |  |  |  |
| `--difficulty` |  |  |  |  |
| `--format` |  | `text` | `text`, `json` |  |

## `openbench inspect-dataset`

Inspect metadata, schema breakdown, and sample records of a dataset.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `dataset_path` | yes |  |  |  |
| `--category` |  |  |  |  |
| `--difficulty` |  |  |  |  |
| `--samples` |  | `2` |  |  |
| `--format` |  | `text` | `text`, `json` |  |

## `openbench dataset-stats`

Display detailed statistics for a dataset file or directory.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `dataset_path` | yes |  |  |  |
| `--category` |  |  |  |  |
| `--difficulty` |  |  |  |  |
| `--format` |  | `text` | `text`, `json` |  |

## `openbench history`

List or inspect local historical evaluation runs.

## `openbench history list`

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `--store` |  |  |  |  |
| `--model` |  |  |  |  |
| `--dataset` |  |  |  |  |
| `--format` |  | `text` | `text`, `json` |  |

## `openbench history show`

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `evaluation_id` | yes |  |  |  |
| `--store` |  |  |  |  |
| `--format` |  | `text` | `text`, `json` |  |

## `openbench export`

Export an existing JSON evaluation or comparison result into CSV or Markdown.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `--input` | yes |  |  |  |
| `--format` | yes |  | `csv`, `markdown`, `json` |  |
| `--mode` |  | `individual` | `individual`, `aggregate` |  |
| `--output` | yes |  |  |  |

## `openbench run`

Execute a model provider against a dataset, evaluate metrics, and create a verifiable experiment run.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `--config` |  |  |  | Path to a YAML experiment configuration profile. |
| `--dataset` |  |  |  | Path to a JSONL dataset file or directory. |
| `--provider` |  |  |  | Model provider identifier (e.g. ollama, huggingface, mock). |
| `--model` |  |  |  | Model identifier (e.g. llama3.2, gpt2, mock-deterministic-v1). |
| `--model-version` |  |  |  | Model version or revision (default: latest or profile setting). |
| `--endpoint` |  |  |  | Optional provider HTTP endpoint (e.g. http://localhost:11434 for Ollama). |
| `--metrics` |  |  |  | Metrics to evaluate (default: exact_match accuracy keyword_coverage, or profile setting). |
| `--concurrency` |  |  |  | Number of concurrent model requests (default: 1). |
| `--timeout` |  |  |  | Per-request timeout in seconds (default: 30.0). |
| `--max-attempts` |  |  |  | Maximum retry attempts per item (default: 3). |
| `--no-cache` |  |  |  | Disable deterministic response caching for this run. |
| `--cache-dir` |  |  |  | Directory for response cache (default: .openbench_cache). |
| `--experiments-dir` |  |  |  | Directory for storing experiment runs (default: experiments). |
| `--output` |  |  |  | Optional output directory to also write top-level .json/.md/.csv reports. |
| `--repetitions` |  |  |  | *(v0.4.0)* number of independent repetitions (>= 2 creates a repeat group; overrides options.repeat_count). |
| `--seed` |  |  |  | *(v0.4.0)* generation base seed (repetition i uses seed + i); overrides generation.seed. |
| `--experiment-id` |  |  |  | *(v0.4.0)* optional experiment ID recorded in run lineage. |
| `--parent-experiment-id` |  |  |  | *(v0.4.0)* optional parent experiment ID recorded in run lineage. |

## `openbench runs`

Manage, inspect, verify, resume, and compare experiment runs.

## `openbench runs list`

List stored experiment runs.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `--experiments-dir` |  |  |  |  |
| `--status` |  |  |  |  |
| `--provider` |  |  |  |  |
| `--format` |  | `text` | `text`, `json` |  |
| `--group` |  |  |  | *(v0.4.0)* list only the runs of this repeat group. |

## `openbench runs show`

Inspect a specific experiment run.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `run_id` | yes |  |  |  |
| `--experiments-dir` |  |  |  |  |
| `--format` |  | `text` | `text`, `json` |  |

## `openbench runs verify`

Verify cryptographic SHA-256 artifact checksums and provenance of a run.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `run_id` | yes |  |  |  |
| `--experiments-dir` |  |  |  |  |
| `--format` |  | `text` | `text`, `json` |  |
| `--quality-checks` |  |  |  | *(v0.4.0)* also run the experiment quality audit (EQ checks). |
| `--audit-output` |  |  |  | *(v0.4.0)* write the audit report (JSON) here. |

## `openbench runs resume`

Resume an interrupted or partial provider-executed experiment run.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `run_id` | yes |  |  |  |
| `--experiments-dir` |  |  |  |  |
| `--cache-dir` |  |  |  |  |

## `openbench runs compare`

Compare two or more experiment runs by their RUN_IDs.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `run_ids` | yes |  |  | Two or more experiment run IDs. |
| `--experiments-dir` |  |  |  |  |
| `--output` |  |  |  |  |
| `--format` |  | `text` | `text`, `json`, `markdown` |  |

## `openbench providers`

List registered model providers or check provider connectivity.

## `openbench providers list`

List all registered model provider adapters.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `--format` |  | `text` | `text`, `json` |  |

## `openbench providers check`

Check connectivity/availability of a provider.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `provider_name` | yes |  |  | Provider name (e.g. ollama, huggingface, mock). |
| `--model` |  |  |  |  |
| `--endpoint` |  |  |  |  |
| `--format` |  | `text` | `text`, `json` |  |

## `openbench cache`

Manage or clear the deterministic response cache.

## `openbench cache clear`

Clear all cached responses.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `--cache-dir` |  |  |  |  |

## `openbench cache stats`

Show cache entry count and location.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `--cache-dir` |  |  |  |  |

## `openbench analyze` *(v0.4.0)*

v0.4.0: statistically compare stored runs, repeat groups or evaluation files (no provider calls), or analyse one repeat group's variability.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `--runs` |  |  |  | Run IDs (run_...) or repeat group IDs (grp_...). |
| `--results` |  |  |  | Evaluation result JSON files. |
| `--experiments-dir` |  |  |  |  |
| `--metrics` |  |  |  |  |
| `--baseline` |  |  |  | Arm ID or model name of the baseline. |
| `--ids-file` |  |  |  | Restrict the analysis to these record IDs. |
| `--confidence-level` |  | `0.95` |  |  |
| `--resamples` |  | `10000` |  |  |
| `--permutations` |  | `9999` |  |  |
| `--seed` |  | `0` |  | Statistical analysis seed (default 0). |
| `--missing-policy` |  | `complete_case` | `complete_case`, `missing_as_failure` |  |
| `--correction` |  | `holm` | `holm`, `bh`, `none` |  |
| `--min-practical-difference` |  |  |  |  |
| `--equivalence-margin` |  |  |  | Pre-specified TOST margin (metric units). |
| `--alpha` |  | `0.05` |  |  |
| `--stratified-bootstrap` |  |  |  |  |
| `--stratum-tests` |  |  |  |  |
| `--exploratory` |  |  |  | Descriptive-only comparison of incompatible inputs. |
| `--exploratory-reason` |  |  |  |  |
| `--output` |  |  |  |  |
| `--format` |  | `json` | `json`, `markdown`, `csv` |  |

## `openbench preferences`

v0.4.0: local blinded pairwise human preference evaluation.

## `openbench preferences create` *(v0.4.0)*

Create blind A/B tasks (tasks.jsonl) and a separate key file.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `--runs` |  |  |  |  |
| `--results` |  |  |  |  |
| `--experiments-dir` |  |  |  |  |
| `--seed` |  | `0` |  | Preference-order seed (default 0). |
| `--output` | yes |  |  | Output directory for tasks.jsonl. |
| `--key-output` |  |  |  | Key file path (default OUTPUT/tasks.key.json). |
| `--rubric` |  |  |  | JSON rubric file. |
| `--show-reference` |  |  |  | Include reference answers in tasks. |
| `--csv` |  |  |  | Also write tasks.csv. |
| `--overwrite` |  |  |  | Allow overwriting existing task files. |

## `openbench preferences validate` *(v0.4.0)*

Validate completed annotation files against a task key.

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `--key` | yes |  |  |  |
| `--annotations` | yes |  |  |  |
| `--annotators` |  |  |  | Optional roster file (one annotator ID per line). |
| `--format` |  | `text` | `text`, `json` |  |

## `openbench preferences analyze` *(v0.4.0)*

Analyse validated annotations (win rates, agreement, optional Elo).

| Argument | Required | Default | Choices | Help |
| :--- | :---: | :--- | :--- | :--- |
| `--key` | yes |  |  |  |
| `--annotations` | yes |  |  |  |
| `--annotators` |  |  |  |  |
| `--seed` |  | `0` |  |  |
| `--confidence-level` |  | `0.95` |  |  |
| `--resamples` |  | `10000` |  |  |
| `--elo` |  |  |  | Also report Elo ratings (relative; see limitations). |
| `--output` |  |  |  |  |
| `--format` |  | `json` | `json`, `markdown`, `csv` |  |
