# Repeated-run benchmarking (v0.4.0)

A single run measures one sample of a model's behaviour. Sampling settings, provider-side nondeterminism and transient failures can change results from run to run. v0.4.0 lets you repeat an experiment, keeps every repetition inspectable, and reports run-to-run variability separately from item-to-item variation.

## Running repetitions

```bash
openbench run --dataset datasets/reasoning/sample.jsonl --provider ollama --model llama3.2 \
  --repetitions 5 --seed 100 --experiment-id reasoning-temp07
```

- `--repetitions 1` (the default) is a **single run** with no group. `--repetitions N` with N ≥ 2 creates a **repeat group** `grp_<12 hex>` with N runs. The flag overrides `options.repeat_count` in a YAML profile.
- `--seed S` is the generation base seed; repetition *i* uses seed `S + i`. Without `--seed`, the profile's `generation.seed` is used as the base, if one is set.
- `--experiment-id` and `--parent-experiment-id` are optional labels recorded in lineage.
- The legacy `repeated_runs_analysis.json` (v0.3.1 format) is still written into the last run directory.

Python API:

```python
from openbench import DatasetLoader, ExperimentManager, MockModelProvider

manager = ExperimentManager("experiments")
group = manager.execute_repeated(
    DatasetLoader.load("datasets/reasoning/sample.jsonl"),
    MockModelProvider(stochastic_error_rate=0.2, stochastic_seed=1),  # simulated sampling noise
    repetitions=3,
    generation_base_seed=100,
)
print(group.repeat_group_id, [r.run_id for r in group.records])
```

## What is recorded (lineage)

Every run created by v0.4.0 has a `lineage` block in `config.json` and `reproducibility.json`, with run `schema_version` `0.4.0`. A single run has `repeat_group_id: null`, `repetition_index: 0` and `repetition_count: 1`.

| Field | Meaning |
| --- | --- |
| `experiment_id`, `parent_experiment_id` | Optional labels; `experiment_id` defaults to the group ID (or the run ID). |
| `repeat_group_id`, `repetition_index`, `repetition_count` | Group identity. |
| `generation_base_seed`, `repetition_seed` | Seeds (`repetition_seed = base + index`). |
| `seed_supplied`, `seed_supported_by_adapter`, `seed_passed_to_provider`, `seed_status` | What happened to the seed. |
| `selected_record_ids`, `selected_record_ids_sha256` | The exact analysed IDs. |
| `record_content_sha256` | `{record_id: SHA-256 of the full record}`, used by comparison compatibility checks. |
| `cache_provenance` | Whether the cache was enabled, the cache context, cached requests, and items preserved from an earlier attempt. |

`seed_status` takes one of these values:

- `not_supplied`
- `unsupported_by_adapter` (mock, local and Hugging Face adapters do not forward seeds)
- `passed_to_provider` (Ollama placed the seed in at least one request)
- `no_provider_requests` (a seed-capable adapter made no requests because everything was cached)
- `unknown` (v0.3.x runs)

**A seed passed to a provider does not guarantee deterministic output.** The provider may still produce different responses.

Run identity, model and provider, dataset, generation configuration, metric configuration, timestamps, failures, retries, timeouts, latency and tokens are recorded as in v0.3.1 (`config.json`, `model_metadata.json`, `performance.json`, `logs.jsonl`).

## The group index

`experiments/grp_<id>.json` summarises a group. It records:

- the planned and completed repetitions;
- the member run IDs by index, with their statuses and seeds;
- shared configuration checksums;
- a checksum over the members' `reproducibility.json` files.

It is a rebuildable index. **Per-run lineage is authoritative**: if the index and the lineage disagree, the index is rebuilt from the lineage and a warning is recorded. The index is a file at the experiments root rather than a sub-directory, so tools that list run directories are unaffected.

## Cache isolation (fix)

v0.3.1 reused cached responses across repetitions, so "repeated" runs returned identical answers and understated variability. In v0.4.0 the cache key of a repetition includes `(repeat_group_id, repetition_index)`. As a result:

- Repetitions in one group never share cache entries.
- Two independent groups with the same configuration never share entries.
- Resuming repetition *i* of group G reuses only (G, *i*) entries.
- Single runs keep the v0.3.1 key, byte for byte, so v0.3.1 caches and single-run sharing behave as before.
- `--no-cache` still performs no reads or writes.

The cache is a local file with in-process locking. It is not safe for several processes writing at once, and OpenBench does not provide distributed exactly-once execution.

## Resuming

```bash
openbench runs resume grp_<id>
```

- Completed members are never re-requested.
- Partial or failed members are resumed in place, keeping their completed items.
- Missing repetition indices are created under the same group with their own seed and cache context.
- Duplicate repetition indices in lineage stop the resume with an error, so the runs can be inspected first.

Resumption restores the original metric options, normalization, exact record IDs, seeds and cache setting.

## Inspecting and verifying

```bash
openbench runs list --group grp_<id>
openbench runs show grp_<id>
openbench runs verify grp_<id> --quality-checks
```

## Variability analysis

```bash
openbench analyze --runs grp_<id> --format markdown --output variability.md
```

`openbench.stats.decompose_variability(evaluations)` reports three things separately:

1. **Generation variability** within items, across repetitions:
   - per-item mean, SD, minimum and maximum, and whether the score changed;
   - mean within-item variance and the share of items that change;
   - for binary metrics, the flip rate and mean pairwise agreement;
   - exact response-text identity.
2. **Item variation**: the spread of per-item means across records, and the question-level percentile bootstrap CI computed over per-item means. The unit is the item, never item × repetition.
3. **Failures**: missing and invalid responses per repetition, and items failing in some or all repetitions. Failures are excluded from per-item means, never counted as wrong answers, and reported on their own.

It also reports the run-to-run distribution of headline scores, latency and token variation, and per-category and per-difficulty breakdowns. A hierarchical bootstrap over items and repetitions is not used in v0.4.0.
