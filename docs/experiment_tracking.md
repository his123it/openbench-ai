# Experiment Management & Run Storage (v0.3.0)

OpenBench AI `v0.3.0` introduces `ExperimentManager` (`openbench.experiments`) to organize model execution runs into self-contained, verifiable directories under `experiments/run_<id>/`.

---

## 1. Run Directory Structure

Every experiment run creates a dedicated directory containing 7 standardized artifacts:

```text
experiments/
└── run_a1b2c3d4e5f6/
    ├── config.json          # Full experiment, dataset, metric, generation, and execution config
    ├── model_metadata.json  # Provider name, model name, revision, mock flag, and execution origin
    ├── responses.json       # Standardized model responses (compatible with `openbench evaluate`)
    ├── evaluation.json      # Complete EvaluationResult with category and difficulty breakdowns
    ├── performance.json     # Latency percentiles (P50/P95/P99), throughput, retries, and token usage
    ├── reproducibility.json # SHA-256 checksums for all files, dataset, and configurations
    └── logs.jsonl           # Per-request execution event stream
```

---

## 2. CLI Experiment Lifecycle Commands

```bash
# 1. Create and execute a new run
openbench run --config configs/mock_experiment.yaml

# 2. List all stored runs (optionally filter by --status or --provider)
openbench runs list

# 3. Inspect a specific run's summary and integrity status
openbench runs show run_a1b2c3d4e5f6

# 4. Verify SHA-256 checksums and provenance of all artifacts in a run
openbench runs verify run_a1b2c3d4e5f6

# 5. Resume an interrupted or partial run (skips already completed items)
openbench runs resume run_a1b2c3d4e5f6

# 6. Compare two or more runs side-by-side
openbench runs compare run_a1b2c3d4e5f6 run_998877665544 --output results/run_comparison.md
```

## v0.4.0: lineage and repeat groups

New runs record a `lineage` block in `config.json` and `reproducibility.json` (schema `0.4.0`). It holds the experiment and parent IDs, repeat group, repetition index and count, seeds and seed status, selected record IDs, per-record content hashes and cache provenance.

`openbench run --repetitions N` creates a repeat group (`grp_<id>`) with a rebuildable index file `experiments/grp_<id>.json`. `runs list --group`, `runs show grp_...`, `runs verify grp_...` and `runs resume grp_...` operate on groups.

v0.3.x runs remain readable and are never rewritten. See [repeated_runs.md](repeated_runs.md) and [reproducibility.md](reproducibility.md).

