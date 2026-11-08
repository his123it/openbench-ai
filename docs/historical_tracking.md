# Local Result History & Longitudinal Tracking (v0.2.0)

OpenBench AI `v0.2.0` includes a lightweight, zero-server evaluation history store (`openbench.history.ResultHistoryStore`) backed by an append-only JSONL file (`results/history.jsonl` by default).

---

## 1. Stored History Fields (`HistoryEntry`)

Each recorded evaluation run appends a single JSON object containing:
- `evaluation_id` (unique identifier, e.g., `eval_a1b2c3d4e5f6`)
- `timestamp` (UTC ISO-8601 timestamp)
- `model_name` and `model_version`
- `dataset_name`, `dataset_version`, and `dataset_checksum` (SHA-256)
- `metric_configuration`
- `openbench_version`
- `result_summary` (mapping of metric names to aggregate scores)
- `total_items`, `evaluated_items`, and `missing_responses_count`
- `report_path`

---

## 2. CLI History Commands

```bash
# Record a run into history during evaluation
openbench evaluate   --dataset datasets/reasoning/sample.jsonl   --responses examples/sample_responses.json   --history-file results/history.jsonl   --output results/

# List all recorded runs (or filter by --model / --dataset)
openbench history list --store results/history.jsonl

# Inspect a specific evaluation run by ID
openbench history show eval_a1b2c3d4e5f6 --store results/history.jsonl
```

---

## 3. Programmatic Historical Comparison

Use `ModelComparator.compare_history([run_v1, run_v2])` to compute chronological score deltas between an earlier baseline run and the latest evaluation run.
