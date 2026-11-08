# Usage Guide

OpenBench AI can be used either from the **Command-Line Interface (`openbench`)** or programmatically via its **Python API**.

---

## 1. Command-Line Interface (CLI)

### Validate a Dataset (`validate-dataset`)

```bash
openbench validate-dataset datasets/reasoning/sample.jsonl
openbench validate-dataset datasets/
```

### Evaluate Model Responses (`evaluate`)

Evaluates a JSON response file against a JSONL dataset. Any evaluation warnings (such as missing responses, unknown response IDs, or active ID filter exclusions) are printed to `stderr` while keeping `stdout` clean:

```bash
openbench evaluate   --dataset datasets/reasoning/sample.jsonl   --responses examples/sample_responses.json   --metrics exact_match accuracy keyword_coverage   --output results/
```

### Like-for-Like Subset Evaluation Using `--ids-file` (New in v0.3.1)

When evaluating a response file collected on an earlier subset of a dataset (such as a 12-item `v0.1.0` response file evaluated against a 20-item `v0.2.0+` dataset), pass `--ids-file` so intentionally excluded dataset records are not counted as missing/failed responses:

```bash
openbench evaluate   --dataset datasets/reasoning/sample.jsonl   --responses legacy_12_responses.json   --ids-file legacy_reasoning_ids.txt   --output results/
```

### Execute a Model Provider & Manage Experiment Runs (`run` / `runs`)

```bash
openbench run --config configs/mock_experiment.yaml
openbench runs list
openbench runs show <RUN_ID>
openbench runs verify <RUN_ID>
openbench runs resume <RUN_ID>
openbench runs compare <RUN_ID_1> <RUN_ID_2> --output results/comparison.md
```
