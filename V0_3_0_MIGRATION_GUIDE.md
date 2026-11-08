# OpenBench AI v0.3.0 — Migration Guide

Upgrading from **OpenBench AI v0.1.0 or v0.2.0** to **v0.3.0** keeps existing imports, CLI commands, dataset files, response JSON files, and YAML profiles working without modification. See the CHANGELOG for intentional behavioral changes.

---

## 1. Upgrading Your Environment

```bash
python -m pip install -e ".[dev]"
```

---

## 2. Continuing to Evaluate Static Response Files (Unchanged)

If you already collect model responses externally, continue using `openbench evaluate` exactly as in `v0.1.0` and `v0.2.0`:

```bash
openbench evaluate   --dataset datasets/reasoning/sample.jsonl   --responses examples/sample_responses.json   --output results/
```

---

## 3. Adopting v0.3.0 Model Execution & Experiment Tracking

### Step 1: Add `provider`, `generation`, and `execution` blocks to a YAML profile (Optional)
Existing `v0.2.0` profiles (`configs/reasoning.yaml`, etc.) continue to work with `openbench evaluate --config ...`. To execute a model directly via `openbench run`, add a `provider:` block (see `configs/local_ollama.yaml` or `configs/mock_experiment.yaml`):

```yaml
provider:
  name: ollama
  model: llama3.2
  endpoint: http://localhost:11434

generation:
  temperature: 0.0
  max_tokens: 128

execution:
  timeout_seconds: 30.0
  concurrency: 2
  use_cache: true
  max_attempts: 3
```

### Step 2: Run, Verify, and Resume Experiments
```bash
openbench run --config configs/mock_experiment.yaml
openbench runs list
openbench runs verify <RUN_ID>
openbench runs resume <RUN_ID>
```
