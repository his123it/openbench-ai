# OpenBench AI v0.3.0 — Build, Test & Verification Instructions

## 1. Requirements & Optional Dependency Groups

- **Python**: `3.10`, `3.11`, or `3.12`
- **Core Runtime Dependencies**: None (`dependencies = []`)
- **Optional Groups (`pyproject.toml`)**:
  - `dev`: `pytest`, `ruff`, `PyYAML`, `jsonschema`
  - `ollama`: `httpx` (optional; `OllamaAdapter` works out-of-the-box using stdlib `urllib.request`)
  - `huggingface`: `transformers`, `torch`
  - `performance`: `tabulate`
  - `config`: `PyYAML`
  - `metrics`: `jsonschema`
  - `semantic`: `sentence-transformers`
  - `all`: All optional dependencies

---

## 2. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

---

## 3. Dataset Validation & Test Execution

```bash
# Validate all 10 built-in datasets (200 unique examples)
python scripts/validate_datasets.py

# Run the full test suite (19 test files, 80 tests)
pytest -v

# Run Ruff linter
ruff check src tests examples scripts
```

---

## 4. CLI Verification Commands (v0.1.0, v0.2.0 & v0.3.0)

```bash
# Version check
openbench --version

# Providers list & health check
openbench providers list
openbench providers check mock

# Execute a mock experiment run via config or CLI flags
openbench run --config configs/mock_experiment.yaml --experiments-dir experiments/
openbench run --dataset datasets/reasoning/sample.jsonl --provider mock --model mock-cli-v1 --experiments-dir experiments/

# List, inspect, verify integrity, and compare runs
openbench runs list --experiments-dir experiments/
openbench runs show <RUN_ID> --experiments-dir experiments/
openbench runs verify <RUN_ID> --experiments-dir experiments/
openbench runs compare <RUN_ID_1> <RUN_ID_2> --experiments-dir experiments/ --output results/runs_cmp.md

# Manage response cache
openbench cache stats
openbench cache clear
```
