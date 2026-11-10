# Installation Guide

OpenBench AI requires **Python 3.10 or newer** (the test suite is run on Python 3.10, 3.11, 3.12 and 3.13 on Linux) and requires zero third-party runtime dependencies for its core library and CLI.

## 1. Local Editable Installation (Recommended for Development)

```bash
git clone https://github.com/lakshaychhaba0001-sys/openbench-ai.git
cd openbench-ai

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip and install OpenBench AI with dev extras
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 2. Core-Only Installation (Zero Third-Party Dependencies)

```bash
python -m pip install -e .
```

## 3. Verify CLI & Package Import

```bash
openbench --version
# Expected output: openbench 0.4.0

python -c "import openbench; print(openbench.__version__)"
# Expected output: 0.4.0
```

## 4. Optional Extras

| Extra | Installs | Used for |
| --- | --- | --- |
| `dev` | pytest, ruff, PyYAML, jsonschema | Tests, linting, YAML profiles, JSON Schema checks |
| `config` | PyYAML | YAML profiles |
| `metrics` | jsonschema | `structured_output` JSON Schema validation |
| `semantic` | sentence-transformers | Neural `semantic_similarity` |
| `huggingface` | transformers, torch | The experimental Hugging Face adapter |
| `stats-ref` | numpy, scipy, statsmodels | **Tests only**: the statistical reference cross-checks in `tests/reference`. The library never imports them. |
| `all` | All runtime integrations above | |

All v0.4.0 features (statistics, repeated runs, paired comparison, preferences, audits, reports) need no extra.

## 5. Offline / Air-Gapped Environment Installation

```bash
python -m pip install --no-build-isolation --no-deps -e .
```
