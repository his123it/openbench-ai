# Installation Guide

OpenBench AI requires **Python 3.10 or newer** and uses zero third-party runtime dependencies for its core library and CLI. Development tools (`pytest` and `ruff`) are provided via the `[dev]` optional extras.

## 1. Local Editable Installation (Recommended for Development)

```bash
git clone https://github.com/<your-username>/openbench-ai.git
cd openbench-ai

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip and install OpenBench AI with dev extras
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 2. Verify CLI & Package Import

Check that the `openbench` entry point is on your `PATH`:

```bash
openbench --version
# Expected output: openbench 0.1.0
```

Check that the Python package imports cleanly:

```bash
python -c "import openbench; print(openbench.__version__)"
# Expected output: 0.1.0
```

## 3. Offline / Air-Gapped Environment Installation

Because OpenBench AI has no external runtime dependencies (`dependencies = []` in `pyproject.toml`), you can install and run the package in an offline environment where `setuptools` and `wheel` are already present:

```bash
python -m pip install --no-build-isolation --no-deps -e .
```
