# Installation Guide

OpenBench AI requires **Python 3.10 or newer** (tested across Python 3.10, 3.11, 3.12, and 3.13) and requires zero third-party runtime dependencies for its core library and CLI.

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
# Expected output: openbench 0.3.1

python -c "import openbench; print(openbench.__version__)"
# Expected output: 0.3.1
```

## 4. Offline / Air-Gapped Environment Installation

```bash
python -m pip install --no-build-isolation --no-deps -e .
```
