# Developer & Contributor Guide

This document supplements the root [CONTRIBUTING.md](../CONTRIBUTING.md) with architectural notes for extending OpenBench AI.

## Directory Layout Conventions

- `src/openbench/`: All importable library and CLI code. Do not place test fixtures or scripts inside `src/openbench/`.
- `datasets/<category>/`: Standardized JSONL files. Every record must pass `DatasetLoader.load_file()`.
- `tests/`: Automated unit and integration tests corresponding 1-to-1 with core modules (`test_dataset.py`, `test_metrics.py`, `test_evaluator.py`, `test_comparison.py`, `test_reporting.py`, `test_cli.py`).

## Adding New CLI Subcommands

When adding new subcommands in `src/openbench/cli.py`:
1. Keep business logic inside core modules (`dataset.py`, `evaluator.py`, `comparison.py`, `reporting.py`).
2. Add the subparser in `build_parser()`.
3. Create a dedicated handler function `_handle_<subcommand>(...) -> int` that returns standard POSIX exit codes (`0` for success, `1` for runtime/validation errors, `2` for CLI usage errors).
4. Add end-to-end tests invoking `main([...])` in `tests/test_cli.py`.
