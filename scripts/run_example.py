#!/usr/bin/env python3
"""End-to-end demonstration script for OpenBench AI v0.1.0."""

import subprocess
import sys
from pathlib import Path


def run_step(description: str, cmd: list[str], cwd: Path) -> None:
    print(f"==> {description}")
    print(f"    $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    if proc.stdout:
        for line in proc.stdout.strip().splitlines():
            print(f"    {line}")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    print()


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    run_step(
        "Step 1: Validate all built-in benchmark datasets",
        [sys.executable, "scripts/validate_datasets.py"],
        cwd=repo_root,
    )
    run_step(
        "Step 2: Run basic single-model evaluation example",
        [sys.executable, "examples/basic_evaluation.py"],
        cwd=repo_root,
    )
    run_step(
        "Step 3: Run multi-model comparison example",
        [sys.executable, "examples/compare_models.py"],
        cwd=repo_root,
    )
    print("OpenBench AI end-to-end example workflow completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
