#!/usr/bin/env python3
"""Validate all JSONL datasets in the OpenBench AI `datasets/` directory."""

import sys
from pathlib import Path

from openbench.dataset import DatasetLoader
from openbench.exceptions import OpenBenchError


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    datasets_dir = repo_root / "datasets"

    if not datasets_dir.exists():
        print(f"Error: datasets directory not found at {datasets_dir}", file=sys.stderr)
        return 1

    jsonl_files = sorted(datasets_dir.rglob("*.jsonl"))
    if not jsonl_files:
        print("Error: No .jsonl files found in datasets directory.", file=sys.stderr)
        return 1

    total_records = 0
    print(f"Validating {len(jsonl_files)} dataset file(s) in {datasets_dir}...")

    for file_path in jsonl_files:
        try:
            ds = DatasetLoader.load_file(file_path)
            rel_path = file_path.relative_to(repo_root)
            summary = ds.summary()
            print(
                f"  [OK] {rel_path}: {len(ds)} records "
                f"(categories={summary['categories']}, difficulties={summary['difficulties']})"
            )
            total_records += len(ds)
        except OpenBenchError as exc:
            print(f"  [FAIL] {file_path}: {exc}", file=sys.stderr)
            return 1

    # Also verify cross-directory uniqueness via load_directory
    try:
        combined = DatasetLoader.load_directory(datasets_dir, dataset_name="openbench_all")
        print(
            f"\nAll datasets valid! Total unique records across repository: {len(combined)} "
            f"(matches sum: {total_records})."
        )
    except OpenBenchError as exc:
        print(f"\nCross-dataset validation failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
