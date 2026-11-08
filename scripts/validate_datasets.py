#!/usr/bin/env python3
"""Validate all JSONL datasets and run reference-answer self-consistency checks (v0.3.1)."""

from __future__ import annotations

import sys
from pathlib import Path

from openbench.dataset import Dataset, DatasetLoader
from openbench.exceptions import OpenBenchError
from openbench.metrics import create_metric

# Five known v0.1.0 legacy records with extra synonym/explanation keywords in metadata.keywords
# that do not appear verbatim in expected_answer (preserved unmodified per compatibility policy),
# plus v0.2.0 records with alternative synonym keywords.
KNOWN_LEGACY_KEYWORD_EXCEPTIONS = {
    "coding_003",
    "coding_007",
    "coding_010",
    "reasoning_004",
    "reasoning_012",
}
KNOWN_V0_2_SYNONYM_KEYWORD_RECORDS = {
    "coding_019",
    "factual_005",
    "factual_007",
    "factual_018",
    "multilingual_006",
    "summarization_004",
    "summarization_017",
}


def check_dataset_self_consistency(dataset: Dataset) -> tuple[list[str], list[str], list[str]]:
    """Verify that every reference answer scores 1.0 against itself on applicable deterministic metrics.

    Returns `(unexpected_failures, reported_v1_legacy_exceptions, reported_v2_synonym_exceptions)`.
    """
    deterministic_metrics = [
        create_metric("exact_match"),
        create_metric("accuracy"),
        create_metric("keyword_coverage"),
        create_metric("precision"),
        create_metric("recall"),
        create_metric("f1"),
        create_metric("rouge_1"),
        create_metric("rouge_l"),
        create_metric("bleu"),
        create_metric("lexical_similarity"),
    ]
    structured_metric = create_metric("structured_output")

    unexpected_failures: list[str] = []
    v1_legacy_reports: list[str] = []
    v2_synonym_reports: list[str] = []

    for rec in dataset.records:
        for m in deterministic_metrics:
            res = m.compute_item(rec, rec.expected_answer)
            if res.score < 0.999999:
                if m.name == "keyword_coverage" and rec.id in KNOWN_LEGACY_KEYWORD_EXCEPTIONS:
                    missing_kw = res.details.get("missing_keywords", [])
                    v1_legacy_reports.append(
                        f"{rec.id} ({rec.category}): score={res.score:.4f}, "
                        f"non-verbatim legacy keywords={missing_kw}"
                    )
                elif m.name == "keyword_coverage" and rec.id in KNOWN_V0_2_SYNONYM_KEYWORD_RECORDS:
                    missing_kw = res.details.get("missing_keywords", [])
                    v2_synonym_reports.append(
                        f"{rec.id} ({rec.category}): score={res.score:.4f}, "
                        f"synonym keywords={missing_kw}"
                    )
                else:
                    unexpected_failures.append(
                        f"Record '{rec.id}' failed self-consistency on metric '{m.name}' "
                        f"(score={res.score:.4f}, details={res.details})"
                    )

        if rec.category == "structured_output":
            s_res = structured_metric.compute_item(rec, rec.expected_answer)
            if s_res.score < 0.999999 or not s_res.passed:
                unexpected_failures.append(
                    f"Record '{rec.id}' failed structured_output self-consistency "
                    f"(score={s_res.score:.4f}, details={s_res.details})"
                )

    return unexpected_failures, v1_legacy_reports, v2_synonym_reports


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

    try:
        combined = DatasetLoader.load_directory(datasets_dir, dataset_name="openbench_all")
        print(
            f"\nAll datasets valid! Total unique records across repository: {len(combined)} "
            f"(matches sum: {total_records})."
        )
    except OpenBenchError as exc:
        print(f"\nCross-dataset validation failed: {exc}", file=sys.stderr)
        return 1

    unexpected, v1_legacy, v2_synonyms = check_dataset_self_consistency(combined)
    print(
        f"Self-consistency check completed across {len(combined)} records: "
        f"{len(unexpected)} unexpected failures, "
        f"{len(v1_legacy)} known v0.1.0 legacy keyword exceptions preserved."
    )
    if v1_legacy:
        print("Known v0.1.0 legacy keyword exceptions (preserved for backward compatibility):")
        for item in v1_legacy:
            print(f"  - {item}")
    if v2_synonyms:
        print("Known v0.2.0 alternative/synonym keyword records (preserved for backward compatibility):")
        for item in v2_synonyms:
            print(f"  - {item}")

    if unexpected:
        for fail_msg in unexpected:
            print(f"  [SELF-CONSISTENCY FAIL] {fail_msg}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
