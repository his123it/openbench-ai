"""Built-in dataset audit computed from the live repository files (plan E2, section 7.4).

Nothing here is a stored expectation of the auditor's output: the audit runs on the
actual dataset files, the DQ007 candidates are recomputed with an independent minimal
implementation of the keyword rule, and the reviewed classification (0 errors, 6
warnings, 8 info) is asserted so any dataset or rule change forces a review.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

from _v040_helpers import snapshot
from openbench.quality import DatasetAuditor

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"

EXPECTED = {
    ("DQ007", "warning"): {"coding_003", "coding_007", "coding_010", "reasoning_004", "reasoning_012"},
    ("DQ011", "warning"): {"multilingual_006"},
    ("DQ010", "info"): {"coding_019", "factual_005", "factual_007", "factual_018"},
}
DQ009B_FILES = {"coding", "instruction_following", "reasoning", "safety"}


def _reference_keyword_rule() -> set[str]:
    """Independent F11 rule: trim, collapse whitespace, lower-case; substring test."""

    def norm(text: str) -> str:
        return " ".join(text.strip().split()).lower()

    flagged = set()
    for path in sorted(DATASETS.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            kws = (rec.get("metadata") or {}).get("keywords")
            if isinstance(kws, list) and any(k.strip() and norm(k) not in norm(rec["expected_answer"]) for k in kws):
                flagged.add(rec["id"])
    return flagged


def test_builtin_audit_matches_reviewed_classification():
    before = snapshot(DATASETS)
    report = DatasetAuditor().audit(DATASETS).to_dict()
    assert snapshot(DATASETS) == before  # read-only: SHA-256, size and mtime unchanged
    s = report["summary"]
    assert (s["errors"], s["warnings"], s["info"]) == (0, 6, 8)
    assert report["subject"]["record_count"] == 200
    for (check, severity), ids in EXPECTED.items():
        found = {f["record_ids"][0] for f in report["findings"] if f["check_id"] == check}
        assert found == ids, check
        assert all(f["severity"] == severity for f in report["findings"] if f["check_id"] == check)
    dq9b = [f for f in report["findings"] if f["check_id"] == "DQ009b"]
    assert {Path(f["details"]["file"]).parent.name for f in dq9b} == DQ009B_FILES
    assert all(f["details"]["unversioned_count"] == 12 for f in dq9b)
    assert sum(len(f["record_ids"]) for f in dq9b) == 48
    assert s["max_similarity_pair"] == ["coding_001", "coding_014"]
    assert abs(s["max_near_duplicate_similarity"] - 0.535) < 0.001
    assert all(
        f["classification"] == "documented_historical_exception"
        for f in report["findings"] if f["check_id"] == "DQ007"
    )
    missing = next(f for f in report["findings"] if f["record_ids"] == ["reasoning_012"])
    assert missing["details"]["missing_keywords"] == ["3", "5"]


def test_auditor_dq007_matches_independent_reference_rule():
    report = DatasetAuditor(use_review_notes=False).audit(DATASETS).to_dict()
    auditor_ids = {f["record_ids"][0] for f in report["findings"] if f["check_id"] == "DQ007"}
    assert auditor_ids == _reference_keyword_rule() == EXPECTED[("DQ007", "warning")]
    assert "summarization_004" not in auditor_ids and "summarization_017" not in auditor_ids
    assert string.punctuation  # punctuation is kept by both rules


def test_notes_never_suppress_or_change_heuristic_findings():
    def heuristic(report):
        return [
            (f["check_id"], f["severity"], tuple(f["record_ids"]), f["message"])
            for f in report["findings"] if f["check_id"] not in ("DQ010", "DQ011")
        ]

    with_notes = DatasetAuditor().audit(DATASETS).to_dict()
    without = DatasetAuditor(use_review_notes=False).audit(DATASETS).to_dict()
    assert heuristic(with_notes) == heuristic(without)
    assert {f["check_id"] for f in without["findings"]} == {"DQ007", "DQ009b"}


def test_validate_dataset_without_flag_is_unchanged(capsys):
    from openbench.cli import main

    assert main(["validate-dataset", str(DATASETS)]) == 0
    out = capsys.readouterr().out
    assert out == (
        "Dataset valid: 'datasets' (200 records; categories: coding: 20, debugging: 20, "
        "factual_knowledge: 20, instruction_following: 20, mathematics: 20, multilingual: 20, "
        "reasoning: 20, safety: 20, structured_output: 20, summarization: 20)\n"
    )
    assert main(["validate-dataset", str(DATASETS), "--quality-checks"]) == 0
