"""`evaluate --confidence-intervals` (plan A1 CLI, D12)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openbench.cli import main

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "reasoning" / "sample.jsonl"
RESPONSES = ROOT / "examples" / "sample_responses.json"


def _run(tmp_path, *extra):
    out = tmp_path / "out"
    assert main(["evaluate", "--dataset", str(DATASET), "--responses", str(RESPONSES), "--metrics",
                 "exact_match", "keyword_coverage", "--output", str(out), *extra]) == 0
    return out


def _load(out):
    return {p.name: json.loads(p.read_text()) for p in sorted(out.glob("*.json"))}


def test_without_flag_output_has_no_statistics_block(tmp_path):
    for payload in _load(_run(tmp_path)).values():
        assert "statistics" not in payload


def test_with_flag_point_estimate_equals_headline(tmp_path):
    files = _load(_run(tmp_path, "--confidence-intervals", "--resamples", "500", "--seed", "3"))
    for payload in files.values():
        stats = payload["statistics"]
        assert stats["schema_version"] == "0.4.0" and stats["analysis_seed"] == 3
        for name, block in stats["metrics"].items():
            assert block["headline"]["estimate"] == pytest.approx(payload["aggregate_results"][name]["score"], abs=1e-6)
            assert block["headline"]["sample_count"] == payload["model"]["total_dataset_items"]
            assert block["evaluated_only"]["sample_count"] == payload["model"]["evaluated_items"]
            assert block["headline"]["lower"] <= block["headline"]["estimate"] <= block["headline"]["upper"]


def test_statistics_round_trip_and_determinism(tmp_path):
    from openbench import EvaluationResult

    a = _load(_run(tmp_path / "a", "--confidence-intervals", "--resamples", "300"))
    b = _load(_run(tmp_path / "b", "--confidence-intervals", "--resamples", "300"))
    for name in a:
        assert a[name]["statistics"] == b[name]["statistics"]
        assert EvaluationResult.from_dict(a[name]).to_dict()["statistics"] == a[name]["statistics"]
