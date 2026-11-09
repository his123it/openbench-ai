"""v0.4.0 report rendering: JSON/Markdown/CSV agreement, exploratory banner, CSV safety,
redaction, provenance, old-artifact readability and no provider calls (plan F1)."""

from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path

import pytest

from _v040_helpers import correct_answers, evaluate, snapshot, write_dataset
from openbench import MockModelProvider
from openbench.cli import main
from openbench.reporting_v040 import fmt, fmt_p, render_analysis, render_audit

ROOT = Path(__file__).resolve().parents[1]
FAST = ["--resamples", "300", "--permutations", "999"]


@pytest.fixture
def two_files(tmp_path):
    ds = write_dataset(tmp_path / "ds.jsonl", 12)
    a = evaluate(ds, correct_answers(12, wrong=(2, 5, 9)), "model-a", tmp_path / "a.json")
    b = evaluate(ds, correct_answers(12, wrong=(3,), missing=(7,)), "model-b", tmp_path / "b.json")
    return a, b


def _analyze(tmp_path, paths, fmt_name, *extra):
    out = tmp_path / f"out.{ 'md' if fmt_name == 'markdown' else fmt_name}"
    assert main(["analyze", "--results", *map(str, paths), "--format", fmt_name, "--output", str(out), *FAST, *extra]) == 0
    return out.read_text(encoding="utf-8")


def test_json_markdown_and_csv_agree(tmp_path, two_files):
    payload = json.loads(_analyze(tmp_path, two_files, "json", "--equivalence-margin", "0.2"))
    md = _analyze(tmp_path, two_files, "markdown", "--equivalence-margin", "0.2")
    rows = list(csv.DictReader(io.StringIO(_analyze(tmp_path, two_files, "csv", "--equivalence-margin", "0.2"))))
    assert len(rows) == len(payload["comparisons"])
    for comp, row in zip(payload["comparisons"], rows, strict=True):
        assert row["comparison_id"] == comp["comparison_id"]
        assert float(row["point_estimate"]) == comp["point_estimate"]
        assert int(row["sample_count"]) == comp["sample_count"]
        assert float(row["p_value"]) == comp["p_value"]
        assert float(row["adjusted_p_value"]) == comp["adjusted_p_value"]
        assert float(row["ci_lower"]) == comp["confidence_interval"]["lower"]
        assert row["equivalence_familywise"] == comp["equivalence"]["familywise_result"]
        assert fmt(comp["point_estimate"]) in md and fmt_p(comp["adjusted_p_value"]) in md
    assert "not multiplicity-adjusted" in md
    assert "Missing-data policy: `complete_case`" in md
    assert "## Provenance" in md and "do not prove" in md
    assert "equivalent" not in md.lower().replace("equivalence", "")


def test_exploratory_reports_carry_banner_and_no_inference(tmp_path, two_files):
    a, b = two_files
    data = json.loads(b.read_text())
    data["configuration"]["normalization"]["case_sensitive"] = True
    b.write_text(json.dumps(data))
    md = _analyze(tmp_path, two_files, "markdown", "--exploratory", "--exploratory-reason", "quick look")
    assert md.splitlines()[2].startswith("> **EXPLORATORY")
    assert "quick look" in md and "n/a" in md
    rows = list(csv.DictReader(io.StringIO(_analyze(tmp_path, two_files, "csv", "--exploratory", "--exploratory-reason", "quick look"))))
    assert all(r["analysis_mode"] == "exploratory" and r["p_value"] == "" and r["ci_lower"] == "" for r in rows)


def test_secrets_never_reach_reports(tmp_path, two_files):
    a, b = two_files
    data = json.loads(b.read_text())
    data["model"]["inference_config"]["extra_options"] = {"api_key": "sk-supersecretvalue123"}
    data["model"]["inference_config"]["endpoint"] = "https://host/x?token=abcdef123456"
    b.write_text(json.dumps(data))
    for f in ("json", "markdown", "csv"):
        text = _analyze(tmp_path, two_files, f)
        assert "sk-supersecretvalue123" not in text and "abcdef123456" not in text


def test_csv_outputs_neutralise_formulas(tmp_path):
    ds = tmp_path / "evil.jsonl"
    rows = [{"id": "=HYPERLINK(\"http://x\")", "category": "c", "prompt": "p?", "expected_answer": "x"}]
    rows += [{"id": f"r{i}", "category": "c", "prompt": f"question {i}?", "expected_answer": "y"} for i in range(40)]
    ds.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    from openbench.quality import DatasetAuditor

    payload = DatasetAuditor(use_review_notes=False).audit(ds).to_dict()
    payload["findings"].append({**payload["findings"][0], "record_ids": ["=HYPERLINK(\"http://x\")"], "message": "@SUM(A1)"}) if payload["findings"] else None
    text = render_audit(payload, "csv")
    for line in text.splitlines()[1:]:
        for cell in next(csv.reader([line])):
            assert not cell.startswith(("=", "+", "@")) or cell.startswith("'")


def test_v031_csv_report_protects_formula_cells_but_keeps_numbers(tmp_path):
    from openbench import DatasetLoader, EvaluationConfig, Evaluator, ModelResponseSet, ResponseItem
    from openbench.reporting import ReportGenerator

    ds = DatasetLoader.load(str(ROOT / "datasets" / "mathematics" / "sample.jsonl"))
    rs = ModelResponseSet("m", "1", [ResponseItem(id="mathematics_014", response="-9"), ResponseItem(id="mathematics_001", response="=1+1")])
    text = ReportGenerator.to_csv(Evaluator(config=EvaluationConfig(metric_names=["exact_match"])).evaluate(ds, rs))
    cells = [c for row in csv.reader(io.StringIO(text)) for c in row]
    assert "-9" in cells and "'=1+1" in cells and "=1+1" not in cells


def test_variability_and_preference_renderers(tmp_path):
    exp = tmp_path / "exp"
    assert main(["run", "--dataset", str(ROOT / "datasets" / "reasoning" / "sample.jsonl"), "--provider", "mock",
                 "--model", "m", "--repetitions", "3", "--no-cache", "--experiments-dir", str(exp)]) == 0
    group = next(p.stem for p in exp.glob("grp_*.json"))
    for f in ("json", "markdown", "csv"):
        out = tmp_path / f"var.{f}"
        assert main(["analyze", "--runs", group, "--experiments-dir", str(exp), "--format", f, "--output", str(out), "--resamples", "200"]) == 0
    md = (tmp_path / "var.markdown").read_text()
    assert "repeated-run variability" in md and "Question-level CI" in md
    rows = list(csv.DictReader(io.StringIO((tmp_path / "var.csv").read_text())))
    assert len(rows) == 20 * 3 and rows[0]["repetition_statuses"] == "valid valid valid"


def test_old_v031_artifacts_analysable_and_unchanged(tmp_path, monkeypatch):
    exp = tmp_path / "exp"
    shutil.copytree(ROOT / "tests" / "data" / "v031_run_fixture", exp)
    before = snapshot(exp)

    def boom(*a, **k):
        raise AssertionError("no provider may be constructed by analysis or reporting")

    monkeypatch.setattr(MockModelProvider, "__init__", boom)
    for f in ("json", "markdown", "csv"):
        out = tmp_path / f"cmp.{f}"
        assert main(["analyze", "--runs", "run_v031fixturea", "run_v031fixtureb", "--experiments-dir", str(exp),
                     "--format", f, "--output", str(out), *FAST]) == 0
    payload = json.loads((tmp_path / "cmp.json").read_text())
    assert payload["analysis_mode"] == "inferential"
    codes = {e["code"] for e in payload["pairs"][0]["compatibility"]["info"]}
    assert "content_verified_by_dataset_checksum" in codes
    assert all(r["seed_status"] == "unknown" for i in payload["provenance"]["inputs"] for r in i["run_lineage"])
    assert snapshot(exp) == before


def test_analysis_never_written_into_run_directory(tmp_path):
    exp = tmp_path / "exp"
    shutil.copytree(ROOT / "tests" / "data" / "v031_run_fixture", exp)
    before = snapshot(exp)
    code = main(["analyze", "--runs", "run_v031fixturea", "run_v031fixtureb", "--experiments-dir", str(exp),
                 "--output", str(exp / "run_v031fixturea" / "analysis.json"), *FAST])
    assert code == 2 and snapshot(exp) == before


def test_render_rejects_unknown_formats():
    from openbench.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        render_analysis({"analysis": "paired_comparison"}, "xml")


def test_evaluation_markdown_statistics_section_only_with_flag(tmp_path):
    base = ["evaluate", "--dataset", str(ROOT / "datasets" / "reasoning" / "sample.jsonl"), "--responses",
            str(ROOT / "examples" / "sample_responses.json"), "--metrics", "exact_match"]
    assert main([*base, "--output", str(tmp_path / "plain")]) == 0
    assert main([*base, "--output", str(tmp_path / "ci"), "--confidence-intervals", "--resamples", "200"]) == 0
    plain = next((tmp_path / "plain").glob("*.md")).read_text()
    ci = next((tmp_path / "ci").glob("*.md")).read_text()
    assert "Confidence Intervals (v0.4.0" not in plain
    assert "Confidence Intervals (v0.4.0" in ci and "not adjusted for multiple comparisons" in ci
