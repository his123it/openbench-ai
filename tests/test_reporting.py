"""Unit tests for OpenBench AI JSON, Markdown, and Reproducibility Card reporting."""

from __future__ import annotations

import json
from pathlib import Path

from openbench.comparison import ModelComparator
from openbench.dataset import Dataset, DatasetRecord
from openbench.evaluator import EvaluationResult, Evaluator, ModelResponseSet, ResponseItem
from openbench.reporting import ReportGenerator


def _create_sample_result() -> EvaluationResult:
    ds = Dataset(
        name="report_ds",
        source_path="datasets/reasoning/sample.jsonl",
        records=[
            DatasetRecord(
                id="r1",
                category="reasoning",
                prompt="12 * 12?",
                expected_answer="144",
                metadata={"difficulty": "easy", "keywords": ["144"]},
            )
        ],
    )
    rs = ModelResponseSet(
        model_name="ReportModel",
        model_version="1.2",
        responses=[ResponseItem(id="r1", response="144")],
    )
    return Evaluator().evaluate(ds, rs)


def test_json_report_serialization_and_roundtrip(tmp_path: Path) -> None:
    """JSON report includes reproducibility card and can be deserialized via EvaluationResult.from_dict."""
    result = _create_sample_result()
    json_path = ReportGenerator.save_json(result, tmp_path / "report.json")

    assert json_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["project_version"] == "0.1.0"
    assert loaded["model"]["model_name"] == "ReportModel"
    assert "reproducibility_card" in loaded
    assert loaded["reproducibility_card"]["number_of_test_cases"] == 1

    roundtripped = EvaluationResult.from_dict(loaded)
    assert roundtripped.model_name == "ReportModel"
    assert roundtripped.aggregate_results["exact_match"].score == 1.0


def test_markdown_report_contains_reproducibility_card_and_sections(tmp_path: Path) -> None:
    """Markdown report contains summary, metric table, limitations, and Benchmark Reproducibility Card."""
    result = _create_sample_result()
    md_path = ReportGenerator.save_markdown(result, tmp_path / "report.md")

    text = md_path.read_text(encoding="utf-8")
    assert "# OpenBench AI Evaluation Report: ReportModel (v1.2)" in text
    assert "## Evaluation Summary" in text
    assert "## Metric Results" in text
    assert "## Limitations" in text
    assert "## Benchmark Reproducibility Card" in text
    assert "`exact_match`" in text


def test_comparison_markdown_report_generation(tmp_path: Path) -> None:
    """Comparison Markdown report renders model comparison table and per-metric leaders."""
    res1 = _create_sample_result()
    res2 = _create_sample_result()
    comparison = ModelComparator.compare([res1, res2])

    cmp_md = ReportGenerator.save_markdown(comparison, tmp_path / "comparison.md")
    content = cmp_md.read_text(encoding="utf-8")
    assert "# OpenBench AI Model Comparison Report" in content
    assert "## Model Comparison Table" in content
    assert "## Highest Score by Metric" in content
