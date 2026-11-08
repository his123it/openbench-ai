"""Unit tests for local evaluation result history store and historical comparisons."""

from __future__ import annotations

from pathlib import Path

import pytest

from openbench.comparison import ModelComparator
from openbench.dataset import Dataset, DatasetRecord
from openbench.evaluator import Evaluator, ModelResponseSet, ResponseItem
from openbench.history import ResultHistoryStore


def test_result_history_store_record_list_and_show(tmp_path: Path) -> None:
    """ResultHistoryStore records runs to JSONL, filters by model/dataset, and retrieves by ID."""
    ds = Dataset(
        name="hist_ds",
        source_path="memory://hist_ds",
        records=[
            DatasetRecord(
                id="h1",
                category="reasoning",
                prompt="2 + 2?",
                expected_answer="4",
            ),
            DatasetRecord(
                id="h2",
                category="reasoning",
                prompt="3 + 3?",
                expected_answer="6",
            ),
        ],
    )
    evaluator = Evaluator()
    run1 = evaluator.evaluate(
        ds,
        ModelResponseSet(
            model_name="TrackedModel",
            model_version="1.0",
            responses=[
                ResponseItem(id="h1", response="4"),
                ResponseItem(id="h2", response="0"),
            ],
        ),
    )
    run2 = evaluator.evaluate(
        ds,
        ModelResponseSet(
            model_name="TrackedModel",
            model_version="1.1",
            responses=[
                ResponseItem(id="h1", response="4"),
                ResponseItem(id="h2", response="6"),
            ],
        ),
    )

    store_path = tmp_path / "history.jsonl"
    store = ResultHistoryStore(store_path)
    store.record(run1, report_path=tmp_path / "run1.json")
    store.record(run2, report_path=tmp_path / "run2.json")

    entries = store.list_entries(model_name="TrackedModel")
    assert len(entries) == 2
    fetched = store.get_entry(run1.evaluation_id)
    assert fetched.evaluation_id == run1.evaluation_id
    assert fetched.result_summary["exact_match"] == 0.5

    with pytest.raises(KeyError):
        store.get_entry("eval_nonexistent")

    hist_cmp = ModelComparator.compare_history([run1, run2])
    assert hist_cmp["total_runs_compared"] == 2
    assert hist_cmp["metric_deltas"]["exact_match"]["delta"] == 0.5
