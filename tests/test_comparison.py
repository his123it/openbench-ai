"""Unit tests for OpenBench AI model comparison module."""

from __future__ import annotations

import pytest

from openbench.comparison import ModelComparator
from openbench.dataset import Dataset, DatasetRecord
from openbench.evaluator import Evaluator, ModelResponseSet, ResponseItem
from openbench.exceptions import ConfigurationError


def _build_eval_results():
    ds = Dataset(
        name="cmp_ds",
        source_path="memory://cmp_ds",
        records=[
            DatasetRecord(
                id="c1",
                category="reasoning",
                prompt="What is 5 * 5?",
                expected_answer="25",
                metadata={"difficulty": "easy", "keywords": ["25"]},
            ),
            DatasetRecord(
                id="c2",
                category="reasoning",
                prompt="What is 8 * 8?",
                expected_answer="64",
                metadata={"difficulty": "easy", "keywords": ["64"]},
            ),
        ],
    )
    evaluator = Evaluator()
    res_a = evaluator.evaluate(
        ds,
        ModelResponseSet(
            model_name="StrictBot",
            model_version="1.0",
            responses=[
                ResponseItem(id="c1", response="25"),
                ResponseItem(id="c2", response="0"),
            ],
        ),
    )
    res_b = evaluator.evaluate(
        ds,
        ModelResponseSet(
            model_name="ChattyBot",
            model_version="2.0",
            responses=[
                ResponseItem(id="c1", response="The result is 25."),
                ResponseItem(id="c2", response="The result is 64."),
            ],
        ),
    )
    return res_a, res_b


def test_model_comparator_identifies_per_metric_leaders() -> None:
    """ModelComparator identifies StrictBot for exact_match and ChattyBot for accuracy/keyword_coverage."""
    res_a, res_b = _build_eval_results()
    comparison = ModelComparator.compare([res_a, res_b])

    assert len(comparison.models) == 2
    # StrictBot has 0.5 exact_match vs ChattyBot 0.0
    assert comparison.highest_by_metric["exact_match"].highest_score == 0.5
    assert comparison.highest_by_metric["exact_match"].leading_models == ["StrictBot (v1.0)"]

    # ChattyBot has 1.0 accuracy and 1.0 keyword_coverage vs StrictBot 0.5
    assert comparison.highest_by_metric["accuracy"].highest_score == 1.0
    assert comparison.highest_by_metric["accuracy"].leading_models == ["ChattyBot (v2.0)"]
    assert comparison.highest_by_metric["keyword_coverage"].highest_score == 1.0
    assert comparison.highest_by_metric["keyword_coverage"].leading_models == ["ChattyBot (v2.0)"]


def test_model_comparator_handles_ties_accurately() -> None:
    """When two models achieve the same top score on a metric, both are listed as leaders."""
    res_a, _ = _build_eval_results()
    comparison = ModelComparator.compare([res_a, res_a])
    leader = comparison.highest_by_metric["exact_match"]
    assert leader.to_dict()["is_tie"] is True
    assert len(leader.leading_models) == 2


def test_model_comparator_requires_at_least_two_results() -> None:
    """Passing fewer than two results to ModelComparator.compare raises ConfigurationError."""
    res_a, _ = _build_eval_results()
    with pytest.raises(ConfigurationError, match="at least 2"):
        ModelComparator.compare([res_a])
