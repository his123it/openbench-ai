"""Stage 1: score_kind declarations on metrics (plan section 1.4, F5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from openbench import DatasetLoader
from openbench.metrics import METRIC_REGISTRY, BaseMetric, MetricResult, create_metric

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_KINDS = {
    "exact_match": "binary",
    "accuracy": "binary",
    "keyword_coverage": "bounded_continuous",
    "keyword_match": "bounded_continuous",
    "precision": "bounded_continuous",
    "recall": "bounded_continuous",
    "f1": "bounded_continuous",
    "rouge": "bounded_continuous",
    "rouge_1": "bounded_continuous",
    "rouge_l": "bounded_continuous",
    "bleu": "bounded_continuous",
    "semantic_similarity": "bounded_continuous",
    "lexical_similarity": "bounded_continuous",
    "structured_output": "bounded_continuous",
}


def test_registry_kinds() -> None:
    # Other tests may register custom metrics, so check the built-ins and validity of all.
    for name, kind in EXPECTED_KINDS.items():
        assert METRIC_REGISTRY[name].score_kind == kind, name
    for name, cls in METRIC_REGISTRY.items():
        assert cls.score_kind in {"binary", "bounded_continuous", "unknown"}, name


def test_custom_metric_defaults_to_unknown() -> None:
    class Custom(BaseMetric):
        name = "custom_kind_probe"

        def compute_item(self, record, response_text):  # pragma: no cover - not called
            return MetricResult(self.name, 0.0, False)

    assert Custom.score_kind == "unknown"
    assert BaseMetric.score_kind == "unknown"


def test_metadata_schema_unchanged() -> None:
    """score_kind is a class attribute only; metadata dictionaries keep their v0.3.1 keys."""
    keys = set(METRIC_REGISTRY["exact_match"].get_metadata().to_dict())
    assert keys == {
        "name",
        "version",
        "description",
        "limitation_note",
        "is_optional",
        "optional_dependencies",
    }


@pytest.mark.parametrize("metric_name", sorted(set(EXPECTED_KINDS) - {"semantic_similarity"}))
def test_declared_kind_matches_scores_on_builtin_datasets(metric_name: str) -> None:
    """Binary metrics only ever score 0.0 or 1.0; bounded metrics stay within [0, 1]."""
    metric = create_metric(metric_name)
    dataset = DatasetLoader.load(str(ROOT / "datasets"))
    responses = ["", "The answer is 42.", '{"a": 1}']
    for record in dataset.records:
        for response in [record.expected_answer, *responses]:
            score = metric.compute_item(record, response).score
            if metric.score_kind == "binary":
                assert score in (0.0, 1.0), (metric_name, record.id, score)
            else:
                assert 0.0 <= score <= 1.0, (metric_name, record.id, score)
