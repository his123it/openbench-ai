"""Repeated-run variability decomposition (plan B5, sections 1.8 and 3.4)."""

from __future__ import annotations

import statistics

import pytest

from _v040_helpers import write_dataset
from openbench import DatasetLoader, EvaluationConfig, Evaluator, ModelResponseSet, ResponseItem
from openbench.stats import decompose_variability


def _evals(tmp_path, matrix, metric="exact_match", missing=()):
    """matrix[item][rep] in {0, 1}: 1 = correct answer."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    ds = DatasetLoader.load(str(write_dataset(tmp_path / "ds.jsonl", len(matrix))))
    evals = []
    for r in range(len(matrix[0])):
        items = []
        for i, row in enumerate(matrix, start=1):
            if (i, r) in missing:
                continue
            items.append(ResponseItem(id=f"item_{i:03d}", response=f"ans{i}" if row[r] else "nope", latency_ms=10.0 + r))
        rs = ModelResponseSet("m", "1", items, model_provider="local_lookup")
        evals.append(Evaluator(config=EvaluationConfig(metric_names=[metric])).evaluate(ds, rs))
    return evals


MATRIX = [[1, 1, 1], [1, 0, 1], [0, 0, 0], [0, 1, 1]]


def test_hand_4x3_binary_matrix(tmp_path):
    rep = decompose_variability(_evals(tmp_path, MATRIX), resamples=500)
    m = rep["metrics"]["exact_match"]
    gen = m["generation_variability"]
    assert gen["within_item_variance_mean"] == pytest.approx(
        statistics.mean([statistics.variance(row) for row in MATRIX])
    )
    assert gen["within_item_variance_mean"] == pytest.approx(1 / 6)
    assert gen["items_with_any_change_rate"] == 0.5
    assert gen["binary_flip_rate"] == 0.5
    assert gen["mean_pairwise_agreement"] == pytest.approx(2 / 3)
    item_means = [statistics.mean(row) for row in MATRIX]
    assert m["item_variation"]["mean_of_item_means"] == pytest.approx(statistics.mean(item_means))
    assert m["item_variation"]["item_spread_sd"] == pytest.approx(statistics.stdev(item_means))
    run_scores = [statistics.mean(col) for col in zip(*MATRIX, strict=True)]
    assert m["run_distribution"]["headline_scores"] == pytest.approx(run_scores)
    assert m["run_distribution"]["headline_summary"]["sd"] == pytest.approx(statistics.stdev(run_scores))
    ci = m["item_variation"]["question_level_confidence_interval"]
    assert ci["sample_count"] == 4  # items, never items x repetitions
    per_item = {e["id"]: e for e in m["per_item"]}
    assert per_item["item_002"]["changed"] is True and per_item["item_001"]["changed"] is False
    assert rep["response_identity"]["exact_response_identity_rate"] == 0.5
    assert rep["latency_and_tokens"]["latency_ms"]["per_repetition_mean"] == [10.0, 11.0, 12.0]


def test_repetition_noise_does_not_change_item_level_ci(tmp_path):
    shuffled = [[1, 1, 1], [0, 1, 1], [0, 0, 0], [1, 1, 0]]  # same per-item means as MATRIX
    a = decompose_variability(_evals(tmp_path / "a", MATRIX), resamples=800, analysis_seed=5)
    b = decompose_variability(_evals(tmp_path / "b", shuffled), resamples=800, analysis_seed=5)
    ci_a = a["metrics"]["exact_match"]["item_variation"]["question_level_confidence_interval"]
    ci_b = b["metrics"]["exact_match"]["item_variation"]["question_level_confidence_interval"]
    assert ci_a == ci_b
    assert a["metrics"]["exact_match"]["generation_variability"]["within_item_variance_mean"] == pytest.approx(
        b["metrics"]["exact_match"]["generation_variability"]["within_item_variance_mean"]
    )


def test_failures_are_separate_and_excluded_from_item_means(tmp_path):
    evals = _evals(tmp_path, MATRIX, missing={(3, 0), (3, 1), (3, 2), (2, 1)})
    rep = decompose_variability(evals, resamples=300)
    fail = rep["failures"]
    assert [p["missing"] for p in fail["per_repetition"]] == [1, 2, 1]
    assert fail["items_failing_in_every_repetition"] == ["item_003"]
    assert fail["items_failing_in_some_repetitions"] == ["item_002"]
    m = rep["metrics"]["exact_match"]
    assert m["items_excluded_no_valid_repetition"] == 1
    per_item = {e["id"]: e for e in m["per_item"]}
    assert per_item["item_002"]["repetitions_used"] == 2
    assert per_item["item_002"]["mean"] == 1.0


def test_breakdowns_and_continuous_metric(tmp_path):
    rep = decompose_variability(_evals(tmp_path, MATRIX, metric="keyword_coverage"), resamples=300)
    m = rep["metrics"]["keyword_coverage"]
    assert m["score_kind"] == "bounded_continuous"
    assert "binary_flip_rate" not in m["generation_variability"]
    assert set(m["by_category"]) == {"alpha", "beta"}
    assert sum(v["items"] for v in m["by_difficulty"].values()) == 4


def test_deterministic_and_requires_input(tmp_path):
    evals = _evals(tmp_path, MATRIX)
    assert decompose_variability(evals, resamples=300) == decompose_variability(evals, resamples=300)
    with pytest.raises(ValueError):
        decompose_variability([])
