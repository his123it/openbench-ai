"""Unit tests for PerformanceMonitor (percentiles, token sources, pricing) and BenchmarkAnalytics."""

from __future__ import annotations

import pytest

from openbench.analytics import BenchmarkAnalytics
from openbench.dataset import Dataset, DatasetRecord
from openbench.evaluator import Evaluator, ModelResponseSet, ResponseItem
from openbench.performance import PerformanceMonitor, PricingConfig, RequestTelemetry


def test_performance_monitor_latency_percentiles_and_token_source_distinction() -> None:
    """PerformanceMonitor computes P50/P95/P99 latency and distinguishes provider vs estimated vs unavailable tokens."""
    items = [
        RequestTelemetry(
            record_id="r1",
            category="math",
            difficulty="easy",
            status="completed",
            latency_ms=10.0,
            token_source="provider_reported",
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
        ),
        RequestTelemetry(
            record_id="r2",
            category="math",
            difficulty="medium",
            status="completed",
            latency_ms=20.0,
            token_source="provider_reported",
            prompt_tokens=30,
            completion_tokens=15,
            total_tokens=45,
        ),
        RequestTelemetry(
            record_id="r3",
            category="math",
            difficulty="hard",
            status="completed",
            latency_ms=100.0,
            token_source="provider_reported",
            prompt_tokens=50,
            completion_tokens=25,
            total_tokens=75,
        ),
    ]
    pricing = PricingConfig(
        prompt_cost_per_1k_tokens=0.01,
        completion_cost_per_1k_tokens=0.02,
        currency="USD",
    )
    report = PerformanceMonitor.summarize(items, total_duration_ms=200.0, pricing=pricing)

    assert report.latency_summary["available"] is True
    assert report.latency_summary["median_ms"] == 20.0
    assert report.latency_summary["p95_ms"] > 80.0
    assert report.token_usage_summary["source"] == "provider_reported"
    assert report.token_usage_summary["total_tokens"] == 150
    assert report.estimated_cost is not None
    # 100 prompt tokens @ $0.01/1k = $0.001; 50 completion tokens @ $0.02/1k = $0.001 -> total $0.002
    assert pytest.approx(report.estimated_cost["estimated_total_cost"], rel=1e-5) == 0.002


def test_performance_monitor_unavailable_tokens_never_fabricated() -> None:
    """When token usage is not supplied, source is 'unavailable' and counts are None."""
    items = [
        RequestTelemetry(
            record_id="r1",
            category="coding",
            difficulty="easy",
            status="completed",
            latency_ms=15.0,
            token_source="unavailable",
        )
    ]
    report = PerformanceMonitor.summarize(items)
    assert report.token_usage_summary["source"] == "unavailable"
    assert report.token_usage_summary["total_tokens"] is None


def test_benchmark_analytics_repeated_runs_variance_and_disagreements() -> None:
    """BenchmarkAnalytics computes sample variance across repeated runs and identifies item disagreements."""
    ds = Dataset(
        name="analytics_ds",
        source_path="memory://analytics_ds",
        records=[
            DatasetRecord(id="q1", category="reasoning", prompt="P1", expected_answer="10"),
            DatasetRecord(id="q2", category="reasoning", prompt="P2", expected_answer="20"),
        ],
    )
    evaluator = Evaluator()
    run_a = evaluator.evaluate(
        ds,
        ModelResponseSet(
            model_name="ModelX",
            model_version="1.0",
            responses=[
                ResponseItem(id="q1", response="10"),
                ResponseItem(id="q2", response="20"),
            ],
        ),
    )
    run_b = evaluator.evaluate(
        ds,
        ModelResponseSet(
            model_name="ModelX",
            model_version="1.0",
            responses=[
                ResponseItem(id="q1", response="10"),
                ResponseItem(id="q2", response="wrong"),
            ],
        ),
    )

    rep_summary = BenchmarkAnalytics.analyze_repeated_runs([run_a, run_b])
    assert rep_summary["sample_size_n"] == 2
    assert rep_summary["consistent_items_count"] == 1
    assert rep_summary["varying_item_ids"] == ["q2"]
    assert rep_summary["metric_variance_summary"]["exact_match"]["mean_score"] == 0.75

    disagreements = BenchmarkAnalytics.analyze_disagreements([run_a, run_b], metric_name="exact_match")
    assert disagreements["disagreement_count"] == 1
    assert disagreements["disagreement_items"][0]["id"] == "q2"
