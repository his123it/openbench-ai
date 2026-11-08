"""Unit tests for ExecutionEngine: retries, exponential backoff, timeouts, concurrency, and caching."""

from __future__ import annotations

import threading
from pathlib import Path

from openbench.adapters import GenerationConfig, MockModelProvider
from openbench.cache import ResponseCache
from openbench.dataset import Dataset, DatasetRecord
from openbench.execution import ExecutionConfig, ExecutionEngine, RetryPolicy


def _three_item_dataset() -> Dataset:
    return Dataset(
        name="exec_test_ds",
        source_path="memory://exec_test_ds",
        version="0.3.0",
        records=[
            DatasetRecord(id="e1", category="reasoning", prompt="P1?", expected_answer="A1"),
            DatasetRecord(id="e2", category="reasoning", prompt="P2?", expected_answer="A2"),
            DatasetRecord(id="e3", category="reasoning", prompt="P3?", expected_answer="A3"),
        ],
    )


def test_execution_engine_retries_transient_failures_and_succeeds(tmp_path: Path) -> None:
    """ExecutionEngine retries transient connection errors according to RetryPolicy and records retry counts."""
    ds = _three_item_dataset()
    mock = MockModelProvider(
        model_name="retry-mock",
        transient_failures_by_id={"e2": 2},
        simulated_latency_ms=0.5,
    )
    cfg = ExecutionConfig(
        concurrency=1,
        use_cache=False,
        retry_policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0.005),
    )
    engine = ExecutionEngine(provider=mock, config=cfg)
    outcome = engine.execute_dataset(ds)

    assert len(outcome.completed_record_ids) == 3
    assert len(outcome.failed_record_ids) == 0
    assert outcome.performance_report.retry_count == 2
    assert mock.call_counts["e2"] == 3


def test_execution_engine_handles_timeouts_and_permanent_failures() -> None:
    """ExecutionEngine classifies timeouts and permanent failures without crashing the rest of the batch."""
    ds = _three_item_dataset()
    mock = MockModelProvider(
        model_name="failure-mock",
        timeout_on_ids=["e1"],
        fail_on_ids=["e3"],
        simulated_latency_ms=0.5,
    )
    cfg = ExecutionConfig(
        concurrency=2,
        use_cache=False,
        retry_policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=0.005, retry_on_timeouts=False),
    )
    engine = ExecutionEngine(provider=mock, config=cfg)
    outcome = engine.execute_dataset(ds)

    assert outcome.completed_record_ids == ["e2"]
    assert sorted(outcome.failed_record_ids) == ["e1", "e3"]
    assert outcome.performance_report.timeout_count == 1
    assert outcome.performance_report.failed_requests == 2
    # e1 timed out with retry_on_timeouts=False -> only 1 attempt; e3 failed twice -> 1 retry
    assert mock.call_counts["e1"] == 1
    assert mock.call_counts["e3"] == 2


def test_execution_engine_cache_isolation_and_reuse(tmp_path: Path) -> None:
    """ResponseCache reuses responses for identical configs and never reuses across different models or temperatures."""
    ds = _three_item_dataset()
    cache = ResponseCache(cache_dir=tmp_path / "cache", enabled=True)

    mock1 = MockModelProvider(
        model_name="model-alpha",
        generation_config=GenerationConfig(temperature=0.0),
        simulated_latency_ms=0.5,
    )
    engine1 = ExecutionEngine(provider=mock1, config=ExecutionConfig(use_cache=True), cache=cache)

    # First run: all 3 are cache misses
    out1 = engine1.execute_dataset(ds)
    assert out1.performance_report.cached_requests == 0
    assert mock1.call_counts == {"e1": 1, "e2": 1, "e3": 1}

    # Second run with identical model & generation config: all 3 are cache hits (provider not called again)
    out2 = engine1.execute_dataset(ds)
    assert out2.performance_report.cached_requests == 3
    assert mock1.call_counts == {"e1": 1, "e2": 1, "e3": 1}

    # Third run with different temperature (0.7): cache keys differ -> 0 cache hits
    mock_temp = MockModelProvider(
        model_name="model-alpha",
        generation_config=GenerationConfig(temperature=0.7),
        simulated_latency_ms=0.5,
    )
    engine_temp = ExecutionEngine(provider=mock_temp, config=ExecutionConfig(use_cache=True), cache=cache)
    out3 = engine_temp.execute_dataset(ds)
    assert out3.performance_report.cached_requests == 0
    assert mock_temp.call_counts == {"e1": 1, "e2": 1, "e3": 1}


def test_execution_engine_cancellation_support() -> None:
    """Setting cancel_event stops execution of remaining pending records gracefully."""
    ds = _three_item_dataset()
    cancel_event = threading.Event()
    cancel_event.set()

    mock = MockModelProvider(model_name="cancel-mock")
    engine = ExecutionEngine(provider=mock, config=ExecutionConfig(use_cache=False))
    outcome = engine.execute_dataset(ds, cancel_event=cancel_event)

    assert outcome.was_cancelled is True
    assert len(outcome.completed_record_ids) == 0
