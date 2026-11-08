"""Unit tests for local and optional Hugging Face model adapters and telemetry tracking."""

from __future__ import annotations

import pytest

from openbench.adapters import (
    DeterministicLookupAdapter,
    HuggingFaceAdapter,
    LocalCallableAdapter,
)
from openbench.dataset import Dataset, DatasetRecord
from openbench.evaluator import Evaluator
from openbench.exceptions import ConfigurationError


def _mini_dataset() -> Dataset:
    return Dataset(
        name="adapter_ds",
        source_path="memory://adapter_ds",
        records=[
            DatasetRecord(
                id="a1",
                category="mathematics",
                prompt="Calculate 15 multiplied by 12.",
                expected_answer="180",
            ),
            DatasetRecord(
                id="a2",
                category="mathematics",
                prompt="What is 2^10?",
                expected_answer="1024",
            ),
        ],
    )


def test_local_callable_adapter_and_telemetry_recording() -> None:
    """LocalCallableAdapter generates responses and records supplied latency and token usage."""
    ds = _mini_dataset()

    def mock_model(prompt: str) -> dict:
        ans = "180" if "15" in prompt else "1024"
        return {
            "response": ans,
            "latency_ms": 14.5,
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }

    adapter = LocalCallableAdapter(
        model_name="LocalRuleBot",
        generate_fn=mock_model,
        model_version="0.2",
    )
    resp_set = adapter.generate_responses(ds)
    assert resp_set.model_provider == "local_callable"
    assert len(resp_set.responses) == 2

    result = Evaluator().evaluate(ds, resp_set)
    assert result.aggregate_results["exact_match"].score == 1.0
    assert result.performance_summary["latency_supplied"] is True
    assert result.performance_summary["latency_ms"]["mean"] == 14.5
    assert result.performance_summary["token_usage"]["total_tokens"] == 24


def test_deterministic_lookup_adapter_never_fabricates_latency() -> None:
    """DeterministicLookupAdapter leaves latency_ms and token_usage as None when not measured."""
    ds = _mini_dataset()
    adapter = DeterministicLookupAdapter(
        model_name="LookupBot",
        lookup_by_id={"a1": "180", "a2": "1024"},
    )
    resp_set = adapter.generate_responses(ds)
    result = Evaluator().evaluate(ds, resp_set)
    assert result.performance_summary["latency_supplied"] is False
    assert result.performance_summary["token_usage_supplied"] is False


def test_huggingface_adapter_mock_pipeline_and_missing_dep_handling() -> None:
    """HuggingFaceAdapter works with an injected mock pipeline and raises clear error when transformers is absent."""
    ds = _mini_dataset()

    # 1. With mock pipeline
    def mock_pipe(prompt, **kwargs):
        return [{"generated_text": "180"}]
    hf_mock = HuggingFaceAdapter(
        model_id="org/test-tiny-llm",
        revision="v1.0",
        pipeline_fn=mock_pipe,
    )
    resp_set = hf_mock.generate_responses(ds)
    assert resp_set.model_provider == "huggingface_transformers"
    assert resp_set.inference_config["local_files_only"] is True
    assert len(resp_set.responses) == 2

    # 2. Without mock pipeline when transformers is not installed
    hf_uninstalled = HuggingFaceAdapter(model_id="org/test-tiny-llm")
    with pytest.raises(ConfigurationError, match="transformers"):
        hf_uninstalled.generate_responses(ds)
