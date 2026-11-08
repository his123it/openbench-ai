"""Unit tests for v0.3.0 model providers: MockModelProvider, OllamaAdapter, HuggingFaceAdapter, and provider registry."""

from __future__ import annotations

import json
from typing import Any

import pytest

from openbench.adapters import (
    GenerationConfig,
    HuggingFaceAdapter,
    MockModelProvider,
    OllamaAdapter,
    OllamaConnectionError,
    create_provider,
    list_registered_providers,
)
from openbench.dataset import DatasetRecord
from openbench.exceptions import ConfigurationError


def _sample_record() -> DatasetRecord:
    return DatasetRecord(
        id="prov_001",
        category="reasoning",
        prompt="What is 15 multiplied by 12?",
        expected_answer="180",
    )


def test_provider_registry_lists_builtin_providers_and_creates_instances() -> None:
    """list_registered_providers lists mock, ollama, huggingface_transformers, local_callable, and local_lookup."""
    providers = list_registered_providers()
    names = {p["provider_name"] for p in providers}
    assert {
        "mock",
        "ollama",
        "huggingface_transformers",
        "local_callable",
        "local_lookup",
    }.issubset(names)

    mock_inst = create_provider("mock", model_name="unit-mock")
    assert mock_inst.is_mock is True
    assert mock_inst.model_name == "unit-mock"

    with pytest.raises(ConfigurationError, match="Unknown model provider"):
        create_provider("non_existent_provider_xyz")


def test_mock_provider_deterministic_and_explicitly_labeled_mock() -> None:
    """MockModelProvider generates deterministic outputs and permanently marks is_mock=True."""
    rec = _sample_record()
    mock = MockModelProvider(
        model_name="mock-v1",
        use_expected_answer=True,
        simulated_latency_ms=1.0,
    )
    resp = mock.generate_one(rec)
    assert resp.id == "prov_001"
    assert resp.response == "180"
    assert resp.token_usage is not None
    assert mock.reproducibility_metadata()["is_mock"] is True
    assert mock.check_connection().available is True


def test_ollama_adapter_non_streaming_and_streaming_with_mock_transport() -> None:
    """OllamaAdapter normalizes both non-streaming JSON and streaming NDJSON responses and extracts tokens."""
    rec = _sample_record()

    # 1. Non-streaming transport
    def mock_transport_sync(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
        if url.endswith("/api/tags"):
            return {"models": [{"name": "llama3.2:latest"}]}
        assert payload is not None
        assert payload["model"] == "llama3.2"
        return {
            "response": "180",
            "done": True,
            "prompt_eval_count": 9,
            "eval_count": 2,
        }

    adapter = OllamaAdapter(
        model_name="llama3.2",
        endpoint="http://localhost:11434",
        transport_fn=mock_transport_sync,
    )
    health = adapter.check_connection()
    assert health.available is True
    assert health.details["model_installed"] is True

    item = adapter.generate_one(rec)
    assert item.response == "180"
    assert item.token_usage == {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}

    # 2. Streaming NDJSON transport
    def mock_transport_stream(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
        return [
            json.dumps({"response": "1", "done": False}),
            json.dumps({"response": "80", "done": True, "prompt_eval_count": 9, "eval_count": 2}),
        ]

    stream_adapter = OllamaAdapter(
        model_name="llama3.2",
        generation_config=GenerationConfig(stream=True),
        transport_fn=mock_transport_stream,
    )
    stream_item = stream_adapter.generate_one(rec)
    assert stream_item.response == "180"
    assert stream_item.token_usage == {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}


def test_ollama_adapter_clear_connection_error_when_server_unreachable() -> None:
    """OllamaAdapter returns available=False on check_connection and raises OllamaConnectionError when server is down."""
    rec = _sample_record()
    # Port 19999 on localhost is unused
    adapter = OllamaAdapter(
        model_name="llama3.2",
        endpoint="http://127.0.0.1:19999",
        timeout_seconds=0.5,
    )
    health = adapter.check_connection()
    assert health.available is False
    assert "127.0.0.1:19999" in health.message

    with pytest.raises(OllamaConnectionError, match="Could not connect to Ollama server"):
        adapter.generate_one(rec)


def test_huggingface_adapter_check_connection_and_no_auto_download() -> None:
    """HuggingFaceAdapter enforces local_files_only=True by default and reports missing dependency cleanly."""
    hf = HuggingFaceAdapter(model_id="gpt2", revision="main")
    assert hf.inference_config["local_files_only"] is True
    health = hf.check_connection()
    assert health.available is False
    assert "transformers" in health.message
