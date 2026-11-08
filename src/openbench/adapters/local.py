"""Local model adapters for custom Python callables and deterministic replay."""

from __future__ import annotations

import time
from typing import Any, Callable

from openbench.adapters.base import BaseModelAdapter
from openbench.dataset import DatasetRecord
from openbench.evaluator import ResponseItem
from openbench.exceptions import ConfigurationError


class LocalCallableAdapter(BaseModelAdapter):
    """Adapter that wraps any local Python function `fn(prompt: str) -> str | dict`.

    Allows researchers and students to connect local inference engines (e.g., a custom
    PyTorch model, llama.cpp binding, or rule-based baseline) directly to OpenBench AI
    without network dependencies.
    """

    provider_name = "local_callable"
    is_experimental = False

    def __init__(
        self,
        model_name: str,
        generate_fn: Callable[[str], str | dict[str, Any]],
        model_version: str = "1.0",
        measure_latency: bool = True,
        inference_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            inference_config=inference_config,
        )
        if not callable(generate_fn):
            raise ConfigurationError("generate_fn must be a callable accepting a prompt string")
        self._generate_fn = generate_fn
        self.measure_latency = measure_latency

    def generate_one(self, record: DatasetRecord) -> ResponseItem:
        start = time.perf_counter()
        raw_out = self._generate_fn(record.prompt)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if isinstance(raw_out, dict):
            text = str(raw_out.get("response", ""))
            latency = raw_out.get("latency_ms", elapsed_ms if self.measure_latency else None)
            token_usage = raw_out.get("token_usage")
        elif isinstance(raw_out, str):
            text = raw_out
            latency = elapsed_ms if self.measure_latency else None
            token_usage = None
        else:
            raise ConfigurationError(
                f"LocalCallableAdapter function must return str or dict, got {type(raw_out).__name__}"
            )

        return ResponseItem(
            id=record.id,
            response=text,
            latency_ms=float(latency) if latency is not None else None,
            token_usage=token_usage,
        )


class DeterministicLookupAdapter(BaseModelAdapter):
    """Deterministic local adapter backed by an in-memory lookup dictionary for testing."""

    provider_name = "local_lookup"
    is_experimental = False

    def __init__(
        self,
        model_name: str,
        lookup_by_id: dict[str, str],
        default_response: str = "",
        model_version: str = "1.0",
    ) -> None:
        super().__init__(model_name=model_name, model_version=model_version)
        self.lookup_by_id = dict(lookup_by_id)
        self.default_response = default_response

    def generate_one(self, record: DatasetRecord) -> ResponseItem:
        text = self.lookup_by_id.get(record.id, self.default_response)
        return ResponseItem(id=record.id, response=text)
