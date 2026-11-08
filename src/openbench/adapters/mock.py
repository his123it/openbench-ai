"""Deterministic MockModelProvider for offline testing, examples, and reliability simulation."""

from __future__ import annotations

import time
from typing import Any

from openbench.adapters.base import BaseModelAdapter, GenerationConfig, ProviderHealthStatus
from openbench.dataset import DatasetRecord
from openbench.evaluator import ResponseItem


class MockModelProvider(BaseModelAdapter):
    """Deterministic mock provider for tests, examples, and retry/timeout simulation.

    Transparency Guarantee:
    `is_mock = True` and `provider_name = "mock"` are permanently set on this adapter
    and embedded into all experiment and reproducibility records so mock outputs can
    NEVER be mistaken for or presented as real language model results.
    """

    provider_name = "mock"
    description = (
        "Deterministic mock provider for offline tests, CLI examples, and retry/failure simulation. "
        "Never represents a real language model."
    )
    is_experimental = False
    is_mock = True

    def __init__(
        self,
        model_name: str = "mock-deterministic-v1",
        model_version: str = "0.3.0",
        response_map: dict[str, str] | None = None,
        use_expected_answer: bool = True,
        simulated_latency_ms: float = 5.0,
        fail_on_ids: list[str] | None = None,
        transient_failures_by_id: dict[str, int] | None = None,
        timeout_on_ids: list[str] | None = None,
        report_tokens: bool = True,
        generation_config: GenerationConfig | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            generation_config=generation_config,
            timeout_seconds=timeout_seconds,
            inference_config={
                "is_mock": True,
                "use_expected_answer": use_expected_answer,
                "simulated_latency_ms": simulated_latency_ms,
            },
        )
        self.response_map = dict(response_map or {})
        self.use_expected_answer = use_expected_answer
        self.simulated_latency_ms = float(simulated_latency_ms)
        self.fail_on_ids = set(fail_on_ids or [])
        self.transient_failures_remaining: dict[str, int] = dict(transient_failures_by_id or {})
        self.timeout_on_ids = set(timeout_on_ids or [])
        self.report_tokens = report_tokens
        self.call_counts: dict[str, int] = {}

    def generate_one(self, record: DatasetRecord) -> ResponseItem:
        self.call_counts[record.id] = self.call_counts.get(record.id, 0) + 1

        if record.id in self.timeout_on_ids:
            raise TimeoutError(f"Simulated request timeout on record '{record.id}'")

        remaining = self.transient_failures_remaining.get(record.id, 0)
        if remaining > 0:
            self.transient_failures_remaining[record.id] = remaining - 1
            raise ConnectionError(
                f"Simulated transient connection failure on record '{record.id}' "
                f"({remaining - 1} transient failures remaining)"
            )

        if record.id in self.fail_on_ids:
            raise RuntimeError(f"Simulated permanent provider failure on record '{record.id}'")

        if record.id in self.response_map:
            text = self.response_map[record.id]
        elif self.use_expected_answer:
            text = record.expected_answer
        else:
            text = f"mock_response_for_{record.id}"

        token_usage: dict[str, int] | None = None
        if self.report_tokens:
            p_tok = max(1, len(record.prompt.split()))
            c_tok = max(1, len(text.split()))
            token_usage = {
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": p_tok + c_tok,
            }

        if self.simulated_latency_ms > 0 and self.simulated_latency_ms <= 2.0:
            time.sleep(self.simulated_latency_ms / 1000.0)

        return ResponseItem(
            id=record.id,
            response=text,
            latency_ms=self.simulated_latency_ms,
            token_usage=token_usage,
        )

    def check_connection(self) -> ProviderHealthStatus:
        return ProviderHealthStatus(
            provider_name=self.provider_name,
            available=True,
            message="Mock provider is available (offline deterministic mode; not a real model).",
            details={"model_name": self.model_name, "is_mock": True},
        )
