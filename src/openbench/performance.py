"""Performance monitoring, latency percentiles, token accounting, and optional cost estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from openbench.exceptions import ConfigurationError


def compute_percentile(sorted_values: list[float], percentile: float) -> float:
    """Compute a percentile (0..100) from a non-empty sorted list of floats using linear interpolation."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return float(sorted_values[lower])
    weight = pos - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


@dataclass(frozen=True)
class PricingConfig:
    """Optional user-supplied pricing configuration for estimated token cost calculation.

    Transparency Guarantee:
    OpenBench AI NEVER hardcodes claims about current commercial provider prices.
    Cost estimation is computed ONLY when the caller explicitly provides `PricingConfig`.
    """

    prompt_cost_per_1k_tokens: float
    completion_cost_per_1k_tokens: float
    currency: str = "USD"

    def __post_init__(self) -> None:
        if self.prompt_cost_per_1k_tokens < 0 or self.completion_cost_per_1k_tokens < 0:
            raise ConfigurationError("Token pricing rates must be non-negative")

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
        prompt_cost = (prompt_tokens / 1000.0) * self.prompt_cost_per_1k_tokens
        completion_cost = (completion_tokens / 1000.0) * self.completion_cost_per_1k_tokens
        return {
            "currency": self.currency,
            "prompt_cost_per_1k_tokens": self.prompt_cost_per_1k_tokens,
            "completion_cost_per_1k_tokens": self.completion_cost_per_1k_tokens,
            "estimated_prompt_cost": round(prompt_cost, 6),
            "estimated_completion_cost": round(completion_cost, 6),
            "estimated_total_cost": round(prompt_cost + completion_cost, 6),
            "note": "Estimated cost calculated using user-supplied pricing rates; not a provider invoice.",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_cost_per_1k_tokens": self.prompt_cost_per_1k_tokens,
            "completion_cost_per_1k_tokens": self.completion_cost_per_1k_tokens,
            "currency": self.currency,
        }


@dataclass(frozen=True)
class RequestTelemetry:
    """Telemetry record for a single prompt execution attempt/item."""

    record_id: str
    category: str
    difficulty: str
    status: str  # "completed" | "cached" | "failed" | "timeout"
    latency_ms: float | None = None
    attempts: int = 1
    retries: int = 0
    timed_out: bool = False
    cache_hit: bool = False
    error_type: str | None = None
    error_message: str | None = None
    token_source: str = "unavailable"  # "provider_reported" | "locally_estimated" | "unavailable"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 3) if self.latency_ms is not None else None,
            "attempts": self.attempts,
            "retries": self.retries,
            "timed_out": self.timed_out,
            "cache_hit": self.cache_hit,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "token_source": self.token_source,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class PerformanceReport:
    """Structured performance report for a model execution or evaluation run."""

    total_duration_ms: float | None
    total_requests: int
    completed_requests: int
    cached_requests: int
    failed_requests: int
    timeout_count: int
    retry_count: int
    latency_summary: dict[str, Any]
    throughput_requests_per_sec: float | None
    token_usage_summary: dict[str, Any]
    estimated_cost: dict[str, Any] | None
    per_request_telemetry: list[RequestTelemetry] = field(default_factory=list)
    category_latency: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_duration_ms": (
                round(self.total_duration_ms, 3)
                if self.total_duration_ms is not None
                else None
            ),
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "cached_requests": self.cached_requests,
            "failed_requests": self.failed_requests,
            "timeout_count": self.timeout_count,
            "retry_count": self.retry_count,
            "latency_summary": dict(self.latency_summary),
            "throughput_requests_per_sec": (
                round(self.throughput_requests_per_sec, 4)
                if self.throughput_requests_per_sec is not None
                else None
            ),
            "token_usage_summary": dict(self.token_usage_summary),
            "estimated_cost": dict(self.estimated_cost) if self.estimated_cost else None,
            "category_latency": dict(self.category_latency),
            "per_request_telemetry": [t.to_dict() for t in self.per_request_telemetry],
        }


class PerformanceMonitor:
    """Aggregates request telemetry into a transparent, non-fabricated PerformanceReport.

    Measurement Integrity Rules:
    - Never fabricates latency or token counts when not supplied.
    - Explicitly distinguishes `provider_reported` token usage, `locally_estimated` token
      usage (only when `estimate_missing_tokens=True` is explicitly enabled), and
      `unavailable` token usage.
    """

    @staticmethod
    def summarize(
        telemetry_items: list[RequestTelemetry],
        total_duration_ms: float | None = None,
        pricing: PricingConfig | None = None,
    ) -> PerformanceReport:
        total_requests = len(telemetry_items)
        completed = sum(1 for t in telemetry_items if t.status in {"completed", "cached"})
        cached = sum(1 for t in telemetry_items if t.cache_hit)
        failed = sum(1 for t in telemetry_items if t.status in {"failed", "timeout"})
        timeouts = sum(1 for t in telemetry_items if t.timed_out)
        retries = sum(t.retries for t in telemetry_items)

        latencies = sorted(
            t.latency_ms for t in telemetry_items if t.latency_ms is not None
        )
        if latencies:
            latency_summary: dict[str, Any] = {
                "available": True,
                "measured_requests": len(latencies),
                "min_ms": round(latencies[0], 3),
                "max_ms": round(latencies[-1], 3),
                "average_ms": round(sum(latencies) / len(latencies), 3),
                "median_ms": round(compute_percentile(latencies, 50.0), 3),
                "p95_ms": round(compute_percentile(latencies, 95.0), 3),
                "p99_ms": round(compute_percentile(latencies, 99.0), 3),
            }
        else:
            latency_summary = {
                "available": False,
                "measured_requests": 0,
                "note": "No per-request latency measurements were supplied.",
            }

        cat_lat_lists: dict[str, list[float]] = {}
        for t in telemetry_items:
            if t.latency_ms is not None:
                cat_lat_lists.setdefault(t.category, []).append(t.latency_ms)

        category_latency: dict[str, dict[str, float]] = {}
        for cat, vals in sorted(cat_lat_lists.items()):
            s_vals = sorted(vals)
            category_latency[cat] = {
                "count": len(s_vals),
                "average_ms": round(sum(s_vals) / len(s_vals), 3),
                "median_ms": round(compute_percentile(s_vals, 50.0), 3),
                "p95_ms": round(compute_percentile(s_vals, 95.0), 3),
            }

        throughput: float | None = None
        if total_duration_ms is not None and total_duration_ms > 0 and completed > 0:
            throughput = completed / (total_duration_ms / 1000.0)

        provider_reported_items = [
            t for t in telemetry_items if t.token_source == "provider_reported" and t.total_tokens is not None
        ]
        estimated_items = [
            t for t in telemetry_items if t.token_source == "locally_estimated" and t.total_tokens is not None
        ]
        unavailable_count = total_requests - len(provider_reported_items) - len(estimated_items)

        total_prompt_tok = sum((t.prompt_tokens or 0) for t in provider_reported_items + estimated_items)
        total_comp_tok = sum((t.completion_tokens or 0) for t in provider_reported_items + estimated_items)
        total_tok = sum((t.total_tokens or 0) for t in provider_reported_items + estimated_items)

        if provider_reported_items and not estimated_items:
            overall_source = "provider_reported"
        elif estimated_items and not provider_reported_items:
            overall_source = "locally_estimated"
        elif provider_reported_items and estimated_items:
            overall_source = "mixed_provider_and_estimated"
        else:
            overall_source = "unavailable"

        tokens_per_sec: float | None = None
        if total_tok > 0 and total_duration_ms is not None and total_duration_ms > 0:
            tokens_per_sec = round(total_tok / (total_duration_ms / 1000.0), 3)

        token_usage_summary: dict[str, Any] = {
            "source": overall_source,
            "provider_reported_count": len(provider_reported_items),
            "locally_estimated_count": len(estimated_items),
            "unavailable_count": unavailable_count,
            "total_prompt_tokens": total_prompt_tok if overall_source != "unavailable" else None,
            "total_completion_tokens": total_comp_tok if overall_source != "unavailable" else None,
            "total_tokens": total_tok if overall_source != "unavailable" else None,
            "tokens_per_second": tokens_per_sec,
        }
        if overall_source in {"locally_estimated", "mixed_provider_and_estimated"}:
            token_usage_summary["estimation_disclaimer"] = (
                "Contains locally estimated token counts (~1.3 tokens/word heuristic); "
                "estimates are NOT exact tokenizer measurements."
            )

        estimated_cost: dict[str, Any] | None = None
        if pricing is not None and overall_source != "unavailable":
            estimated_cost = pricing.estimate_cost(total_prompt_tok, total_comp_tok)
            estimated_cost["token_source"] = overall_source

        return PerformanceReport(
            total_duration_ms=total_duration_ms,
            total_requests=total_requests,
            completed_requests=completed,
            cached_requests=cached,
            failed_requests=failed,
            timeout_count=timeouts,
            retry_count=retries,
            latency_summary=latency_summary,
            throughput_requests_per_sec=throughput,
            token_usage_summary=token_usage_summary,
            estimated_cost=estimated_cost,
            per_request_telemetry=list(telemetry_items),
            category_latency=category_latency,
        )
