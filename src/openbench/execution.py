"""Provider-independent Model Execution Engine with retry, backoff, timeout, concurrency, and caching."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from openbench.adapters.base import BaseModelAdapter
from openbench.cache import ResponseCache
from openbench.dataset import Dataset, DatasetRecord
from openbench.evaluator import ModelResponseSet, ResponseItem
from openbench.exceptions import ConfigurationError
from openbench.performance import (
    PerformanceMonitor,
    PerformanceReport,
    PricingConfig,
    RequestTelemetry,
)
from openbench.security import redact_prompt_if_private, sanitize_error_message


@dataclass(frozen=True)
class RetryPolicy:
    """Configurable retry and exponential backoff policy for model execution."""

    max_attempts: int = 3
    initial_backoff_seconds: float = 0.05
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 2.0
    retry_on_timeouts: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ConfigurationError("RetryPolicy.max_attempts must be at least 1")
        if self.initial_backoff_seconds < 0 or self.backoff_multiplier < 1.0:
            raise ConfigurationError("Invalid backoff configuration in RetryPolicy")

    def compute_delay(self, attempt_number: int) -> float:
        """Compute exponential backoff delay in seconds before the next retry (`attempt_number` >= 1)."""
        if self.initial_backoff_seconds <= 0:
            return 0.0
        raw = self.initial_backoff_seconds * (self.backoff_multiplier ** (attempt_number - 1))
        return min(raw, self.max_backoff_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "initial_backoff_seconds": self.initial_backoff_seconds,
            "backoff_multiplier": self.backoff_multiplier,
            "max_backoff_seconds": self.max_backoff_seconds,
            "retry_on_timeouts": self.retry_on_timeouts,
        }


@dataclass(frozen=True)
class ExecutionConfig:
    """Configuration for the provider-independent ExecutionEngine."""

    timeout_seconds: float = 30.0
    concurrency: int = 1
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    use_cache: bool = True
    estimate_missing_tokens: bool = False
    privacy_mode: bool = False
    pricing: PricingConfig | None = None
    request_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ConfigurationError("ExecutionConfig.timeout_seconds must be positive")
        if self.concurrency < 1:
            raise ConfigurationError("ExecutionConfig.concurrency must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "concurrency": self.concurrency,
            "retry_policy": self.retry_policy.to_dict(),
            "use_cache": self.use_cache,
            "estimate_missing_tokens": self.estimate_missing_tokens,
            "privacy_mode": self.privacy_mode,
            "pricing": self.pricing.to_dict() if self.pricing else None,
            "request_metadata": dict(self.request_metadata),
        }


def classify_execution_error(exc: BaseException) -> str:
    """Classify an exception raised during model generation into a standard failure category."""
    msg = str(exc).lower()
    if isinstance(exc, TimeoutError) or "timed out" in msg or "timeout" in msg:
        return "timeout"
    if "rate limit" in msg or "429" in msg:
        return "rate_limit"
    if isinstance(exc, ConnectionError) or "connect" in msg or "unreachable" in msg:
        return "connection_error"
    if isinstance(exc, (ValueError, ConfigurationError)):
        return "validation_error"
    return "provider_error"


@dataclass
class ExecutionOutcome:
    """Result of executing a dataset against a model provider."""

    response_set: ModelResponseSet
    performance_report: PerformanceReport
    execution_logs: list[dict[str, Any]]
    failed_record_ids: list[str]
    completed_record_ids: list[str]
    was_cancelled: bool = False
    started_at: str = ""
    completed_at: str = ""


class ExecutionEngine:
    """Provider-independent execution engine for running datasets against model adapters.

    Architectural Separation:
    `ExecutionEngine` handles prompt dispatch, concurrency, retries, timeouts, response
    caching, duplicate-request protection, and telemetry collection. It produces a standard
    `ModelResponseSet` that can then be passed to `Evaluator`, keeping model execution
    strictly decoupled from evaluation metrics.
    """

    def __init__(
        self,
        provider: BaseModelAdapter,
        config: ExecutionConfig | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or ExecutionConfig()
        self.cache = cache if cache is not None else ResponseCache(enabled=self.config.use_cache)
        if not self.config.use_cache:
            self.cache.enabled = False

    def execute_dataset(
        self,
        dataset: Dataset,
        existing_responses: dict[str, ResponseItem] | None = None,
        cancel_event: threading.Event | None = None,
        on_item_complete: Callable[[ResponseItem | None, RequestTelemetry], None] | None = None,
    ) -> ExecutionOutcome:
        """Execute all (or remaining) records in `dataset` against `self.provider`.

        Supports:
        - Resuming interrupted runs via `existing_responses`.
        - Duplicate request protection across dataset records.
        - Deterministic configuration-aware caching via `self.cache`.
        - Configurable thread-pool concurrency and exponential retry backoff.
        - Cooperative cancellation via `cancel_event`.
        """
        started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        run_start_perf = time.perf_counter()

        preserved_map: dict[str, ResponseItem] = dict(existing_responses or {})
        responses_by_id: dict[str, ResponseItem] = dict(preserved_map)
        telemetry_by_id: dict[str, RequestTelemetry] = {}
        logs: list[dict[str, Any]] = []
        lock = threading.Lock()
        in_flight_ids: set[str] = set()
        was_cancelled = False

        # Identify records that already have responses from an earlier interrupted run
        pending_records: list[DatasetRecord] = []
        for rec in dataset.records:
            if rec.id in preserved_map:
                prev_item = preserved_map[rec.id]
                tok_src = "provider_reported" if prev_item.token_usage else "unavailable"
                p_tok = prev_item.token_usage.get("prompt_tokens") if prev_item.token_usage else None
                c_tok = prev_item.token_usage.get("completion_tokens") if prev_item.token_usage else None
                t_tok = prev_item.token_usage.get("total_tokens") if prev_item.token_usage else None
                telemetry_by_id[rec.id] = RequestTelemetry(
                    record_id=rec.id,
                    category=rec.category,
                    difficulty=rec.difficulty,
                    status="completed",
                    latency_ms=prev_item.latency_ms,
                    attempts=1,
                    retries=0,
                    cache_hit=False,
                    token_source=tok_src,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=t_tok,
                )
            else:
                pending_records.append(rec)

        def _process_one(record: DatasetRecord) -> tuple[ResponseItem | None, RequestTelemetry, dict[str, Any]]:
            # Duplicate in-flight protection
            with lock:
                if record.id in in_flight_ids:
                    raise ConfigurationError(f"Duplicate concurrent request detected for ID '{record.id}'")
                in_flight_ids.add(record.id)

            try:
                # 1. Check response cache first
                cached_rec = self.cache.get(
                    provider_name=self.provider.provider_name,
                    model_name=self.provider.model_name,
                    model_version=self.provider.model_version,
                    dataset_name=dataset.name,
                    dataset_version=dataset.version,
                    record_id=record.id,
                    prompt=record.prompt,
                    generation_config=self.provider.inference_config,
                )
                if cached_rec is not None:
                    item = ResponseItem(
                        id=record.id,
                        response=cached_rec.response,
                        latency_ms=0.0,
                        token_usage=cached_rec.token_usage,
                    )
                    tok_src = "provider_reported" if cached_rec.token_usage else "unavailable"
                    p_tok = cached_rec.token_usage.get("prompt_tokens") if cached_rec.token_usage else None
                    c_tok = cached_rec.token_usage.get("completion_tokens") if cached_rec.token_usage else None
                    t_tok = cached_rec.token_usage.get("total_tokens") if cached_rec.token_usage else None
                    telem = RequestTelemetry(
                        record_id=record.id,
                        category=record.category,
                        difficulty=record.difficulty,
                        status="cached",
                        latency_ms=0.0,
                        attempts=1,
                        retries=0,
                        cache_hit=True,
                        token_source=tok_src,
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        total_tokens=t_tok,
                    )
                    log_entry = {
                        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                        "record_id": record.id,
                        "event": "cache_hit",
                        "cache_key": cached_rec.cache_key,
                        "prompt": redact_prompt_if_private(record.prompt, self.config.privacy_mode),
                    }
                    return item, telem, log_entry

                # 2. Execute with retry policy
                max_attempts = self.config.retry_policy.max_attempts
                last_error: BaseException | None = None
                last_error_type: str = "provider_error"
                timed_out = False

                for attempt in range(1, max_attempts + 1):
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("Execution cancelled by caller")

                    call_start = time.perf_counter()
                    try:
                        resp_item = self.provider.generate_one(record)
                        measured_ms = (time.perf_counter() - call_start) * 1000.0
                        latency_val = (
                            resp_item.latency_ms
                            if resp_item.latency_ms is not None
                            else round(measured_ms, 3)
                        )

                        # Determine token usage source
                        tok_usage = resp_item.token_usage
                        if tok_usage is not None:
                            tok_source = "provider_reported"
                            p_tok = tok_usage.get("prompt_tokens")
                            c_tok = tok_usage.get("completion_tokens")
                            t_tok = tok_usage.get("total_tokens")
                        elif self.config.estimate_missing_tokens:
                            tok_source = "locally_estimated"
                            p_tok = max(1, int(round(len(record.prompt.split()) * 1.3)))
                            c_tok = max(1, int(round(len(resp_item.response.split()) * 1.3)))
                            t_tok = p_tok + c_tok
                            tok_usage = {
                                "prompt_tokens": p_tok,
                                "completion_tokens": c_tok,
                                "total_tokens": t_tok,
                            }
                        else:
                            tok_source = "unavailable"
                            p_tok = None
                            c_tok = None
                            t_tok = None

                        final_item = ResponseItem(
                            id=record.id,
                            response=resp_item.response,
                            latency_ms=latency_val,
                            token_usage=tok_usage,
                        )

                        # Store in cache
                        self.cache.put(
                            provider_name=self.provider.provider_name,
                            model_name=self.provider.model_name,
                            model_version=self.provider.model_version,
                            dataset_name=dataset.name,
                            dataset_version=dataset.version,
                            record_id=record.id,
                            prompt=record.prompt,
                            response_item=final_item,
                            generation_config=self.provider.inference_config,
                        )

                        telem = RequestTelemetry(
                            record_id=record.id,
                            category=record.category,
                            difficulty=record.difficulty,
                            status="completed",
                            latency_ms=latency_val,
                            attempts=attempt,
                            retries=attempt - 1,
                            timed_out=False,
                            cache_hit=False,
                            token_source=tok_source,
                            prompt_tokens=p_tok,
                            completion_tokens=c_tok,
                            total_tokens=t_tok,
                        )
                        log_entry = {
                            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                            "record_id": record.id,
                            "event": "completed",
                            "attempts": attempt,
                            "latency_ms": latency_val,
                            "prompt": redact_prompt_if_private(record.prompt, self.config.privacy_mode),
                        }
                        return final_item, telem, log_entry

                    except Exception as exc:
                        last_error = exc
                        last_error_type = classify_execution_error(exc)
                        if last_error_type == "timeout":
                            timed_out = True
                            if not self.config.retry_policy.retry_on_timeouts:
                                break
                        if last_error_type == "validation_error":
                            break
                        if attempt < max_attempts:
                            delay = self.config.retry_policy.compute_delay(attempt)
                            if delay > 0:
                                time.sleep(delay)

                clean_err_msg = sanitize_error_message(str(last_error)) if last_error else "Unknown error"
                telem = RequestTelemetry(
                    record_id=record.id,
                    category=record.category,
                    difficulty=record.difficulty,
                    status="timeout" if timed_out else "failed",
                    latency_ms=None,
                    attempts=max_attempts if last_error_type != "validation_error" else 1,
                    retries=(max_attempts - 1) if last_error_type != "validation_error" else 0,
                    timed_out=timed_out,
                    cache_hit=False,
                    error_type=last_error_type,
                    error_message=clean_err_msg,
                    token_source="unavailable",
                )
                log_entry = {
                    "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    "record_id": record.id,
                    "event": "failed",
                    "error_type": last_error_type,
                    "error_message": clean_err_msg,
                    "prompt": redact_prompt_if_private(record.prompt, self.config.privacy_mode),
                }
                return None, telem, log_entry
            finally:
                with lock:
                    in_flight_ids.discard(record.id)

        if self.config.concurrency == 1:
            for rec in pending_records:
                if cancel_event is not None and cancel_event.is_set():
                    was_cancelled = True
                    break
                item, telem, log_entry = _process_one(rec)
                telemetry_by_id[rec.id] = telem
                logs.append(log_entry)
                if item is not None:
                    responses_by_id[rec.id] = item
                if on_item_complete is not None:
                    on_item_complete(item, telem)
        else:
            with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
                future_to_rec = {
                    pool.submit(_process_one, rec): rec for rec in pending_records
                }
                for fut in as_completed(future_to_rec):
                    rec = future_to_rec[fut]
                    if cancel_event is not None and cancel_event.is_set():
                        was_cancelled = True
                    item, telem, log_entry = fut.result()
                    with lock:
                        telemetry_by_id[rec.id] = telem
                        logs.append(log_entry)
                        if item is not None:
                            responses_by_id[rec.id] = item
                        if on_item_complete is not None:
                            on_item_complete(item, telem)

        total_duration_ms = (time.perf_counter() - run_start_perf) * 1000.0
        completed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

        ordered_responses = [
            responses_by_id[rec.id]
            for rec in dataset.records
            if rec.id in responses_by_id
        ]
        ordered_telemetry = [
            telemetry_by_id[rec.id]
            for rec in dataset.records
            if rec.id in telemetry_by_id
        ]
        failed_ids = [
            rec.id
            for rec in dataset.records
            if rec.id not in responses_by_id
        ]
        completed_ids = [rec.id for rec in ordered_responses]

        response_set = ModelResponseSet(
            model_name=self.provider.model_name,
            model_version=self.provider.model_version,
            responses=ordered_responses,
            model_provider=self.provider.provider_name,
            inference_config=self.provider.reproducibility_metadata(),
        )
        perf_report = PerformanceMonitor.summarize(
            ordered_telemetry,
            total_duration_ms=total_duration_ms,
            pricing=self.config.pricing,
        )

        return ExecutionOutcome(
            response_set=response_set,
            performance_report=perf_report,
            execution_logs=logs,
            failed_record_ids=failed_ids,
            completed_record_ids=completed_ids,
            was_cancelled=was_cancelled,
            started_at=started_at,
            completed_at=completed_at,
        )
