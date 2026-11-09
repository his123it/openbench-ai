"""Structured Experiment & Run Management System for OpenBench AI (v0.3.0; lineage and repeat groups v0.4.0)."""

from __future__ import annotations

import json
import os
import platform
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openbench.adapters.base import BaseModelAdapter
from openbench.cache import ResponseCache
from openbench.comparison import ComparisonResult, ModelComparator
from openbench.dataset import Dataset, DatasetLoader
from openbench.evaluator import (
    EvaluationConfig,
    EvaluationResult,
    Evaluator,
    ModelResponseSet,
    ResponseItem,
    ResponseLoader,
)
from openbench.exceptions import ConfigurationError, OpenBenchError
from openbench.execution import ExecutionConfig, ExecutionEngine
from openbench.history import ResultHistoryStore
from openbench.integrity import (
    IntegrityCheckResult,
    RunIntegrityVerifier,
    compute_dict_sha256,
    compute_file_sha256,
)
from openbench.lineage import (
    GROUP_INDEX_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    SEED_DISCLAIMER,
    RunLineage,
    generation_config_without_seed,
    ids_sha256,
    legacy_lineage,
    new_repeat_group_id,
    record_content_hashes,
    seed_status,
    validate_repeat_group_id,
    validate_run_id,
)
from openbench.performance import PerformanceMonitor, PerformanceReport, RequestTelemetry
from openbench.reporting import ReportGenerator
from openbench.security import redact_sensitive_data


@dataclass
class ExperimentRecord:
    """Complete metadata record for a managed OpenBench AI experiment run."""

    run_id: str
    created_at: str
    started_at: str
    completed_at: str | None
    status: str  # "completed" | "partial" | "failed"
    execution_origin: str  # "provider_executed" | "external_responses_evaluated" | "manually_imported"
    dataset_identifier: str
    dataset_path: str
    dataset_version: str
    dataset_checksum: str
    model_provider: str
    model_name: str
    model_version: str
    is_mock_provider: bool
    metric_configuration: dict[str, Any]
    generation_configuration: dict[str, Any]
    execution_configuration: dict[str, Any]
    openbench_version: str
    environment_metadata: dict[str, Any]
    run_dir: str
    aggregate_scores: dict[str, float] = field(default_factory=dict)
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    failure_info: dict[str, Any] = field(default_factory=dict)
    #: v0.4.0 lineage block (empty for records built without one; legacy runs are mapped
    #: to :func:`openbench.lineage.legacy_lineage` when loaded).
    lineage: dict[str, Any] = field(default_factory=dict)

    @property
    def repeat_group_id(self) -> str | None:
        return self.lineage.get("repeat_group_id") if self.lineage else None

    @property
    def repetition_index(self) -> int:
        return int(self.lineage.get("repetition_index", 0)) if self.lineage else 0

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "execution_origin": self.execution_origin,
            "dataset_identifier": self.dataset_identifier,
            "dataset_path": self.dataset_path,
            "dataset_version": self.dataset_version,
            "dataset_checksum": self.dataset_checksum,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "is_mock_provider": self.is_mock_provider,
            "metric_configuration": redact_sensitive_data(dict(self.metric_configuration)),
            "generation_configuration": redact_sensitive_data(dict(self.generation_configuration)),
            "execution_configuration": redact_sensitive_data(dict(self.execution_configuration)),
            "openbench_version": self.openbench_version,
            "environment_metadata": dict(self.environment_metadata),
            "run_dir": self.run_dir,
            "aggregate_scores": {k: round(v, 6) for k, v in self.aggregate_scores.items()},
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "failure_info": dict(self.failure_info),
        }
        if self.lineage:
            payload["lineage"] = dict(self.lineage)
        return payload


@dataclass
class RepeatGroupResult:
    """Result of executing (or resuming) a repeat group of runs (v0.4.0)."""

    repeat_group_id: str | None
    planned_repetitions: int
    records: list[ExperimentRecord]
    evaluations: list[EvaluationResult]
    performance_reports: list[PerformanceReport]
    index_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def completed_repetitions(self) -> int:
        return sum(1 for r in self.records if r.status == "completed")

    @property
    def is_complete(self) -> bool:
        return (
            len(self.records) == self.planned_repetitions
            and self.completed_repetitions == self.planned_repetitions
        )


class ExperimentManager:
    """Manages structured experiment execution, storage, resumption, comparison, and integrity verification.

    Directory Layout per Run (`<experiments_root>/run_<id>/`):
    - `config.json`: Full experiment, dataset, metric, execution, and generation configuration.
    - `model_metadata.json`: Provider, model name, revision, mock status, and execution origin.
    - `responses.json`: Standard OpenBench response payload (`model_name`, `model_version`, `responses`).
    - `evaluation.json`: Full `EvaluationResult` report with category/difficulty breakdowns.
    - `performance.json`: Full `PerformanceReport` with latency percentiles, token accounting, and retries.
    - `reproducibility.json`: Cryptographic SHA-256 checksums for all artifacts, configs, and dataset.
    - `logs.jsonl`: Per-request execution event stream.
    """

    def __init__(
        self,
        experiments_root: str | Path | None = None,
        history_store: ResultHistoryStore | None = None,
    ) -> None:
        if experiments_root is not None:
            self.root = Path(experiments_root)
        else:
            self.root = Path.cwd() / "experiments"
        self.history_store = history_store

    def _run_dir(self, run_id: str) -> Path:
        try:
            folder_name = validate_run_id(run_id)
        except ConfigurationError as exc:
            # Unsafe IDs are reported as unknown runs (never resolved outside the root).
            raise KeyError(str(exc)) from exc
        return self.root / folder_name

    def _group_index_path(self, repeat_group_id: str) -> Path:
        validate_repeat_group_id(repeat_group_id)
        # Stored as a file at the experiments root (not a sub-directory) so tools that
        # enumerate run directories, including v0.3.1 readers, are unaffected.
        return self.root / f"{repeat_group_id}.json"

    @staticmethod
    def _environment_metadata() -> dict[str, Any]:
        return {
            "python_version": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
        }

    def execute_and_record(
        self,
        dataset: Dataset,
        provider: BaseModelAdapter,
        evaluation_config: EvaluationConfig | None = None,
        execution_config: ExecutionConfig | None = None,
        cache: ResponseCache | None = None,
        run_id: str | None = None,
        existing_responses: dict[str, ResponseItem] | None = None,
        lineage: RunLineage | None = None,
    ) -> tuple[ExperimentRecord, EvaluationResult, PerformanceReport]:
        """Run a dataset through `ExecutionEngine`, evaluate via `Evaluator`, and persist all 7 run artifacts.

        v0.4.0: ``lineage`` identifies the run's repeat group and repetition (``None`` means
        a single run). A repetition seed in the lineage is applied to the provider, and the
        repetition context isolates the run's cache entries.
        """
        from openbench import __version__

        run_lineage = lineage or RunLineage()
        if (
            run_lineage.repetition_seed is not None
            and provider.generation_config.seed != run_lineage.repetition_seed
        ):
            provider = provider.with_generation_seed(run_lineage.repetition_seed)

        eval_cfg = evaluation_config or EvaluationConfig()
        exec_cfg = execution_config or ExecutionConfig(timeout_seconds=provider.timeout_seconds)

        evaluator = Evaluator(config=eval_cfg)
        active_dataset = evaluator._apply_configured_filters(dataset)

        resolved_run_id = run_id or self._new_run_id()
        if not resolved_run_id.startswith("run_"):
            resolved_run_id = f"run_{resolved_run_id}"
        run_dir = self._run_dir(resolved_run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        forwarded_before = int(getattr(provider, "seed_forward_count", 0))
        engine = ExecutionEngine(
            provider=provider,
            config=exec_cfg,
            cache=cache,
            repetition_context=run_lineage.repetition_context,
        )
        outcome = engine.execute_dataset(
            active_dataset,
            existing_responses=existing_responses,
        )
        forwarded = int(getattr(provider, "seed_forward_count", 0)) - forwarded_before

        eval_result = evaluator.evaluate(active_dataset, outcome.response_set)
        perf_report = outcome.performance_report

        if len(outcome.failed_record_ids) == 0 and not outcome.was_cancelled:
            status = "completed"
        elif len(outcome.completed_record_ids) > 0:
            status = "partial"
        else:
            status = "failed"

        failure_info = {
            "failed_record_ids": list(outcome.failed_record_ids),
            "failed_count": len(outcome.failed_record_ids),
            "timeout_count": perf_report.timeout_count,
            "retry_count": perf_report.retry_count,
            "was_cancelled": outcome.was_cancelled,
        }

        env_meta = self._environment_metadata()
        gen_cfg_dict = redact_sensitive_data(provider.reproducibility_metadata())
        eval_cfg_dict = redact_sensitive_data(eval_cfg.to_dict())
        exec_cfg_dict = redact_sensitive_data(exec_cfg.to_dict())

        record = ExperimentRecord(
            run_id=resolved_run_id,
            created_at=created_at,
            started_at=outcome.started_at,
            completed_at=outcome.completed_at,
            status=status,
            execution_origin="provider_executed",
            dataset_identifier=active_dataset.name,
            dataset_path=active_dataset.source_path,
            dataset_version=active_dataset.version,
            dataset_checksum=active_dataset.checksum,
            model_provider=provider.provider_name,
            model_name=provider.model_name,
            model_version=provider.model_version,
            is_mock_provider=bool(provider.is_mock),
            metric_configuration=eval_cfg_dict,
            generation_configuration=gen_cfg_dict,
            execution_configuration=exec_cfg_dict,
            openbench_version=__version__,
            environment_metadata=env_meta,
            run_dir=str(run_dir),
            aggregate_scores={
                k: round(v.score, 6) for k, v in eval_result.aggregate_results.items()
            },
            total_items=len(active_dataset),
            completed_items=len(outcome.completed_record_ids),
            failed_items=len(outcome.failed_record_ids),
            failure_info=failure_info,
            lineage=self._build_lineage(
                run_lineage,
                provider=provider,
                active_dataset=active_dataset,
                forwarded_requests=forwarded,
                cache_enabled=bool(engine.cache.enabled),
                cached_requests=perf_report.cached_requests,
                preserved_items=len(existing_responses or {}),
                run_id=resolved_run_id,
            ),
        )

        self._write_run_artifacts(
            run_dir=run_dir,
            record=record,
            response_set=outcome.response_set,
            eval_result=eval_result,
            perf_report=perf_report,
            logs=outcome.execution_logs,
        )

        if self.history_store is not None:
            self.history_store.record(eval_result, report_path=run_dir / "evaluation.json")

        if run_lineage.repeat_group_id is not None:
            self.rebuild_group_index(run_lineage.repeat_group_id)

        return record, eval_result, perf_report

    def _new_run_id(self) -> str:
        while True:
            candidate = f"run_{uuid.uuid4().hex[:12]}"
            if not (self.root / candidate).exists():
                return candidate

    @staticmethod
    def _build_lineage(
        run_lineage: RunLineage,
        provider: BaseModelAdapter,
        active_dataset: Dataset,
        forwarded_requests: int,
        cache_enabled: bool,
        cached_requests: int,
        preserved_items: int,
        run_id: str,
    ) -> dict[str, Any]:
        seed_supplied = provider.generation_config.seed
        supported = bool(getattr(provider, "supports_generation_seed", False))
        base_seed = run_lineage.generation_base_seed
        rep_seed = run_lineage.repetition_seed
        if base_seed is None and rep_seed is None and seed_supplied is not None:
            # Single run with a seed from the profile or GenerationConfig.
            base_seed = rep_seed = seed_supplied
        selected = sorted(rec.id for rec in active_dataset.records)
        identity = run_lineage.to_dict()
        identity["experiment_id"] = run_lineage.experiment_id or (
            run_lineage.repeat_group_id or run_id
        )
        identity["generation_base_seed"] = base_seed
        identity["repetition_seed"] = rep_seed
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            **identity,
            "seed_supplied": seed_supplied,
            "seed_supported_by_adapter": supported,
            "seed_passed_to_provider": forwarded_requests > 0,
            "seed_status": seed_status(seed_supplied, supported, forwarded_requests),
            "seed_disclaimer": SEED_DISCLAIMER,
            "selected_record_ids": selected,
            "selected_record_ids_sha256": ids_sha256(selected),
            "record_content_sha256": record_content_hashes(active_dataset.records),
            "cache_provenance": {
                "cache_enabled": cache_enabled,
                "repetition_context": (
                    list(run_lineage.repetition_context)
                    if run_lineage.repetition_context is not None
                    else None
                ),
                "cached_requests": int(cached_requests),
                "preserved_items_from_previous_attempt": int(preserved_items),
            },
        }

    def register_external_evaluation(
        self,
        dataset: Dataset,
        response_set: ModelResponseSet,
        eval_result: EvaluationResult,
        execution_origin: str = "external_responses_evaluated",
        run_id: str | None = None,
    ) -> ExperimentRecord:
        """Record an evaluation of externally supplied responses into the experiment store with honest provenance."""
        from openbench import __version__

        if execution_origin not in {"external_responses_evaluated", "manually_imported"}:
            raise ConfigurationError(
                "External evaluations must use execution_origin='external_responses_evaluated' or 'manually_imported'"
            )

        resolved_run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        if not resolved_run_id.startswith("run_"):
            resolved_run_id = f"run_{resolved_run_id}"
        run_dir = self._run_dir(resolved_run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        telemetry_items: list[RequestTelemetry] = []
        resp_map = response_set.response_item_map
        for rec in dataset.records:
            if rec.id in resp_map:
                ri = resp_map[rec.id]
                tok_src = "provider_reported" if ri.token_usage else "unavailable"
                telemetry_items.append(
                    RequestTelemetry(
                        record_id=rec.id,
                        category=rec.category,
                        difficulty=rec.difficulty,
                        status="completed",
                        latency_ms=ri.latency_ms,
                        token_source=tok_src,
                        prompt_tokens=ri.token_usage.get("prompt_tokens") if ri.token_usage else None,
                        completion_tokens=ri.token_usage.get("completion_tokens") if ri.token_usage else None,
                        total_tokens=ri.token_usage.get("total_tokens") if ri.token_usage else None,
                    )
                )
            else:
                telemetry_items.append(
                    RequestTelemetry(
                        record_id=rec.id,
                        category=rec.category,
                        difficulty=rec.difficulty,
                        status="failed",
                        error_type="missing_response",
                    )
                )

        perf_report = PerformanceMonitor.summarize(telemetry_items, total_duration_ms=None)
        record = ExperimentRecord(
            run_id=resolved_run_id,
            created_at=now_iso,
            started_at=now_iso,
            completed_at=now_iso,
            status="completed" if eval_result.missing_responses_count == 0 else "partial",
            execution_origin=execution_origin,
            dataset_identifier=dataset.name,
            dataset_path=dataset.source_path,
            dataset_version=dataset.version,
            dataset_checksum=dataset.checksum,
            model_provider=response_set.model_provider,
            model_name=response_set.model_name,
            model_version=response_set.model_version,
            is_mock_provider=False,
            metric_configuration=redact_sensitive_data(eval_result.config),
            generation_configuration=redact_sensitive_data(response_set.inference_config),
            execution_configuration={"execution_origin": execution_origin},
            openbench_version=__version__,
            environment_metadata=self._environment_metadata(),
            run_dir=str(run_dir),
            aggregate_scores={
                k: round(v.score, 6) for k, v in eval_result.aggregate_results.items()
            },
            total_items=len(dataset),
            completed_items=eval_result.evaluated_items_count,
            failed_items=eval_result.missing_responses_count,
            failure_info={"missing_response_ids": list(eval_result.missing_response_ids)},
        )

        self._write_run_artifacts(
            run_dir=run_dir,
            record=record,
            response_set=response_set,
            eval_result=eval_result,
            perf_report=perf_report,
            logs=[
                {
                    "timestamp": now_iso,
                    "event": execution_origin,
                    "source_path": response_set.source_path,
                }
            ],
        )
        return record

    def _write_run_artifacts(
        self,
        run_dir: Path,
        record: ExperimentRecord,
        response_set: ModelResponseSet,
        eval_result: EvaluationResult,
        perf_report: PerformanceReport,
        logs: list[dict[str, Any]],
    ) -> None:
        config_path = run_dir / "config.json"
        model_meta_path = run_dir / "model_metadata.json"
        responses_path = run_dir / "responses.json"
        eval_path = run_dir / "evaluation.json"
        perf_path = run_dir / "performance.json"
        repro_path = run_dir / "reproducibility.json"
        logs_path = run_dir / "logs.jsonl"

        config_payload = record.to_dict()
        config_path.write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")

        model_meta_payload = {
            "run_id": record.run_id,
            "execution_origin": record.execution_origin,
            "model_provider": record.model_provider,
            "model_name": record.model_name,
            "model_version": record.model_version,
            "is_mock_provider": record.is_mock_provider,
            "generation_configuration": record.generation_configuration,
        }
        if record.lineage:
            model_meta_payload["repeat_group_id"] = record.lineage.get("repeat_group_id")
            model_meta_payload["repetition_index"] = record.lineage.get("repetition_index", 0)
        model_meta_path.write_text(json.dumps(model_meta_payload, indent=2) + "\n", encoding="utf-8")

        responses_payload = {
            "model_name": response_set.model_name,
            "model_version": response_set.model_version,
            "model_provider": response_set.model_provider,
            "inference_config": redact_sensitive_data(response_set.inference_config),
            "responses": [item.to_dict() for item in response_set.responses],
        }
        responses_path.write_text(json.dumps(responses_payload, indent=2) + "\n", encoding="utf-8")

        ReportGenerator.save_json(eval_result, eval_path)
        perf_path.write_text(json.dumps(perf_report.to_dict(), indent=2) + "\n", encoding="utf-8")

        log_lines = [json.dumps(entry, ensure_ascii=False) for entry in logs]
        logs_path.write_text(("\n".join(log_lines) + "\n") if log_lines else "", encoding="utf-8")

        repro_payload = {
            "schema_version": RUN_SCHEMA_VERSION if record.lineage else "0.3.0",
            "run_id": record.run_id,
            "execution_origin": record.execution_origin,
            "created_at": record.created_at,
            "started_at": record.started_at,
            "completed_at": record.completed_at,
            "openbench_version": record.openbench_version,
            "environment_metadata": record.environment_metadata,
            "dataset_identifier": record.dataset_identifier,
            "dataset_version": record.dataset_version,
            "dataset_checksum": record.dataset_checksum,
            "model_provider": record.model_provider,
            "model_name": record.model_name,
            "model_version": record.model_version,
            "is_mock_provider": record.is_mock_provider,
            "metrics": [
                {"name": m.get("name"), "version": m.get("version", "0.1.0")}
                for m in eval_result.metrics_used
            ],
            "evaluation_configuration_checksum": compute_dict_sha256(record.metric_configuration),
            "generation_configuration_checksum": compute_dict_sha256(record.generation_configuration),
            "artifact_checksums": {
                "config_sha256": compute_file_sha256(config_path),
                "model_metadata_sha256": compute_file_sha256(model_meta_path),
                "responses_sha256": compute_file_sha256(responses_path),
                "evaluation_sha256": compute_file_sha256(eval_path),
                "performance_sha256": compute_file_sha256(perf_path),
            },
            "integrity_disclaimer": (
                "SHA-256 checksums verify file and configuration integrity on disk; "
                "they do NOT prove the factual truthfulness of a model result."
            ),
        }
        if record.lineage:
            repro_payload["lineage"] = dict(record.lineage)
            repro_payload["generation_configuration_without_seed_checksum"] = compute_dict_sha256(
                generation_config_without_seed(record.generation_configuration)
            )
        repro_path.write_text(json.dumps(repro_payload, indent=2) + "\n", encoding="utf-8")

    def list_runs(
        self,
        status: str | None = None,
        provider: str | None = None,
        repeat_group_id: str | None = None,
    ) -> list[ExperimentRecord]:
        """List all experiment runs stored under `self.root` (optionally one repeat group)."""
        if not self.root.exists():
            return []
        records: list[ExperimentRecord] = []
        for d in sorted(self.root.iterdir()):
            if not d.is_dir() or not d.name.startswith("run_"):
                continue
            cfg_file = d / "config.json"
            if not cfg_file.exists():
                continue
            try:
                rec = self._load_record_from_dir(d)
            except OpenBenchError:
                # Skip corrupted run directories during listing rather than crashing list_runs()
                continue
            if status and rec.status.lower() != status.lower():
                continue
            if provider and rec.model_provider.lower() != provider.lower():
                continue
            if repeat_group_id is not None and rec.repeat_group_id != repeat_group_id:
                continue
            records.append(rec)
        return records

    def inspect_run(self, run_id: str) -> dict[str, Any]:
        """Load complete metadata, evaluation summary, performance summary, and integrity status for a run."""
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            raise KeyError(f"Experiment run '{run_id}' does not exist in {self.root}")
        rec = self._load_record_from_dir(run_dir)
        integrity = RunIntegrityVerifier.verify_run_directory(run_dir)
        try:
            perf_data = (
                json.loads((run_dir / "performance.json").read_text(encoding="utf-8"))
                if (run_dir / "performance.json").exists()
                else {}
            )
        except json.JSONDecodeError as exc:
            perf_data = {"error": f"Corrupted performance.json: {exc.msg}"}
        try:
            repro_data = (
                json.loads((run_dir / "reproducibility.json").read_text(encoding="utf-8"))
                if (run_dir / "reproducibility.json").exists()
                else {}
            )
        except json.JSONDecodeError as exc:
            repro_data = {"error": f"Corrupted reproducibility.json: {exc.msg}"}
        return {
            "experiment": rec.to_dict(),
            "integrity": integrity.to_dict(),
            "performance": perf_data,
            "reproducibility": repro_data,
        }

    def verify_run(self, run_id: str) -> IntegrityCheckResult:
        """Run cryptographic checksum and provenance verification on a stored run."""
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            raise KeyError(f"Experiment run '{run_id}' does not exist in {self.root}")
        return RunIntegrityVerifier.verify_run_directory(run_dir)

    def resume_run(
        self,
        run_id: str,
        provider: BaseModelAdapter,
        cache: ResponseCache | None = None,
    ) -> tuple[ExperimentRecord, EvaluationResult, PerformanceReport]:
        """Resume an interrupted or partial run, preserving already completed response items."""
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            raise KeyError(f"Experiment run '{run_id}' does not exist in {self.root}")

        rec = self._load_record_from_dir(run_dir)
        if rec.execution_origin != "provider_executed":
            raise ConfigurationError(
                f"Run '{rec.run_id}' has origin '{rec.execution_origin}' and is not eligible for provider resumption"
            )
        if rec.status == "completed" and rec.failed_items == 0:
            eval_res = EvaluationResult.from_dict(
                json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
            )
            perf_data = json.loads((run_dir / "performance.json").read_text(encoding="utf-8"))
            perf_rep = PerformanceReport(
                total_duration_ms=perf_data.get("total_duration_ms"),
                total_requests=perf_data.get("total_requests", rec.total_items),
                completed_requests=perf_data.get("completed_requests", rec.completed_items),
                cached_requests=perf_data.get("cached_requests", 0),
                failed_requests=perf_data.get("failed_requests", 0),
                timeout_count=perf_data.get("timeout_count", 0),
                retry_count=perf_data.get("retry_count", 0),
                latency_summary=perf_data.get("latency_summary", {}),
                throughput_requests_per_sec=perf_data.get("throughput_requests_per_sec"),
                token_usage_summary=perf_data.get("token_usage_summary", {}),
                estimated_cost=perf_data.get("estimated_cost"),
            )
            return rec, eval_res, perf_rep

        dataset = DatasetLoader.load(rec.dataset_path)
        existing_resp_set = ResponseLoader.load_file(run_dir / "responses.json")
        preserved_items = existing_resp_set.response_item_map

        is_v040 = bool(rec.lineage) and not rec.lineage.get("legacy_run", False)
        eval_cfg = evaluation_config_from_record(rec)
        lineage = RunLineage.from_dict(rec.lineage) if is_v040 else None
        exec_cfg = execution_config_from_record(rec, provider) if is_v040 else None

        return self.execute_and_record(
            dataset=dataset,
            provider=provider,
            evaluation_config=eval_cfg,
            execution_config=exec_cfg,
            cache=cache,
            run_id=rec.run_id,
            existing_responses=preserved_items,
            lineage=lineage,
        )

    def compare_runs(self, run_ids: list[str]) -> ComparisonResult:
        """Compare two or more stored experiment runs using `ModelComparator`."""
        if len(run_ids) < 2:
            raise ConfigurationError("At least 2 run IDs are required for comparison")
        eval_results: list[EvaluationResult] = []
        for rid in run_ids:
            run_dir = self._run_dir(rid)
            eval_file = run_dir / "evaluation.json"
            if not eval_file.exists():
                raise KeyError(f"Evaluation artifact not found for run '{rid}' at {eval_file}")
            payload = json.loads(eval_file.read_text(encoding="utf-8"))
            eval_results.append(EvaluationResult.from_dict(payload))
        return ModelComparator.compare(eval_results)

    def load_run_record(self, run_id: str) -> ExperimentRecord:
        """Load a run's :class:`ExperimentRecord` (read-only; v0.4.0)."""
        run_dir = self._run_dir(run_id)
        if not (run_dir / "config.json").exists():
            raise KeyError(f"Experiment run '{run_id}' does not exist in {self.root}")
        return self._load_record_from_dir(run_dir)

    def load_evaluation(self, run_id: str) -> EvaluationResult:
        """Load a stored run's evaluation without constructing any provider (read-only; v0.4.0)."""
        run_dir = self._run_dir(run_id)
        eval_file = run_dir / "evaluation.json"
        if not eval_file.exists():
            raise KeyError(f"Evaluation artifact not found for run '{run_id}' at {eval_file}")
        try:
            payload = json.loads(eval_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise OpenBenchError(f"Malformed evaluation.json in {run_dir}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise OpenBenchError(f"Malformed evaluation.json in {run_dir}: expected an object")
        try:
            return EvaluationResult.from_dict(payload)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise OpenBenchError(f"Malformed evaluation.json in {run_dir}: {type(exc).__name__}: {exc}") from exc

    # ------------------------------------------------------------------
    # Repeat groups (v0.4.0)
    # ------------------------------------------------------------------

    def execute_repeated(
        self,
        dataset: Dataset,
        provider: BaseModelAdapter,
        repetitions: int,
        generation_base_seed: int | None = None,
        evaluation_config: EvaluationConfig | None = None,
        execution_config: ExecutionConfig | None = None,
        cache: ResponseCache | None = None,
        repeat_group_id: str | None = None,
        experiment_id: str | None = None,
        parent_experiment_id: str | None = None,
        on_repetition: Callable[[ExperimentRecord], None] | None = None,
    ) -> RepeatGroupResult:
        """Execute ``repetitions`` independent repetitions of one configuration.

        ``repetitions == 1`` is a single run (no group). ``repetitions >= 2`` creates a repeat
        group ``grp_<12 hex>``: every repetition is its own run with lineage, its own cache
        context and, when ``generation_base_seed`` is given, the seed
        ``generation_base_seed + repetition_index``.
        """
        if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
            raise ConfigurationError("repetitions must be an integer >= 1")
        if generation_base_seed is not None and (
            isinstance(generation_base_seed, bool) or not isinstance(generation_base_seed, int)
        ):
            raise ConfigurationError("generation_base_seed must be an integer or None")

        if repetitions == 1:
            lineage = RunLineage(
                generation_base_seed=generation_base_seed,
                repetition_seed=generation_base_seed,
                experiment_id=experiment_id,
                parent_experiment_id=parent_experiment_id,
            )
            rec, ev, perf = self.execute_and_record(
                dataset=dataset,
                provider=provider,
                evaluation_config=evaluation_config,
                execution_config=execution_config,
                cache=cache,
                lineage=lineage,
            )
            if on_repetition is not None:
                on_repetition(rec)
            return RepeatGroupResult(None, 1, [rec], [ev], [perf])

        group_id = repeat_group_id or new_repeat_group_id()
        validate_repeat_group_id(group_id)
        if self._group_index_path(group_id).exists() or self.list_runs(repeat_group_id=group_id):
            raise ConfigurationError(
                f"Repeat group '{group_id}' already exists; use resume_group() to continue it"
            )
        self._write_group_index(group_id, planned=repetitions, members=[], extra={
            "experiment_id": experiment_id or group_id,
            "parent_experiment_id": parent_experiment_id,
            "generation_base_seed": generation_base_seed,
        })

        records: list[ExperimentRecord] = []
        evaluations: list[EvaluationResult] = []
        perfs: list[PerformanceReport] = []
        for index in range(repetitions):
            lineage = RunLineage.for_repetition(
                group_id,
                index,
                repetitions,
                generation_base_seed=generation_base_seed,
                experiment_id=experiment_id or group_id,
                parent_experiment_id=parent_experiment_id,
            )
            rec, ev, perf = self.execute_and_record(
                dataset=dataset,
                provider=provider,
                evaluation_config=evaluation_config,
                execution_config=execution_config,
                cache=cache,
                lineage=lineage,
            )
            records.append(rec)
            evaluations.append(ev)
            perfs.append(perf)
            if on_repetition is not None:
                on_repetition(rec)
        index_path = self._group_index_path(group_id)
        return RepeatGroupResult(group_id, repetitions, records, evaluations, perfs, str(index_path))

    def _scan_group_members(self, repeat_group_id: str) -> list[ExperimentRecord]:
        members = self.list_runs(repeat_group_id=repeat_group_id)
        return sorted(members, key=lambda r: (r.repetition_index, r.run_id))

    def _read_group_index(self, repeat_group_id: str) -> dict[str, Any] | None:
        path = self._group_index_path(repeat_group_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def _write_group_index(
        self,
        repeat_group_id: str,
        planned: int,
        members: list[ExperimentRecord],
        extra: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        previous = self._read_group_index(repeat_group_id) or {}
        member_entries = []
        repro_checksums = []
        for rec in members:
            repro_file = Path(rec.run_dir) / "reproducibility.json"
            repro_sha = compute_file_sha256(repro_file) if repro_file.exists() else None
            repro_checksums.append(repro_sha or "")
            member_entries.append(
                {
                    "repetition_index": rec.repetition_index,
                    "run_id": rec.run_id,
                    "status": rec.status,
                    "repetition_seed": rec.lineage.get("repetition_seed"),
                    "seed_status": rec.lineage.get("seed_status"),
                    "reproducibility_sha256": repro_sha,
                }
            )
        completed = sorted({m["repetition_index"] for m in member_entries if m["status"] == "completed"})
        shared: dict[str, Any] = {}
        if members:
            first = members[0]
            shared = {
                "dataset_identifier": first.dataset_identifier,
                "dataset_checksum": first.dataset_checksum,
                "model_provider": first.model_provider,
                "model_name": first.model_name,
                "model_version": first.model_version,
                "is_mock_provider": first.is_mock_provider,
                "evaluation_configuration_checksum": compute_dict_sha256(first.metric_configuration),
                "generation_configuration_without_seed_checksum": compute_dict_sha256(
                    generation_config_without_seed(first.generation_configuration)
                ),
                "selected_record_ids_sha256": first.lineage.get("selected_record_ids_sha256"),
            }
        base = {k: previous.get(k) for k in ("experiment_id", "parent_experiment_id", "generation_base_seed", "created_at")}
        if members:
            base["experiment_id"] = members[0].lineage.get("experiment_id", base.get("experiment_id"))
            base["parent_experiment_id"] = members[0].lineage.get("parent_experiment_id")
            base["generation_base_seed"] = members[0].lineage.get("generation_base_seed")
        if extra:
            base.update(extra)
        index = {
            "schema_version": GROUP_INDEX_SCHEMA_VERSION,
            "repeat_group_id": repeat_group_id,
            "experiment_id": base.get("experiment_id") or repeat_group_id,
            "parent_experiment_id": base.get("parent_experiment_id"),
            "generation_base_seed": base.get("generation_base_seed"),
            "created_at": base.get("created_at") or now_iso,
            "updated_at": now_iso,
            "planned_repetitions": planned,
            "member_count": len(member_entries),
            "completed_repetitions": len(completed),
            "missing_repetition_indices": [i for i in range(planned) if i not in {m["repetition_index"] for m in member_entries}],
            "status": "complete" if len(completed) == planned and len(member_entries) == planned else "incomplete",
            "members": member_entries,
            "shared_configuration": shared,
            "members_reproducibility_sha256": compute_dict_sha256({"checksums": repro_checksums}),
            "rebuild_warnings": list(warnings or []),
            "note": (
                "Rebuildable index. Per-run lineage in each run's config.json and "
                "reproducibility.json is authoritative; if they disagree, lineage wins."
            ),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._group_index_path(repeat_group_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return index

    def rebuild_group_index(self, repeat_group_id: str) -> dict[str, Any]:
        """Rebuild a group index from member lineage (lineage wins over a stale index)."""
        validate_repeat_group_id(repeat_group_id)
        members = self._scan_group_members(repeat_group_id)
        previous = self._read_group_index(repeat_group_id)
        warnings: list[str] = []
        counts = {int(m.lineage.get("repetition_count", 1)) for m in members}
        if len(counts) > 1:
            warnings.append(
                f"Members disagree on repetition_count {sorted(counts)}; using the largest."
            )
        if counts:
            planned = max(counts)
        elif previous is not None:
            planned = int(previous.get("planned_repetitions", 0))
        else:
            raise KeyError(f"Repeat group '{repeat_group_id}' does not exist in {self.root}")
        indices = [m.repetition_index for m in members]
        duplicates = sorted({i for i in indices if indices.count(i) > 1})
        if duplicates:
            warnings.append(f"Duplicate repetition indices in lineage: {duplicates}")
        if previous is not None:
            prev_members = {
                (m.get("repetition_index"), m.get("run_id"))
                for m in previous.get("members", [])
                if isinstance(m, dict)
            }
            now_members = {(m.repetition_index, m.run_id) for m in members}
            if prev_members and prev_members != now_members:
                warnings.append(
                    "Group index disagreed with run lineage; the index was rebuilt from lineage."
                )
            if counts and int(previous.get("planned_repetitions", planned)) != planned:
                warnings.append(
                    "Group index planned_repetitions disagreed with lineage; lineage used."
                )
        return self._write_group_index(repeat_group_id, planned, members, warnings=warnings)

    def load_group(self, repeat_group_id: str) -> dict[str, Any]:
        """Return the (rebuilt) group index and its member records, without writing runs."""
        index = self.rebuild_group_index(repeat_group_id)
        members = self._scan_group_members(repeat_group_id)
        return {"index": index, "members": members}

    def list_groups(self) -> list[dict[str, Any]]:
        """List repeat-group index files under the experiments root (read-only)."""
        if not self.root.exists():
            return []
        out = []
        for path in sorted(self.root.glob("grp_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("repeat_group_id"):
                out.append(data)
        return out

    def load_group_evaluations(self, repeat_group_id: str) -> list[EvaluationResult]:
        """Evaluations of a group's members in repetition order (read-only; no providers)."""
        validate_repeat_group_id(repeat_group_id)
        members = self._scan_group_members(repeat_group_id)
        if not members:
            raise KeyError(f"Repeat group '{repeat_group_id}' has no member runs in {self.root}")
        return [self.load_evaluation(m.run_id) for m in members]

    def resume_group(
        self,
        repeat_group_id: str,
        provider: BaseModelAdapter,
        cache: ResponseCache | None = None,
    ) -> RepeatGroupResult:
        """Resume a repeat group: finish partial members and create missing repetitions.

        Completed members are never re-requested. Partial members are resumed in place,
        keeping their completed items. Missing repetition indices are executed under the
        same group ID with their own lineage, seed and cache context.
        """
        group = self.load_group(repeat_group_id)
        index = group["index"]
        members: list[ExperimentRecord] = group["members"]
        planned = int(index["planned_repetitions"])
        by_index: dict[int, list[ExperimentRecord]] = {}
        for m in members:
            by_index.setdefault(m.repetition_index, []).append(m)
        duplicates = sorted(i for i, ms in by_index.items() if len(ms) > 1)
        if duplicates:
            raise ConfigurationError(
                f"Cannot resume repeat group '{repeat_group_id}': duplicate repetition indices "
                f"{duplicates} in run lineage. Inspect the runs before continuing."
            )
        if not members:
            raise ConfigurationError(
                f"Repeat group '{repeat_group_id}' has no member runs to take its configuration "
                "from; it cannot be resumed."
            )
        template = members[0]
        records: list[ExperimentRecord] = []
        evaluations: list[EvaluationResult] = []
        perfs: list[PerformanceReport] = []
        for rep_index in range(planned):
            existing = by_index.get(rep_index)
            if existing:
                rec, ev, perf = self.resume_run(existing[0].run_id, provider=provider, cache=cache)
            else:
                lineage = RunLineage.for_repetition(
                    repeat_group_id,
                    rep_index,
                    planned,
                    generation_base_seed=template.lineage.get("generation_base_seed"),
                    experiment_id=template.lineage.get("experiment_id"),
                    parent_experiment_id=template.lineage.get("parent_experiment_id"),
                )
                rec, ev, perf = self.execute_and_record(
                    dataset=DatasetLoader.load(template.dataset_path),
                    provider=provider,
                    evaluation_config=evaluation_config_from_record(template),
                    execution_config=execution_config_from_record(template, provider),
                    cache=cache,
                    lineage=lineage,
                )
            records.append(rec)
            evaluations.append(ev)
            perfs.append(perf)
        final_index = self.rebuild_group_index(repeat_group_id)
        return RepeatGroupResult(
            repeat_group_id,
            planned,
            records,
            evaluations,
            perfs,
            str(self._group_index_path(repeat_group_id)),
            warnings=list(final_index.get("rebuild_warnings", [])),
        )

    def verify_group(self, repeat_group_id: str) -> dict[str, Any]:
        """Verify every member run and the group's lineage consistency (read-only)."""
        validate_repeat_group_id(repeat_group_id)
        members = self._scan_group_members(repeat_group_id)
        stored = self._read_group_index(repeat_group_id)
        problems: list[str] = []
        warnings: list[str] = []
        if not members:
            problems.append("No member runs carry this repeat_group_id in their lineage.")
        member_checks = []
        for m in members:
            check = RunIntegrityVerifier.verify_run_directory(Path(m.run_dir))
            member_checks.append(check.to_dict())
            if not check.valid:
                problems.append(f"Member {m.run_id} failed integrity verification.")
        indices = [m.repetition_index for m in members]
        if len(set(indices)) != len(indices):
            problems.append("Duplicate repetition indices in lineage.")
        counts = {int(m.lineage.get("repetition_count", 1)) for m in members}
        if len(counts) > 1:
            problems.append(f"Members disagree on repetition_count: {sorted(counts)}")
        for field_name in ("dataset_checksum", "model_name", "model_version", "model_provider"):
            values = {getattr(m, field_name) for m in members}
            if len(values) > 1:
                problems.append(f"Members disagree on {field_name}: {sorted(map(str, values))}")
        eval_checksums = {compute_dict_sha256(m.metric_configuration) for m in members}
        if len(eval_checksums) > 1:
            problems.append("Members disagree on the evaluation configuration.")
        planned = max(counts) if counts else (stored or {}).get("planned_repetitions", 0)
        missing = [i for i in range(int(planned or 0)) if i not in set(indices)]
        if missing:
            warnings.append(f"Missing repetition indices: {missing}")
        incomplete = [m.run_id for m in members if m.status != "completed"]
        if incomplete:
            warnings.append(f"Members not completed: {incomplete}")
        if stored is None:
            warnings.append("Group index file is missing; it can be rebuilt from lineage.")
        else:
            recomputed = compute_dict_sha256(
                {
                    "checksums": [
                        compute_file_sha256(Path(m.run_dir) / "reproducibility.json")
                        if (Path(m.run_dir) / "reproducibility.json").exists()
                        else ""
                        for m in members
                    ]
                }
            )
            if stored.get("members_reproducibility_sha256") != recomputed:
                warnings.append(
                    "Group index is stale relative to member reproducibility.json files; "
                    "lineage is authoritative (rebuild the index to refresh it)."
                )
        return {
            "repeat_group_id": repeat_group_id,
            "valid": not problems,
            "planned_repetitions": planned,
            "member_count": len(members),
            "problems": problems,
            "warnings": warnings,
            "members": member_checks,
            "integrity_disclaimer": (
                "SHA-256 checksums verify file integrity relative to recorded checksums; they "
                "do not prove that results are factually correct or authentic."
            ),
        }

    @staticmethod
    def _load_record_from_dir(run_dir: Path) -> ExperimentRecord:
        cfg_file = run_dir / "config.json"
        try:
            data = json.loads(cfg_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise OpenBenchError(f"Malformed config.json in {run_dir}: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise OpenBenchError(f"Malformed config.json in {run_dir}: expected a JSON object")

        raw_lineage = data.get("lineage")
        lineage = dict(raw_lineage) if isinstance(raw_lineage, dict) else legacy_lineage()

        try:
            return ExperimentManager._record_from_config(data, run_dir, lineage)
        except (KeyError, TypeError, ValueError) as exc:
            raise OpenBenchError(f"Malformed config.json in {run_dir}: {exc}") from exc

    @staticmethod
    def _record_from_config(
        data: dict[str, Any], run_dir: Path, lineage: dict[str, Any]
    ) -> ExperimentRecord:
        return ExperimentRecord(
            run_id=str(data["run_id"]),
            created_at=str(data["created_at"]),
            started_at=str(data["started_at"]),
            completed_at=data.get("completed_at"),
            status=str(data["status"]),
            execution_origin=str(data.get("execution_origin", "provider_executed")),
            dataset_identifier=str(data["dataset_identifier"]),
            dataset_path=str(data["dataset_path"]),
            dataset_version=str(data.get("dataset_version", "0.2.0")),
            dataset_checksum=str(data.get("dataset_checksum", "unknown")),
            model_provider=str(data.get("model_provider", "unspecified")),
            model_name=str(data["model_name"]),
            model_version=str(data["model_version"]),
            is_mock_provider=bool(data.get("is_mock_provider", False)),
            metric_configuration=dict(data.get("metric_configuration", {})),
            generation_configuration=dict(data.get("generation_configuration", {})),
            execution_configuration=dict(data.get("execution_configuration", {})),
            openbench_version=str(data.get("openbench_version", "0.3.0")),
            environment_metadata=dict(data.get("environment_metadata", {})),
            run_dir=str(run_dir),
            aggregate_scores=dict(data.get("aggregate_scores", {})),
            total_items=int(data.get("total_items", 0)),
            completed_items=int(data.get("completed_items", 0)),
            failed_items=int(data.get("failed_items", 0)),
            failure_info=dict(data.get("failure_info", {})),
            lineage=lineage,
        )


def evaluation_config_from_record(rec: ExperimentRecord) -> EvaluationConfig:
    """Rebuild the evaluation configuration of a stored run (for resumption; v0.4.0).

    Restores metric names, metric options, normalization and filters. For v0.4.0 runs the
    exact analysed record IDs from lineage are used, so a resumed run never widens its
    item set (v0.3.1 dropped ID filters, normalization and metric options on resume).
    """
    from openbench.metrics.normalization import NormalizationConfig

    cfg = rec.metric_configuration
    filters = cfg.get("filters") or {}
    selected = rec.lineage.get("selected_record_ids") if rec.lineage else None
    norm_raw = cfg.get("normalization")
    normalization = NormalizationConfig()
    if isinstance(norm_raw, dict):
        allowed = {k: v for k, v in norm_raw.items() if k in NormalizationConfig.__dataclass_fields__}
        normalization = NormalizationConfig(**allowed)
    metric_options = cfg.get("metric_options") or {}
    ids_file = None if selected else filters.get("ids_file")
    if ids_file and not Path(ids_file).exists():
        raise ConfigurationError(
            f"Run '{rec.run_id}' was evaluated with --ids-file '{ids_file}', which no longer "
            "exists; resuming would change the evaluated item set."
        )
    return EvaluationConfig(
        metric_names=list(cfg.get("metric_names", ["exact_match", "accuracy", "keyword_coverage"])),
        normalization=normalization,
        strict_unknown_ids=bool(cfg.get("strict_unknown_ids", False)),
        metric_options={k: dict(v) for k, v in metric_options.items() if isinstance(v, dict)},
        categories=filters.get("categories"),
        difficulties=filters.get("difficulties"),
        tags=filters.get("tags"),
        languages=filters.get("languages"),
        selected_ids=list(selected) if selected else filters.get("selected_ids"),
        ids_file=ids_file,
        profile_name=cfg.get("profile_name"),
    )


def execution_config_from_record(
    rec: ExperimentRecord, provider: BaseModelAdapter
) -> ExecutionConfig:
    """Rebuild a stored run's execution configuration (timeout, concurrency, retries, cache)."""
    from openbench.execution import RetryPolicy

    raw = rec.execution_configuration or {}
    retry_raw = raw.get("retry_policy") or {}
    try:
        retry = RetryPolicy(
            max_attempts=int(retry_raw.get("max_attempts", 3)),
            initial_backoff_seconds=float(retry_raw.get("initial_backoff_seconds", 0.05)),
            backoff_multiplier=float(retry_raw.get("backoff_multiplier", 2.0)),
            max_backoff_seconds=float(retry_raw.get("max_backoff_seconds", 2.0)),
            retry_on_timeouts=bool(retry_raw.get("retry_on_timeouts", True)),
        )
        return ExecutionConfig(
            timeout_seconds=float(raw.get("timeout_seconds", provider.timeout_seconds)),
            concurrency=int(raw.get("concurrency", 1)),
            retry_policy=retry,
            use_cache=bool(raw.get("use_cache", True)),
            estimate_missing_tokens=bool(raw.get("estimate_missing_tokens", False)),
            privacy_mode=bool(raw.get("privacy_mode", False)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Stored execution configuration is invalid: {exc}") from exc
