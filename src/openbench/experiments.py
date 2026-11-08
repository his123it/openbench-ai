"""Structured Experiment & Run Management System for OpenBench AI (v0.3.0)."""

from __future__ import annotations

import json
import platform
import uuid
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

    def to_dict(self) -> dict[str, Any]:
        return {
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
        folder_name = run_id if run_id.startswith("run_") else f"run_{run_id}"
        return self.root / folder_name

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
    ) -> tuple[ExperimentRecord, EvaluationResult, PerformanceReport]:
        """Run a dataset through `ExecutionEngine`, evaluate via `Evaluator`, and persist all 7 run artifacts."""
        from openbench import __version__

        eval_cfg = evaluation_config or EvaluationConfig()
        exec_cfg = execution_config or ExecutionConfig(timeout_seconds=provider.timeout_seconds)

        evaluator = Evaluator(config=eval_cfg)
        active_dataset = evaluator._apply_configured_filters(dataset)

        resolved_run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        if not resolved_run_id.startswith("run_"):
            resolved_run_id = f"run_{resolved_run_id}"
        run_dir = self._run_dir(resolved_run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        engine = ExecutionEngine(provider=provider, config=exec_cfg, cache=cache)
        outcome = engine.execute_dataset(
            active_dataset,
            existing_responses=existing_responses,
        )

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

        return record, eval_result, perf_report

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
            "schema_version": "0.3.0",
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
        repro_path.write_text(json.dumps(repro_payload, indent=2) + "\n", encoding="utf-8")

    def list_runs(
        self,
        status: str | None = None,
        provider: str | None = None,
    ) -> list[ExperimentRecord]:
        """List all experiment runs stored under `self.root`."""
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

        filters = rec.metric_configuration.get("filters", {})
        eval_cfg = EvaluationConfig(
            metric_names=rec.metric_configuration.get("metric_names", ["exact_match", "accuracy", "keyword_coverage"]),
            categories=filters.get("categories"),
            difficulties=filters.get("difficulties"),
            tags=filters.get("tags"),
            languages=filters.get("languages"),
        )

        return self.execute_and_record(
            dataset=dataset,
            provider=provider,
            evaluation_config=eval_cfg,
            cache=cache,
            run_id=rec.run_id,
            existing_responses=preserved_items,
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

    @staticmethod
    def _load_record_from_dir(run_dir: Path) -> ExperimentRecord:
        cfg_file = run_dir / "config.json"
        try:
            data = json.loads(cfg_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise OpenBenchError(f"Malformed config.json in {run_dir}: {exc.msg}") from exc

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
        )
