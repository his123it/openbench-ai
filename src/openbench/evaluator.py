"""Core evaluation engine and model response loader for OpenBench AI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openbench.dataset import Dataset, DatasetLoader
from openbench.exceptions import ConfigurationError, ResponseValidationError
from openbench.metrics import (
    AggregateMetricResult,
    BaseMetric,
    MetricResult,
    NormalizationConfig,
    create_metric,
)

DEFAULT_METRIC_NAMES = ("exact_match", "accuracy", "keyword_coverage")


@dataclass(frozen=True)
class ResponseItem:
    """Represents a single model response entry."""

    id: str
    response: str


@dataclass
class ModelResponseSet:
    """Represents a validated set of responses produced by a single language model."""

    model_name: str
    model_version: str
    responses: list[ResponseItem]
    source_path: str | None = None

    @property
    def response_map(self) -> dict[str, str]:
        """Return mapping from dataset item ID to model response string."""
        return {item.id: item.response for item in self.responses}


class ResponseLoader:
    """Loads and validates model response JSON files."""

    @staticmethod
    def load_file(file_path: str | Path) -> ModelResponseSet:
        """Load a single model response set (or throw error if file contains multiple without load_all)."""
        sets = ResponseLoader.load_all(file_path)
        if len(sets) != 1:
            raise ResponseValidationError(
                f"Expected a single model response object in {file_path}, "
                f"but found {len(sets)} model entries. Use ResponseLoader.load_all() for multi-model files.",
                file_path=file_path,
            )
        return sets[0]

    @staticmethod
    def load_all(file_path: str | Path) -> list[ModelResponseSet]:
        """Load one or more model response sets from a JSON file.

        Supports either:
        - A single JSON object: `{"model_name": "...", "model_version": "...", "responses": [...]}`
        - A JSON object with `"models": [...]`
        - A top-level JSON list of model objects
        """
        target = Path(file_path)
        if not target.exists():
            raise FileNotFoundError(f"Response file does not exist: {target}")
        if target.is_dir():
            raise ResponseValidationError(
                "Expected a JSON response file path, but received a directory",
                file_path=target,
            )

        raw_text = target.read_text(encoding="utf-8")
        if not raw_text.strip():
            raise ResponseValidationError(
                "Response JSON file is empty",
                file_path=target,
            )

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ResponseValidationError(
                f"Malformed JSON in response file: {exc.msg} at line {exc.lineno}",
                file_path=target,
            ) from exc

        if isinstance(payload, dict) and "models" in payload:
            models_list = payload["models"]
            if not isinstance(models_list, list) or not models_list:
                raise ResponseValidationError(
                    "Field 'models' must be a non-empty list of model response objects",
                    file_path=target,
                )
            return [
                ResponseLoader._validate_model_payload(item, file_path=target)
                for item in models_list
            ]

        if isinstance(payload, list):
            if not payload:
                raise ResponseValidationError(
                    "Response file contains an empty JSON array",
                    file_path=target,
                )
            return [
                ResponseLoader._validate_model_payload(item, file_path=target)
                for item in payload
            ]

        if isinstance(payload, dict):
            return [ResponseLoader._validate_model_payload(payload, file_path=target)]

        raise ResponseValidationError(
            f"Expected JSON object or array in response file, got {type(payload).__name__}",
            file_path=target,
        )

    @staticmethod
    def _validate_model_payload(payload: Any, file_path: Path | None = None) -> ModelResponseSet:
        """Validate a single model response dictionary."""
        if not isinstance(payload, dict):
            raise ResponseValidationError(
                f"Expected model response object to be a dict, got {type(payload).__name__}",
                file_path=file_path,
            )

        for req_field in ("model_name", "model_version"):
            if req_field not in payload or payload[req_field] is None:
                raise ResponseValidationError(
                    f"Missing required field '{req_field}' in response payload",
                    file_path=file_path,
                )
            if not isinstance(payload[req_field], str) or not payload[req_field].strip():
                raise ResponseValidationError(
                    f"Field '{req_field}' must be a non-empty string",
                    file_path=file_path,
                )

        if "responses" not in payload or not isinstance(payload["responses"], list):
            raise ResponseValidationError(
                "Field 'responses' must be a list of response objects",
                file_path=file_path,
            )

        items: list[ResponseItem] = []
        seen_ids: set[str] = set()

        for idx, raw_item in enumerate(payload["responses"], start=1):
            if not isinstance(raw_item, dict):
                raise ResponseValidationError(
                    f"Response entry #{idx} must be a JSON object",
                    file_path=file_path,
                )
            if "id" not in raw_item or raw_item["id"] is None:
                raise ResponseValidationError(
                    f"Response entry #{idx} is missing required field 'id'",
                    file_path=file_path,
                )
            if not isinstance(raw_item["id"], str) or not raw_item["id"].strip():
                raise ResponseValidationError(
                    f"Response entry #{idx} field 'id' must be a non-empty string",
                    file_path=file_path,
                )
            item_id = raw_item["id"].strip()

            if item_id in seen_ids:
                raise ResponseValidationError(
                    f"Duplicate response ID '{item_id}' detected in model '{payload['model_name']}'",
                    file_path=file_path,
                    record_id=item_id,
                )
            seen_ids.add(item_id)

            if "response" not in raw_item or raw_item["response"] is None:
                raise ResponseValidationError(
                    f"Response entry for ID '{item_id}' is missing required field 'response'",
                    file_path=file_path,
                    record_id=item_id,
                )
            if not isinstance(raw_item["response"], str):
                raise ResponseValidationError(
                    f"Response field for ID '{item_id}' must be a string",
                    file_path=file_path,
                    record_id=item_id,
                )

            items.append(ResponseItem(id=item_id, response=raw_item["response"]))

        return ModelResponseSet(
            model_name=payload["model_name"].strip(),
            model_version=payload["model_version"].strip(),
            responses=items,
            source_path=str(file_path) if file_path else None,
        )


@dataclass
class EvaluationConfig:
    """Configuration for an evaluation run."""

    metric_names: list[str] = field(default_factory=lambda: list(DEFAULT_METRIC_NAMES))
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    strict_unknown_ids: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_names": list(self.metric_names),
            "normalization": self.normalization.to_dict(),
            "strict_unknown_ids": self.strict_unknown_ids,
        }


@dataclass
class ItemEvaluationResult:
    """Evaluation outcome for a single dataset item."""

    id: str
    category: str
    difficulty: str
    prompt: str
    expected_answer: str
    model_response: str | None
    status: str  # "evaluated" | "missing"
    metrics: dict[str, MetricResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "expected_answer": self.expected_answer,
            "model_response": self.model_response,
            "status": self.status,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }


@dataclass
class EvaluationResult:
    """Complete structured result of evaluating one model on one dataset."""

    project_version: str
    evaluation_timestamp: str
    dataset_summary: dict[str, Any]
    model_name: str
    model_version: str
    config: dict[str, Any]
    metrics_used: list[dict[str, Any]]
    aggregate_results: dict[str, AggregateMetricResult]
    individual_results: list[ItemEvaluationResult]
    missing_response_ids: list[str]
    unknown_response_ids: list[str]
    warnings: list[str]
    limitations: list[str]

    @property
    def total_dataset_items(self) -> int:
        return int(self.dataset_summary.get("total_records", len(self.individual_results)))

    @property
    def evaluated_items_count(self) -> int:
        return sum(1 for item in self.individual_results if item.status == "evaluated")

    @property
    def missing_responses_count(self) -> int:
        return len(self.missing_response_ids)

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete evaluation result to a JSON-compatible dictionary."""
        return {
            "project_version": self.project_version,
            "evaluation_timestamp": self.evaluation_timestamp,
            "dataset": dict(self.dataset_summary),
            "model": {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "total_dataset_items": self.total_dataset_items,
                "evaluated_items": self.evaluated_items_count,
                "missing_responses_count": self.missing_responses_count,
                "unknown_responses_count": len(self.unknown_response_ids),
            },
            "configuration": dict(self.config),
            "metrics_used": list(self.metrics_used),
            "aggregate_results": {
                k: v.to_dict() for k, v in self.aggregate_results.items()
            },
            "individual_results": [item.to_dict() for item in self.individual_results],
            "missing_responses": list(self.missing_response_ids),
            "unknown_response_ids": list(self.unknown_response_ids),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvaluationResult:
        """Reconstruct an EvaluationResult from a previously saved JSON report dictionary."""
        required_keys = (
            "project_version",
            "evaluation_timestamp",
            "dataset",
            "model",
            "aggregate_results",
            "individual_results",
        )
        for key in required_keys:
            if key not in payload:
                raise ConfigurationError(f"Saved evaluation result is missing required key '{key}'")

        agg_map: dict[str, AggregateMetricResult] = {}
        for m_name, m_data in payload["aggregate_results"].items():
            agg_map[m_name] = AggregateMetricResult(
                metric_name=m_data["metric_name"],
                metric_version=m_data.get("metric_version", "0.1.0"),
                score=float(m_data["score"]),
                total_items=int(m_data["total_items"]),
                evaluated_items=int(m_data["evaluated_items"]),
                passed_items=int(m_data["passed_items"]),
                details=dict(m_data.get("details", {})),
            )

        indiv_list: list[ItemEvaluationResult] = []
        for item_data in payload["individual_results"]:
            m_results: dict[str, MetricResult] = {}
            for m_name, mr in item_data.get("metrics", {}).items():
                m_results[m_name] = MetricResult(
                    metric_name=mr["metric_name"],
                    score=float(mr["score"]),
                    passed=bool(mr["passed"]),
                    details=dict(mr.get("details", {})),
                )
            indiv_list.append(
                ItemEvaluationResult(
                    id=item_data["id"],
                    category=item_data.get("category", "unknown"),
                    difficulty=item_data.get("difficulty", "unspecified"),
                    prompt=item_data.get("prompt", ""),
                    expected_answer=item_data.get("expected_answer", ""),
                    model_response=item_data.get("model_response"),
                    status=item_data.get("status", "evaluated"),
                    metrics=m_results,
                )
            )

        model_info = payload["model"]
        return cls(
            project_version=str(payload["project_version"]),
            evaluation_timestamp=str(payload["evaluation_timestamp"]),
            dataset_summary=dict(payload["dataset"]),
            model_name=str(model_info["model_name"]),
            model_version=str(model_info["model_version"]),
            config=dict(payload.get("configuration", {})),
            metrics_used=list(payload.get("metrics_used", [])),
            aggregate_results=agg_map,
            individual_results=indiv_list,
            missing_response_ids=list(payload.get("missing_responses", [])),
            unknown_response_ids=list(payload.get("unknown_response_ids", [])),
            warnings=list(payload.get("warnings", [])),
            limitations=list(payload.get("limitations", [])),
        )


class Evaluator:
    """Central evaluation engine for matching model responses to datasets and computing metrics."""

    def __init__(
        self,
        metrics: list[BaseMetric] | None = None,
        config: EvaluationConfig | None = None,
    ) -> None:
        self.config = config or EvaluationConfig()
        if metrics is not None:
            if not metrics:
                raise ConfigurationError("At least one metric must be provided to Evaluator")
            self.metrics = list(metrics)
        else:
            if not self.config.metric_names:
                raise ConfigurationError("EvaluationConfig.metric_names cannot be empty")
            self.metrics = [
                create_metric(name, normalization=self.config.normalization)
                for name in self.config.metric_names
            ]

    def evaluate(
        self,
        dataset: Dataset,
        response_set: ModelResponseSet,
    ) -> EvaluationResult:
        """Evaluate a single ModelResponseSet against a validated Dataset."""
        from openbench import __version__

        dataset_ids = {rec.id for rec in dataset.records}
        response_map = response_set.response_map

        unknown_ids = sorted(rid for rid in response_map if rid not in dataset_ids)
        if unknown_ids and self.config.strict_unknown_ids:
            raise ResponseValidationError(
                f"Response set contains {len(unknown_ids)} unknown dataset ID(s): "
                f"{', '.join(unknown_ids)}"
            )

        missing_ids: list[str] = []
        individual_results: list[ItemEvaluationResult] = []
        per_metric_items: dict[str, list[MetricResult]] = {m.name: [] for m in self.metrics}

        for record in dataset.records:
            if record.id not in response_map:
                missing_ids.append(record.id)
                individual_results.append(
                    ItemEvaluationResult(
                        id=record.id,
                        category=record.category,
                        difficulty=record.difficulty,
                        prompt=record.prompt,
                        expected_answer=record.expected_answer,
                        model_response=None,
                        status="missing",
                        metrics={},
                    )
                )
                continue

            resp_text = response_map[record.id]
            item_metrics: dict[str, MetricResult] = {}
            for metric in self.metrics:
                m_res = metric.compute_item(record, resp_text)
                item_metrics[metric.name] = m_res
                per_metric_items[metric.name].append(m_res)

            individual_results.append(
                ItemEvaluationResult(
                    id=record.id,
                    category=record.category,
                    difficulty=record.difficulty,
                    prompt=record.prompt,
                    expected_answer=record.expected_answer,
                    model_response=resp_text,
                    status="evaluated",
                    metrics=item_metrics,
                )
            )

        aggregate_results: dict[str, AggregateMetricResult] = {}
        for metric in self.metrics:
            aggregate_results[metric.name] = metric.aggregate(
                per_metric_items[metric.name],
                total_dataset_items=len(dataset.records),
            )

        warnings: list[str] = []
        if missing_ids:
            warnings.append(
                f"{len(missing_ids)} dataset item(s) had no matching response from model "
                f"'{response_set.model_name}' and were scored as 0.0 in aggregate metrics: "
                f"{', '.join(missing_ids)}"
            )
        if unknown_ids:
            warnings.append(
                f"{len(unknown_ids)} response ID(s) did not match any record in dataset "
                f"'{dataset.name}' and were ignored: {', '.join(unknown_ids)}"
            )

        limitations = [
            "OpenBench AI v0.1.0 uses deterministic reference-based metrics and does not "
            "measure general LLM intelligence or open-ended reasoning quality.",
            "Keyword coverage checks lexical presence only and is not equivalent to semantic correctness.",
        ]
        if "safety" in dataset.categories:
            limitations.append(
                "Safety dataset evaluation in v0.1.0 uses simple reference-based checks and "
                "does NOT provide comprehensive safety or alignment certification."
            )

        timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        return EvaluationResult(
            project_version=__version__,
            evaluation_timestamp=timestamp,
            dataset_summary=dataset.summary(),
            model_name=response_set.model_name,
            model_version=response_set.model_version,
            config=self.config.to_dict(),
            metrics_used=[m.config_summary() for m in self.metrics],
            aggregate_results=aggregate_results,
            individual_results=individual_results,
            missing_response_ids=missing_ids,
            unknown_response_ids=unknown_ids,
            warnings=warnings,
            limitations=limitations,
        )

    def evaluate_multiple(
        self,
        dataset: Dataset,
        response_sets: list[ModelResponseSet],
    ) -> list[EvaluationResult]:
        """Evaluate multiple models independently on the same dataset."""
        if not response_sets:
            raise ConfigurationError("At least one ModelResponseSet is required for evaluation")
        return [self.evaluate(dataset, rs) for rs in response_sets]

    def evaluate_files(
        self,
        dataset_path: str | Path,
        responses_path: str | Path,
    ) -> list[EvaluationResult]:
        """Convenience method to load a dataset and response file and evaluate all models inside."""
        dataset = DatasetLoader.load(dataset_path)
        response_sets = ResponseLoader.load_all(responses_path)
        return self.evaluate_multiple(dataset, response_sets)
