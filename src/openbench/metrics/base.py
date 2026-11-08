"""Base metric interfaces, metadata, and structured result dataclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from openbench.dataset import DatasetRecord


@dataclass(frozen=True)
class MetricMetadata:
    """Descriptive metadata for a registered OpenBench metric."""

    name: str
    version: str
    description: str
    limitation_note: str
    is_optional: bool = False
    optional_dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "limitation_note": self.limitation_note,
            "is_optional": self.is_optional,
            "optional_dependencies": list(self.optional_dependencies),
        }


@dataclass(frozen=True)
class MetricResult:
    """Structured result for a single evaluation item on a single metric."""

    metric_name: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize item metric result to a dictionary."""
        return {
            "metric_name": self.metric_name,
            "score": round(self.score, 6),
            "passed": self.passed,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class AggregateMetricResult:
    """Aggregate metric summary across a collection of evaluated items."""

    metric_name: str
    metric_version: str
    score: float
    total_items: int
    evaluated_items: int
    passed_items: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize aggregate metric result to a dictionary."""
        return {
            "metric_name": self.metric_name,
            "metric_version": self.metric_version,
            "score": round(self.score, 6),
            "total_items": self.total_items,
            "evaluated_items": self.evaluated_items,
            "passed_items": self.passed_items,
            "details": dict(self.details),
        }


class BaseMetric(ABC):
    """Abstract base interface for all OpenBench AI evaluation metrics.

    Developers can subclass `BaseMetric` to implement custom evaluation metrics
    that return structured item-level and aggregate results.
    """

    name: str = "base_metric"
    version: str = "0.1.0"
    description: str = "Abstract base metric."
    limitation_note: str = ""
    is_optional: bool = False
    optional_dependencies: tuple[str, ...] = ()

    @classmethod
    def get_metadata(cls) -> MetricMetadata:
        """Return structured metadata describing this metric class."""
        return MetricMetadata(
            name=cls.name,
            version=cls.version,
            description=cls.description,
            limitation_note=cls.limitation_note,
            is_optional=cls.is_optional,
            optional_dependencies=list(cls.optional_dependencies),
        )

    @abstractmethod
    def compute_item(self, record: DatasetRecord, response_text: str) -> MetricResult:
        """Evaluate a single model response against a dataset record."""

    def aggregate(
        self,
        item_results: list[MetricResult],
        total_dataset_items: int,
    ) -> AggregateMetricResult:
        """Compute aggregate statistics over all evaluated item results.

        Note: Missing responses are counted in `total_dataset_items` and penalized
        as 0.0 in `score` so models cannot inflate scores by omitting difficult prompts.
        Both `score` and `evaluated_only_score` are reported.
        """
        evaluated_count = len(item_results)
        passed_count = sum(1 for r in item_results if r.passed)
        total_score_sum = sum(r.score for r in item_results)

        denominator = max(total_dataset_items, 1)
        overall_score = total_score_sum / denominator if total_dataset_items > 0 else 0.0
        evaluated_only_score = (
            total_score_sum / evaluated_count if evaluated_count > 0 else 0.0
        )

        return AggregateMetricResult(
            metric_name=self.name,
            metric_version=self.version,
            score=overall_score,
            total_items=total_dataset_items,
            evaluated_items=evaluated_count,
            passed_items=passed_count,
            details={
                "evaluated_only_score": round(evaluated_only_score, 6),
                "missing_items": max(total_dataset_items - evaluated_count, 0),
                "limitation": self.limitation_note,
            },
        )

    def config_summary(self) -> dict[str, Any]:
        """Return a dictionary describing the metric's active configuration."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
        }
