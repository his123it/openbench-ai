"""Model comparison module for OpenBench AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from openbench.evaluator import EvaluationResult
from openbench.exceptions import ConfigurationError


@dataclass(frozen=True)
class ModelComparisonEntry:
    """Comparison summary row for a single evaluated model."""

    model_name: str
    model_version: str
    dataset_name: str
    total_dataset_items: int
    evaluated_items: int
    missing_responses_count: int
    exact_match_score: float | None
    accuracy_score: float | None
    keyword_coverage_score: float | None
    metric_scores: dict[str, float] = field(default_factory=dict)
    metric_details: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def model_identifier(self) -> str:
        return f"{self.model_name} (v{self.model_version})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_identifier": self.model_identifier,
            "dataset_name": self.dataset_name,
            "total_dataset_items": self.total_dataset_items,
            "evaluated_items": self.evaluated_items,
            "missing_responses_count": self.missing_responses_count,
            "exact_match": (
                round(self.exact_match_score, 6)
                if self.exact_match_score is not None
                else None
            ),
            "accuracy": (
                round(self.accuracy_score, 6)
                if self.accuracy_score is not None
                else None
            ),
            "keyword_coverage": (
                round(self.keyword_coverage_score, 6)
                if self.keyword_coverage_score is not None
                else None
            ),
            "metric_scores": {k: round(v, 6) for k, v in self.metric_scores.items()},
            "metric_details": dict(self.metric_details),
        }


@dataclass(frozen=True)
class MetricLeader:
    """Identifies the highest score and leading model(s) for a specific metric."""

    metric_name: str
    highest_score: float
    leading_models: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "highest_score": round(self.highest_score, 6),
            "leading_models": list(self.leading_models),
            "is_tie": len(self.leading_models) > 1,
        }


@dataclass
class ComparisonResult:
    """Structured result comparing two or more model EvaluationResults."""

    comparison_timestamp: str
    dataset_name: str
    total_dataset_items: int
    models: list[ModelComparisonEntry]
    metrics_compared: list[str]
    highest_by_metric: dict[str, MetricLeader]
    warnings: list[str]
    methodology_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_timestamp": self.comparison_timestamp,
            "dataset_name": self.dataset_name,
            "total_dataset_items": self.total_dataset_items,
            "metrics_compared": list(self.metrics_compared),
            "models": [m.to_dict() for m in self.models],
            "highest_by_metric": {
                k: v.to_dict() for k, v in self.highest_by_metric.items()
            },
            "warnings": list(self.warnings),
            "methodology_note": self.methodology_note,
        }


class ModelComparator:
    """Compares two or more EvaluationResult objects across individual metrics.

    Design Principle:
    Different evaluation metrics measure fundamentally different properties
    (e.g., strict string equality vs. lexical keyword presence). Therefore,
    `ModelComparator` reports per-metric leaders and explicitly avoids synthesizing
    an arbitrary single 'overall winner' rank.
    """

    @staticmethod
    def compare(results: list[EvaluationResult]) -> ComparisonResult:
        """Compare two or more EvaluationResult instances."""
        if len(results) < 2:
            raise ConfigurationError(
                f"Model comparison requires at least 2 evaluation results, got {len(results)}"
            )

        warnings: list[str] = []
        dataset_names = {
            str(r.dataset_summary.get("name", "unknown")) for r in results
        }
        dataset_counts = {r.total_dataset_items for r in results}

        if len(dataset_names) > 1:
            warnings.append(
                f"Models were evaluated on different dataset names: {sorted(dataset_names)}. "
                "Cross-dataset score comparisons should be interpreted with caution."
            )
        if len(dataset_counts) > 1:
            warnings.append(
                f"Models were evaluated on datasets with differing item counts: {sorted(dataset_counts)}."
            )

        # Collect ordered union of all metrics present
        metrics_compared: list[str] = []
        for res in results:
            for m_name in res.aggregate_results:
                if m_name not in metrics_compared:
                    metrics_compared.append(m_name)

        entries: list[ModelComparisonEntry] = []
        for res in results:
            scores_map = {
                m_name: agg.score for m_name, agg in res.aggregate_results.items()
            }
            details_map = {
                m_name: agg.to_dict() for m_name, agg in res.aggregate_results.items()
            }
            entries.append(
                ModelComparisonEntry(
                    model_name=res.model_name,
                    model_version=res.model_version,
                    dataset_name=str(res.dataset_summary.get("name", "unknown")),
                    total_dataset_items=res.total_dataset_items,
                    evaluated_items=res.evaluated_items_count,
                    missing_responses_count=res.missing_responses_count,
                    exact_match_score=scores_map.get("exact_match"),
                    accuracy_score=scores_map.get("accuracy"),
                    keyword_coverage_score=scores_map.get("keyword_coverage"),
                    metric_scores=scores_map,
                    metric_details=details_map,
                )
            )

        highest_by_metric: dict[str, MetricLeader] = {}
        for m_name in metrics_compared:
            candidates = [
                (entry.model_identifier, entry.metric_scores[m_name])
                for entry in entries
                if m_name in entry.metric_scores
            ]
            if not candidates:
                continue
            max_score = max(score for _, score in candidates)
            leaders = [
                ident
                for ident, score in candidates
                if abs(score - max_score) < 1e-9
            ]
            highest_by_metric[m_name] = MetricLeader(
                metric_name=m_name,
                highest_score=max_score,
                leading_models=leaders,
            )

        primary_dataset = str(results[0].dataset_summary.get("name", "unknown"))
        primary_count = results[0].total_dataset_items
        timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

        methodology_note = (
            "Each metric evaluates a distinct aspect of model output (strict formatting vs. "
            "deterministic answer containment vs. lexical keyword coverage). No composite "
            "overall score is computed because averaging heterogeneous metrics can obscure "
            "important failure modes."
        )

        return ComparisonResult(
            comparison_timestamp=timestamp,
            dataset_name=primary_dataset,
            total_dataset_items=primary_count,
            models=entries,
            metrics_compared=metrics_compared,
            highest_by_metric=highest_by_metric,
            warnings=warnings,
            methodology_note=methodology_note,
        )
