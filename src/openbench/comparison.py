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
    category_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    difficulty_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    evaluation_id: str | None = None
    evaluation_timestamp: str | None = None
    weighted_score: float | None = None

    @property
    def model_identifier(self) -> str:
        return f"{self.model_name} (v{self.model_version})"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
            "category_scores": dict(self.category_scores),
            "difficulty_scores": dict(self.difficulty_scores),
        }
        if self.evaluation_id is not None:
            payload["evaluation_id"] = self.evaluation_id
        if self.evaluation_timestamp is not None:
            payload["evaluation_timestamp"] = self.evaluation_timestamp
        if self.weighted_score is not None:
            payload["weighted_score"] = round(self.weighted_score, 6)
        return payload


@dataclass(frozen=True)
class MetricLeader:
    """Identifies the highest score and leading model(s) for a specific metric."""

    metric_name: str
    highest_score: float
    leading_models: list[str]
    runner_up_score: float | None = None
    score_margin: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "highest_score": round(self.highest_score, 6),
            "leading_models": list(self.leading_models),
            "is_tie": len(self.leading_models) > 1,
            "runner_up_score": (
                round(self.runner_up_score, 6)
                if self.runner_up_score is not None
                else None
            ),
            "score_margin": round(self.score_margin, 6),
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
    category_comparison: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    difficulty_comparison: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    score_differences: dict[str, dict[str, float]] = field(default_factory=dict)
    relative_performance: list[str] = field(default_factory=list)
    weights_used: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "comparison_timestamp": self.comparison_timestamp,
            "dataset_name": self.dataset_name,
            "total_dataset_items": self.total_dataset_items,
            "metrics_compared": list(self.metrics_compared),
            "models": [m.to_dict() for m in self.models],
            "highest_by_metric": {
                k: v.to_dict() for k, v in self.highest_by_metric.items()
            },
            "category_comparison": dict(self.category_comparison),
            "difficulty_comparison": dict(self.difficulty_comparison),
            "score_differences": dict(self.score_differences),
            "relative_performance": list(self.relative_performance),
            "warnings": list(self.warnings),
            "methodology_note": self.methodology_note,
        }
        if self.weights_used is not None:
            payload["weights_used"] = {k: round(v, 6) for k, v in self.weights_used.items()}
        return payload


class ModelComparator:
    """Compares two or more EvaluationResult objects across individual metrics, categories, and difficulties.

    Design Principle:
    Different evaluation metrics measure fundamentally different properties
    (e.g., strict string equality vs. lexical keyword presence). Therefore,
    `ModelComparator` reports per-metric leaders and explicitly avoids synthesizing
    an arbitrary single 'overall winner' rank.

    Optional User-Weighted Score:
    If the caller explicitly supplies a valid `weights: dict[str, float]` mapping
    metric names to positive numeric weights, `ModelComparator` normalizes the weights
    to sum to 1.0, computes the user-defined weighted score alongside all underlying
    metric scores, and documents that the weighted score reflects user preferences
    rather than universal superiority.
    """

    @staticmethod
    def compare(
        results: list[EvaluationResult],
        weights: dict[str, float] | None = None,
    ) -> ComparisonResult:
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

        metrics_compared: list[str] = []
        for res in results:
            for m_name in res.aggregate_results:
                if m_name not in metrics_compared:
                    metrics_compared.append(m_name)

        normalized_weights: dict[str, float] | None = None
        if weights is not None:
            if not isinstance(weights, dict) or not weights:
                raise ConfigurationError("Explicit weights must be a non-empty dictionary of metric weights")
            for m_name, w_val in weights.items():
                if not isinstance(w_val, (int, float)) or isinstance(w_val, bool) or w_val <= 0:
                    raise ConfigurationError(
                        f"Weight for metric '{m_name}' must be a positive number, got {w_val}"
                    )
                if m_name not in metrics_compared:
                    raise ConfigurationError(
                        f"Weighted metric '{m_name}' is not present in the compared evaluation results "
                        f"({metrics_compared})"
                    )
            total_w = sum(float(v) for v in weights.values())
            normalized_weights = {k: float(v) / total_w for k, v in weights.items()}

        entries: list[ModelComparisonEntry] = []
        category_comparison: dict[str, dict[str, dict[str, float]]] = {}
        difficulty_comparison: dict[str, dict[str, dict[str, float]]] = {}

        # Handle duplicate model identifiers (e.g., historical comparisons of the same model at different timestamps)
        seen_identifiers: dict[str, int] = {}

        for res in results:
            scores_map = {
                m_name: agg.score for m_name, agg in res.aggregate_results.items()
            }
            details_map = {
                m_name: agg.to_dict() for m_name, agg in res.aggregate_results.items()
            }

            cat_scores: dict[str, dict[str, float]] = {}
            for cat_name, cat_data in res.category_breakdown.items():
                cat_m = dict(cat_data.get("metrics", {}))
                cat_scores[cat_name] = cat_m
                category_comparison.setdefault(cat_name, {})[
                    f"{res.model_name} (v{res.model_version})"
                ] = cat_m

            diff_scores: dict[str, dict[str, float]] = {}
            for diff_name, diff_data in res.difficulty_breakdown.items():
                diff_m = dict(diff_data.get("metrics", {}))
                diff_scores[diff_name] = diff_m
                difficulty_comparison.setdefault(diff_name, {})[
                    f"{res.model_name} (v{res.model_version})"
                ] = diff_m

            w_score: float | None = None
            if normalized_weights is not None:
                if all(m in scores_map for m in normalized_weights):
                    w_score = sum(
                        scores_map[m] * w for m, w in normalized_weights.items()
                    )

            entry = ModelComparisonEntry(
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
                category_scores=cat_scores,
                difficulty_scores=diff_scores,
                evaluation_id=res.evaluation_id,
                evaluation_timestamp=res.evaluation_timestamp,
                weighted_score=w_score,
            )
            seen_identifiers[entry.model_identifier] = (
                seen_identifiers.get(entry.model_identifier, 0) + 1
            )
            entries.append(entry)

        highest_by_metric: dict[str, MetricLeader] = {}
        score_differences: dict[str, dict[str, float]] = {}
        relative_performance: list[str] = []

        for m_name in metrics_compared:
            candidates = [
                (entry.model_identifier, entry.metric_scores[m_name])
                for entry in entries
                if m_name in entry.metric_scores
            ]
            if not candidates:
                continue
            max_score = max(score for _, score in candidates)
            min_score = min(score for _, score in candidates)
            leaders = [
                ident
                for ident, score in candidates
                if abs(score - max_score) < 1e-9
            ]
            non_leaders = [
                score for _, score in candidates if abs(score - max_score) >= 1e-9
            ]
            runner_up = max(non_leaders) if non_leaders else (max_score if len(leaders) > 1 else None)
            margin = (max_score - runner_up) if runner_up is not None else 0.0

            highest_by_metric[m_name] = MetricLeader(
                metric_name=m_name,
                highest_score=max_score,
                leading_models=leaders,
                runner_up_score=runner_up,
                score_margin=margin,
            )

            score_differences[m_name] = {
                "max_score": round(max_score, 6),
                "min_score": round(min_score, 6),
                "spread": round(max_score - min_score, 6),
                "deltas_from_leader": {
                    ident: round(score - max_score, 6) for ident, score in candidates
                },
            }

            if len(leaders) > 1:
                relative_performance.append(
                    f"Metric '{m_name}': Tie between {', '.join(leaders)} at {max_score:.4f} ({max_score * 100:.2f}%)."
                )
            elif runner_up is not None:
                relative_performance.append(
                    f"Metric '{m_name}': {leaders[0]} leads with {max_score:.4f} "
                    f"(+{margin:.4f} ahead of runner-up at {runner_up:.4f})."
                )
            else:
                relative_performance.append(
                    f"Metric '{m_name}': {leaders[0]} leads with {max_score:.4f}."
                )

        primary_dataset = str(results[0].dataset_summary.get("name", "unknown"))
        primary_count = results[0].total_dataset_items
        timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

        methodology_note = (
            "Each metric evaluates a distinct aspect of model output (strict formatting vs. "
            "deterministic answer containment vs. lexical or structural alignment). No default composite "
            "overall score is computed because averaging heterogeneous metrics can obscure "
            "important failure modes. Higher scores on a single dataset do not imply universal superiority."
        )
        if normalized_weights is not None:
            methodology_note += (
                " An optional user-weighted score was explicitly requested and computed using "
                f"normalized weights {normalized_weights}; this reflects user-specified trade-offs only."
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
            category_comparison=category_comparison,
            difficulty_comparison=difficulty_comparison,
            score_differences=score_differences,
            relative_performance=relative_performance,
            weights_used=normalized_weights,
        )

    @staticmethod
    def compare_history(runs: list[EvaluationResult]) -> dict[str, Any]:
        """Compare multiple historical evaluation runs chronologically."""
        if len(runs) < 2:
            raise ConfigurationError("Historical comparison requires at least 2 evaluation runs")
        ordered = sorted(runs, key=lambda r: r.evaluation_timestamp)
        baseline = ordered[0]
        latest = ordered[-1]

        metric_deltas: dict[str, dict[str, float]] = {}
        for m_name, latest_agg in latest.aggregate_results.items():
            if m_name in baseline.aggregate_results:
                base_score = baseline.aggregate_results[m_name].score
                metric_deltas[m_name] = {
                    "baseline_score": round(base_score, 6),
                    "latest_score": round(latest_agg.score, 6),
                    "delta": round(latest_agg.score - base_score, 6),
                }

        return {
            "baseline_evaluation_id": baseline.evaluation_id,
            "baseline_timestamp": baseline.evaluation_timestamp,
            "latest_evaluation_id": latest.evaluation_id,
            "latest_timestamp": latest.evaluation_timestamp,
            "total_runs_compared": len(ordered),
            "metric_deltas": metric_deltas,
            "runs": [
                {
                    "evaluation_id": r.evaluation_id,
                    "timestamp": r.evaluation_timestamp,
                    "model_name": r.model_name,
                    "model_version": r.model_version,
                    "dataset_name": r.dataset_summary.get("name", "unknown"),
                    "scores": {k: round(v.score, 6) for k, v in r.aggregate_results.items()},
                }
                for r in ordered
            ],
        }
