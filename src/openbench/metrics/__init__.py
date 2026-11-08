"""Metrics package and registry for OpenBench AI."""

from __future__ import annotations

from openbench.exceptions import MetricError
from openbench.metrics.accuracy import AccuracyMetric
from openbench.metrics.base import AggregateMetricResult, BaseMetric, MetricResult
from openbench.metrics.exact_match import ExactMatchMetric
from openbench.metrics.keyword_match import KeywordCoverageMetric
from openbench.metrics.normalization import NormalizationConfig, normalize_text

METRIC_REGISTRY: dict[str, type[BaseMetric]] = {
    ExactMatchMetric.name: ExactMatchMetric,
    AccuracyMetric.name: AccuracyMetric,
    KeywordCoverageMetric.name: KeywordCoverageMetric,
    # Convenient alias so both "keyword_coverage" and "keyword_match" work seamlessly
    "keyword_match": KeywordCoverageMetric,
}


def create_metric(
    name: str,
    normalization: NormalizationConfig | None = None,
) -> BaseMetric:
    """Instantiate a registered metric by name."""
    key = name.strip().lower()
    if key not in METRIC_REGISTRY:
        available = sorted({m.name for m in METRIC_REGISTRY.values()})
        raise MetricError(
            f"Unknown metric '{name}'. Available metrics: {', '.join(available)}"
        )
    metric_cls = METRIC_REGISTRY[key]
    return metric_cls(normalization=normalization)


def register_metric(metric_cls: type[BaseMetric]) -> None:
    """Register a custom BaseMetric subclass for lookup by name."""
    if not issubclass(metric_cls, BaseMetric):
        raise MetricError("Custom metric class must inherit from BaseMetric")
    METRIC_REGISTRY[metric_cls.name] = metric_cls


__all__ = [
    "AccuracyMetric",
    "AggregateMetricResult",
    "BaseMetric",
    "ExactMatchMetric",
    "KeywordCoverageMetric",
    "METRIC_REGISTRY",
    "MetricResult",
    "NormalizationConfig",
    "create_metric",
    "normalize_text",
    "register_metric",
]
