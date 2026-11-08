"""Metrics package and registry for OpenBench AI."""

from __future__ import annotations

import inspect
from typing import Any

from openbench.exceptions import MetricError
from openbench.metrics.accuracy import AccuracyMetric
from openbench.metrics.base import (
    AggregateMetricResult,
    BaseMetric,
    MetricMetadata,
    MetricResult,
)
from openbench.metrics.bleu import BleuMetric
from openbench.metrics.exact_match import ExactMatchMetric
from openbench.metrics.f1 import F1Metric
from openbench.metrics.keyword_match import KeywordCoverageMetric
from openbench.metrics.normalization import NormalizationConfig, normalize_text
from openbench.metrics.precision import PrecisionMetric
from openbench.metrics.recall import RecallMetric
from openbench.metrics.rouge import Rouge1Metric, RougeLMetric, RougeMetric
from openbench.metrics.semantic_similarity import (
    CallableEmbeddingProvider,
    EmbeddingProvider,
    HuggingFaceEmbeddingProvider,
    LexicalSimilarityMetric,
    LocalNgramEmbeddingProvider,
    SemanticSimilarityMetric,
    cosine_similarity,
)
from openbench.metrics.structured_output import StructuredOutputMetric, extract_fenced_json

METRIC_REGISTRY: dict[str, type[BaseMetric]] = {
    # v0.1.0 core metrics
    ExactMatchMetric.name: ExactMatchMetric,
    AccuracyMetric.name: AccuracyMetric,
    KeywordCoverageMetric.name: KeywordCoverageMetric,
    "keyword_match": KeywordCoverageMetric,
    # v0.2.0 & v0.3.1 advanced metrics
    PrecisionMetric.name: PrecisionMetric,
    RecallMetric.name: RecallMetric,
    F1Metric.name: F1Metric,
    RougeMetric.name: RougeMetric,
    Rouge1Metric.name: Rouge1Metric,
    RougeLMetric.name: RougeLMetric,
    BleuMetric.name: BleuMetric,
    SemanticSimilarityMetric.name: SemanticSimilarityMetric,
    LexicalSimilarityMetric.name: LexicalSimilarityMetric,
    StructuredOutputMetric.name: StructuredOutputMetric,
}


def create_metric(
    name: str,
    normalization: NormalizationConfig | None = None,
    **kwargs: Any,
) -> BaseMetric:
    """Instantiate a registered metric by name with optional metric-specific parameters."""
    key = name.strip().lower()
    if key not in METRIC_REGISTRY:
        available = sorted({m.name for m in METRIC_REGISTRY.values()})
        raise MetricError(
            f"Unknown metric '{name}'. Available metrics: {', '.join(available)}"
        )
    metric_cls = METRIC_REGISTRY[key]
    sig = inspect.signature(metric_cls.__init__)
    accepted_kwargs: dict[str, Any] = {}
    if "normalization" in sig.parameters:
        accepted_kwargs["normalization"] = normalization
    for k, v in kwargs.items():
        if k in sig.parameters:
            accepted_kwargs[k] = v
        else:
            raise MetricError(
                f"Metric '{name}' does not accept configuration parameter '{k}'"
            )
    return metric_cls(**accepted_kwargs)


def register_metric(metric_cls: type[BaseMetric]) -> None:
    """Register a custom BaseMetric subclass for lookup by name."""
    if not isinstance(metric_cls, type) or not issubclass(metric_cls, BaseMetric):
        raise MetricError("Custom metric class must inherit from BaseMetric")
    if not getattr(metric_cls, "name", "").strip():
        raise MetricError("Custom metric class must define a non-empty 'name' attribute")
    METRIC_REGISTRY[metric_cls.name.strip().lower()] = metric_cls


def list_registered_metrics() -> list[MetricMetadata]:
    """Return sorted MetricMetadata objects for all unique registered metrics."""
    unique_classes = {cls.name: cls for cls in METRIC_REGISTRY.values()}
    return [unique_classes[name].get_metadata() for name in sorted(unique_classes)]


__all__ = [
    "AccuracyMetric",
    "AggregateMetricResult",
    "BaseMetric",
    "BleuMetric",
    "CallableEmbeddingProvider",
    "EmbeddingProvider",
    "ExactMatchMetric",
    "F1Metric",
    "HuggingFaceEmbeddingProvider",
    "KeywordCoverageMetric",
    "LexicalSimilarityMetric",
    "LocalNgramEmbeddingProvider",
    "METRIC_REGISTRY",
    "MetricMetadata",
    "MetricResult",
    "NormalizationConfig",
    "PrecisionMetric",
    "RecallMetric",
    "Rouge1Metric",
    "RougeLMetric",
    "RougeMetric",
    "SemanticSimilarityMetric",
    "StructuredOutputMetric",
    "cosine_similarity",
    "create_metric",
    "extract_fenced_json",
    "list_registered_metrics",
    "normalize_text",
    "register_metric",
]
