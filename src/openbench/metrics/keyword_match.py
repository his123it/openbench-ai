"""Keyword Coverage metric implementation."""

from __future__ import annotations

from typing import Any

from openbench.dataset import DatasetRecord
from openbench.metrics.base import AggregateMetricResult, BaseMetric, MetricResult
from openbench.metrics.normalization import NormalizationConfig, normalize_text


class KeywordCoverageMetric(BaseMetric):
    """Checks whether configured expected keywords appear in model responses.

    Important Limitation:
    Keyword coverage measures lexical presence of expected terms. It is NOT
    equivalent to semantic correctness or factual accuracy (for example, a
    response containing the keyword 'encryption' could state 'do not use encryption').
    """

    name = "keyword_coverage"
    version = "0.1.0"
    score_kind = "bounded_continuous"
    description = (
        "Calculates the percentage of expected keywords present in the model response. "
        "Accepts custom keyword lists via record.metadata['keywords'] or falls back "
        "to the expected answer."
    )
    limitation_note = (
        "Keyword coverage is NOT equivalent to semantic correctness. A response may "
        "include all expected keywords while negating them or using them in an incorrect context."
    )

    def __init__(
        self,
        normalization: NormalizationConfig | None = None,
        pass_threshold: float = 1.0,
    ) -> None:
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be between 0.0 and 1.0")
        self.normalization = normalization or NormalizationConfig()
        self.pass_threshold = pass_threshold

    def compute_item(self, record: DatasetRecord, response_text: str) -> MetricResult:
        """Check presence of each expected keyword in the normalized response."""
        expected_keywords = record.keywords
        norm_response = normalize_text(response_text, self.normalization)

        matched_keywords: list[str] = []
        missing_keywords: list[str] = []

        for kw in expected_keywords:
            norm_kw = normalize_text(kw, self.normalization)
            if norm_kw and norm_kw in norm_response:
                matched_keywords.append(kw)
            else:
                missing_keywords.append(kw)

        total_kw = len(expected_keywords)
        coverage_ratio = (len(matched_keywords) / total_kw) if total_kw > 0 else 1.0
        coverage_percentage = round(coverage_ratio * 100.0, 2)
        passed = coverage_ratio >= self.pass_threshold

        return MetricResult(
            metric_name=self.name,
            score=coverage_ratio,
            passed=passed,
            details={
                "matched_keywords": matched_keywords,
                "missing_keywords": missing_keywords,
                "total_keywords": total_kw,
                "coverage_percentage": coverage_percentage,
                "limitation": self.limitation_note,
            },
        )

    def aggregate(
        self,
        item_results: list[MetricResult],
        total_dataset_items: int,
    ) -> AggregateMetricResult:
        base_agg = super().aggregate(item_results, total_dataset_items)
        merged_details = dict(base_agg.details)
        merged_details["coverage_percentage"] = round(base_agg.score * 100.0, 2)
        merged_details["pass_threshold"] = self.pass_threshold
        return AggregateMetricResult(
            metric_name=base_agg.metric_name,
            metric_version=base_agg.metric_version,
            score=base_agg.score,
            total_items=base_agg.total_items,
            evaluated_items=base_agg.evaluated_items,
            passed_items=base_agg.passed_items,
            details=merged_details,
        )

    def config_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "pass_threshold": self.pass_threshold,
            "normalization": self.normalization.to_dict(),
        }
