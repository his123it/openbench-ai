"""Exact Match metric implementation."""

from __future__ import annotations

from typing import Any

from openbench.dataset import DatasetRecord
from openbench.metrics.base import BaseMetric, MetricResult
from openbench.metrics.normalization import NormalizationConfig, normalize_text


class ExactMatchMetric(BaseMetric):
    """Compares normalized model output directly against normalized expected answer."""

    name = "exact_match"
    version = "0.1.0"
    description = (
        "Checks whether the normalized model response is identical to the "
        "normalized expected answer."
    )
    limitation_note = (
        "Exact match is strict: any extra conversational framing (e.g., 'The answer is 42') "
        "will score 0.0 unless the expected answer includes that exact phrasing."
    )

    def __init__(self, normalization: NormalizationConfig | None = None) -> None:
        self.normalization = normalization or NormalizationConfig()

    def compute_item(self, record: DatasetRecord, response_text: str) -> MetricResult:
        """Compare normalized expected answer and response text."""
        norm_expected = normalize_text(record.expected_answer, self.normalization)
        norm_response = normalize_text(response_text, self.normalization)
        matched = norm_expected == norm_response

        return MetricResult(
            metric_name=self.name,
            score=1.0 if matched else 0.0,
            passed=matched,
            details={
                "normalized_expected": norm_expected,
                "normalized_response": norm_response,
            },
        )

    def config_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "normalization": self.normalization.to_dict(),
        }
