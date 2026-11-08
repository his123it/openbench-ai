"""Accuracy metric implementation with deterministic correctness rules."""

from __future__ import annotations

import re
from typing import Any

from openbench.dataset import DatasetRecord
from openbench.exceptions import MetricError
from openbench.metrics.base import BaseMetric, MetricResult
from openbench.metrics.normalization import NormalizationConfig, normalize_text

ALLOWED_CORRECTNESS_RULES = {"exact_or_substring", "exact_only", "word_boundary"}


class AccuracyMetric(BaseMetric):
    """Calculates proportion of correctly answered items under a deterministic rule.

    Supported deterministic correctness rules (`rule` parameter):
    - `"exact_or_substring"` (default): An item is marked correct (1.0) if:
      1. The normalized response equals the normalized expected answer, OR
      2. If `record.metadata` specifies `"disallow_substring": true`, only exact match counts;
         otherwise, the normalized expected answer appears as a token/phrase match within
         the normalized response (using word boundaries when the expected answer starts/ends
         with alphanumeric characters, preventing false matches like `'18'` inside `'180'`).
    - `"exact_only"`: Requires exact match after normalization.
    - `"word_boundary"`: Requires the normalized expected answer to appear with word boundaries
      inside the normalized response.
    """

    name = "accuracy"
    version = "0.1.0"
    description = (
        "Calculates the proportion of evaluation items answered correctly according "
        "to a deterministic correctness rule (default: exact match or bounded answer containment)."
    )
    limitation_note = (
        "Deterministic accuracy verifies whether the expected answer appears in the response "
        "using string/boundary rules; it does not parse negation or complex semantic equivalence."
    )

    def __init__(
        self,
        rule: str = "exact_or_substring",
        normalization: NormalizationConfig | None = None,
    ) -> None:
        if rule not in ALLOWED_CORRECTNESS_RULES:
            raise MetricError(
                f"Invalid accuracy correctness rule '{rule}'. "
                f"Allowed rules: {sorted(ALLOWED_CORRECTNESS_RULES)}"
            )
        self.rule = rule
        self.normalization = normalization or NormalizationConfig()

    def compute_item(self, record: DatasetRecord, response_text: str) -> MetricResult:
        """Evaluate whether a response is correct according to the configured rule."""
        norm_expected = normalize_text(record.expected_answer, self.normalization)
        norm_response = normalize_text(response_text, self.normalization)

        if not norm_expected:
            raise MetricError(f"Expected answer for record '{record.id}' normalized to empty string")

        exact_matched = norm_expected == norm_response
        disallow_substring = bool(record.metadata.get("disallow_substring", False))

        if self.rule == "exact_only" or disallow_substring:
            is_correct = exact_matched
            match_mode = "exact" if is_correct else "none"
        elif self.rule == "word_boundary":
            is_correct = self._bounded_contains(norm_expected, norm_response)
            match_mode = "word_boundary" if is_correct else "none"
        else:  # exact_or_substring
            if exact_matched:
                is_correct = True
                match_mode = "exact"
            elif self._bounded_contains(norm_expected, norm_response):
                is_correct = True
                match_mode = "bounded_substring"
            else:
                is_correct = False
                match_mode = "none"

        return MetricResult(
            metric_name=self.name,
            score=1.0 if is_correct else 0.0,
            passed=is_correct,
            details={
                "rule": self.rule,
                "match_mode": match_mode,
                "normalized_expected": norm_expected,
                "normalized_response": norm_response,
            },
        )

    @staticmethod
    def _bounded_contains(expected: str, response: str) -> bool:
        """Check whether `expected` appears inside `response` respecting word boundaries.

        If `expected` starts or ends with an alphanumeric character, require a word
        boundary on that side so `'18'` does not falsely match `'180'` and `'cat'`
        does not falsely match `'category'`.
        """
        if not expected or not response:
            return False
        prefix = r"\b" if expected[0].isalnum() else ""
        suffix = r"\b" if expected[-1].isalnum() else ""
        pattern = f"{prefix}{re.escape(expected)}{suffix}"
        return re.search(pattern, response) is not None

    def config_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "rule": self.rule,
            "normalization": self.normalization.to_dict(),
        }
