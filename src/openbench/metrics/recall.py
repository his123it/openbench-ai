"""Recall metric for binary correctness decisions and deterministic set-prediction tasks."""

from __future__ import annotations

from typing import Any

from openbench.dataset import DatasetRecord
from openbench.metrics.base import AggregateMetricResult, BaseMetric, MetricResult
from openbench.metrics.normalization import NormalizationConfig
from openbench.metrics.precision import extract_decision_elements


class RecallMetric(BaseMetric):
    """Calculates recall for binary correctness decisions and element-set predictions.

    Definition:
    - Item-level recall: $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$
      (For binary classification items where expected is negative and TN=1, item score is 1.0;
      corpus micro-recall across all TP and FN is reported in `aggregate().details["micro_recall"]`.)
    """

    name = "recall"
    version = "0.2.0"
    description = (
        "Computes recall (TP / (TP + FN)) for binary classification decisions "
        "and structured element/token set predictions."
    )
    limitation_note = (
        "Recall measures coverage of expected positive decisions or target elements and "
        "does not penalize extraneous false-positive tokens on its own."
    )

    def __init__(
        self,
        normalization: NormalizationConfig | None = None,
        pass_threshold: float = 1.0,
    ) -> None:
        self.normalization = normalization or NormalizationConfig()
        self.pass_threshold = pass_threshold

    def compute_item(self, record: DatasetRecord, response_text: str) -> MetricResult:
        mode, tp, fp, fn, tn, extra = extract_decision_elements(
            record, response_text, self.normalization
        )
        if tp + fn > 0:
            score = tp / (tp + fn)
        elif tn > 0 and fp == 0:
            score = 1.0
        else:
            score = 0.0

        passed = score >= self.pass_threshold
        details = {
            "decision_mode": mode,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            **extra,
        }
        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=passed,
            details=details,
        )

    def aggregate(
        self,
        item_results: list[MetricResult],
        total_dataset_items: int,
    ) -> AggregateMetricResult:
        base_agg = super().aggregate(item_results, total_dataset_items)
        total_tp = sum(int(r.details.get("tp", 0)) for r in item_results)
        total_fn = sum(int(r.details.get("fn", 0)) for r in item_results)
        micro_recall = (
            total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        )
        details = dict(base_agg.details)
        details.update(
            {
                "total_tp": total_tp,
                "total_fn": total_fn,
                "micro_recall": round(micro_recall, 6),
                "macro_recall": round(base_agg.score, 6),
            }
        )
        return AggregateMetricResult(
            metric_name=base_agg.metric_name,
            metric_version=base_agg.metric_version,
            score=base_agg.score,
            total_items=base_agg.total_items,
            evaluated_items=base_agg.evaluated_items,
            passed_items=base_agg.passed_items,
            details=details,
        )

    def config_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "pass_threshold": self.pass_threshold,
            "normalization": self.normalization.to_dict(),
        }
