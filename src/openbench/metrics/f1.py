"""F1 score metric combining Precision and Recall."""

from __future__ import annotations

from typing import Any

from openbench.dataset import DatasetRecord
from openbench.metrics.base import AggregateMetricResult, BaseMetric, MetricResult
from openbench.metrics.normalization import NormalizationConfig
from openbench.metrics.precision import extract_decision_elements


class F1Metric(BaseMetric):
    r"""Calculates F1 score as the harmonic mean of Precision and Recall.

    Formula:
    $$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$
    """

    name = "f1"
    version = "0.2.0"
    score_kind = "bounded_continuous"
    description = (
        "Computes the harmonic mean of precision and recall for binary decisions "
        "and element-set predictions."
    )
    limitation_note = (
        "F1 balances precision and recall over discrete elements or binary labels; "
        "it does not evaluate word order or syntactic structure."
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
        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tn > 0 and fn == 0 else 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if tn > 0 and fp == 0 else 0.0)

        if precision + recall > 0:
            f1_score = 2.0 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0

        passed = f1_score >= self.pass_threshold
        details = {
            "decision_mode": mode,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            **extra,
        }
        return MetricResult(
            metric_name=self.name,
            score=f1_score,
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
        total_fp = sum(int(r.details.get("fp", 0)) for r in item_results)
        total_fn = sum(int(r.details.get("fn", 0)) for r in item_results)
        denom = 2 * total_tp + total_fp + total_fn
        micro_f1 = (2.0 * total_tp / denom) if denom > 0 else 0.0

        details = dict(base_agg.details)
        details.update(
            {
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
                "micro_f1": round(micro_f1, 6),
                "macro_f1": round(base_agg.score, 6),
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
