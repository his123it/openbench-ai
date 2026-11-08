"""Precision metric for binary classification and deterministic set-prediction tasks."""

from __future__ import annotations

import re
from typing import Any

from openbench.dataset import DatasetRecord
from openbench.metrics.base import AggregateMetricResult, BaseMetric, MetricResult
from openbench.metrics.normalization import NormalizationConfig, normalize_text

_DELIMITER_RE = re.compile(r"[,;|]+")
_POSITIVE_LABELS = {"yes", "true", "positive", "1", "safe", "valid", "correct", "pass"}
_NEGATIVE_LABELS = {"no", "false", "negative", "0", "unsafe", "invalid", "incorrect", "fail"}


def extract_decision_elements(
    record: DatasetRecord,
    response_text: str,
    normalization: NormalizationConfig,
) -> tuple[str, int, int, int, int, dict[str, Any]]:
    """Derive True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN).

    Task Modes:
    1. Binary Classification Decision (`mode="binary_classification"`):
       Activated when `record.metadata.get("task_type") == "binary_classification"` or when
       the normalized expected answer is a canonical binary label (`yes`/`no`, `true`/`false`,
       `positive`/`negative`, `safe`/`unsafe`, `valid`/`invalid`).
       - Positive class: `{"yes", "true", "positive", "1", "safe", "valid", "correct", "pass"}`
         (or custom `record.metadata["positive_label"]`).
       - Negative class: `{"no", "false", "negative", "0", "unsafe", "invalid", "incorrect", "fail"}`.
       - Extracts the predicted binary decision from the normalized response:
         * TP = 1 if expected is Positive and predicted is Positive
         * FP = 1 if expected is Negative (or abstained/wrong) and predicted is Positive
         * FN = 1 if expected is Positive and predicted is NOT Positive
         * TN = 1 if expected is Negative and predicted is Negative
    2. Set / Element Prediction (`mode="element_set"`):
       Activated for multi-label or delimited element extraction tasks (e.g., comma/semicolon/pipe
       lists or explicit `record.metadata["expected_elements"]`) or keyword/element prediction:
       - Reference set $E$: explicit `record.metadata["expected_elements"]`, or delimited elements
         in `expected_answer`, or unique normalized tokens of `expected_answer`.
       - Predicted set $P$: delimited elements in `response_text` (if delimiter matches) or unique
         normalized tokens in `response_text`.
       - $\text{TP} = |P \cap E|$, $\text{FP} = |P \setminus E|$, $\text{FN} = |E \setminus P|$, $\text{TN} = 0$.
    """
    norm_expected = normalize_text(record.expected_answer, normalization)
    norm_response = normalize_text(response_text, normalization)

    pos_label = record.metadata.get("positive_label")
    pos_set = {normalize_text(str(pos_label), normalization)} if pos_label else _POSITIVE_LABELS
    neg_set = _NEGATIVE_LABELS

    is_binary_task = (
        record.metadata.get("task_type") == "binary_classification"
        or norm_expected in pos_set
        or norm_expected in neg_set
    )

    if is_binary_task:
        expected_positive = norm_expected in pos_set
        resp_tokens = re.findall(r"[a-z0-9_]+", norm_response)
        first_token = resp_tokens[0] if resp_tokens else ""

        if norm_response in pos_set or first_token in pos_set:
            predicted_positive = True
            predicted_negative = False
        elif norm_response in neg_set or first_token in neg_set:
            predicted_positive = False
            predicted_negative = True
        else:
            predicted_positive = False
            predicted_negative = False

        tp = 1 if (expected_positive and predicted_positive) else 0
        fp = 1 if ((not expected_positive) and predicted_positive) else 0
        fn = 1 if (expected_positive and (not predicted_positive)) else 0
        tn = 1 if ((not expected_positive) and predicted_negative) else 0

        return (
            "binary_classification",
            tp,
            fp,
            fn,
            tn,
            {
                "expected_class": "positive" if expected_positive else "negative",
                "predicted_class": (
                    "positive"
                    if predicted_positive
                    else ("negative" if predicted_negative else "unrecognized")
                ),
            },
        )

    # Mode 2: Element-set / token-set decision
    raw_elements = record.metadata.get("expected_elements")
    if isinstance(raw_elements, list) and raw_elements:
        expected_elems = [
            normalize_text(str(e), normalization)
            for e in raw_elements
            if normalize_text(str(e), normalization)
        ]
        pred_elems = [
            normalize_text(part, normalization)
            for part in _DELIMITER_RE.split(norm_response)
            if normalize_text(part, normalization)
        ]
    elif _DELIMITER_RE.search(norm_expected):
        expected_elems = [
            normalize_text(part, normalization)
            for part in _DELIMITER_RE.split(norm_expected)
            if normalize_text(part, normalization)
        ]
        pred_elems = [
            normalize_text(part, normalization)
            for part in _DELIMITER_RE.split(norm_response)
            if normalize_text(part, normalization)
        ]
    else:
        expected_elems = re.findall(r"[\w./+-]+", norm_expected)
        pred_elems = re.findall(r"[\w./+-]+", norm_response)

    exp_set = set(expected_elems)
    pred_set = set(pred_elems)

    tp_items = sorted(pred_set & exp_set)
    fp_items = sorted(pred_set - exp_set)
    fn_items = sorted(exp_set - pred_set)

    tp = len(tp_items)
    fp = len(fp_items)
    fn = len(fn_items)
    tn = 0

    return (
        "element_set",
        tp,
        fp,
        fn,
        tn,
        {
            "true_positive_elements": tp_items,
            "false_positive_elements": fp_items,
            "false_negative_elements": fn_items,
        },
    )


class PrecisionMetric(BaseMetric):
    """Calculates precision for binary correctness decisions and element-set predictions.

    Definition:
    - Item-level precision: $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$
      (For binary classification items where both expected and predicted are negative,
      i.e. TN=1 and TP=FP=0, item score is 1.0 if TN=1 else 0.0; corpus micro-precision
      is also computed across all TP and FP counts in `aggregate().details["micro_precision"]`.)
    """

    name = "precision"
    version = "0.2.0"
    description = (
        "Computes precision (TP / (TP + FP)) for binary classification decisions "
        "and structured element/token set predictions."
    )
    limitation_note = (
        "Precision evaluates discrete binary decisions or predicted element sets; it should not "
        "be interpreted as open-ended semantic truthfulness."
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
        if tp + fp > 0:
            score = tp / (tp + fp)
        elif tn > 0 and fn == 0:
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
        total_fp = sum(int(r.details.get("fp", 0)) for r in item_results)
        micro_precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        details = dict(base_agg.details)
        details.update(
            {
                "total_tp": total_tp,
                "total_fp": total_fp,
                "micro_precision": round(micro_precision, 6),
                "macro_precision": round(base_agg.score, 6),
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
