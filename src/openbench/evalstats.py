"""Opt-in confidence intervals for a single evaluation (``evaluate --confidence-intervals``; v0.4.0).

Without the option, evaluation output is unchanged. With it, an additional ``statistics``
block (its own ``schema_version``) holds, per metric, two percentile bootstrap CIs of the
mean per-item score over items:

* ``headline``: missing responses count as 0.0, exactly as the headline score does, so the
  point estimate equals the headline score;
* ``evaluated_only``: only items with a response (the ``evaluated_only_score`` view).

Each records its sample count, method, resamples, confidence level and analysis seed.
"""

from __future__ import annotations

from typing import Any

from openbench.evaluator import EvaluationResult
from openbench.itemscores import metric_score_kind
from openbench.stats.bootstrap import DEFAULT_RESAMPLES, bootstrap_mean_ci

SCHEMA_VERSION = "0.4.0"


def compute_evaluation_statistics(
    result: EvaluationResult,
    *,
    confidence_level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    analysis_seed: int = 0,
) -> dict[str, Any]:
    """Return the ``statistics`` block for ``result`` (does not modify ``result``)."""
    items = sorted(result.individual_results, key=lambda i: i.id)
    per_metric: dict[str, Any] = {}
    for metric in result.aggregate_results:
        headline_vals: list[float] = []
        evaluated_vals: list[float] = []
        for item in items:
            if item.status != "evaluated" or metric not in item.metrics:
                headline_vals.append(0.0)
                continue
            score = float(item.metrics[metric].score)
            headline_vals.append(score)
            evaluated_vals.append(score)
        arm = result.model_name
        headline = bootstrap_mean_ci(
            headline_vals,
            confidence_level=confidence_level,
            resamples=resamples,
            analysis_seed=analysis_seed,
            metric=metric,
            arm_a=arm,
        )
        evaluated = bootstrap_mean_ci(
            evaluated_vals,
            confidence_level=confidence_level,
            resamples=resamples,
            analysis_seed=analysis_seed,
            metric=f"{metric}|evaluated_only",
            arm_a=arm,
        )
        per_metric[metric] = {
            "score_kind": metric_score_kind(metric),
            "headline": {
                **headline.to_dict(),
                "missing_policy": "missing_as_failure",
                "matches_headline_score": True,
            },
            "evaluated_only": {**evaluated.to_dict(), "missing_policy": "complete_case"},
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "percentile_bootstrap_over_items",
        "confidence_level": confidence_level,
        "resamples": resamples,
        "analysis_seed": analysis_seed,
        "unit_of_analysis": "item",
        "metrics": per_metric,
        "interpretation": (
            "Each interval treats the evaluated items as a sample of similar items and describes "
            "uncertainty about the mean score for this model, prompts, settings and references. "
            "It does not include run-to-run generation noise or performance on other datasets. "
            "The percentile bootstrap tends to under-cover for fewer than about 20 items."
        ),
    }
