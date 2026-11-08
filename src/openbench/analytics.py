"""Advanced analytics: failure analysis, error categorization, repeated-run variance, and item disagreement."""

from __future__ import annotations

import math
from typing import Any

from openbench.evaluator import EvaluationResult
from openbench.exceptions import ConfigurationError
from openbench.performance import PerformanceReport


class BenchmarkAnalytics:
    """Provides statistical and diagnostic analysis across evaluations and repeated runs.

    Statistical Honesty Policy:
    - Reports explicit sample size ($n$) on all variance and standard deviation summaries.
    - Documents assumptions (e.g., independent runs on identical dataset items).
    - Never claims statistical significance or confidence intervals when $n < 30$ or when
      distributional assumptions are not satisfied.
    """

    @staticmethod
    def analyze_failures_and_errors(
        eval_result: EvaluationResult,
        perf_report: PerformanceReport | None = None,
    ) -> dict[str, Any]:
        """Categorize missing, invalid, failed-metric, and transport-error items for a model run."""
        missing_by_cat: dict[str, list[str]] = {}
        invalid_items: list[dict[str, Any]] = []
        metric_failures_by_cat: dict[str, int] = {}
        error_categories: dict[str, int] = {}

        for item in eval_result.individual_results:
            if item.status == "missing":
                missing_by_cat.setdefault(item.category, []).append(item.id)
                error_categories["missing_response"] = error_categories.get("missing_response", 0) + 1
                continue

            if not item.is_valid_response:
                reason = item.invalid_reason or "invalid_response"
                invalid_items.append({"id": item.id, "category": item.category, "reason": reason})
                bucket = reason.split(":")[0]
                error_categories[bucket] = error_categories.get(bucket, 0) + 1

            # Check if item failed primary accuracy or exact_match metric
            failed_any = any(not mr.passed for mr in item.metrics.values())
            if failed_any:
                metric_failures_by_cat[item.category] = (
                    metric_failures_by_cat.get(item.category, 0) + 1
                )

        transport_errors: dict[str, int] = {}
        if perf_report is not None:
            for t in perf_report.per_request_telemetry:
                if t.error_type:
                    transport_errors[t.error_type] = transport_errors.get(t.error_type, 0) + 1

        return {
            "model_name": eval_result.model_name,
            "model_version": eval_result.model_version,
            "total_items": eval_result.total_dataset_items,
            "evaluated_items": eval_result.evaluated_items_count,
            "missing_count": eval_result.missing_responses_count,
            "missing_by_category": {k: list(v) for k, v in sorted(missing_by_cat.items())},
            "invalid_responses": invalid_items,
            "metric_failures_by_category": dict(sorted(metric_failures_by_cat.items())),
            "error_categories": dict(sorted(error_categories.items())),
            "transport_error_counts": dict(sorted(transport_errors.items())),
        }

    @staticmethod
    def analyze_repeated_runs(runs: list[EvaluationResult]) -> dict[str, Any]:
        """Compute score mean, sample variance, standard deviation, and consistency across repeated runs.

        Documents sample size `n = len(runs)` and explicitly avoids unsupported statistical
        significance claims.
        """
        n = len(runs)
        if n < 2:
            raise ConfigurationError("Repeated-run analysis requires at least 2 EvaluationResult objects")

        metric_names: list[str] = []
        for r in runs:
            for m in r.aggregate_results:
                if m not in metric_names:
                    metric_names.append(m)

        metric_stats: dict[str, dict[str, Any]] = {}
        for m_name in metric_names:
            scores = [
                r.aggregate_results[m_name].score
                for r in runs
                if m_name in r.aggregate_results
            ]
            k = len(scores)
            if k == 0:
                continue
            mean_val = sum(scores) / k
            if k > 1:
                sample_var = sum((s - mean_val) ** 2 for s in scores) / (k - 1)
                std_dev = math.sqrt(sample_var)
            else:
                sample_var = 0.0
                std_dev = 0.0

            metric_stats[m_name] = {
                "sample_size_n": k,
                "mean_score": round(mean_val, 6),
                "min_score": round(min(scores), 6),
                "max_score": round(max(scores), 6),
                "range": round(max(scores) - min(scores), 6),
                "sample_variance": round(sample_var, 8),
                "sample_std_dev": round(std_dev, 6),
                "scores": [round(s, 6) for s in scores],
            }

        # Compute item-level response consistency across runs
        item_ids = [item.id for item in runs[0].individual_results]
        consistent_items = 0
        varying_item_ids: list[str] = []

        for item_id in item_ids:
            observed_outputs: set[str | None] = set()
            for r in runs:
                for it in r.individual_results:
                    if it.id == item_id:
                        observed_outputs.add(
                            it.model_response.strip() if it.model_response is not None else None
                        )
                        break
            if len(observed_outputs) <= 1:
                consistent_items += 1
            else:
                varying_item_ids.append(item_id)

        total_items = max(len(item_ids), 1)
        consistency_rate = consistent_items / total_items

        return {
            "sample_size_n": n,
            "total_dataset_items": len(item_ids),
            "exact_output_consistency_rate": round(consistency_rate, 6),
            "consistent_items_count": consistent_items,
            "varying_item_ids": varying_item_ids,
            "metric_variance_summary": metric_stats,
            "statistical_assumptions_and_limitations": (
                f"Computed over n={n} repeated runs using Bessel-corrected sample variance (n-1 denominator). "
                "No parametric confidence intervals or p-values are claimed because small sample sizes (n < 30) "
                "and discrete benchmark items do not satisfy asymptotic normality assumptions."
            ),
        }

    @staticmethod
    def analyze_disagreements(
        results: list[EvaluationResult],
        metric_name: str = "accuracy",
    ) -> dict[str, Any]:
        """Identify per-example pass/fail and output disagreements across multiple models or runs."""
        if len(results) < 2:
            raise ConfigurationError("Disagreement analysis requires at least 2 EvaluationResult objects")

        all_ids: list[str] = [item.id for item in results[0].individual_results]
        disagreements: list[dict[str, Any]] = []
        unanimous_pass: list[str] = []
        unanimous_fail: list[str] = []

        for item_id in all_ids:
            outcomes: dict[str, bool] = {}
            responses: dict[str, str | None] = {}
            prompt_text = ""
            expected_text = ""

            for idx, res in enumerate(results, start=1):
                label = f"{res.model_name} (v{res.model_version}) [#{idx}]"
                found = next((it for it in res.individual_results if it.id == item_id), None)
                if found is None or found.status == "missing":
                    outcomes[label] = False
                    responses[label] = None
                else:
                    prompt_text = found.prompt
                    expected_text = found.expected_answer
                    responses[label] = found.model_response
                    m_res = found.metrics.get(metric_name)
                    outcomes[label] = bool(m_res.passed) if m_res is not None else False

            unique_pass_states = set(outcomes.values())
            if len(unique_pass_states) == 1:
                if True in unique_pass_states:
                    unanimous_pass.append(item_id)
                else:
                    unanimous_fail.append(item_id)
            else:
                disagreements.append(
                    {
                        "id": item_id,
                        "prompt": prompt_text,
                        "expected_answer": expected_text,
                        "metric_evaluated": metric_name,
                        "passed_by_model": outcomes,
                        "responses_by_model": responses,
                    }
                )

        total = max(len(all_ids), 1)
        return {
            "metric_evaluated": metric_name,
            "models_compared": len(results),
            "total_items": len(all_ids),
            "agreement_rate": round((len(unanimous_pass) + len(unanimous_fail)) / total, 6),
            "unanimous_pass_count": len(unanimous_pass),
            "unanimous_fail_count": len(unanimous_fail),
            "disagreement_count": len(disagreements),
            "disagreement_items": disagreements,
        }
