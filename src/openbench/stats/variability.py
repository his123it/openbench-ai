"""Repeated-run variability decomposition (plan sections 1.8 and 3.4; v0.4.0).

Three sources of variation are reported separately and never combined:

1. **Generation variability**: for the same record, how much the metric and the response
   text change across repetitions (``within_item_variance_mean``, ``items_with_any_change_rate``,
   ``exact_response_identity_rate``, binary flip rate and pairwise agreement).
2. **Item (prompt / dataset) variation**: how much per-item mean scores differ between
   records (``item_spread_sd``), and the question-level percentile bootstrap CI of the mean
   over items, computed on per-item means (unit = item, never item x repetition).
3. **Provider failures and missing responses**: per-repetition failure counts and rates,
   and items that failed in some but not all repetitions.

The run-to-run distribution of headline scores is also reported. A hierarchical bootstrap
over items and repetitions is deliberately not used in v0.4.0 (plan U7).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

from openbench.evaluator import EvaluationResult
from openbench.itemscores import (
    ABSENT,
    INVALID,
    MISSING,
    VALID,
    ItemObservation,
    collect_item_scores,
    metric_score_kind,
)
from openbench.stats.bootstrap import DEFAULT_RESAMPLES, bootstrap_mean_ci

SCHEMA_VERSION = "0.4.0"

INTERPRETATION = (
    "Generation variability (within items, across repetitions) is reported next to, and "
    "never folded into, the question-level confidence interval. The interval treats the "
    "analysed items as a sample of similar items and uses per-item means over valid "
    "repetitions; it does not describe run-to-run generation noise, other datasets, or "
    "errors in references."
)


def _sd(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": math.fsum(values) / len(values),
        "sd": _sd(values),
        "min": min(values),
        "max": max(values),
    }


def _item_entry(obs: ItemObservation, binary: bool) -> dict[str, Any]:
    vals = obs.valid_values
    entry: dict[str, Any] = {
        "id": obs.id,
        "category": obs.category,
        "difficulty": obs.difficulty,
        "repetition_values": list(obs.values),
        "repetition_statuses": list(obs.statuses),
        "repetitions_used": len(vals),
        "mean": obs.mean,
        "sd": _sd(vals),
        "variance": statistics.variance(vals) if len(vals) >= 2 else None,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "changed": (len(set(vals)) > 1) if vals else None,
    }
    if binary and len(vals) >= 2:
        pairs = len(vals) * (len(vals) - 1) / 2
        ones = sum(1 for v in vals if v == 1.0)
        zeros = len(vals) - ones
        agree = (ones * (ones - 1) + zeros * (zeros - 1)) / 2
        entry["pairwise_agreement"] = agree / pairs
    return entry


def _generation_summary(entries: list[dict[str, Any]], binary: bool) -> dict[str, Any]:
    multi = [e for e in entries if e["repetitions_used"] >= 2]
    variances = [e["variance"] for e in multi]
    out: dict[str, Any] = {
        "items_with_two_or_more_valid_repetitions": len(multi),
        "within_item_variance_mean": (math.fsum(variances) / len(variances)) if variances else None,
        "within_item_sd_mean": (
            math.fsum(e["sd"] for e in multi) / len(multi) if multi else None
        ),
        "items_with_any_change_rate": (
            sum(1 for e in multi if e["changed"]) / len(multi) if multi else None
        ),
    }
    if binary:
        out["binary_flip_rate"] = out["items_with_any_change_rate"]
        agreements = [e["pairwise_agreement"] for e in multi if "pairwise_agreement" in e]
        out["mean_pairwise_agreement"] = (
            math.fsum(agreements) / len(agreements) if agreements else None
        )
    return out


def _response_identity(observations: dict[str, ItemObservation]) -> dict[str, Any]:
    complete = [
        obs
        for obs in observations.values()
        if obs.statuses and all(s in {VALID, INVALID} for s in obs.statuses)
        and all(r is not None for r in obs.responses)
    ]
    identical = sum(
        1 for obs in complete if len({(r or "").strip() for r in obs.responses}) == 1
    )
    return {
        "items_with_response_in_every_repetition": len(complete),
        "exact_response_identity_rate": (identical / len(complete)) if complete else None,
        "items_with_identical_responses": identical,
    }


def _failures(observations: dict[str, ItemObservation], repetitions: int) -> dict[str, Any]:
    per_rep = []
    for r in range(repetitions):
        statuses = [obs.statuses[r] for obs in observations.values()]
        evaluated = [s for s in statuses if s != ABSENT]
        missing = sum(1 for s in evaluated if s == MISSING)
        invalid = sum(1 for s in evaluated if s == INVALID)
        per_rep.append(
            {
                "repetition_index": r,
                "items": len(evaluated),
                "missing": missing,
                "invalid": invalid,
                "failure_rate": ((missing + invalid) / len(evaluated)) if evaluated else None,
            }
        )
    sometimes = sorted(
        obs.id
        for obs in observations.values()
        if any(s in {MISSING, INVALID} for s in obs.statuses)
        and any(s == VALID for s in obs.statuses)
    )
    always = sorted(
        obs.id
        for obs in observations.values()
        if obs.in_universe and not any(s == VALID for s in obs.statuses)
        and any(s in {MISSING, INVALID} for s in obs.statuses)
    )
    rates = [p["failure_rate"] for p in per_rep if p["failure_rate"] is not None]
    return {
        "per_repetition": per_rep,
        "failure_rate_summary": _summary(rates),
        "items_failing_in_some_repetitions": sometimes,
        "items_failing_in_every_repetition": always,
        "note": (
            "Missing and invalid responses are provider/format failures. They are excluded "
            "from per-item means (an item with no valid repetition is excluded entirely) and "
            "are never counted as wrong answers in this analysis."
        ),
    }


def _latency_tokens(observations: dict[str, ItemObservation], repetitions: int) -> dict[str, Any]:
    lat_means = []
    tok_means = []
    for r in range(repetitions):
        lats = [o.latencies[r] for o in observations.values() if o.latencies and o.latencies[r] is not None]
        toks = [o.total_tokens[r] for o in observations.values() if o.total_tokens and o.total_tokens[r] is not None]
        lat_means.append(math.fsum(lats) / len(lats) if lats else None)
        tok_means.append(math.fsum(toks) / len(toks) if toks else None)
    within_lat = []
    within_tok = []
    for o in observations.values():
        lats = [v for v in o.latencies if v is not None]
        toks = [float(v) for v in o.total_tokens if v is not None]
        if len(lats) >= 2:
            within_lat.append(statistics.stdev(lats))
        if len(toks) >= 2:
            within_tok.append(statistics.stdev(toks))
    return {
        "latency_ms": {
            "per_repetition_mean": lat_means,
            "across_repetition": _summary([v for v in lat_means if v is not None]),
            "within_item_sd_mean": (math.fsum(within_lat) / len(within_lat)) if within_lat else None,
            "available": any(v is not None for v in lat_means),
        },
        "total_tokens": {
            "per_repetition_mean": tok_means,
            "across_repetition": _summary([v for v in tok_means if v is not None]),
            "within_item_sd_mean": (math.fsum(within_tok) / len(within_tok)) if within_tok else None,
            "available": any(v is not None for v in tok_means),
        },
    }


def _breakdown(entries: list[dict[str, Any]], key: str, binary: bool) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        groups.setdefault(str(e[key]), []).append(e)
    out = {}
    for label in sorted(groups):
        members = groups[label]
        means = [e["mean"] for e in members if e["mean"] is not None]
        out[label] = {
            "items": len(members),
            "items_with_valid_values": len(means),
            "mean_of_item_means": (math.fsum(means) / len(means)) if means else None,
            "item_spread_sd": _sd(means),
            "generation_variability": _generation_summary(members, binary),
        }
    return out


def decompose_variability(
    evaluations: Sequence[EvaluationResult],
    metrics: Sequence[str] | None = None,
    *,
    analysis_seed: int = 0,
    confidence_level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    arm_label: str = "",
) -> dict[str, Any]:
    """Decompose variability across the repetitions of one configuration.

    ``evaluations`` are the repetitions in repetition order (at least one). ``metrics``
    defaults to every metric present in the first evaluation.
    """
    evals = list(evaluations)
    if not evals:
        raise ValueError("at least one evaluation is required")
    metric_names = list(metrics) if metrics else list(evals[0].aggregate_results)
    repetitions = len(evals)
    per_metric: dict[str, Any] = {}
    shared_obs: dict[str, ItemObservation] | None = None
    for metric in metric_names:
        kind = metric_score_kind(metric)
        binary = kind == "binary"
        observations = collect_item_scores(evals, metric)
        if shared_obs is None:
            shared_obs = observations
        entries = [_item_entry(obs, binary) for obs in observations.values() if obs.in_universe]
        item_means = [e["mean"] for e in entries if e["mean"] is not None]
        ci = bootstrap_mean_ci(
            item_means,
            confidence_level=confidence_level,
            resamples=resamples,
            analysis_seed=analysis_seed,
            metric=metric,
            arm_a=arm_label,
        )
        headline = [
            ev.aggregate_results[metric].score for ev in evals if metric in ev.aggregate_results
        ]
        evaluated_only = []
        for ev in evals:
            agg = ev.aggregate_results.get(metric)
            if agg is not None and agg.details.get("evaluated_only_score") is not None:
                evaluated_only.append(float(agg.details["evaluated_only_score"]))
        per_metric[metric] = {
            "score_kind": kind,
            "items_in_universe": len(entries),
            "items_with_valid_values": len(item_means),
            "items_excluded_no_valid_repetition": len(entries) - len(item_means),
            "generation_variability": _generation_summary(entries, binary),
            "item_variation": {
                "mean_of_item_means": (math.fsum(item_means) / len(item_means)) if item_means else None,
                "item_spread_sd": _sd(item_means),
                "question_level_confidence_interval": ci.to_dict(),
            },
            "run_distribution": {
                "headline_scores": headline,
                "headline_summary": _summary(headline),
                "evaluated_only_scores": evaluated_only,
                "note": (
                    "Headline scores count missing responses as 0.0 (v0.3.1 semantics); "
                    "evaluated-only scores exclude them."
                ),
            },
            "by_category": _breakdown(entries, "category", binary),
            "by_difficulty": _breakdown(entries, "difficulty", binary),
            "per_item": entries,
        }
    assert shared_obs is not None
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "repeated_run_variability",
        "repetitions": repetitions,
        "evaluation_ids": [ev.evaluation_id for ev in evals],
        "analysis_seed": analysis_seed,
        "parameters": {
            "confidence_level": confidence_level,
            "resamples": resamples,
            "unit_of_analysis": "item (per-item mean over valid repetitions)",
            "missing_policy": "exclude_missing_and_invalid_repetitions",
        },
        "response_identity": _response_identity(shared_obs),
        "failures": _failures(shared_obs, repetitions),
        "latency_and_tokens": _latency_tokens(shared_obs, repetitions),
        "metrics": per_metric,
        "interpretation": INTERPRETATION,
        "limitations": [
            "With few repetitions, within-item variance estimates are imprecise.",
            "Generation variability depends on the provider, its sampling settings and any "
            "caching; cached responses are identical by construction (v0.4.0 isolates cache "
            "entries per repetition).",
            "A seed passed to a provider does not guarantee deterministic output.",
        ],
    }
