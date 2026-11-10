"""Per-item score extraction from stored evaluations (v0.4.0).

Shared by repeated-run variability analysis and paired model comparison. Everything here
is read-only and keyed by stable record IDs; input file order never matters.

Per repetition, an item's status for one metric is:

* ``valid``: a response was evaluated, it is a valid response and the metric was computed;
* ``missing``: the item was in the evaluated universe but had no response;
* ``invalid``: a response existed but was empty or structurally invalid;
* ``metric_unavailable``: the metric was not computed for this item;
* ``absent``: the item was not part of that repetition's evaluated universe.

An item's value for an arm is the mean of its valid repetitions (plan section 1.3, D7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from openbench.evaluator import EvaluationResult

VALID = "valid"
MISSING = "missing"
INVALID = "invalid"
METRIC_UNAVAILABLE = "metric_unavailable"
ABSENT = "absent"


def metric_score_kind(metric_name: str) -> str:
    """Return the declared ``score_kind`` of a registered metric, or ``"unknown"``."""
    from openbench.metrics import METRIC_REGISTRY

    cls = METRIC_REGISTRY.get(metric_name.strip().lower())
    kind = getattr(cls, "score_kind", "unknown") if cls is not None else "unknown"
    return kind if kind in {"binary", "bounded_continuous", "unknown"} else "unknown"


def metric_version_of(evaluation: EvaluationResult, metric_name: str) -> str | None:
    """Metric version as recorded in an evaluation (``metrics_used`` or the aggregate)."""
    for entry in evaluation.metrics_used:
        if isinstance(entry, dict) and entry.get("name") == metric_name:
            return str(entry.get("version")) if entry.get("version") is not None else None
    agg = evaluation.aggregate_results.get(metric_name)
    return agg.metric_version if agg is not None else None


@dataclass(frozen=True)
class ItemObservation:
    """One item's per-repetition scores for one metric in one arm."""

    id: str
    category: str
    difficulty: str
    values: tuple[float | None, ...]
    statuses: tuple[str, ...]
    responses: tuple[str | None, ...] = ()
    latencies: tuple[float | None, ...] = ()
    total_tokens: tuple[int | None, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def valid_values(self) -> list[float]:
        return [v for v, s in zip(self.values, self.statuses, strict=True) if s == VALID and v is not None]

    @property
    def repetitions_used(self) -> int:
        return len(self.valid_values)

    @property
    def mean(self) -> float | None:
        vals = self.valid_values
        return math.fsum(vals) / len(vals) if vals else None

    @property
    def failure_share(self) -> float | None:
        """Share of repetitions (in which the item was evaluated) that are missing or invalid."""
        considered = [s for s in self.statuses if s in {VALID, MISSING, INVALID}]
        if not considered:
            return None
        return sum(1 for s in considered if s != VALID) / len(considered)

    @property
    def in_universe(self) -> bool:
        return any(s != ABSENT for s in self.statuses)


def _total_tokens(token_usage: dict[str, Any] | None) -> int | None:
    if not token_usage:
        return None
    if "total_tokens" in token_usage:
        try:
            return int(token_usage["total_tokens"])
        except (TypeError, ValueError):
            return None
    if "prompt_tokens" in token_usage and "completion_tokens" in token_usage:
        try:
            return int(token_usage["prompt_tokens"]) + int(token_usage["completion_tokens"])
        except (TypeError, ValueError):
            return None
    return None


def collect_item_scores(
    evaluations: list[EvaluationResult], metric_name: str
) -> dict[str, ItemObservation]:
    """Collect ``{record_id: ItemObservation}`` sorted by ID for one metric.

    ``evaluations`` are the repetitions of one arm, in repetition order. Item metadata
    (category, difficulty) comes from the first repetition that contains the item.
    """
    if not evaluations:
        raise ValueError("at least one evaluation is required")
    per_rep: list[dict[str, Any]] = []
    for ev in evaluations:
        mapping: dict[str, Any] = {}
        for item in ev.individual_results:
            if item.id in mapping:
                raise ValueError(
                    f"Evaluation '{ev.evaluation_id}' contains duplicate record ID '{item.id}'"
                )
            mapping[item.id] = item
        per_rep.append(mapping)
    all_ids = sorted(set().union(*[set(m) for m in per_rep]))
    out: dict[str, ItemObservation] = {}
    for rid in all_ids:
        values: list[float | None] = []
        statuses: list[str] = []
        responses: list[str | None] = []
        latencies: list[float | None] = []
        tokens: list[int | None] = []
        category = "unknown"
        difficulty = "unspecified"
        found_meta = False
        for mapping in per_rep:
            item = mapping.get(rid)
            if item is None:
                values.append(None)
                statuses.append(ABSENT)
                responses.append(None)
                latencies.append(None)
                tokens.append(None)
                continue
            if not found_meta:
                category, difficulty, found_meta = item.category, item.difficulty, True
            responses.append(item.model_response)
            latencies.append(item.latency_ms)
            tokens.append(_total_tokens(item.token_usage))
            if item.status != "evaluated" or item.model_response is None:
                values.append(None)
                statuses.append(MISSING)
            elif not item.is_valid_response:
                values.append(None)
                statuses.append(INVALID)
            elif metric_name not in item.metrics:
                values.append(None)
                statuses.append(METRIC_UNAVAILABLE)
            else:
                score = float(item.metrics[metric_name].score)
                if not math.isfinite(score):
                    values.append(None)
                    statuses.append(METRIC_UNAVAILABLE)
                else:
                    values.append(score)
                    statuses.append(VALID)
        out[rid] = ItemObservation(
            id=rid,
            category=category,
            difficulty=difficulty,
            values=tuple(values),
            statuses=tuple(statuses),
            responses=tuple(responses),
            latencies=tuple(latencies),
            total_tokens=tuple(tokens),
        )
    return out
