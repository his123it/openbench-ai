"""Paired percentile bootstrap confidence intervals (plan section 1.5)."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from openbench.performance import compute_percentile
from openbench.stats.results import ConfidenceInterval, StatWarning
from openbench.stats.rng import derive_stream

DEFAULT_RESAMPLES = 10_000
MIN_RECOMMENDED_RESAMPLES = 1_000
METHOD = "paired_percentile_bootstrap"


def _validate(values: Sequence[float], confidence_level: float, resamples: int) -> list[float]:
    if not 0.5 <= confidence_level <= 0.999:
        raise ValueError("confidence_level must be within [0.5, 0.999]")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    out = [float(v) for v in values]
    for v in out:
        if not math.isfinite(v):
            raise ValueError("values must be finite numbers")
    return out


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence_level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    analysis_seed: int = 0,
    metric: str = "",
    arm_a: str = "",
    arm_b: str = "",
    strata: Sequence[str] | None = None,
) -> ConfidenceInterval:
    """Percentile bootstrap CI for the mean of ``values``.

    ``values`` are per-item values (one model) or paired differences y_B - y_A, already
    aligned, filtered and sorted by item ID by the caller. With ``strata``, resampling is
    done within each stratum, keeping each stratum's size, using one stream per stratum
    (purpose ``stratum:<label>``).
    """
    data = _validate(values, confidence_level, resamples)
    n = len(data)
    warnings: list[StatWarning] = []
    if resamples < MIN_RECOMMENDED_RESAMPLES:
        warnings.append(
            StatWarning(
                "few_resamples",
                f"Only {resamples} bootstrap resamples; at least "
                f"{MIN_RECOMMENDED_RESAMPLES} are recommended.",
            )
        )
    common = {
        "confidence_level": confidence_level,
        "method": METHOD,
        "resamples": resamples,
        "sample_count": n,
        "analysis_seed": analysis_seed,
        "stratified": strata is not None,
    }
    if n < 2:
        return ConfidenceInterval(
            estimate=data[0] if n == 1 else None,
            lower=None,
            upper=None,
            standard_error=None,
            unavailable_reason="insufficient_items",
            warnings=tuple(warnings),
            **common,
        )
    estimate = math.fsum(data) / n
    if all(v == data[0] for v in data):
        warnings.append(
            StatWarning(
                "degenerate_zero_variance",
                "All values are equal; the interval is a single point.",
            )
        )
        return ConfidenceInterval(
            estimate=estimate,
            lower=data[0],
            upper=data[0],
            standard_error=0.0,
            warnings=tuple(warnings),
            **common,
        )

    means: list[float] = []
    if strata is None:
        rng = derive_stream(analysis_seed, "bootstrap", metric, arm_a, arm_b)
        randrange = rng.randrange
        for _ in range(resamples):
            means.append(math.fsum(data[randrange(n)] for _ in range(n)) / n)
    else:
        labels = list(strata)
        if len(labels) != n:
            raise ValueError("strata must have one label per value")
        groups: dict[str, list[float]] = {}
        for label, value in zip(labels, data, strict=True):
            groups.setdefault(str(label), []).append(value)
        streams = {
            label: derive_stream(analysis_seed, f"stratum:{label}", metric, arm_a, arm_b)
            for label in groups
        }
        ordered = sorted(groups)
        for _ in range(resamples):
            parts: list[float] = []
            for label in ordered:
                group = groups[label]
                size = len(group)
                randrange = streams[label].randrange
                parts.extend(group[randrange(size)] for _ in range(size))
            means.append(math.fsum(parts) / n)

    alpha = 1.0 - confidence_level
    ordered_means = sorted(means)
    lower = compute_percentile(ordered_means, 100.0 * alpha / 2.0)
    upper = compute_percentile(ordered_means, 100.0 * (1.0 - alpha / 2.0))
    se = statistics.stdev(means) if len(means) > 1 else None
    return ConfidenceInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        standard_error=se,
        warnings=tuple(warnings),
        **common,
    )


def paired_bootstrap_ci(
    differences: Sequence[float],
    **kwargs: object,
) -> ConfidenceInterval:
    """Percentile bootstrap CI for the mean paired difference (alias of
    :func:`bootstrap_mean_ci` applied to aligned differences y_B - y_A)."""
    return bootstrap_mean_ci(differences, **kwargs)  # type: ignore[arg-type]
