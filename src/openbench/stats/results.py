"""Result containers for OpenBench statistical procedures.

All containers are frozen dataclasses with a ``to_dict`` method. Floats are reported
unrounded; non-finite values are serialised as ``None`` with an explanatory field so the
output is always valid JSON.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def finite_or_none(value: float | None) -> float | None:
    """Return ``value`` if it is a finite float, otherwise ``None``."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class StatWarning:
    """A machine-readable warning attached to a statistical result."""

    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


def _warnings_to_list(warnings: tuple[StatWarning, ...]) -> list[dict[str, str]]:
    return [w.to_dict() for w in warnings]


@dataclass(frozen=True)
class ConfidenceInterval:
    """A bootstrap confidence interval for a mean or a mean paired difference."""

    estimate: float | None
    lower: float | None
    upper: float | None
    confidence_level: float
    method: str
    resamples: int
    standard_error: float | None
    sample_count: int
    analysis_seed: int
    stratified: bool = False
    unavailable_reason: str | None = None
    warnings: tuple[StatWarning, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": finite_or_none(self.estimate),
            "lower": finite_or_none(self.lower),
            "upper": finite_or_none(self.upper),
            "confidence_level": self.confidence_level,
            "method": self.method,
            "resamples": self.resamples,
            "standard_error": finite_or_none(self.standard_error),
            "sample_count": self.sample_count,
            "analysis_seed": self.analysis_seed,
            "stratified": self.stratified,
            "unavailable_reason": self.unavailable_reason,
            "warnings": _warnings_to_list(self.warnings),
        }


@dataclass(frozen=True)
class TestResult:
    """Result of a hypothesis test of no difference between two paired arms."""

    __test__ = False  # not a pytest test class

    test: str | None
    p_value: float | None
    sample_count: int
    statistic: float | None = None
    statistic_name: str | None = None
    alternative: str = "two-sided"
    permutation_distribution: str | None = None
    validity: dict[str, str] = field(default_factory=dict)
    permutations: int | None = None
    monte_carlo_se: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    unavailable_reason: str | None = None
    warnings: tuple[StatWarning, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "test": self.test,
            "p_value": finite_or_none(self.p_value),
            "sample_count": self.sample_count,
            "statistic": finite_or_none(self.statistic),
            "statistic_name": self.statistic_name,
            "alternative": self.alternative,
            "permutation_distribution": self.permutation_distribution,
            "validity": dict(self.validity),
            "permutations": self.permutations,
            "monte_carlo_se": finite_or_none(self.monte_carlo_se),
            "details": dict(self.details),
            "description": self.description,
            "unavailable_reason": self.unavailable_reason,
            "warnings": _warnings_to_list(self.warnings),
        }


@dataclass(frozen=True)
class AdjustedPValues:
    """Raw and multiplicity-adjusted p-values for one family of tests."""

    method: str
    raw: tuple[float, ...]
    adjusted: tuple[float, ...]
    family_size: int
    warnings: tuple[StatWarning, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "raw": list(self.raw),
            "adjusted": list(self.adjusted),
            "family_size": self.family_size,
            "warnings": _warnings_to_list(self.warnings),
        }


@dataclass(frozen=True)
class EquivalenceInterval:
    """The per-comparison (1 - 2 alpha) interval dual to a TOST.

    It is never adjusted for multiple comparisons.
    """

    lower: float | None
    upper: float | None
    level: float
    method: str
    multiplicity_adjusted: bool = False
    label: str = ""
    flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": finite_or_none(self.lower),
            "upper": finite_or_none(self.upper),
            "level": self.level,
            "method": self.method,
            "multiplicity_adjusted": self.multiplicity_adjusted,
            "label": self.label,
            "flags": list(self.flags),
        }


@dataclass(frozen=True)
class EquivalenceResult:
    """Result of a two one-sided tests (TOST) equivalence analysis for one comparison."""

    procedure: str
    margin: float | None
    alpha: float
    sample_count: int
    estimate: float | None
    p_lower: float | None
    p_upper: float | None
    p_tost: float | None
    p_value_method: str
    result_unadjusted: str
    interval: EquivalenceInterval | None
    reason: str | None = None
    descriptive_only: bool = False
    statistic: dict[str, float | None] = field(default_factory=dict)
    minimum_n: dict[str, Any] = field(default_factory=dict)
    numerical: dict[str, Any] = field(default_factory=dict)
    adjusted_p_tost: float | None = None
    familywise_result: str | None = None
    family: dict[str, Any] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[StatWarning, ...] = ()

    @property
    def established_familywise(self) -> bool:
        return self.familywise_result == "established"

    def to_dict(self) -> dict[str, Any]:
        return {
            "procedure": self.procedure,
            "margin": self.margin,
            "alpha": self.alpha,
            "sample_count": self.sample_count,
            "estimate": finite_or_none(self.estimate),
            "statistic": {k: finite_or_none(v) for k, v in self.statistic.items()},
            "p_lower": finite_or_none(self.p_lower),
            "p_upper": finite_or_none(self.p_upper),
            "p_tost": finite_or_none(self.p_tost),
            "p_value_method": self.p_value_method,
            "result_unadjusted": self.result_unadjusted,
            "adjusted_p_tost": finite_or_none(self.adjusted_p_tost),
            "familywise_result": self.familywise_result,
            "family": dict(self.family),
            "interval": self.interval.to_dict() if self.interval is not None else None,
            "reason": self.reason,
            "descriptive_only": self.descriptive_only,
            "minimum_n": dict(self.minimum_n),
            "numerical": dict(self.numerical),
            "assumptions": list(self.assumptions),
            "diagnostics": dict(self.diagnostics),
            "warnings": _warnings_to_list(self.warnings),
        }
