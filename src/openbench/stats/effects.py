"""Effect sizes and practical-importance classification (plan sections 1.4 and 1.9)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openbench.stats.results import finite_or_none

#: Paired differences with |d| <= this tolerance count as ties.
TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class BinaryPairedCounts:
    """Discordance table for a binary metric with one repetition per arm.

    ``b`` = #(A = 0, B = 1) and ``c`` = #(A = 1, B = 0), with A the baseline and B the
    challenger.
    """

    both_correct: int
    b: int
    c: int
    both_wrong: int

    @property
    def n(self) -> int:
        return self.both_correct + self.b + self.c + self.both_wrong

    @property
    def risk_difference(self) -> float | None:
        return (self.b - self.c) / self.n if self.n else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "both_correct": self.both_correct,
            "b_challenger_only_correct": self.b,
            "c_baseline_only_correct": self.c,
            "both_wrong": self.both_wrong,
            "risk_difference": self.risk_difference,
            "wins": self.b,
            "losses": self.c,
            "ties": self.both_correct + self.both_wrong,
        }


def binary_paired_counts(
    baseline: Sequence[float], challenger: Sequence[float]
) -> BinaryPairedCounts:
    """Count the 2 x 2 paired table from aligned binary scores (0.0 or 1.0)."""
    if len(baseline) != len(challenger):
        raise ValueError("baseline and challenger must have the same length")
    counts = {(1, 1): 0, (0, 1): 0, (1, 0): 0, (0, 0): 0}
    for a, b in zip(baseline, challenger, strict=True):
        if a not in (0, 1) or b not in (0, 1):
            raise ValueError("binary scores must be 0 or 1")
        counts[(int(a), int(b))] += 1
    return BinaryPairedCounts(
        both_correct=counts[(1, 1)],
        b=counts[(0, 1)],
        c=counts[(1, 0)],
        both_wrong=counts[(0, 0)],
    )


@dataclass(frozen=True)
class PairedEffects:
    """Descriptive effect sizes for paired continuous differences d = y_B - y_A."""

    sample_count: int
    mean_difference: float | None
    sd_difference: float | None
    d_z: float | None
    probability_of_superiority: float | None
    wins: int
    losses: int
    ties: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "mean_difference": finite_or_none(self.mean_difference),
            "sd_difference": finite_or_none(self.sd_difference),
            "d_z": finite_or_none(self.d_z),
            "d_z_note": "descriptive standardised mean difference (mean / SD of differences)",
            "probability_of_superiority": finite_or_none(self.probability_of_superiority),
            "probability_of_superiority_definition": "(wins + 0.5 * ties) / n",
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "tie_tolerance": TIE_TOLERANCE,
        }


def paired_effects(differences: Sequence[float]) -> PairedEffects:
    """Mean difference, SD, d_z, probability of superiority and wins/losses/ties."""
    data = [float(d) for d in differences]
    n = len(data)
    wins = sum(1 for d in data if d > TIE_TOLERANCE)
    losses = sum(1 for d in data if d < -TIE_TOLERANCE)
    ties = n - wins - losses
    if n == 0:
        return PairedEffects(0, None, None, None, None, 0, 0, 0)
    mean = math.fsum(data) / n
    sd: float | None = None
    d_z: float | None = None
    if n >= 2:
        sd = math.sqrt(math.fsum((d - mean) ** 2 for d in data) / (n - 1))
        d_z = mean / sd if sd > 0 else None
    pos = (wins + 0.5 * ties) / n
    return PairedEffects(n, mean, sd, d_z, pos, wins, losses, ties)


PRACTICAL_CLASSES = (
    "exceeds_practical_threshold",
    "below_practical_threshold",
    "inconclusive_relative_to_threshold",
)


def classify_practical_importance(
    lower: float | None, upper: float | None, threshold: float
) -> str | None:
    """Classify a CI [lower, upper] of a difference against the practical threshold dp.

    Returns ``None`` when the interval is unavailable. This is descriptive and never an
    equivalence claim.
    """
    if not threshold > 0 or not math.isfinite(threshold):
        raise ValueError("threshold must be a positive finite number")
    if lower is None or upper is None:
        return None
    if lower > upper:
        raise ValueError("lower must not exceed upper")
    if lower > threshold or upper < -threshold:
        return "exceeds_practical_threshold"
    if -threshold < lower and upper < threshold:
        return "below_practical_threshold"
    return "inconclusive_relative_to_threshold"
