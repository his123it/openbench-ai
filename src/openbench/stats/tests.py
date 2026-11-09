"""Difference tests and multiplicity corrections (plan sections 1.4, 1.6, 1.7, 1.11).

* Exact McNemar test for binary metrics with one repetition per arm.
* Paired sign-flip permutation test for bounded-continuous metrics and per-item means.
* Holm, Benjamini-Hochberg and no correction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

from openbench.stats._validation import finite_floats, require_int
from openbench.stats.results import AdjustedPValues, StatWarning, TestResult
from openbench.stats.rng import derive_stream

#: Provisional limit (U4) on the number of non-zero differences for exact enumeration.
DEFAULT_EXACT_PERMUTATION_MAX = 20
#: Default number of Monte Carlo sign vectors; alpha * (P + 1) is an integer for
#: alpha in {0.01, 0.05, 0.10}.
DEFAULT_PERMUTATIONS = 9_999

SIGN_FLIP_VALIDITY = {
    "sharp_exchangeability_null": "exact",
    "mean_difference_null": "approximate",
}
SIGN_FLIP_DESCRIPTION = (
    "Sign-flip permutation test (exact under the hypothesis that the two models are "
    "interchangeable on every item; approximate as a test of zero mean difference)."
)
MCNEMAR_DESCRIPTION = (
    "Exact McNemar test (exact binomial test on the discordant pairs; two-sided p-value "
    "by doubling the smaller tail)."
)
SCORE_KINDS = ("binary", "bounded_continuous", "unknown")


def select_difference_test(
    score_kind: str,
    repetitions_a: int = 1,
    repetitions_b: int = 1,
) -> str | None:
    """Return the difference test for a metric kind: ``"mcnemar_exact"``,
    ``"sign_flip"`` or ``None`` (unsupported; CI only)."""
    if score_kind not in SCORE_KINDS:
        raise ValueError(f"score_kind must be one of {SCORE_KINDS}")
    require_int("repetitions_a", repetitions_a, minimum=1)
    require_int("repetitions_b", repetitions_b, minimum=1)
    if score_kind == "binary":
        if repetitions_a == 1 and repetitions_b == 1:
            return "mcnemar_exact"
        return "sign_flip"
    if score_kind == "bounded_continuous":
        return "sign_flip"
    return None


def _sample_size_warnings(n: int) -> list[StatWarning]:
    warnings: list[StatWarning] = []
    if n < 10:
        warnings.append(StatWarning("very_small_sample", f"Only {n} items were analysed."))
    elif n < 30:
        warnings.append(
            StatWarning(
                "low_power_wide_intervals",
                f"Only {n} items were analysed; power is low and intervals are wide.",
            )
        )
    return warnings


def mcnemar_exact(b: int, c: int) -> TestResult:
    """Exact McNemar test.

    ``b`` = #(A wrong, B right) and ``c`` = #(A right, B wrong). Two-sided
    p = min(1, 2 * P(X <= min(b, c))) with X ~ Binomial(b + c, 1/2), computed in exact
    rational arithmetic.
    """
    require_int("b", b, minimum=0)
    require_int("c", c, minimum=0)
    m = b + c
    warnings: list[StatWarning] = []
    details = {"b": b, "c": c, "discordant_pairs": m}
    if m == 0:
        warnings.append(
            StatWarning("no_discordant_pairs", "No discordant pairs; p-value set to 1.0.")
        )
        return TestResult(
            test="mcnemar_exact",
            p_value=1.0,
            sample_count=m,
            statistic=0.0,
            statistic_name="b - c",
            details=details,
            description=MCNEMAR_DESCRIPTION,
            warnings=tuple(warnings),
        )
    if m < 10:
        warnings.append(
            StatWarning("few_discordant_pairs", f"Only {m} discordant pairs (b + c < 10).")
        )
    k = min(b, c)
    tail = Fraction(sum(math.comb(m, j) for j in range(k + 1)), 2**m)
    p = min(Fraction(1), 2 * tail)
    return TestResult(
        test="mcnemar_exact",
        p_value=float(p),
        sample_count=m,
        statistic=float(b - c),
        statistic_name="b - c",
        details=details,
        description=MCNEMAR_DESCRIPTION,
        warnings=tuple(warnings),
    )


def studentized_mean(differences: Sequence[float]) -> float | None:
    """t = mean / (sd / sqrt(n)); ``None`` when n < 2, +/-inf when the SD is zero."""
    n = len(differences)
    if n < 2:
        return None
    mean = math.fsum(differences) / n
    var = math.fsum((d - mean) ** 2 for d in differences) / (n - 1)
    if var <= 0.0:
        if mean == 0.0:
            return 0.0
        return math.copysign(math.inf, mean)
    return mean / math.sqrt(var / n)


def sign_flip_test(
    differences: Sequence[float],
    *,
    exact_max: int = DEFAULT_EXACT_PERMUTATION_MAX,
    permutations: int = DEFAULT_PERMUTATIONS,
    analysis_seed: int = 0,
    metric: str = "",
    arm_a: str = "",
    arm_b: str = "",
) -> TestResult:
    """Two-sided paired sign-flip permutation test of no difference.

    ``differences`` must already be aligned and sorted by item ID. Sign vectors are
    compared through the sum of signed differences, which orders them exactly as the
    studentized t statistic does (sum of squares is invariant under sign flips), so the
    p-value is identical for both statistics. ``details`` also carries the one-sided
    p-values (``p_greater``, ``p_less``).
    """
    data = finite_floats("differences", differences)
    require_int("exact_max", exact_max, minimum=0)
    require_int("permutations", permutations, minimum=1)
    require_int("analysis_seed", analysis_seed)
    n = len(data)
    warnings = _sample_size_warnings(n) if n >= 2 else []
    base = {
        "test": "sign_flip",
        "sample_count": n,
        "statistic_name": "studentized_mean_difference",
        "validity": dict(SIGN_FLIP_VALIDITY),
        "description": SIGN_FLIP_DESCRIPTION,
    }
    if n < 2:
        return TestResult(
            p_value=None,
            unavailable_reason="insufficient_items",
            warnings=tuple(warnings),
            **base,
        )
    scale = max(1.0, max(abs(d) for d in data))
    tau = 1e-12 * scale
    nonzero = [d for d in data if abs(d) > tau]
    k = len(nonzero)
    if 0 < k < 10:
        # Power depends on the k non-zero differences, not on n (audit finding F-004).
        warnings.append(
            StatWarning(
                "few_nonzero_differences",
                f"Only {k} of {n} paired differences are non-zero; the smallest attainable "
                f"two-sided p-value is {2 / 2**k:.6g}.",
            )
        )
    t_stat = studentized_mean(data)
    mean = math.fsum(data) / n
    details: dict[str, object] = {
        "mean_difference": mean,
        "nonzero_differences": k,
        "zero_tolerance": tau,
    }
    if 0 < k <= exact_max:
        details["minimum_attainable_p_two_sided"] = 2 / 2**k
    elif k > exact_max:
        details["minimum_attainable_p_two_sided"] = 1 / (permutations + 1)
    if t_stat is not None and not math.isfinite(t_stat):
        details["statistic_note"] = "zero variance; studentized statistic is infinite"
    if k == 0:
        warnings.append(
            StatWarning("all_differences_zero", "All paired differences are zero; p = 1.0.")
        )
        details.update({"p_greater": 1.0, "p_less": 1.0})
        return TestResult(
            p_value=1.0,
            statistic=0.0,
            permutation_distribution="exact_enumeration",
            permutations=1,
            details=details,
            warnings=tuple(warnings),
            **base,
        )
    # Exact integer arithmetic: every float is a dyadic rational, so scaling by a common
    # power of two makes all signed sums exact (no drift over 2^k Gray-code steps).
    denominator = max(Fraction(d).denominator for d in nonzero)
    ints = [int(Fraction(d) * denominator) for d in nonzero]
    s_obs = sum(ints)
    tau_s = int(Fraction(1e-12 * max(1.0, math.fsum(abs(d) for d in nonzero))) * denominator)
    abs_obs = abs(s_obs) - tau_s
    if k <= exact_max:
        # Gray-code enumeration of all 2^k sign patterns of the non-zero differences.
        signs = [1] * k
        current = s_obs
        count_two = 1
        count_ge = 1
        count_le = 1
        total = 1 << k
        for i in range(1, total):
            j = (i & -i).bit_length() - 1
            current -= 2 * signs[j] * ints[j]
            signs[j] = -signs[j]
            if abs(current) >= abs_obs:
                count_two += 1
            if current >= s_obs - tau_s:
                count_ge += 1
            if current <= s_obs + tau_s:
                count_le += 1
        details.update(
            {
                "p_greater": count_ge / total,
                "p_less": count_le / total,
                "sign_patterns": total,
            }
        )
        return TestResult(
            p_value=count_two / total,
            statistic=t_stat,
            permutation_distribution="exact_enumeration",
            permutations=total,
            details=details,
            warnings=tuple(warnings),
            **base,
        )
    rng = derive_stream(analysis_seed, "permutation", metric, arm_a, arm_b)
    count_two = count_ge = count_le = 0
    indexed = list(enumerate(ints))
    for _ in range(permutations):
        bits = rng.getrandbits(k)
        total_s = 0
        for j, value in indexed:
            total_s += -value if (bits >> j) & 1 else value
        if abs(total_s) >= abs_obs:
            count_two += 1
        if total_s >= s_obs - tau_s:
            count_ge += 1
        if total_s <= s_obs + tau_s:
            count_le += 1
    p = (1 + count_two) / (permutations + 1)
    details.update(
        {
            "p_greater": (1 + count_ge) / (permutations + 1),
            "p_less": (1 + count_le) / (permutations + 1),
        }
    )
    return TestResult(
        p_value=p,
        statistic=t_stat,
        permutation_distribution="monte_carlo",
        permutations=permutations,
        monte_carlo_se=math.sqrt(p * (1 - p) / permutations),
        details=details,
        warnings=tuple(warnings),
        **base,
    )


def adjust_p_values(p_values: Sequence[float], method: str = "holm") -> AdjustedPValues:
    """Adjust a family of p-values: ``"holm"`` (default), ``"bh"`` or ``"none"``."""
    raw = [float(p) for p in p_values]
    for p in raw:
        if not 0.0 <= p <= 1.0 or math.isnan(p):
            raise ValueError("p-values must lie in [0, 1]")
    k = len(raw)
    warnings: list[StatWarning] = []
    if method == "none":
        warnings.append(
            StatWarning("no_multiplicity_correction", "p-values are not adjusted.")
        )
        return AdjustedPValues("none", tuple(raw), tuple(raw), k, tuple(warnings))
    order = sorted(range(k), key=lambda i: (raw[i], i))
    adjusted = [0.0] * k
    if method == "holm":
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (k - rank) * raw[i]))
            adjusted[i] = running
    elif method == "bh":
        warnings.append(
            StatWarning(
                "controls_fdr_not_fwer",
                "Benjamini-Hochberg controls the false discovery rate, not the "
                "familywise error rate.",
            )
        )
        running = 1.0
        for rank in range(k - 1, -1, -1):
            i = order[rank]
            running = min(running, min(1.0, k * raw[i] / (rank + 1)))
            adjusted[i] = running
    else:
        raise ValueError("method must be 'holm', 'bh' or 'none'")
    return AdjustedPValues(method, tuple(raw), tuple(adjusted), k, tuple(warnings))
