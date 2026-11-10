"""Regression tests for the independent Stage 1 audit findings F-001 to F-006 and the U4
limits (see STAGE1_FINDINGS_RESOLUTION.md in the handoff package)."""

from __future__ import annotations

import math
import os
import random
from decimal import Decimal, getcontext
from fractions import Fraction

import pytest

from openbench.stats import (
    binary_exact_interval,
    binary_exact_p_lower,
    binary_exact_p_upper,
    bootstrap_mean_ci,
    classify_practical_importance,
    mcnemar_exact,
    paired_effects,
    select_difference_test,
    sign_flip_test,
    tost_bootstrap_t,
    tost_paired_binary_exact,
)
from openbench.stats import equivalence as eq
from openbench.stats.tests import DEFAULT_EXACT_PERMUTATION_MAX

ALPHA = 0.05
slow = pytest.mark.skipif(
    not os.environ.get("OPENBENCH_RUN_SLOW"), reason="set OPENBENCH_RUN_SLOW=1"
)

# --------------------------------------------------------------------------------- F-001


@pytest.mark.parametrize("n", [1, 2, 5, pytest.param(10, marks=slow), pytest.param(25, marks=slow)])
def test_f001_all_discordant_one_direction(n: int) -> None:
    """b = n, c = 0 (and the mirror) must not crash and must match the analytic answer.

    With b = n only the all-(0, 1) outcome is as extreme, so
    p_lower(Delta0) = sup_theta q01^n = ((1 + Delta0) / 2)^n and the interval's lower
    end is the crossing L* = 2 * alpha^(1/n) - 1; the upper end is the limit +1.
    """
    exact_lower = 2 * ALPHA ** (1 / n) - 1
    result = tost_paired_binary_exact(n, 0, n, 0.10, ALPHA)
    assert result.result_unadjusted == "not_established"
    assert result.p_upper == 1.0
    expected_p1 = ((1 - 0.10) / 2) ** n
    assert expected_p1 <= result.p_lower <= expected_p1 + 1e-8
    interval = result.interval
    assert interval.upper == 1.0
    assert exact_lower - 2e-5 <= interval.lower <= exact_lower + 1e-9

    mirror = tost_paired_binary_exact(0, n, n, 0.10, ALPHA)
    assert mirror.result_unadjusted == "not_established"
    assert mirror.p_lower == 1.0 and mirror.p_upper == result.p_lower
    assert mirror.interval.lower == -1.0
    assert mirror.interval.upper == pytest.approx(-interval.lower, abs=1e-12)


def test_f001_boundary_rules_are_exact() -> None:
    # b = n: H: Delta >= 1 is not rejected, so the upper end is the limit +1.
    assert binary_exact_interval(6, 0, 6, ALPHA).upper == 1.0
    # b < n: H: Delta >= Delta0 is rejected as Delta0 -> 1, so the upper end is below 1.
    assert binary_exact_interval(5, 0, 6, ALPHA).upper < 1.0
    assert binary_exact_interval(0, 5, 6, ALPHA).lower > -1.0


@pytest.mark.parametrize("n", [1, 2, 3, 4, pytest.param(5, marks=slow), pytest.param(6, marks=slow)])
@pytest.mark.parametrize("margin", [0.1, 0.5])
def test_f001_every_table_small_n(n: int, margin: float) -> None:
    """Exhaustive sweep of all (b, c) tables with the full TOST and interval."""
    for b in range(n + 1):
        for c in range(n + 1 - b):
            result = tost_paired_binary_exact(b, c, n, margin, ALPHA)
            lo, hi = result.interval.lower, result.interval.upper
            assert -1.0 <= lo < hi <= 1.0
            assert lo <= (b - c) / n <= hi
            established = result.result_unadjusted == "established"
            assert established == (lo >= -margin and hi <= margin)
            assert "interval_non_monotone" not in result.interval.flags


# --------------------------------------------------------------------------------- F-002


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_f002_paired_effects_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError):
        paired_effects([bad, 0.2, -0.1])


def test_f002_practical_importance_rejects_non_finite_bounds() -> None:
    with pytest.raises(ValueError):
        classify_practical_importance(float("nan"), 0.1, 0.05)
    with pytest.raises(ValueError):
        classify_practical_importance(-0.1, float("inf"), 0.05)


# --------------------------------------------------------------------------------- F-003


@pytest.mark.parametrize("values", [[], [0.5], [0.5, 0.5, 0.5]])
def test_f003_strata_length_checked_before_early_returns(values) -> None:
    with pytest.raises(ValueError):
        bootstrap_mean_ci(values, strata=["only"] * (len(values) + 1))


# --------------------------------------------------------------------------------- F-004


def test_f004_warns_on_few_nonzero_differences() -> None:
    result = sign_flip_test([0.5] * 3 + [0.0] * 32)
    codes = [w.code for w in result.warnings]
    assert codes == ["few_nonzero_differences"]
    assert result.p_value == 0.25
    assert result.details["minimum_attainable_p_two_sided"] == 0.25
    assert "smallest attainable two-sided p-value is 0.25" in result.warnings[0].message


def test_f004_no_warning_with_ten_nonzero_differences() -> None:
    rng = random.Random(2)
    diffs = [round(rng.uniform(0.05, 0.5), 3) * rng.choice((1, -1)) for _ in range(10)]
    result = sign_flip_test(diffs + [0.0] * 30)
    assert "few_nonzero_differences" not in [w.code for w in result.warnings]
    assert result.details["minimum_attainable_p_two_sided"] == 2 / 2**10


# --------------------------------------------------------------------------------- F-005


def test_f005_decision_mode_uses_reported_rounding() -> None:
    assert eq._reported_p(0.05) > 0.05
    assert eq._reported_p(0.05) == math.nextafter(0.05, math.inf)
    assert eq._reported_p(1.0) == 1.0


@pytest.mark.parametrize(
    ("n", "delta"), [(7, 0.1), (7, 0.25), pytest.param(12, 0.1, marks=slow), pytest.param(12, 0.25, marks=slow)]
)
def test_f005_decision_mode_agrees_with_full_computation(n: int, delta: float) -> None:
    for b in range(n + 1):
        for c in range(n + 1 - b):
            full = binary_exact_p_upper(b, c, n, delta).p_value
            quick = binary_exact_p_upper(b, c, n, delta, decision_alpha=ALPHA)
            assert (quick.decided == "reject") == (full <= ALPHA), (b, c)
            full_l = binary_exact_p_lower(b, c, n, -delta).p_value
            quick_l = binary_exact_p_lower(b, c, n, -delta, decision_alpha=ALPHA)
            assert (quick_l.decided == "reject") == (full_l <= ALPHA), (b, c)


def test_f005_walk_steps_counted_without_monotonicity_assumption() -> None:
    monotone = [-1, 0, 1, 1, 2, 3]
    zigzag = [-1, 1, 0, 3, 0, 5]
    assert eq._walk_steps(5, monotone) == 6 + 5 + 4
    assert eq._walk_steps(5, zigzag) > eq._walk_steps(5, monotone)
    assert eq._fp_allowance(5, [zigzag]) > eq._fp_allowance(5, [monotone])


def _exact_mixture(n: int, psi: float, pi: float, thresholds) -> Fraction:
    p, q = Fraction(psi), Fraction(pi)
    total = Fraction(0)
    for m in range(n + 1):
        w = math.comb(n, m) * p**m * (1 - p) ** (n - m)
        k = thresholds[m]
        inner = sum(
            (math.comb(m, j) * q**j * (1 - q) ** (m - j) for j in range(0, min(k, m) + 1)),
            Fraction(0),
        )
        total += w * inner
    return total


def test_f005_allowance_covers_measured_floating_point_error() -> None:
    rng = random.Random(11)
    worst_ratio = 0.0
    for _ in range(60):
        n = rng.randint(1, 30)
        psi, pi = rng.random(), rng.random()
        thresholds = [rng.randint(-1, m) for m in range(n + 1)]
        got = eq._mixture_cdf(n, psi, pi, thresholds)
        exact = _exact_mixture(n, psi, pi, thresholds)
        error = abs(Fraction(got) - exact)
        allowance = eq._fp_allowance(n, [thresholds])
        assert error <= allowance
        worst_ratio = max(worst_ratio, float(error) / allowance)
    assert worst_ratio < 0.01  # the allowance is far from tight


def test_f005_lgamma_error_within_stated_assumption() -> None:
    """The allowance assumes math.lgamma(k + 1) is within 8 ulp; check k <= 1001."""
    getcontext().prec = 60
    for k in list(range(2, 200)) + list(range(200, 1002, 7)):
        exact = Decimal(math.factorial(k)).ln()
        got = math.lgamma(k + 1.0)
        assert abs(Decimal(got) - exact) <= 8 * Decimal(math.ulp(got)), k


# --------------------------------------------------------------------------------- F-006


@pytest.mark.parametrize(
    "call",
    [
        lambda: select_difference_test("binary", True, 1),
        lambda: select_difference_test("binary", 1, False),
        lambda: sign_flip_test([0.1, -0.2, 0.3], permutations=True),
        lambda: sign_flip_test([0.1, -0.2, 0.3], exact_max=False),
        lambda: sign_flip_test([0.1, -0.2, 0.3], analysis_seed=True),
        lambda: tost_bootstrap_t([0.01 * i for i in range(30)], 0.1, resamples=True),
        lambda: tost_bootstrap_t([0.01 * i for i in range(30)], 0.1, analysis_seed=True),
        lambda: bootstrap_mean_ci([0.1, 0.2, 0.4], resamples=True),
        lambda: bootstrap_mean_ci([0.1, 0.2, 0.4], analysis_seed=False),
        lambda: mcnemar_exact(True, 1),
        lambda: tost_paired_binary_exact(1, 1, 10, 0.1, max_n=True),
        lambda: sign_flip_test([0.1, -0.2, 0.3], permutations=99.0),
    ],
)
def test_f006_booleans_and_floats_rejected_as_integers(call) -> None:
    with pytest.raises(TypeError):
        call()


# ------------------------------------------------------------------------------ U4 limits


def test_u4_binary_limit_is_500_and_never_falls_back() -> None:
    assert eq.BINARY_EXACT_MAX_N == 500
    over = tost_paired_binary_exact(30, 20, 501, 0.1)
    assert over.reason == "exact_computation_limit"
    assert over.result_unadjusted == "not_established"
    assert over.p_lower is None and over.p_upper is None and over.p_tost is None
    assert over.interval is None
    assert "asymptotic_score_p_upper" not in over.diagnostics


def test_u4_sign_flip_exact_limit_is_20() -> None:
    assert DEFAULT_EXACT_PERMUTATION_MAX == 20
    rng = random.Random(8)
    twenty = [round(rng.uniform(0.01, 0.5), 3) * rng.choice((1, -1)) for _ in range(20)]
    assert sign_flip_test(twenty).permutation_distribution == "exact_enumeration"
    assert (
        sign_flip_test(twenty + [0.123], permutations=99).permutation_distribution
        == "monte_carlo"
    )


def test_supremum_of_one_terminates_without_refinement_limit() -> None:
    """When every outcome is as extreme as the observed one the supremum is exactly 1;
    cell bounds are capped at 1 (a valid bound for a probability), so the search stops."""
    result = binary_exact_p_upper(60, 0, 60, 0.1)
    assert result.p_value == 1.0
    assert not result.refinement_limit_reached
    assert result.refinements < 100
