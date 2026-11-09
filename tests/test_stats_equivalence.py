"""Stage 1: equivalence testing (plan A7, sections 1.9.1.3-1.9.1.7).

Slow reference cases (n = 200) run only with OPENBENCH_RUN_SLOW=1.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
from dataclasses import replace
from fractions import Fraction

import pytest

from openbench.stats import (
    binary_exact_p_lower,
    binary_exact_p_upper,
    equivalence_family_holm,
    score_statistic,
    tost_bootstrap_t,
    tost_paired_binary_exact,
)
from openbench.stats.equivalence import (
    bootstrap_t_statistic,
    centre_differences,
)

slow = pytest.mark.skipif(
    not os.environ.get("OPENBENCH_RUN_SLOW"), reason="set OPENBENCH_RUN_SLOW=1"
)
DELTA = 0.10
ALPHA = 0.05


# --------------------------------------------------------------------------- score test


def test_score_statistic_worked_example() -> None:
    z_up, q_up, v_up = score_statistic(19, 9, 200, 0.10)
    assert q_up == pytest.approx(0.03330383, abs=1e-8)
    assert v_up == pytest.approx(31.321530, abs=1e-6)
    assert z_up == pytest.approx(-1.786811, abs=1e-6)
    z_lo, q_lo, v_lo = score_statistic(19, 9, 200, -0.10)
    assert q_lo == pytest.approx(0.15112256, abs=1e-8)
    assert v_lo == pytest.approx(38.449026, abs=1e-6)
    assert z_lo == pytest.approx(4.838142, abs=1e-6)


def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _asymptotic_interval(b: int, c: int, n: int, alpha_side: float) -> tuple[float, float]:
    est = (b - c) / n

    def crossing(fn, good: float, bad: float) -> float:
        for _ in range(80):
            mid = (good + bad) / 2
            if fn(mid) > alpha_side:
                good = mid
            else:
                bad = mid
        return (good + bad) / 2

    upper = crossing(lambda d: _phi(score_statistic(b, c, n, d)[0]), est, 1 - 1e-12)
    lower = crossing(lambda d: 1 - _phi(score_statistic(b, c, n, d)[0]), est, -1 + 1e-12)
    return lower, upper


def test_published_tango_asymptotic_interval() -> None:
    """contingencytables / ratesci AS(Tango) 95% CI for n12 = 1, n21 = 7, N = 21."""
    lower, upper = _asymptotic_interval(1, 7, 21, 0.025)
    assert lower == pytest.approx(-0.5172, abs=5e-5)
    assert upper == pytest.approx(-0.0260, abs=5e-5)


def test_restricted_mle_matches_direct_maximisation() -> None:
    def loglik(q: float, b: int, c: int, n: int, d0: float) -> float:
        out = 0.0
        for k, p in ((b, q + d0), (c, q), (n - b - c, 1 - 2 * q - d0)):
            if k:
                if p <= 0:
                    return -math.inf
                out += k * math.log(p)
        return out

    for b, c, n in [(19, 9, 200), (0, 0, 50), (0, 9, 50), (9, 0, 50), (1, 7, 21), (15, 15, 30)]:
        for d0 in (-0.2, -0.1, 0.02, 0.1, 0.2):
            q = score_statistic(b, c, n, d0)[1]
            lo, hi = max(0.0, -d0), (1 - d0) / 2
            grid = [lo + (hi - lo) * i / 20000 for i in range(20001)]
            best = max(grid, key=lambda x: loglik(x, b, c, n, d0))
            assert loglik(q, b, c, n, d0) >= loglik(best, b, c, n, d0) - 1e-9


# ---------------------------------------- independent certified check (exact rationals)


def _poly_mul(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _poly_pow(a: list[Fraction], k: int) -> list[Fraction]:
    out = [Fraction(1)]
    for _ in range(k):
        out = _poly_mul(out, a)
    return out


def _independent_z(b: int, c: int, n: int, d0: float) -> float:
    # Textbook quadratic formula (deliberately not the production code path).
    big_a = 2 * n
    big_b = -(b + c) + d0 * (2 * n - b + c)
    big_c = -c * d0 * (1 - d0)
    q = (-big_b + math.sqrt(max(0.0, big_b * big_b - 4 * big_a * big_c))) / (2 * big_a)
    v = n * (2 * q + d0 * (1 - d0))
    return (b - c - n * d0) / math.sqrt(v) if v > 0 else 0.0


def _independent_sup_bounds(b0: int, c0: int, n: int, d0: Fraction) -> tuple[float, float]:
    """Exact-rational polynomial f(theta) = P_theta(Z <= z_obs); returns (lower, upper)
    bounds on its supremum from a dense grid plus a Lipschitz bound on f'."""
    z_obs = _independent_z(b0, c0, n, float(d0))
    poly = [Fraction(0)]
    q01 = [d0, Fraction(1)]
    q10 = [Fraction(0), Fraction(1)]
    r = [1 - d0, Fraction(-2)]
    for b in range(n + 1):
        for c in range(n + 1 - b):
            if _independent_z(b, c, n, float(d0)) <= z_obs + 1e-12:
                coef = Fraction(math.factorial(n), math.factorial(b) * math.factorial(c) * math.factorial(n - b - c))
                term = _poly_mul(_poly_mul(_poly_pow(q01, b), _poly_pow(q10, c)), _poly_pow(r, n - b - c))
                term = [coef * t for t in term]
                if len(term) > len(poly):
                    poly += [Fraction(0)] * (len(term) - len(poly))
                for i, t in enumerate(term):
                    poly[i] += t
    lo, hi = max(Fraction(0), -d0), (1 - d0) / 2
    deriv = [i * a for i, a in enumerate(poly)][1:]
    radius = max(abs(lo), abs(hi))
    lipschitz = sum(abs(a) * radius**i for i, a in enumerate(deriv))
    steps = 4000
    h = (hi - lo) / steps

    def value(t: Fraction) -> Fraction:
        acc = Fraction(0)
        for a in reversed(poly):
            acc = acc * t + a
        return acc

    best = max(value(lo + h * i) for i in range(steps + 1))
    return float(best), float(best + lipschitz * h / 2)


@pytest.mark.parametrize(
    ("b", "c", "n", "d0"),
    [
        (1, 0, 8, Fraction(1, 10)),
        (2, 1, 8, Fraction(1, 5)),
        (0, 0, 10, Fraction(1, 10)),
        (3, 3, 10, Fraction(1, 10)),
        (4, 1, 12, Fraction(1, 5)),
        (1, 2, 9, Fraction(-3, 20)),
    ],
)
def test_certified_p_brackets_independent_supremum(b, c, n, d0) -> None:
    lower, upper = _independent_sup_bounds(b, c, n, d0)
    p = binary_exact_p_upper(b, c, n, float(d0)).p_value
    assert p >= lower  # never below a value the supremum certainly reaches
    assert p <= upper + 1e-8  # and not looser than an independent upper bound


# --------------------------------------------------------------------- reference cases

FAST_CASES = [
    # (n, b, c, p1, p2, L, U, established)
    (50, 0, 0, 0.011598, 0.011598, -0.0582, 0.0582, True),
    (20, 1, 1, 0.157476, 0.157476, -0.1661, 0.1661, False),
    (20, 0, 0, 0.121577, 0.121577, -0.1391, 0.1391, False),
]
SLOW_CASES = [
    (200, 6, 6, 0.000029, 0.000029, -0.0315, 0.0315, True),
    (200, 19, 9, 7.0e-7, 0.038941, 0.0041, 0.0963, True),
    (200, 9, 19, 0.038941, 7.0e-7, -0.0963, -0.0041, True),
    (200, 20, 9, None, 0.056562, 0.0080, 0.1022, False),
    (200, 30, 10, None, 0.559175, 0.0467, 0.1535, False),
    (200, 40, 8, None, 0.973040, 0.1057, 0.2185, False),
    (200, 0, 0, 1e-6, 1e-6, -0.0149, 0.0149, True),
]


def _check_case(n, b, c, p1, p2, lo, hi, established) -> None:
    result = tost_paired_binary_exact(b, c, n, DELTA, ALPHA)
    for got, ref in ((result.p_lower, p1), (result.p_upper, p2)):
        if ref is None:
            assert got < 1e-6
        else:
            assert got == pytest.approx(ref, abs=1e-6)
            assert got >= ref - 5e-7  # conservative: never below the (rounded) reference
    assert (result.result_unadjusted == "established") is established
    assert result.interval.lower == pytest.approx(lo, abs=1e-4)
    assert result.interval.upper == pytest.approx(hi, abs=1e-4)
    inside = result.interval.lower >= -DELTA and result.interval.upper <= DELTA
    assert inside is established
    assert result.interval.multiplicity_adjusted is False
    assert "not adjusted" in result.interval.label
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("case", FAST_CASES)
def test_binary_reference_cases_fast(case) -> None:
    _check_case(*case)


@slow
@pytest.mark.parametrize("case", SLOW_CASES)
def test_binary_reference_cases_slow(case) -> None:
    _check_case(*case)


def test_analytic_zero_discordance_case() -> None:
    expected = 0.9**20
    result = binary_exact_p_upper(0, 0, 20, 0.1)
    assert expected <= result.p_value <= expected + 2e-9
    assert result.lower_bound <= expected


def test_lower_test_is_mirror_of_upper() -> None:
    a = binary_exact_p_lower(3, 7, 30, -0.1).p_value
    b = binary_exact_p_upper(7, 3, 30, 0.1).p_value
    assert a == b


def test_tighter_tolerance_changes_little() -> None:
    a = binary_exact_p_upper(6, 3, 40, 0.1, tolerance=1e-9).p_value
    b = binary_exact_p_upper(6, 3, 40, 0.1, tolerance=1e-11).p_value
    assert b <= a + 1e-12
    assert a - b <= 1e-7


def test_decision_mode_agrees_with_full_computation() -> None:
    for b, c, n in [(2, 0, 30), (5, 1, 40), (1, 1, 25)]:
        full = binary_exact_p_upper(b, c, n, 0.15).p_value
        quick = binary_exact_p_upper(b, c, n, 0.15, decision_alpha=ALPHA)
        assert (quick.decided == "reject") == (full <= ALPHA)


def test_binary_not_established_cases() -> None:
    assert tost_paired_binary_exact(3, 3, 40, None).reason == "no_margin"
    over = tost_paired_binary_exact(3, 3, 40, 0.1, max_n=30)
    assert over.reason == "exact_computation_limit"
    assert over.result_unadjusted == "not_established" and over.p_tost is None
    small = tost_paired_binary_exact(0, 0, 20, 0.1)
    assert small.result_unadjusted == "not_established"
    assert "low_power_wide_intervals" in [w.code for w in small.warnings]
    assert "few_discordant_pairs" in [w.code for w in small.warnings]
    with pytest.raises(ValueError):
        tost_paired_binary_exact(5, 6, 10, 0.1)
    with pytest.raises(ValueError):
        tost_paired_binary_exact(1, 1, 10, 1.5)


def test_binary_result_reports_numerics_and_no_minimum_n() -> None:
    result = tost_paired_binary_exact(2, 1, 30, 0.2, compute_interval=False)
    data = result.to_dict()
    assert data["minimum_n"] == {"value": None, "status": "none_required_for_validity"}
    assert data["numerical"]["supremum_method"] == "certified_branch_and_bound"
    upper = data["numerical"]["upper_test"]
    assert upper["supremum_lower_bound"] <= upper["p_value_reported"]
    assert data["p_value_method"].startswith("exact_unconditional")
    assert "asymptotic_score_p_upper" in data["diagnostics"]


# ------------------------------------------------------------------ continuous (disabled)

EXACT_BT_CASES = [
    ([0.02, -0.03, 0.05, -0.01, 0.00, -0.02], 8, 123),
    ([0.08, 0.03, 0.11, 0.05, 0.06, 0.04], 2, 2563),
    ([0.12, 0.07, 0.15, 0.09, 0.10, 0.07], 2, 25757),
]


@pytest.mark.parametrize(("data", "ge", "le"), EXACT_BT_CASES)
def test_bootstrap_t_exact_enumeration_counts(data, ge, le) -> None:
    centred, mean, tau = centre_differences(data)
    n = len(data)
    sd = math.sqrt(math.fsum((d - mean) ** 2 for d in data) / (n - 1))
    se = sd / math.sqrt(n)
    t1, t2 = (mean + DELTA) / se, (mean - DELTA) / se
    count_ge = count_le = 0
    for idx in itertools.product(range(n), repeat=n):
        t_star, _ = bootstrap_t_statistic([centred[i] for i in idx], tau)
        count_ge += t_star >= t1 - 1e-12
        count_le += t_star <= t2 + 1e-12
    assert (count_ge, count_le) == (ge, le)


def test_continuous_equivalence_is_never_established() -> None:
    rng = random.Random(4)
    cases = [
        [rng.uniform(-0.01, 0.01) for _ in range(300)],  # clearly "equivalent" data
        [0.0] * 50,
        [0.3],
        [rng.uniform(-0.5, 0.5) for _ in range(40)],
    ]
    for data in cases:
        result = tost_bootstrap_t(data, 0.1, resamples=999)
        assert result.result_unadjusted == "not_established"
        assert result.descriptive_only is True
        assert result.reason in {
            "method_not_validated",
            "degenerate_zero_variance",
            "insufficient_items",
        }
        assert "method_not_validated" in [w.code for w in result.warnings]
        json.dumps(result.to_dict(), allow_nan=False)
    family = equivalence_family_holm([tost_bootstrap_t(cases[0], 0.1, resamples=999)])
    assert family[0].familywise_result == "not_established"
    assert family[0].family["member"] is False


def test_bootstrap_t_interval_duality() -> None:
    rng = random.Random(21)
    for trial in range(60):
        n = rng.randint(5, 60)
        shift = rng.uniform(-0.15, 0.15)
        data = [round(shift + rng.gauss(0, 0.08), 3) for _ in range(n)]
        if len(set(data)) == 1:
            continue
        r = tost_bootstrap_t(data, DELTA, resamples=999, analysis_seed=trial)
        would = r.p_tost <= ALPHA
        lo, hi = r.interval.lower, r.interval.upper
        inside = lo is not None and hi is not None and -DELTA < lo and hi < DELTA
        assert would is inside


def test_bootstrap_t_reports_alpha_resample_warning() -> None:
    data = [0.01 * i for i in range(30)]
    r = tost_bootstrap_t(data, 0.1, resamples=1000)
    assert "alpha_resamples_not_integral" in [w.code for w in r.warnings]


# ------------------------------------------------------------------------- Holm family


def test_holm_family_worked_example() -> None:
    base = tost_paired_binary_exact(0, 0, 50, 0.1, compute_interval=False)
    results = [replace(base, p_tost=p) for p in (0.012, 0.030, 0.040)]
    family = equivalence_family_holm(results, ALPHA, family_id="fam-1")
    assert [r.adjusted_p_tost for r in family] == pytest.approx([0.036, 0.060, 0.060])
    assert [r.familywise_result for r in family] == [
        "established",
        "not_established",
        "not_established",
    ]
    assert all(r.family["size"] == 3 and r.family["id"] == "fam-1" for r in family)
    # Unadjusted decisions are left untouched by the family adjustment.
    assert [r.result_unadjusted for r in family] == [base.result_unadjusted] * 3


def test_holm_family_excludes_untestable_members() -> None:
    testable = tost_paired_binary_exact(0, 0, 50, 0.1, compute_interval=False)
    untestable = tost_paired_binary_exact(0, 0, 50, None)
    family = equivalence_family_holm([testable, untestable])
    assert family[0].family["size"] == 1 and family[0].adjusted_p_tost == testable.p_tost
    assert family[1].adjusted_p_tost is None and family[1].familywise_result == "not_established"
