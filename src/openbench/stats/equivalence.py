"""Equivalence testing by two one-sided tests (TOST) - plan section 1.9.1.

Binary metrics, one repetition per arm
    Tango's restricted-MLE score statistic with an **exact unconditional** p-value: the
    supremum of the tail probability over the nuisance parameter. The supremum is bounded
    from above by a certified branch-and-bound procedure (monotone enclosures plus a
    derivative enclosure), and an explicit floating-point allowance is added before the
    p-value is rounded upward, so the reported p-value is never below the true supremum
    (up to the stated floating-point model). The dual interval is obtained by inverting the
    same conservative p-values.

Bounded-continuous metrics and per-item means
    Null-imposed bootstrap-t TOST. **Disabled for equivalence claims in v0.4.0**: the
    calibration did not validate a minimum sample size, so results are descriptive only
    and are always ``not_established`` with reason ``method_not_validated``.
"""

from __future__ import annotations

import heapq
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

from openbench.stats._validation import finite_floats, require_int
from openbench.stats.results import (
    EquivalenceInterval,
    EquivalenceResult,
    StatWarning,
)
from openbench.stats.rng import derive_stream
from openbench.stats.tests import adjust_p_values

EPS = sys.float_info.epsilon

#: Largest n for the exact binary procedure (U4, decided 2026-11-09). Above it the result
#: is ``exact_computation_limit``; there is never an asymptotic fallback.
BINARY_EXACT_MAX_N = 500
#: Target gap between the certified upper bound and the best lower bound of the supremum.
SUPREMUM_TOLERANCE = 1e-9
#: Initial number of nuisance-parameter cells in the branch-and-bound.
INITIAL_CELLS = 32
#: Cells narrower than this also use the first-order (derivative) enclosure.
TAYLOR_MAX_WIDTH = 1e-4
#: Safety cap on branch-and-bound refinements; the bound stays valid if it is reached.
MAX_REFINEMENTS = 20_000
#: Score-statistic ties within this tolerance count as at least as extreme.
Z_TIE_TOLERANCE = 1e-12
#: Outward widening (absolute) of enclosure parameters to cover their rounding error.
PARAMETER_WIDENING = 8 * EPS
#: Bisection tolerance for the endpoints of the binary dual interval.
INTERVAL_TOLERANCE = 1e-5
#: Default bootstrap resamples for the continuous procedure (alpha * (B + 1) integral).
BOOTSTRAP_T_RESAMPLES = 9_999

ESTABLISHED = "established"
NOT_ESTABLISHED = "not_established"

BINARY_PROCEDURE = "exact_unconditional_score"
BINARY_P_METHOD = "exact_unconditional_supremum_certified_upper_bound"
CONTINUOUS_PROCEDURE = "bootstrap_t_null_imposed"
CONTINUOUS_P_METHOD = "bootstrap_t_monte_carlo"

BINARY_ASSUMPTIONS = (
    "Items are independent and form a random sample from a population of items "
    "(the discordance counts are trinomial).",
    "One binary outcome per item per arm, with the same metric definition in both arms.",
    "The equivalence margin was fixed before seeing the data.",
)
CONTINUOUS_ASSUMPTIONS = (
    "Items are independent and form a random sample from a population of items.",
    "Finite variance (automatic for bounded metrics).",
    "Asymptotic only; second-order accuracy needs a non-lattice distribution.",
    "Not validated by the v0.4.0 calibration: descriptive only, never an equivalence claim.",
)
INTERVAL_LABEL = (
    "{pct}% interval for this comparison alone (not adjusted for multiple comparisons). "
    "The familywise equivalence decision uses the Holm-adjusted p-value."
)


class NonMonotoneRejectionRegionError(RuntimeError):
    """Raised when the rejection region is not a prefix in b for some m."""


# ---------------------------------------------------------------------------
# Tango score statistic
# ---------------------------------------------------------------------------


def score_statistic(b: int, c: int, n: int, delta0: float) -> tuple[float, float, float]:
    """Return ``(z, q10_tilde, variance)`` of Tango's score statistic at ``delta0``.

    ``b`` = #(A wrong, B right), ``c`` = #(A right, B wrong), estimand
    Delta = q01 - q10. The restricted MLE of q10 is the admissible root of
    ``2n q^2 + B q + C = 0`` with ``B = -(b + c) + delta0 (2n - b + c)`` and
    ``C = -c delta0 (1 - delta0)``, computed in a cancellation-free form.
    """
    coef_b = -(b + c) + delta0 * (2 * n - b + c)
    coef_c = -c * delta0 * (1.0 - delta0)
    disc = max(0.0, coef_b * coef_b - 8.0 * n * coef_c)
    root = math.sqrt(disc)
    if coef_b <= 0:
        q = (-coef_b + root) / (4.0 * n)
    else:
        denom = -coef_b - root
        q = (2.0 * coef_c) / denom if denom != 0 else 0.0
    lo, hi = max(0.0, -delta0), (1.0 - delta0) / 2.0
    q = min(max(q, lo), hi)
    variance = n * (2.0 * q + delta0 * (1.0 - delta0))
    if variance <= 0.0:
        return 0.0, q, variance
    return (b - c - n * delta0) / math.sqrt(variance), q, variance


def _normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@lru_cache(maxsize=32)
def _log_factorials(n: int) -> tuple[float, ...]:
    return tuple(math.lgamma(k + 1.0) for k in range(n + 1))


def _walk_steps(size: int, thresholds: Sequence[int]) -> int:
    """Number of probability terms one :func:`_walk` of ``size`` evaluates (<= its cost).

    size + 1 binomial weights, at most ``size`` pmf terms for the m -> m + 1 CDF updates,
    plus one pmf term per unit change of the (clamped) threshold. Counting the actual
    changes avoids assuming that k_m is non-decreasing in m (audit finding F-005).
    """
    clamped = [max(-1, min(thresholds[m], m)) for m in range(size + 1)]
    moves = sum(abs(clamped[m + 1] - clamped[m]) for m in range(size))
    return (size + 1) + size + moves


def _fp_allowance(n: int, threshold_sets: Sequence[Sequence[int]] = ()) -> float:
    """Absolute floating-point allowance for one walk evaluation (sizes n and n - 1).

    Error model (stated assumptions, audit finding F-005):

    * ``math.lgamma(k + 1)`` has absolute error <= 8 ulp of its result for k <= n
      (measured: <= 3.21 ulp for k <= 1,200; checked in the test suite).
    * A pmf exp(x) with x = lf[m] - lf[k] - lf[m-k] + k log(pi) + (m-k) log(1-pi) has
      absolute error <= pmf * |dx| + eps * pmf, with |dx| <= 24 eps lf[n] + 5 eps * (sum of
      the term magnitudes). Because pmf <= exp(lf[m] - |k log pi|), pmf * |k log pi| <=
      lf[m] + 1 (and the same for the (m-k) log(1-pi) term), so every pmf or weight term
      carries an absolute error <= 64 eps (lf[n] + 1).
    * Each term is used in at most one addition or subtraction of magnitude <= 1, adding
      <= 8 eps; the final weighted sum adds <= (size + 1) eps, included in the 8 eps.

    The step count is taken from the actual threshold arrays (see :func:`_walk_steps`).
    A further 16 n eps covers evaluation points whose (psi, pi) differ by rounding from
    the exact boundary parameterisation.
    """
    lf_n = _log_factorials(n)[n] if n > 0 else 0.0
    steps = 3 * n + 4
    for thresholds in threshold_sets:
        size = len(thresholds) - 1
        steps = max(steps, _walk_steps(size, thresholds))
    return steps * (64.0 * EPS * (lf_n + 1.0) + 8.0 * EPS) + 16.0 * n * EPS


def _rejection_thresholds(b_obs: int, c_obs: int, n: int, delta0: float) -> list[int]:
    """Return k_m = max{b : Z(b, m - b) <= z_obs + tol} for m = 0..n (-1 if none).

    Verifies that each set is a prefix {0..k_m}; raises otherwise.
    """
    z_obs = score_statistic(b_obs, c_obs, n, delta0)[0]
    limit = z_obs + Z_TIE_TOLERANCE
    thresholds: list[int] = []
    for m in range(n + 1):
        k = -1
        broken = False
        for b in range(m + 1):
            if score_statistic(b, m - b, n, delta0)[0] <= limit:
                if broken:
                    raise NonMonotoneRejectionRegionError(
                        f"rejection region not a prefix at n={n}, m={m}, delta0={delta0}"
                    )
                k = b
            else:
                broken = True
        thresholds.append(k)
    return thresholds


def _binomial_weight_fn(n: int, psi_lo: float, psi_hi: float):
    """Return w(m) = max over psi in [psi_lo, psi_hi] of Binom(m; n, psi).

    psi^m (1 - psi)^(n - m) is unimodal in psi with mode m / n, so the maximum over the
    interval is attained at clip(m / n, psi_lo, psi_hi). With psi_lo == psi_hi this is the
    ordinary binomial pmf.
    """
    lf = _log_factorials(n)
    psi_lo = min(max(psi_lo, 0.0), 1.0)
    psi_hi = min(max(psi_hi, psi_lo), 1.0)

    def weight(m: int) -> float:
        psi = min(max(m / n, psi_lo), psi_hi) if n else psi_lo
        if psi == 0.0:
            return 1.0 if m == 0 else 0.0
        if psi == 1.0:
            return 1.0 if m == n else 0.0
        return math.exp(
            lf[n] - lf[m] - lf[n - m] + m * math.log(psi) + (n - m) * math.log1p(-psi)
        )

    return weight


def _walk(n: int, weight, pi: float, thresholds: Sequence[int]) -> float:
    """sum_m weight(m) * BinomCDF(thresholds[m]; m, pi), walking the CDF in O(n)."""
    lf = _log_factorials(n)
    pi = min(max(pi, 0.0), 1.0)
    if pi == 0.0 or pi == 1.0:
        total = 0.0
        for m in range(n + 1):
            k = thresholds[m]
            if (k >= 0) if pi == 0.0 else (k >= m):
                total += weight(m)
        return total

    lp, lq = math.log(pi), math.log1p(-pi)

    def pmf(k: int, m: int) -> float:
        return math.exp(lf[m] - lf[k] - lf[m - k] + k * lp + (m - k) * lq)

    k = max(-1, min(thresholds[0], 0))
    cdf = 1.0 if k >= 0 else 0.0
    total = weight(0) * cdf
    for m in range(n):
        if 0 <= k <= m:
            cdf -= pi * pmf(k, m)
        m1 = m + 1
        target = max(-1, min(thresholds[m1], m1))
        while k < target:
            k += 1
            if 0 <= k <= m1:
                cdf += pmf(k, m1)
        while k > target:
            if 0 <= k <= m1:
                cdf -= pmf(k, m1)
            k -= 1
        if k >= m1:
            cdf = 1.0
        elif k < 0:
            cdf = 0.0
        else:
            cdf = min(max(cdf, 0.0), 1.0)
        if cdf > 0.0:
            total += weight(m1) * cdf
    return max(total, 0.0)


def _mixture_cdf(n: int, psi: float, pi: float, thresholds: Sequence[int]) -> float:
    """sum_m Binom(m; n, psi) * BinomCDF(thresholds[m]; m, pi).

    This is P{(b, c) in S} for a trinomial (n; q01 = psi*pi, q10 = psi*(1 - pi)) when S is
    the prefix set described by ``thresholds``. Evaluated in O(n) by walking the CDF.
    """
    return min(_walk(n, _binomial_weight_fn(n, psi, psi), pi, thresholds), 1.0)


def _cell_enclosure(
    n: int, delta0: float, t1: float, t2: float, thresholds: Sequence[int]
) -> float:
    """Certified upper bound of P_theta(S) over theta in [t1, t2] (zeroth order).

    Each term is Binom(m; n, psi) * Binom(b; m, pi) with psi = 2 theta + delta0 and
    pi = (theta + delta0) / psi. Over the cell, Binom(m; n, psi) <= w_hi(m) (mode clipped to
    [psi(t1), psi(t2)]) and BinomCDF(k; m, pi) is decreasing in pi, so it is maximised at
    the smaller of pi(t1), pi(t2) (pi is monotone in theta). The product of the two maxima
    bounds every term.
    """
    psi1, psi2 = 2.0 * t1 + delta0, 2.0 * t2 + delta0
    pi1 = (t1 + delta0) / psi1 if psi1 > 0 else (0.0 if delta0 < 0 else 1.0)
    pi2 = (t2 + delta0) / psi2 if psi2 > 0 else (0.0 if delta0 < 0 else 1.0)
    # Widen outward by a few ulp so rounding in psi and pi cannot shrink the enclosure.
    psi_lo = max(0.0, psi1 - PARAMETER_WIDENING)
    psi_hi = min(1.0, psi2 + PARAMETER_WIDENING)
    pi_min = max(0.0, min(pi1, pi2) - PARAMETER_WIDENING)
    weight = _binomial_weight_fn(n, psi_lo, psi_hi)
    return _walk(n, weight, pi_min, thresholds)


def _scaled_mixture(
    n: int, q01: float, q10: float, r: float, thresholds: Sequence[int]
) -> tuple[float, float]:
    """Return (value, scale) with value = sum_S Trinom(n; b, c) q01^b q10^c r^(n-b-c)
    for non-normalised non-negative weights, and scale = (q01 + q10 + r)^n."""
    q01, q10, r = max(q01, 0.0), max(q10, 0.0), max(r, 0.0)
    total = q01 + q10 + r
    if n == 0:
        return (1.0 if thresholds[0] >= 0 else 0.0), 1.0
    if total <= 0.0:
        return 0.0, 0.0
    s = q01 + q10
    psi = s / total
    pi = q01 / s if s > 0 else 0.5
    scale = total**n
    return scale * _mixture_cdf(n, psi, pi, thresholds), scale


@dataclass(frozen=True)
class SupremumBound:
    """Certified bounds on sup over theta of P_theta(Z <= z_obs)."""

    p_value: float
    lower_bound: float
    upper_bound_raw: float
    allowance: float
    refinements: int
    cells: int
    decided: str | None
    refinement_limit_reached: bool
    argmax_theta: float

    def to_dict(self) -> dict[str, object]:
        return {
            "p_value_reported": self.p_value,
            "supremum_lower_bound": self.lower_bound,
            "supremum_upper_bound": self.upper_bound_raw,
            "floating_point_allowance": self.allowance,
            "gap": self.p_value - self.lower_bound,
            "refinements": self.refinements,
            "cells": self.cells,
            "decided": self.decided,
            "refinement_limit_reached": self.refinement_limit_reached,
            "argmax_theta_lower_bound": self.argmax_theta,
        }


def _reported_p(upper_raw: float) -> float:
    """Round a certified upper bound up to the next float and cap it at 1."""
    return min(1.0, math.nextafter(upper_raw, math.inf))


def _supremum(
    n: int,
    delta0: float,
    thresholds: Sequence[int],
    *,
    tolerance: float = SUPREMUM_TOLERANCE,
    decision_alpha: float | None = None,
) -> SupremumBound:
    """Certified upper bound of sup_theta P_theta(S) over the boundary nuisance range.

    For theta in [t1, t2] each trinomial term q01^b q10^c R^r is bounded above by using
    q01, q10 at t2 and R at t1 (zeroth-order enclosure, scale (1 + 2h)^n), and the
    derivative f'(theta) = n[G1 + G2] - 2n G3 (three shifted prefix sets of size n - 1) is
    enclosed the same way, giving f <= f(mid) + (h/2) max(0, U', -L'). The minimum of the
    two bounds is used. Branch-and-bound refines the cell with the largest bound until
    bound - (best point value) <= tolerance.
    """
    lo, hi = max(0.0, -delta0), (1.0 - delta0) / 2.0
    thr1 = [thresholds[m + 1] - 1 for m in range(n)]
    thr2 = [thresholds[m + 1] for m in range(n)]
    thr3 = list(thresholds[:n])
    allowance = _fp_allowance(n, (thresholds, thr1, thr2, thr3) if n else (thresholds,))
    widen = PARAMETER_WIDENING

    def point(theta: float) -> float:
        q01 = theta + delta0
        psi = 2.0 * theta + delta0
        pi = q01 / psi if psi > 0 else 0.5
        return _mixture_cdf(n, psi, pi, thresholds)

    def cell_bound(t1: float, t2: float, f_mid: float) -> float:
        # f is a probability, so 1 is always an exact upper bound; capping lets the search
        # stop when the supremum is 1 instead of refining to the cap.
        bound0 = min(1.0, _cell_enclosure(n, delta0, t1, t2, thresholds) + allowance)
        h = t2 - t1
        if n == 0 or h > TAYLOR_MAX_WIDTH:
            return bound0
        # Outward-rounded enclosure of (q01, q10, R) over the cell.
        up_q01, up_q10, up_r = t2 + delta0 + widen, t2 + widen, 1.0 - 2.0 * t1 - delta0 + widen
        lo_q01, lo_q10, lo_r = t1 + delta0 - widen, t1 - widen, 1.0 - 2.0 * t2 - delta0 - widen
        g1h, sh = _scaled_mixture(n - 1, up_q01, up_q10, up_r, thr1)
        g2h, _ = _scaled_mixture(n - 1, up_q01, up_q10, up_r, thr2)
        g3h, _ = _scaled_mixture(n - 1, up_q01, up_q10, up_r, thr3)
        g1l, _ = _scaled_mixture(n - 1, lo_q01, lo_q10, lo_r, thr1)
        g2l, _ = _scaled_mixture(n - 1, lo_q01, lo_q10, lo_r, thr2)
        g3l, _ = _scaled_mixture(n - 1, lo_q01, lo_q10, lo_r, thr3)
        upper_slope = n * (g1h + g2h) - 2.0 * n * g3l
        lower_slope = n * (g1l + g2l) - 2.0 * n * g3h
        slope = max(0.0, upper_slope, -lower_slope)
        slope_err = 4.0 * n * max(1.0, sh) * (allowance + 16.0 * n * EPS)
        bound1 = f_mid + allowance + 0.5 * h * (slope + slope_err)
        return min(bound0, bound1)

    best = -1.0
    arg = lo
    for theta in (lo, hi):
        value = point(theta)
        if value > best:
            best, arg = value, theta
    heap: list[tuple[float, float, float]] = []
    width = (hi - lo) / INITIAL_CELLS
    for i in range(INITIAL_CELLS):
        t1 = lo + i * width
        t2 = hi if i == INITIAL_CELLS - 1 else lo + (i + 1) * width
        mid = 0.5 * (t1 + t2)
        f_mid = point(mid)
        if f_mid > best:
            best, arg = f_mid, mid
        heapq.heappush(heap, (-cell_bound(t1, t2, f_mid), t1, t2))

    refinements = 0
    limit_reached = False
    decided: str | None = None
    while True:
        top = -heap[0][0]
        if decision_alpha is not None:
            # Compare the value that would be reported (rounded up), so decision mode and
            # full computation always agree (audit finding F-005).
            if _reported_p(max(top, best + allowance)) <= decision_alpha:
                decided = "reject"
                break
            if best - allowance > decision_alpha:
                decided = "not_reject"
                break
        if top - (best + allowance) <= tolerance:
            break
        if refinements >= MAX_REFINEMENTS:
            limit_reached = True
            break
        _, t1, t2 = heapq.heappop(heap)
        mid = 0.5 * (t1 + t2)
        if not t1 < mid < t2:
            # Cannot split further in floating point; keep the (valid) bound and stop.
            heapq.heappush(heap, (-top, t1, t2))
            limit_reached = True
            break
        for a, b in ((t1, mid), (mid, t2)):
            m = 0.5 * (a + b)
            f_m = point(m)
            if f_m > best:
                best, arg = f_m, m
            heapq.heappush(heap, (-cell_bound(a, b, f_m), a, b))
        refinements += 1
    upper_raw = max(-heap[0][0], best + allowance)
    reported = _reported_p(upper_raw)
    return SupremumBound(
        p_value=reported,
        lower_bound=max(0.0, best - allowance),
        upper_bound_raw=upper_raw,
        allowance=allowance,
        refinements=refinements,
        cells=len(heap),
        decided=decided,
        refinement_limit_reached=limit_reached,
        argmax_theta=arg,
    )


def binary_exact_p_upper(
    b: int,
    c: int,
    n: int,
    delta0: float,
    *,
    tolerance: float = SUPREMUM_TOLERANCE,
    decision_alpha: float | None = None,
) -> SupremumBound:
    """Conservative exact unconditional p-value for H: Delta >= delta0 (reject for small Z).

    With ``decision_alpha`` the search stops as soon as ``p <= alpha`` or ``p > alpha`` is
    certified; the returned ``p_value`` is then only a valid upper bound, not a tight one.
    """
    _check_counts(b, c, n)
    if not -1.0 < delta0 < 1.0:
        raise ValueError("delta0 must lie in (-1, 1)")
    thresholds = _rejection_thresholds(b, c, n, delta0)
    return _supremum(
        n, delta0, thresholds, tolerance=tolerance, decision_alpha=decision_alpha
    )


def binary_exact_p_lower(
    b: int,
    c: int,
    n: int,
    delta0: float,
    *,
    tolerance: float = SUPREMUM_TOLERANCE,
    decision_alpha: float | None = None,
) -> SupremumBound:
    """Conservative exact p-value for H: Delta <= delta0 (by symmetry: swap b, c; negate)."""
    return binary_exact_p_upper(
        c, b, n, -delta0, tolerance=tolerance, decision_alpha=decision_alpha
    )


def _check_counts(b: int, c: int, n: int) -> None:
    require_int("b", b, minimum=0)
    require_int("c", c, minimum=0)
    require_int("n", n, minimum=1)
    if b + c > n:
        raise ValueError("b + c must not exceed n")


def _invert_side(
    rejects, start: float, end: float, extra_points: Sequence[float]
) -> tuple[float, bool]:
    """Find the outward endpoint on one side of the estimate.

    ``rejects(x)`` returns True if the null at x is (certifiably) rejected. Points run from
    ``start`` (the estimate, not rejected) towards ``end``. Returns (endpoint, non_monotone):
    the first certified-rejected value beyond the last non-rejected scan point, refined by
    bisection, or ``end`` if nothing is rejected.
    """
    direction = 1.0 if end >= start else -1.0
    grid = [start + (end - start) * i / 20 for i in range(21)]
    grid.extend(x for x in extra_points if (x - start) * direction > 0 and (end - x) * direction >= 0)
    grid = sorted(set(grid), key=lambda x: (x - start) * direction)
    flags = [rejects(x) for x in grid]
    last_ok = max(i for i, r in enumerate(flags) if not r) if not all(flags) else -1
    non_monotone = any(flags[i] for i in range(last_ok))
    if last_ok == len(grid) - 1:
        return end, non_monotone
    if last_ok < 0:
        # The estimate itself was rejected; report the degenerate side and flag it.
        return start, True
    good, bad = grid[last_ok], grid[last_ok + 1]
    while abs(bad - good) > INTERVAL_TOLERANCE:
        mid = 0.5 * (good + bad)
        if rejects(mid):
            bad = mid
        else:
            good = mid
    return bad, non_monotone


def binary_exact_interval(
    b: int, c: int, n: int, alpha: float, *, margin: float | None = None
) -> EquivalenceInterval:
    """(1 - 2 alpha) interval by inverting the conservative one-sided exact tests.

    Reported as an open interval (L, U): every Delta0 not rejected by either test lies
    strictly inside. L and U are certified-rejected values (or the parameter-space limits
    -1 and +1) found by a 21-point scan plus bisection to 1e-5; the margin points are added
    to the scan.

    At the limits Delta0 = +/-1 the tests are decided exactly instead of being computed
    (audit finding F-001):

    * H: Delta >= -1 and H: Delta <= +1 are the whole parameter space: never rejected.
    * H: Delta >= +1 means q01 = 1, so every item is (0, 1) with probability 1. It is the
      limit of the tests at Delta0 -> 1 and is rejected exactly when b < n (the p-value
      tends to 0 if b < n and equals 1 if b = n). Symmetrically, H: Delta <= -1 is
      rejected exactly when c < n.
    """
    _check_alpha(alpha)
    _check_counts(b, c, n)
    estimate = (b - c) / n
    extra = [margin, -margin] if margin else []

    def rejects_upper(x: float) -> bool:  # H: Delta >= x
        if x >= 1.0:
            return b < n
        if x <= -1.0:
            return False
        r = binary_exact_p_upper(b, c, n, x, decision_alpha=alpha)
        return r.decided == "reject" or (r.decided is None and r.p_value <= alpha)

    def rejects_lower(x: float) -> bool:  # H: Delta <= x
        if x <= -1.0:
            return c < n
        if x >= 1.0:
            return False
        r = binary_exact_p_lower(b, c, n, x, decision_alpha=alpha)
        return r.decided == "reject" or (r.decided is None and r.p_value <= alpha)

    upper, nm_u = _invert_side(rejects_upper, estimate, 1.0, extra)
    lower, nm_l = _invert_side(rejects_lower, estimate, -1.0, extra)
    flags = ("interval_non_monotone",) if (nm_u or nm_l) else ()
    pct = round(100 * (1 - 2 * alpha), 6)
    return EquivalenceInterval(
        lower=lower,
        upper=upper,
        level=1 - 2 * alpha,
        method="inversion_of_conservative_exact_unconditional_score_tests",
        multiplicity_adjusted=False,
        label=INTERVAL_LABEL.format(pct=f"{pct:g}"),
        flags=flags + ("open_interval",),
    )


def _sample_warnings(n: int) -> list[StatWarning]:
    out: list[StatWarning] = []
    if n < 10:
        out.append(StatWarning("very_small_sample", f"Only {n} items were analysed."))
    elif n < 30:
        out.append(
            StatWarning(
                "low_power_wide_intervals",
                f"Only {n} items were analysed; power is low and intervals are wide.",
            )
        )
    return out


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5)")


def _check_margin(margin: float | None) -> None:
    if margin is not None and not (0.0 < margin < 1.0):
        raise ValueError("margin must lie in (0, 1)")


def tost_paired_binary_exact(
    b: int,
    c: int,
    n: int,
    margin: float | None,
    alpha: float = 0.05,
    *,
    compute_interval: bool = True,
    max_n: int = BINARY_EXACT_MAX_N,
    tolerance: float = SUPREMUM_TOLERANCE,
) -> EquivalenceResult:
    """Exact unconditional score TOST for a paired binary metric (one repetition per arm).

    ``b`` = #(baseline wrong, challenger right), ``c`` = #(baseline right, challenger
    wrong), ``n`` = number of aligned items. ``margin`` (delta_e) must be pre-specified.
    """
    _check_alpha(alpha)
    _check_margin(margin)
    _check_counts(b, c, n)
    require_int("max_n", max_n, minimum=1)
    estimate = (b - c) / n
    warnings = _sample_warnings(n)
    if b + c < 10:
        warnings.append(
            StatWarning("few_discordant_pairs", f"Only {b + c} discordant pairs (b + c < 10).")
        )
    base = {
        "procedure": BINARY_PROCEDURE,
        "margin": margin,
        "alpha": alpha,
        "sample_count": n,
        "estimate": estimate,
        "p_value_method": BINARY_P_METHOD,
        "minimum_n": {"value": None, "status": "none_required_for_validity"},
        "assumptions": BINARY_ASSUMPTIONS,
    }
    if margin is None:
        return EquivalenceResult(
            p_lower=None, p_upper=None, p_tost=None, result_unadjusted=NOT_ESTABLISHED,
            interval=None, reason="no_margin", warnings=tuple(warnings), **base,
        )
    if n > max_n:
        return EquivalenceResult(
            p_lower=None, p_upper=None, p_tost=None, result_unadjusted=NOT_ESTABLISHED,
            interval=None, reason="exact_computation_limit", warnings=tuple(warnings), **base,
        )
    z_lower = score_statistic(b, c, n, -margin)[0]
    z_upper = score_statistic(b, c, n, margin)[0]
    try:
        lower = binary_exact_p_lower(b, c, n, -margin, tolerance=tolerance)
        upper = binary_exact_p_upper(b, c, n, margin, tolerance=tolerance)
    except NonMonotoneRejectionRegionError:
        return EquivalenceResult(
            p_lower=None, p_upper=None, p_tost=None, result_unadjusted=NOT_ESTABLISHED,
            interval=None, reason="non_monotone_rejection_region",
            warnings=tuple(warnings), **base,
        )
    p_tost = max(lower.p_value, upper.p_value)
    values = (lower.p_value, upper.p_value, p_tost)
    if not all(math.isfinite(v) for v in values):
        return EquivalenceResult(
            p_lower=None, p_upper=None, p_tost=None, result_unadjusted=NOT_ESTABLISHED,
            interval=None, reason="numerical_failure", warnings=tuple(warnings), **base,
        )
    established = p_tost <= alpha
    interval = binary_exact_interval(b, c, n, alpha, margin=margin) if compute_interval else None
    diagnostics = {
        "asymptotic_score_p_lower": 1.0 - _normal_cdf(z_lower),
        "asymptotic_score_p_upper": _normal_cdf(z_upper),
        "asymptotic_note": "diagnostic only; never used for the decision",
        "discordant_b": b,
        "discordant_c": c,
    }
    return EquivalenceResult(
        p_lower=lower.p_value,
        p_upper=upper.p_value,
        p_tost=p_tost,
        result_unadjusted=ESTABLISHED if established else NOT_ESTABLISHED,
        interval=interval,
        reason=None if established else "p_tost_above_alpha",
        statistic={"z_lower": z_lower, "z_upper": z_upper},
        numerical={
            "supremum_method": "certified_branch_and_bound",
            "rounding": "upward (allowance added, then next float up)",
            "lower_test": lower.to_dict(),
            "upper_test": upper.to_dict(),
        },
        diagnostics=diagnostics,
        warnings=tuple(warnings),
        **base,
    )


# ---------------------------------------------------------------------------
# Continuous: null-imposed bootstrap-t (descriptive only in v0.4.0)
# ---------------------------------------------------------------------------


def bootstrap_t_statistic(
    sample: Sequence[float], tau: float, sqrt_n: float | None = None
) -> tuple[float, bool]:
    """Studentized mean of one resample of centred differences (section 1.9.1.4 step 6).

    Returns ``(t_star, degenerate)``; a zero-variance resample gives 0 or +/-inf and is
    kept (never discarded).
    """
    n = len(sample)
    if sqrt_n is None:
        sqrt_n = math.sqrt(n)
    m_star = math.fsum(sample) / n
    if abs(m_star) <= tau:
        m_star = 0.0
    s_star = math.sqrt(math.fsum((x - m_star) ** 2 for x in sample) / (n - 1))
    if s_star <= tau:
        return (0.0 if m_star == 0.0 else math.copysign(math.inf, m_star)), True
    return m_star / (s_star / sqrt_n), False


def centre_differences(data: Sequence[float]) -> tuple[list[float], float, float]:
    """Return (centred values with |c| <= tau snapped to 0, mean, tau)."""
    n = len(data)
    mean = math.fsum(data) / n
    tau = 1e-12 * max(1.0, max(abs(d) for d in data))
    centred = [d - mean for d in data]
    return [0.0 if abs(x) <= tau else x for x in centred], mean, tau


def tost_bootstrap_t(
    differences: Sequence[float],
    margin: float | None,
    alpha: float = 0.05,
    *,
    resamples: int = BOOTSTRAP_T_RESAMPLES,
    analysis_seed: int = 0,
    metric: str = "",
    arm_a: str = "",
    arm_b: str = "",
) -> EquivalenceResult:
    """Null-imposed bootstrap-t TOST for paired continuous differences (section 1.9.1.4).

    **Disabled for equivalence claims in v0.4.0.** The p-values and the dual interval are
    computed and returned for description only; ``result_unadjusted`` is always
    ``not_established`` (reason ``method_not_validated`` unless an earlier gate applies).
    """
    _check_alpha(alpha)
    if margin is not None and not margin > 0:
        raise ValueError("margin must be positive")
    require_int("resamples", resamples, minimum=1)
    require_int("analysis_seed", analysis_seed)
    data = finite_floats("differences", differences)
    n = len(data)
    warnings = _sample_warnings(n)
    warnings.append(
        StatWarning(
            "method_not_validated",
            "Continuous equivalence testing is disabled in v0.4.0; values are descriptive.",
        )
    )
    k = math.floor(alpha * (resamples + 1) + 1e-9)
    if abs(alpha * (resamples + 1) - round(alpha * (resamples + 1))) > 1e-9:
        warnings.append(
            StatWarning(
                "alpha_resamples_not_integral",
                "alpha * (resamples + 1) is not an integer; the Monte Carlo test is not "
                "exactly sized relative to the ideal bootstrap test.",
            )
        )
    base = {
        "procedure": CONTINUOUS_PROCEDURE,
        "margin": margin,
        "alpha": alpha,
        "sample_count": n,
        "p_value_method": CONTINUOUS_P_METHOD,
        "result_unadjusted": NOT_ESTABLISHED,
        "descriptive_only": True,
        "minimum_n": {"value": None, "status": "not_validated"},
        "assumptions": CONTINUOUS_ASSUMPTIONS,
    }
    if margin is None:
        return EquivalenceResult(
            estimate=math.fsum(data) / n if n else None, p_lower=None, p_upper=None,
            p_tost=None, interval=None, reason="no_margin", warnings=tuple(warnings), **base,
        )
    if n < 2:
        return EquivalenceResult(
            estimate=data[0] if n == 1 else None, p_lower=None, p_upper=None, p_tost=None,
            interval=None, reason="insufficient_items", warnings=tuple(warnings), **base,
        )
    centred, mean, tau = centre_differences(data)
    sd = math.sqrt(math.fsum((d - mean) ** 2 for d in data) / (n - 1))
    if sd <= tau:
        return EquivalenceResult(
            estimate=mean, p_lower=None, p_upper=None, p_tost=None, interval=None,
            reason="degenerate_zero_variance", warnings=tuple(warnings),
            diagnostics={"common_value": data[0]}, **base,
        )
    se = sd / math.sqrt(n)
    rng = derive_stream(analysis_seed, "equivalence", metric, arm_a, arm_b)
    randrange = rng.randrange
    sqrt_n = math.sqrt(n)
    stats: list[float] = []
    zero_variance = 0
    for _ in range(resamples):
        t_star, degenerate = bootstrap_t_statistic(
            [centred[randrange(n)] for _ in range(n)], tau, sqrt_n
        )
        zero_variance += degenerate
        stats.append(t_star)
    tol_t = 1e-12
    t1 = (mean + margin) / se
    t2 = (mean - margin) / se
    p1 = (1 + sum(1 for t in stats if t >= t1 - tol_t)) / (resamples + 1)
    p2 = (1 + sum(1 for t in stats if t <= t2 + tol_t)) / (resamples + 1)
    p_tost = max(p1, p2)
    if zero_variance > 0.01 * resamples:
        warnings.append(
            StatWarning(
                "many_degenerate_resamples",
                f"{zero_variance} of {resamples} resamples had zero variance.",
            )
        )
    interval: EquivalenceInterval | None = None
    if k >= 1:
        ordered = sorted(stats)
        upper_t = ordered[resamples - k]  # T*_(B+1-k), 1-indexed
        lower_t = ordered[k - 1]  # T*_(k)
        pct = round(100 * (1 - 2 * alpha), 6)
        interval = EquivalenceInterval(
            lower=mean - (upper_t + tol_t) * se,
            upper=mean - (lower_t - tol_t) * se,
            level=1 - 2 * alpha,
            method="bootstrap_t_order_statistics",
            multiplicity_adjusted=False,
            label=INTERVAL_LABEL.format(pct=f"{pct:g}") + " Descriptive only.",
        )
    return EquivalenceResult(
        estimate=mean,
        p_lower=p1,
        p_upper=p2,
        p_tost=p_tost,
        interval=interval,
        reason="method_not_validated",
        statistic={"t_lower": t1, "t_upper": t2},
        numerical={
            "resamples": resamples,
            "monte_carlo_se_p_tost": math.sqrt(p_tost * (1 - p_tost) / resamples),
            "zero_variance_resamples": zero_variance,
            "standard_error": se,
            "analysis_seed": analysis_seed,
        },
        warnings=tuple(warnings),
        **base,
    )


# ---------------------------------------------------------------------------
# Holm family
# ---------------------------------------------------------------------------


def equivalence_family_holm(
    results: Sequence[EquivalenceResult],
    alpha: float = 0.05,
    family_id: str = "equivalence",
) -> list[EquivalenceResult]:
    """Holm-adjust the p_TOST values of one equivalence family.

    Members are the results with a p_TOST that are not descriptive-only. The familywise
    decision is ``established`` iff the Holm-adjusted p_TOST <= alpha. Intervals are left
    untouched and remain labelled as not multiplicity-adjusted.
    """
    _check_alpha(alpha)
    members = [
        i for i, r in enumerate(results) if r.p_tost is not None and not r.descriptive_only
    ]
    adjusted = adjust_p_values([results[i].p_tost for i in members], "holm").adjusted
    adj_by_index = dict(zip(members, adjusted, strict=True))
    size = len(members)
    out: list[EquivalenceResult] = []
    for i, r in enumerate(results):
        family = {"id": family_id, "size": size, "correction": "holm", "member": i in adj_by_index}
        if i in adj_by_index:
            adj = adj_by_index[i]
            decision = ESTABLISHED if adj <= alpha else NOT_ESTABLISHED
            out.append(replace(r, adjusted_p_tost=adj, familywise_result=decision, family=family))
        else:
            out.append(
                replace(r, adjusted_p_tost=None, familywise_result=NOT_ESTABLISHED, family=family)
            )
    return out
