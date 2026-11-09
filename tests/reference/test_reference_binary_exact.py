"""Reference cross-checks and exact calibration for the binary exact TOST."""

from __future__ import annotations

import math
import os

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from _ref_binary import (  # noqa: E402
    asymptotic_and_exact_max_sizes,
    max_size_curve,
    outcomes,
    p_upper,
    score_z,
)
from scipy.optimize import brentq, minimize_scalar  # noqa: E402
from scipy.stats import norm  # noqa: E402

from openbench.stats import binary_exact_p_upper, score_statistic  # noqa: E402

slow = pytest.mark.skipif(
    not os.environ.get("OPENBENCH_RUN_SLOW"), reason="set OPENBENCH_RUN_SLOW=1"
)


@pytest.mark.parametrize(
    ("b", "c", "n", "d0"),
    [(0, 0, 20, 0.1), (1, 1, 20, 0.1), (0, 0, 50, 0.1), (5, 2, 40, 0.15), (12, 6, 100, 0.1),
     (3, 9, 60, -0.05), (2, 0, 30, 0.2)],
)
def test_production_p_is_conservative_and_tight(b, c, n, d0) -> None:
    reference = p_upper(b, c, n, d0)  # lower bound of the true supremum
    ours = binary_exact_p_upper(b, c, n, d0).p_value
    assert ours >= reference - 1e-12
    assert ours - reference <= 1e-7


def test_score_statistic_matches_numpy_reference() -> None:
    for b, c, n in [(19, 9, 200), (0, 0, 50), (7, 1, 21), (30, 10, 200)]:
        for d0 in (-0.2, -0.1, 0.05, 0.1, 0.2):
            assert score_statistic(b, c, n, d0)[0] == pytest.approx(
                float(score_z(b, c, n, d0)), abs=1e-9
            )


def test_restricted_mle_matches_scipy_optimiser() -> None:
    for b, c, n in [(19, 9, 200), (0, 9, 50), (9, 0, 50), (1, 7, 21)]:
        for d0 in (-0.2, -0.1, 0.1, 0.2):
            lo, hi = max(0.0, -d0), (1 - d0) / 2

            def nll(q, b=b, c=c, n=n, d0=d0):
                total = 0.0
                for k, p in ((b, q + d0), (c, q), (n - b - c, 1 - 2 * q - d0)):
                    if k:
                        total -= k * math.log(max(p, 1e-300))
                return total

            res = minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-13})
            best = min((nll(res.x), res.x), (nll(lo), lo), (nll(hi), hi))[1]
            assert score_statistic(b, c, n, d0)[1] == pytest.approx(best, abs=1e-7)


def test_published_tango_interval_with_scipy_root_finding() -> None:
    """contingencytables / ratesci AS(Tango) 95% CI for n12 = 1, n21 = 7, N = 21."""
    b, c, n = 1, 7, 21
    est = (b - c) / n
    upper = brentq(lambda d: norm.cdf(score_statistic(b, c, n, d)[0]) - 0.025, est, 0.999)
    lower = brentq(lambda d: norm.sf(score_statistic(b, c, n, d)[0]) - 0.025, -0.999, est)
    assert (round(lower, 4), round(upper, 4)) == (-0.5172, -0.0260)


DOCUMENTED_MAX_SIZES = {
    # (n, delta): (asymptotic, exact) - plan section 1.9.1.6, upper boundary
    (30, 0.02): (0.0525, 0.0450), (30, 0.05): (0.0599, 0.0434),
    (30, 0.10): (0.0714, 0.0463), (30, 0.20): (0.0550, 0.0490),
    (50, 0.02): (0.0557, 0.0493), (50, 0.05): (0.0524, 0.0400),
    (50, 0.10): (0.0526, 0.0492), (50, 0.20): (0.0573, 0.0480),
    (100, 0.02): (0.0518, 0.0496), (100, 0.05): (0.0546, 0.0461),
    (100, 0.10): (0.0576, 0.0494), (100, 0.20): (0.0542, 0.0494),
    (200, 0.02): (0.0519, 0.0465), (200, 0.05): (0.0518, 0.0464),
    (200, 0.10): (0.0566, 0.0460), (200, 0.20): (0.0522, 0.0492),
}


@slow
@pytest.mark.parametrize(("key", "expected"), sorted(DOCUMENTED_MAX_SIZES.items()))
def test_reference_reproduces_documented_binary_calibration(key, expected) -> None:
    n, delta = key
    asym, exact = asymptotic_and_exact_max_sizes(n, delta)
    assert asym == pytest.approx(expected[0], abs=6e-5)
    assert exact == pytest.approx(expected[1], abs=6e-5)
    assert exact <= 0.05 < asym


@slow
@pytest.mark.parametrize(
    ("n", "delta"),
    [(n, d) for n in (2, 5, 10, 15) for d in (0.02, 0.05, 0.1, 0.2)]
    + [(20, 0.1), (30, 0.1), (30, 0.2), (40, 0.05), (50, 0.1), (60, 0.2), (100, 0.1)],
)
def test_production_procedure_controls_size_exactly(n, delta) -> None:
    """Exact size of the *production* decision rule (p_upper <= alpha) over every outcome."""
    b, c, _ = outcomes(n)
    rejected = np.array(
        [
            binary_exact_p_upper(int(bi), int(ci), n, delta, decision_alpha=0.05).decided
            == "reject"
            for bi, ci in zip(b, c, strict=True)
        ]
    )
    thetas = np.linspace(0, (1 - delta) / 2, 2001)
    sizes = max_size_curve(n, delta, rejected, thetas)
    assert sizes.max() <= 0.05
    # Interior of the null: rejection probability no larger than on the boundary.
    for eta in (0.02, 0.05):
        inner = max_size_curve(n, delta + eta, rejected, np.linspace(0, (1 - delta - eta) / 2, 801))
        assert inner.max() <= sizes.max() + 1e-12
