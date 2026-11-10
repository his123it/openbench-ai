"""Reference cross-checks for the (disabled) continuous bootstrap-t TOST.

The procedure is descriptive only in v0.4.0; these tests check that the computation matches
an independent NumPy implementation and that the frozen calibration outcome stands.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from fractions import Fraction
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from openbench.stats import tost_bootstrap_t  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
calibration = pytest.mark.skipif(
    not os.environ.get("OPENBENCH_RUN_CALIBRATION"),
    reason="set OPENBENCH_RUN_CALIBRATION=1 (about 15 minutes)",
)

# Reference values: NumPy, B = 399,999, PCG64 seed 20261109 (plan section 1.9.1.7).
MC_CASES = [
    (0.000, 0.00001, -0.0346, 0.0349),
    (0.062, 0.03723, 0.0275, 0.0970),
    (0.068, 0.06544, 0.0334, 0.1029),
    (0.100, 0.50672, 0.0655, 0.1350),
    (0.250, 1.00000, 0.2155, 0.2850),
]


def _fixture() -> list[float]:
    payload = json.loads((DATA / "ref_z120_fixture.json").read_text())
    base = payload["base"]
    assert hashlib.sha256(json.dumps(base).encode()).hexdigest() == payload["sha256"]
    return base


@pytest.mark.parametrize(("shift", "p_ref", "lo_ref", "hi_ref"), MC_CASES)
def test_monte_carlo_reference_cases(shift, p_ref, lo_ref, hi_ref) -> None:
    data = [x + shift for x in _fixture()]
    result = tost_bootstrap_t(data, 0.10, 0.05, resamples=9_999)
    se = math.sqrt(max(p_ref, 1e-4) * (1 - p_ref) / 9_999)
    assert abs(result.p_tost - p_ref) <= 4 * se + 1e-4
    assert (result.p_tost <= 0.05) == (p_ref <= 0.05)  # same would-be decision
    assert result.interval.lower == pytest.approx(lo_ref, abs=0.005)
    assert result.interval.upper == pytest.approx(hi_ref, abs=0.005)
    assert result.result_unadjusted == "not_established"
    assert result.reason == "method_not_validated"


def _numpy_exact_counts(data):
    d = np.asarray(data)
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    se = sd / math.sqrt(n)
    c = d - mean
    idx = np.array(list(itertools.product(range(n), repeat=n)))
    x = c[idx]
    mu = x.mean(1)
    s = x.std(1, ddof=1)
    mu = np.where(np.abs(mu) <= 1e-12, 0.0, mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(s <= 1e-12, np.where(mu == 0, 0.0, np.sign(mu) * np.inf), mu / (s / math.sqrt(n)))
    t1, t2 = (mean + 0.1) / se, (mean - 0.1) / se
    return int(np.sum(t >= t1 - 1e-12)), int(np.sum(t <= t2 + 1e-12))


def _rational_exact_counts(data):
    d = [Fraction(str(x)) for x in data]
    n = len(d)
    m = sum(d) / n
    s2 = sum((x - m) ** 2 for x in d) / (n - 1)
    c = [x - m for x in d]
    delta = Fraction(1, 10)

    def compare(mu, v, target_num):
        # sign of (t* - target), with t* = mu / sqrt(v / n) and target = target_num / sqrt(s2 / n)
        if v == 0:
            return 0 if mu == 0 and target_num == 0 else (
                (1 if mu > 0 else -1) if mu != 0 else (-1 if target_num > 0 else 1)
            )
        sa, sb = (mu > 0) - (mu < 0), (target_num > 0) - (target_num < 0)
        if sa != sb:
            return (sa > sb) - (sa < sb)
        if sa == 0:
            return 0
        lhs, rhs = mu * mu / v, target_num * target_num / s2
        return ((lhs > rhs) - (lhs < rhs)) * sa

    ge = le = 0
    for idx in itertools.product(range(n), repeat=n):
        x = [c[i] for i in idx]
        mu = sum(x) / n
        v = sum((xi - mu) ** 2 for xi in x) / (n - 1)
        ge += compare(mu, v, m + delta) >= 0
        le += compare(mu, v, m - delta) <= 0
    return ge, le


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([0.02, -0.03, 0.05, -0.01, 0.00, -0.02], (8, 123)),
        ([0.08, 0.03, 0.11, 0.05, 0.06, 0.04], (2, 2563)),
        ([0.12, 0.07, 0.15, 0.09, 0.10, 0.07], (2, 25757)),
    ],
)
def test_exact_enumeration_counts_numpy_and_rational(data, expected) -> None:
    assert _numpy_exact_counts(data) == expected
    assert _rational_exact_counts(data) == expected


def test_recorded_calibration_outcome_disables_continuous_equivalence() -> None:
    """Apply the unchanged pre-specified rule to the recorded results: no validated n."""
    payload = json.loads((DATA / "continuous_calibration_results.json").read_text())
    rows = payload["results"]
    assert len(rows) == 150 and payload["replicates"] == 4000 and payload["resamples"] == 999
    bound = 0.05 + 2 * math.sqrt(0.05 * 0.95 / 4000)
    for row in rows:
        assert row["passes"] == (row["size"] <= bound and row["size"] <= 0.06)
    grid = sorted({row["n"] for row in rows})
    validated = [n for n in grid if all(r["passes"] for r in rows if r["n"] >= n)]
    assert validated == []
    assert sum(not r["passes"] for r in rows) == 6


@calibration
def test_rerun_frozen_continuous_calibration() -> None:
    """Re-run the frozen protocol with the recorded seeds; results must be identical."""
    import _ref_continuous_calibration as cal

    recorded = json.loads((DATA / "continuous_calibration_results.json").read_text())["results"]
    rerun = cal.run_grid()
    key = lambda r: (r["scenario"], r["boundary"], r["delta"], r["n"])  # noqa: E731
    assert {key(r): (r["size"], r["seed"]) for r in rerun} == {
        key(r): (r["size"], r["seed"]) for r in recorded
    }
