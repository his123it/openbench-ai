"""Regression tests for the Stage 1 post-fix audit findings PF-001 to PF-003.

PF-001: ``tost_bootstrap_t`` accepted an infinite margin and ``EquivalenceResult.to_dict``
serialised a raw ``inf``. PF-002: exported low-level helpers did not validate direct
calls. PF-003: ``binary_paired_counts`` accepted booleans. The related helpers found in
the follow-up sweep (``binary_exact_interval`` margin, ``alpha``/``margin`` types) are
covered as well.
"""

from __future__ import annotations

import json
import math

import pytest

from openbench.stats import (
    EquivalenceResult,
    binary_exact_interval,
    binary_exact_p_lower,
    binary_exact_p_upper,
    binary_paired_counts,
    score_statistic,
    studentized_mean,
    tost_bootstrap_t,
    tost_paired_binary_exact,
)
from openbench.stats.equivalence import _score_statistic

INF = float("inf")
NAN = float("nan")


@pytest.mark.parametrize("margin", [INF, -INF, NAN, 0.0, -0.1])
def test_pf001_bootstrap_t_rejects_non_finite_or_non_positive_margin(margin):
    with pytest.raises(ValueError, match="positive finite"):
        tost_bootstrap_t([0.1, 0.2, 0.3], margin=margin, resamples=99)


@pytest.mark.parametrize("margin", [True, "0.1"])
def test_pf001_bootstrap_t_rejects_non_numeric_margin(margin):
    with pytest.raises(TypeError):
        tost_bootstrap_t([0.1, 0.2, 0.3], margin=margin, resamples=99)


def test_pf001_bootstrap_t_finite_margin_still_works_and_serialises():
    r = tost_bootstrap_t([0.1, 0.2, 0.3, 0.05, 0.12], margin=0.5, resamples=99)
    payload = r.to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["margin"] == 0.5
    assert r.result_unadjusted == "not_established"


def test_pf001_equivalence_result_margin_serialised_finite_or_null():
    r = EquivalenceResult(
        procedure="x",
        margin=INF,
        alpha=0.05,
        sample_count=0,
        estimate=None,
        p_lower=None,
        p_upper=None,
        p_tost=None,
        p_value_method="x",
        result_unadjusted="not_established",
        interval=None,
    )
    payload = r.to_dict()
    assert payload["margin"] is None
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("values", [[NAN, 1.0], [INF, 1.0], [1.0, -INF]])
def test_pf002_studentized_mean_rejects_non_finite(values):
    with pytest.raises(ValueError, match="finite"):
        studentized_mean(values)


def test_pf002_studentized_mean_valid_inputs_unchanged():
    assert studentized_mean([1.0]) is None
    assert studentized_mean([0.0, 0.0]) == 0.0
    assert studentized_mean([1.0, 1.0]) == math.inf
    assert studentized_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0 / (1.0 / math.sqrt(3.0)))


@pytest.mark.parametrize(
    ("args", "exc"),
    [
        ((True, 2, 20, 0.1), TypeError),
        ((5, 2, 20.0, 0.1), TypeError),
        ((5, 2, 20, 1.5), ValueError),
        ((5, 2, 20, -1.0), ValueError),
        ((5, 2, 20, NAN), ValueError),
        ((5, 2, 20, INF), ValueError),
        ((15, 10, 20, 0.1), ValueError),
        ((-1, 2, 20, 0.1), ValueError),
        ((5, 2, 0, 0.1), ValueError),
        ((5, 2, 20, True), TypeError),
    ],
)
def test_pf002_score_statistic_validates_direct_calls(args, exc):
    with pytest.raises(exc):
        score_statistic(*args)


def test_pf002_score_statistic_valid_value_matches_unchecked_core():
    assert score_statistic(5, 2, 20, 0.1) == _score_statistic(5, 2, 20, 0.1)


@pytest.mark.parametrize("fn", [binary_exact_p_upper, binary_exact_p_lower])
@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"decision_alpha": 1.5}, ValueError),
        ({"decision_alpha": 0.0}, ValueError),
        ({"decision_alpha": NAN}, ValueError),
        ({"decision_alpha": True}, TypeError),
        ({"tolerance": 0.0}, ValueError),
        ({"tolerance": -1e-9}, ValueError),
        ({"tolerance": INF}, ValueError),
        ({"tolerance": NAN}, ValueError),
        ({"tolerance": True}, TypeError),
    ],
)
def test_pf002_binary_exact_p_validates_tolerance_and_decision_alpha(fn, kwargs, exc):
    with pytest.raises(exc):
        fn(2, 1, 10, 0.1, **kwargs)


def test_pf002_binary_exact_p_valid_decision_alpha_unchanged():
    full = binary_exact_p_upper(2, 1, 10, 0.1)
    decided = binary_exact_p_upper(2, 1, 10, 0.1, decision_alpha=0.05)
    assert decided.decided in {"reject", "not_reject"}
    assert (decided.decided == "reject") == (full.p_value <= 0.05)


@pytest.mark.parametrize("margin", [INF, NAN, 1.5, -0.1, 0.0])
def test_sweep_binary_exact_interval_rejects_invalid_margin(margin):
    with pytest.raises(ValueError):
        binary_exact_interval(3, 1, 10, 0.05, margin=margin)


@pytest.mark.parametrize("alpha", [True, "0.05"])
def test_sweep_alpha_type_checked(alpha):
    with pytest.raises(TypeError):
        tost_paired_binary_exact(3, 1, 10, 0.1, alpha=alpha)


def test_sweep_binary_tost_rejects_boolean_margin():
    with pytest.raises(TypeError):
        tost_paired_binary_exact(3, 1, 10, True)


def test_pf003_binary_paired_counts_rejects_booleans():
    with pytest.raises(TypeError):
        binary_paired_counts([True, False], [1.0, 0.0])
    with pytest.raises(TypeError):
        binary_paired_counts([1, 0], [False, 1])


def test_pf003_binary_paired_counts_numeric_inputs_unchanged():
    counts = binary_paired_counts([1, 0, 1.0, 0.0], [1.0, 1, 0, 0.0])
    assert (counts.both_correct, counts.b, counts.c, counts.both_wrong) == (1, 1, 1, 1)
