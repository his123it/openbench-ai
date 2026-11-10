"""Stage 1: difference tests, selection and multiplicity (plan A4, A5, A6, A9)."""

from __future__ import annotations

import itertools
import math
import random

import pytest

from openbench.stats import (
    adjust_p_values,
    mcnemar_exact,
    select_difference_test,
    sign_flip_test,
    studentized_mean,
)
from openbench.stats.tests import SIGN_FLIP_VALIDITY


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [(7, 1, 0.0703125), (6, 0, 0.03125), (1, 7, 0.0703125), (5, 5, 1.0), (10, 2, 0.0385742187)],
)
def test_mcnemar_exact_golden(b: int, c: int, expected: float) -> None:
    result = mcnemar_exact(b, c)
    assert result.p_value == pytest.approx(expected, abs=1e-10)
    assert result.test == "mcnemar_exact"


def test_mcnemar_no_discordant_pairs() -> None:
    result = mcnemar_exact(0, 0)
    assert result.p_value == 1.0
    assert [w.code for w in result.warnings] == ["no_discordant_pairs"]


def test_mcnemar_few_discordant_warning() -> None:
    assert "few_discordant_pairs" in [w.code for w in mcnemar_exact(4, 1).warnings]
    assert mcnemar_exact(8, 4).warnings == ()


@pytest.mark.parametrize(
    ("kind", "ra", "rb", "expected"),
    [
        ("binary", 1, 1, "mcnemar_exact"),
        ("binary", 3, 1, "sign_flip"),
        ("binary", 1, 5, "sign_flip"),
        ("bounded_continuous", 1, 1, "sign_flip"),
        ("bounded_continuous", 4, 4, "sign_flip"),
        ("unknown", 1, 1, None),
    ],
)
def test_selection_matrix(kind: str, ra: int, rb: int, expected: str | None) -> None:
    assert select_difference_test(kind, ra, rb) == expected


def test_selection_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        select_difference_test("ordinal")


def _brute_force_sign_flip(diffs: list[float]) -> tuple[float, float, float]:
    """Independent brute force over all 2^n sign vectors, using the plain mean."""
    n = len(diffs)
    obs = sum(diffs)
    two = ge = le = 0
    for signs in itertools.product((1, -1), repeat=n):
        s = sum(si * d for si, d in zip(signs, diffs, strict=True))
        two += abs(s) >= abs(obs) - 1e-9
        ge += s >= obs - 1e-9
        le += s <= obs + 1e-9
    total = 2**n
    return two / total, ge / total, le / total


@pytest.mark.parametrize("seed", range(12))
def test_sign_flip_exact_matches_brute_force(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(2, 11)
    diffs = [rng.choice([-0.5, -0.25, 0.0, 0.125, 0.25, 0.5, 1.0]) for _ in range(n)]
    result = sign_flip_test(diffs)
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        assert result.p_value == 1.0
        return
    two, ge, le = _brute_force_sign_flip(nonzero)
    assert result.p_value == pytest.approx(two, abs=1e-12)
    assert result.details["p_greater"] == pytest.approx(ge, abs=1e-12)
    assert result.details["p_less"] == pytest.approx(le, abs=1e-12)
    assert result.permutation_distribution == "exact_enumeration"


def test_studentized_and_mean_orderings_identical() -> None:
    """t is strictly increasing in the mean under sign flips (sum of squares fixed)."""
    rng = random.Random(7)
    for _ in range(50):
        n = rng.randint(3, 9)
        diffs = [round(rng.uniform(-1, 1), 3) for _ in range(n)]
        obs_t = abs(studentized_mean(diffs))
        obs_m = abs(math.fsum(diffs) / n)
        count_t = count_m = 0
        for signs in itertools.product((1, -1), repeat=n):
            v = [s * d for s, d in zip(signs, diffs, strict=True)]
            t = studentized_mean(v)
            count_t += abs(t) >= obs_t - 1e-12
            count_m += abs(math.fsum(v) / n) >= obs_m - 1e-12
        assert count_t == count_m


def test_sign_flip_labels_are_honest() -> None:
    result = sign_flip_test([0.1, -0.2, 0.3, 0.05])
    assert result.validity == SIGN_FLIP_VALIDITY
    assert result.validity["sharp_exchangeability_null"] == "exact"
    assert result.validity["mean_difference_null"] == "approximate"
    assert "approximate as a test of zero mean difference" in result.description
    data = result.to_dict()
    assert data["permutation_distribution"] == "exact_enumeration"
    assert "exact test" not in data["description"].lower()


def test_sign_flip_monte_carlo_path() -> None:
    rng = random.Random(3)
    diffs = [round(rng.uniform(-0.4, 0.6), 4) for _ in range(40)]
    result = sign_flip_test(diffs, exact_max=10, permutations=999, analysis_seed=5, metric="m")
    assert result.permutation_distribution == "monte_carlo"
    assert result.permutations == 999
    assert result.p_value > 0
    assert result.p_value >= 1 / 1000
    assert result.monte_carlo_se == pytest.approx(
        math.sqrt(result.p_value * (1 - result.p_value) / 999)
    )
    again = sign_flip_test(diffs, exact_max=10, permutations=999, analysis_seed=5, metric="m")
    assert again.p_value == result.p_value


def test_sign_flip_never_zero_p() -> None:
    diffs = [0.5 + 0.001 * i for i in range(30)]
    result = sign_flip_test(diffs, exact_max=5, permutations=99)
    assert result.p_value == pytest.approx(1 / 100)


def test_sign_flip_edge_cases() -> None:
    single = sign_flip_test([0.3])
    assert single.p_value is None
    assert single.unavailable_reason == "insufficient_items"
    zeros = sign_flip_test([0.0, 0.0, 0.0])
    assert zeros.p_value == 1.0
    assert "all_differences_zero" in [w.code for w in zeros.warnings]
    equal = sign_flip_test([0.2] * 5)
    assert equal.statistic is None or not math.isfinite(equal.statistic)
    assert equal.to_dict()["statistic"] is None
    assert equal.p_value == pytest.approx(2 / 32)


def test_sign_flip_sample_size_warnings() -> None:
    assert "very_small_sample" in [w.code for w in sign_flip_test([0.1, 0.2, -0.1]).warnings]
    rng = random.Random(1)
    mid = [rng.uniform(-1, 1) for _ in range(15)]
    assert "low_power_wide_intervals" in [w.code for w in sign_flip_test(mid).warnings]


def test_holm_and_bh_hand_values() -> None:
    raw = [0.01, 0.04, 0.03, 0.20]
    holm = adjust_p_values(raw, "holm")
    assert holm.adjusted == pytest.approx((0.04, 0.09, 0.09, 0.20))
    bh = adjust_p_values(raw, "bh")
    assert bh.adjusted == pytest.approx((0.04, 0.05333333333, 0.05333333333, 0.20))
    assert [w.code for w in bh.warnings] == ["controls_fdr_not_fwer"]
    none = adjust_p_values(raw, "none")
    assert none.adjusted == tuple(raw)
    assert [w.code for w in none.warnings] == ["no_multiplicity_correction"]


def test_holm_worked_example_from_plan() -> None:
    adj = adjust_p_values([0.012, 0.030, 0.040], "holm").adjusted
    assert adj == pytest.approx((0.036, 0.060, 0.060))


def test_adjust_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        adjust_p_values([0.1, 1.2])
    with pytest.raises(ValueError):
        adjust_p_values([0.1], "bonferroni")
