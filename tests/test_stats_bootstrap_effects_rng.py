"""Stage 1: bootstrap CIs, effect sizes, practical importance and RNG (plan A1-A3, A8-A10)."""

from __future__ import annotations

import math
import random
import statistics

import pytest

from openbench.performance import compute_percentile
from openbench.stats import (
    binary_paired_counts,
    bootstrap_mean_ci,
    classify_practical_importance,
    derive_seed,
    derive_stream,
    paired_bootstrap_ci,
    paired_effects,
)


def test_derive_seed_formula_and_golden_value() -> None:
    import hashlib

    expected = int.from_bytes(
        hashlib.sha256(b"0|bootstrap|exact_match|model-a|model-b").digest()[:8], "big"
    )
    assert derive_seed(0, "bootstrap", "exact_match", "model-a", "model-b") == expected
    assert derive_seed(0, "bootstrap") != derive_seed(0, "permutation")
    assert derive_seed(0, "stratum:math") != derive_seed(0, "stratum:code")


def test_derive_seed_rejects_unknown_purpose() -> None:
    with pytest.raises(ValueError):
        derive_seed(0, "shuffle")
    with pytest.raises(TypeError):
        derive_seed(True, "bootstrap")


def _brute_force_ci(values: list[float], resamples: int, seed: int, level: float):
    rng = derive_stream(seed, "bootstrap", "m", "a", "b")
    n = len(values)
    means = sorted(math.fsum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    alpha = 1 - level
    return compute_percentile(means, 100 * alpha / 2), compute_percentile(means, 100 * (1 - alpha / 2))


def test_bootstrap_matches_independent_brute_force() -> None:
    rng = random.Random(11)
    values = [round(rng.uniform(-1, 1), 4) for _ in range(25)]
    ci = paired_bootstrap_ci(
        values, resamples=2_000, analysis_seed=9, metric="m", arm_a="a", arm_b="b"
    )
    lo, hi = _brute_force_ci(values, 2_000, 9, 0.95)
    assert ci.lower == pytest.approx(lo, abs=1e-12)
    assert ci.upper == pytest.approx(hi, abs=1e-12)
    assert ci.estimate == pytest.approx(statistics.fmean(values), abs=1e-12)
    assert ci.method == "paired_percentile_bootstrap"
    assert ci.sample_count == 25 and ci.resamples == 2_000 and ci.analysis_seed == 9


def test_bootstrap_deterministic_and_seed_sensitive() -> None:
    values = [0.1 * (i % 7) - 0.2 for i in range(30)]
    a = bootstrap_mean_ci(values, resamples=500, analysis_seed=1)
    b = bootstrap_mean_ci(values, resamples=500, analysis_seed=1)
    c = bootstrap_mean_ci(values, resamples=500, analysis_seed=2)
    assert a == b
    assert (a.lower, a.upper) != (c.lower, c.upper)


def test_bootstrap_edge_cases() -> None:
    assert bootstrap_mean_ci([]).unavailable_reason == "insufficient_items"
    one = bootstrap_mean_ci([0.4])
    assert one.lower is None and one.unavailable_reason == "insufficient_items"
    flat = bootstrap_mean_ci([0.3, 0.3, 0.3])
    assert (flat.lower, flat.upper) == (0.3, 0.3)
    assert "degenerate_zero_variance" in [w.code for w in flat.warnings]
    few = bootstrap_mean_ci([0.1, 0.2, 0.4], resamples=100)
    assert "few_resamples" in [w.code for w in few.warnings]
    with pytest.raises(ValueError):
        bootstrap_mean_ci([0.1, 0.2], confidence_level=0.9999)
    with pytest.raises(ValueError):
        bootstrap_mean_ci([0.1, float("nan")])


def test_stratified_bootstrap_keeps_stratum_sizes() -> None:
    values = [0.0] * 10 + [1.0] * 5
    strata = ["a"] * 10 + ["b"] * 5
    ci = bootstrap_mean_ci(values, strata=strata, resamples=300)
    # Within-stratum resampling of constant strata keeps every resampled mean at 5/15.
    assert ci.lower == pytest.approx(1 / 3) and ci.upper == pytest.approx(1 / 3)
    assert ci.stratified is True
    with pytest.raises(ValueError):
        bootstrap_mean_ci(values, strata=["a"])


def test_bootstrap_to_dict_is_json_safe() -> None:
    import json

    ci = bootstrap_mean_ci([0.1, 0.5, 0.9, 0.2], resamples=200)
    json.dumps(ci.to_dict(), allow_nan=False)


def test_binary_paired_counts_and_risk_difference() -> None:
    a = [1, 1, 0, 0, 1, 0]
    b = [1, 0, 1, 1, 1, 0]
    counts = binary_paired_counts(a, b)
    assert (counts.both_correct, counts.b, counts.c, counts.both_wrong) == (2, 2, 1, 1)
    assert counts.risk_difference == pytest.approx(1 / 6)
    assert counts.to_dict()["ties"] == 3
    with pytest.raises(ValueError):
        binary_paired_counts([1, 0.5], [1, 1])


def test_paired_effects_hand_values() -> None:
    eff = paired_effects([0.2, -0.1, 0.0, 0.3])
    assert eff.mean_difference == pytest.approx(0.1)
    sd = statistics.stdev([0.2, -0.1, 0.0, 0.3])
    assert eff.sd_difference == pytest.approx(sd)
    assert eff.d_z == pytest.approx(0.1 / sd)
    assert (eff.wins, eff.losses, eff.ties) == (2, 1, 1)
    assert eff.probability_of_superiority == pytest.approx((2 + 0.5) / 4)
    flat = paired_effects([0.0, 0.0])
    assert flat.d_z is None
    assert paired_effects([]).mean_difference is None


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (0.06, 0.2, "exceeds_practical_threshold"),
        (-0.3, -0.06, "exceeds_practical_threshold"),
        (-0.04, 0.04, "below_practical_threshold"),
        (-0.04, 0.07, "inconclusive_relative_to_threshold"),
        (-0.05, 0.03, "inconclusive_relative_to_threshold"),
        (None, None, None),
    ],
)
def test_practical_importance(lower, upper, expected) -> None:
    assert classify_practical_importance(lower, upper, 0.05) == expected


def test_practical_importance_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        classify_practical_importance(0.0, 0.1, 0.0)
