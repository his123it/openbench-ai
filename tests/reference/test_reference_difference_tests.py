"""Reference cross-checks: McNemar, sign-flip and multiplicity (SciPy / statsmodels)."""

from __future__ import annotations

import random

import pytest

np = pytest.importorskip("numpy")
stats = pytest.importorskip("scipy.stats")

from openbench.stats import adjust_p_values, mcnemar_exact, sign_flip_test  # noqa: E402


@pytest.mark.parametrize(("b", "c"), [(7, 1), (6, 0), (0, 5), (12, 9), (30, 18), (3, 3)])
def test_mcnemar_matches_scipy_binomtest(b: int, c: int) -> None:
    expected = stats.binomtest(min(b, c), b + c, 0.5).pvalue
    assert mcnemar_exact(b, c).p_value == pytest.approx(min(1.0, expected), abs=1e-12)


def test_mcnemar_matches_statsmodels() -> None:
    contingency = pytest.importorskip("statsmodels.stats.contingency_tables")
    for b, c in [(7, 1), (6, 0), (12, 9), (25, 10)]:
        table = [[5, c], [b, 7]]
        expected = contingency.mcnemar(table, exact=True).pvalue
        assert mcnemar_exact(b, c).p_value == pytest.approx(expected, abs=1e-12)


def _scipy_sign_flip(diffs, alternative: str) -> float:
    res = stats.permutation_test(
        (np.asarray(diffs),),
        lambda x, axis: np.mean(x, axis=axis),
        permutation_type="samples",
        alternative=alternative,
        n_resamples=np.inf,
        vectorized=True,
    )
    return float(res.pvalue)


@pytest.mark.parametrize("seed", range(10))
def test_sign_flip_matches_scipy_exact(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(4, 12)
    diffs = [rng.choice([-0.4, -0.2, -0.1, 0.05, 0.1, 0.3, 0.6]) for _ in range(n)]
    ours = sign_flip_test(diffs)
    assert ours.permutation_distribution == "exact_enumeration"
    assert ours.details["p_greater"] == pytest.approx(_scipy_sign_flip(diffs, "greater"), abs=1e-12)
    assert ours.details["p_less"] == pytest.approx(_scipy_sign_flip(diffs, "less"), abs=1e-12)
    assert ours.p_value == pytest.approx(_scipy_sign_flip(diffs, "two-sided"), abs=1e-12)


def test_multiplicity_matches_statsmodels() -> None:
    multitest = pytest.importorskip("statsmodels.stats.multitest")
    rng = random.Random(5)
    for _ in range(20):
        raw = [rng.random() ** 2 for _ in range(rng.randint(1, 12))]
        for ours, theirs in (("holm", "holm"), ("bh", "fdr_bh")):
            expected = multitest.multipletests(raw, method=theirs)[1]
            assert adjust_p_values(raw, ours).adjusted == pytest.approx(list(expected), abs=1e-12)


def test_bootstrap_percentile_matches_numpy_linear_quantile() -> None:
    from openbench.performance import compute_percentile

    rng = random.Random(9)
    values = sorted(rng.random() for _ in range(101))
    for q in (2.5, 50.0, 97.5):
        assert compute_percentile(values, q) == pytest.approx(
            float(np.quantile(values, q / 100, method="linear")), abs=1e-12
        )
