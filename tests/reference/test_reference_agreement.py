"""Reference cross-checks for inter-annotator agreement (statsmodels; plan D4)."""

from __future__ import annotations

import random

import pytest

np = pytest.importorskip("numpy")
irr = pytest.importorskip("statsmodels.stats.inter_rater")

from openbench.preferences import cohens_kappa, fleiss_kappa  # noqa: E402

CATS = ("response_1", "response_2", "tie", "neither")


def _random_labels(rng, n, k=4, bias=0.6):
    base = [rng.randrange(k) for _ in range(n)]
    other = [b if rng.random() < bias else rng.randrange(k) for b in base]
    return [CATS[x] for x in base], [CATS[x] for x in other]


@pytest.mark.parametrize("seed", range(20))
def test_cohens_kappa_matches_statsmodels(seed):
    rng = random.Random(seed)
    a, b = _random_labels(rng, rng.randrange(5, 60))
    table = np.zeros((4, 4))
    for x, y in zip(a, b, strict=True):
        table[CATS.index(x), CATS.index(y)] += 1
    expected = irr.cohens_kappa(table, return_results=False)
    assert cohens_kappa(a, b) == pytest.approx(float(expected), abs=1e-12)


@pytest.mark.parametrize("seed", range(20))
def test_fleiss_kappa_matches_statsmodels(seed):
    rng = random.Random(1000 + seed)
    m = rng.randrange(2, 7)
    rows = []
    for _ in range(rng.randrange(3, 40)):
        counts = [0, 0, 0, 0]
        for _ in range(m):
            counts[min(3, int(rng.random() ** 2 * 4))] += 1
        rows.append(counts)
    expected = irr.fleiss_kappa(np.array(rows), method="fleiss")
    ours = fleiss_kappa(rows)
    if ours is None:
        assert not np.isfinite(expected)  # expected agreement is 1: kappa undefined
    else:
        assert ours == pytest.approx(float(expected), abs=1e-12)
