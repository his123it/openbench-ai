"""Version-independent standard deviation (v0.4.0 final validation finding).

Python 3.10/3.11 ``statistics.stdev`` rounds twice (variance to float, then sqrt), while 3.12+
is correctly rounded, so bootstrap standard errors differed in the last bit across versions.
"""

from __future__ import annotations

import math
import random
import statistics
import sys
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from openbench.stats import bootstrap_mean_ci
from openbench.stats._numeric import sample_variance_exact, sqrt_fraction, stdev


def _reference_sqrt(value: Fraction) -> float:
    """Correctly rounded sqrt via 120-digit decimal arithmetic (independent of the implementation)."""
    with localcontext() as ctx:
        ctx.prec = 120
        root = (Decimal(value.numerator) / Decimal(value.denominator)).sqrt()
    return float(root)  # Decimal -> float conversion is correctly rounded


def _samples():
    rng = random.Random(20261110)
    yield [0.0, 1.0]
    yield [1.0, 1.0, 1.0]
    yield [0.1, 0.2, 0.3, 0.4]
    yield [1e-300, 2e-300, 5e-300]
    yield [1e150, -1e150, 3e149]
    for _ in range(300):
        n = rng.randint(2, 60)
        yield [rng.choice((rng.random(), float(rng.randint(0, 1)), rng.uniform(-1e6, 1e6))) for _ in range(n)]


def test_stdev_is_correctly_rounded():
    for values in _samples():
        expected = _reference_sqrt(sample_variance_exact(values))
        assert stdev(values) == expected, values


@pytest.mark.skipif(sys.version_info < (3, 12), reason="statistics.stdev is correctly rounded only on 3.12+")
def test_matches_statistics_stdev_on_312_plus():
    for values in _samples():
        assert stdev(values) == statistics.stdev(values)


def test_known_case_where_double_rounding_differs():
    # A case found by search where sqrt(float(variance)) is not the correctly rounded result.
    rng = random.Random(7)
    for _ in range(20000):
        values = [rng.random() for _ in range(rng.randint(2, 8))]
        var = sample_variance_exact(values)
        if math.sqrt(float(var)) != _reference_sqrt(var):
            assert stdev(values) == _reference_sqrt(var)
            return
    pytest.fail("no double-rounding case found; the search seed should be revisited")


def test_sqrt_fraction_edge_cases():
    assert sqrt_fraction(Fraction(0)) == 0.0
    assert sqrt_fraction(Fraction(4)) == 2.0
    assert sqrt_fraction(Fraction(1, 4)) == 0.5
    assert sqrt_fraction(Fraction(2)) == math.sqrt(2.0)
    with pytest.raises(ValueError):
        sqrt_fraction(Fraction(-1))
    with pytest.raises(ValueError):
        sample_variance_exact([1.0])


PINNED_SE = 0.07503108827081979


def test_bootstrap_standard_error_pinned():
    """Pinned value: identical on every supported Python version (checked on 3.10–3.13)."""
    values = [float(i % 3 == 0) for i in range(40)]
    ci = bootstrap_mean_ci(values, analysis_seed=11, resamples=2000)
    assert ci.standard_error == PINNED_SE

