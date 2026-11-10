"""Version-independent numeric helpers (v0.4.0).

``statistics.stdev`` became correctly rounded in Python 3.12; Python 3.10 and 3.11 round
the exact variance to a float and then take ``math.sqrt``, which can differ in the last
bit. OpenBench promises identical results across Python 3.10–3.13 for identical inputs
and seeds, so standard deviations are computed here exactly and rounded once.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

_TARGET_BITS = 64  # > 53 + 2 guard bits, so round-to-odd followed by float() rounds correctly


def sample_variance_exact(values: Sequence[float]) -> Fraction:
    """Exact sample variance (denominator n - 1) of finite floats, as a Fraction."""
    n = len(values)
    if n < 2:
        raise ValueError("sample variance needs at least two values")
    exact = [Fraction(v) for v in values]
    mean = sum(exact, Fraction(0)) / n
    return sum(((x - mean) ** 2 for x in exact), Fraction(0)) / (n - 1)


def sqrt_fraction(value: Fraction) -> float:
    """Correctly rounded square root of a non-negative Fraction (round half to even)."""
    if value < 0:
        raise ValueError("square root of a negative number")
    if value == 0:
        return 0.0
    num, den = value.numerator, value.denominator
    # Scale by 4**k so that the integer square root carries at least _TARGET_BITS bits.
    k = max(0, _TARGET_BITS - (num.bit_length() - den.bit_length()) // 2 + 2)
    scaled_num = num << (2 * k)
    root = math.isqrt(scaled_num // den)  # floor(sqrt(value) * 2**k)
    if root * root * den != scaled_num:
        root |= 1  # round to odd: records that the true root lies strictly above `root`
    return root / (1 << k)  # int / int true division is correctly rounded in CPython


def stdev(values: Sequence[float]) -> float:
    """Sample standard deviation, correctly rounded and identical on Python 3.10–3.13.

    Matches ``statistics.stdev`` on Python 3.12+ for finite float input.
    """
    return sqrt_fraction(sample_variance_exact(values))
