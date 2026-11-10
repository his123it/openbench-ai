"""Shared input validation for :mod:`openbench.stats` (internal)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def require_int(name: str, value: object, minimum: int | None = None) -> int:
    """Return ``value`` if it is an ``int`` (not ``bool``) and >= ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def finite_floats(name: str, values: Sequence[float]) -> list[float]:
    """Convert to floats and reject NaN and infinities."""
    out = [float(v) for v in values]
    for v in out:
        if not math.isfinite(v):
            raise ValueError(f"{name} must contain only finite numbers")
    return out
