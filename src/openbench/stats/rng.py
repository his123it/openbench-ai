"""Deterministic, order-independent random streams for statistical procedures.

Each random procedure draws from its own stream, derived from the analysis seed and a
purpose label, so results do not depend on the order in which analyses run and the
difference test, confidence interval and equivalence test never share random numbers.
"""

from __future__ import annotations

import hashlib
import random

#: Purposes recognised by :func:`derive_seed`. ``stratum:<category>`` is also accepted.
PURPOSES = ("bootstrap", "permutation", "equivalence")


def _check_purpose(purpose: str) -> None:
    if purpose in PURPOSES:
        return
    if purpose.startswith("stratum:") and len(purpose) > len("stratum:"):
        return
    raise ValueError(
        f"Unknown random-stream purpose {purpose!r}; expected one of {PURPOSES} "
        "or 'stratum:<category>'"
    )


def derive_seed(
    analysis_seed: int,
    purpose: str,
    metric: str = "",
    arm_a: str = "",
    arm_b: str = "",
) -> int:
    """Return the 64-bit seed for one random stream.

    ``int.from_bytes(sha256(f"{analysis_seed}|{purpose}|{metric}|{arm_a}|{arm_b}")[:8], "big")``
    """
    if isinstance(analysis_seed, bool) or not isinstance(analysis_seed, int):
        raise TypeError("analysis_seed must be an int")
    _check_purpose(purpose)
    key = f"{analysis_seed}|{purpose}|{metric}|{arm_a}|{arm_b}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def derive_stream(
    analysis_seed: int,
    purpose: str,
    metric: str = "",
    arm_a: str = "",
    arm_b: str = "",
) -> random.Random:
    """Return a fresh :class:`random.Random` seeded by :func:`derive_seed`."""
    return random.Random(derive_seed(analysis_seed, purpose, metric, arm_a, arm_b))
