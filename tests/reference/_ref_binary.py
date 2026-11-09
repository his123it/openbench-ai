"""Independent NumPy/SciPy implementation of the exact unconditional score test.

Full-outcome enumeration (no m-decomposition, no branch-and-bound), dense nuisance grid
plus SciPy bounded refinement. Its supremum is a *lower* bound of the true supremum.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import norm


def score_z(b, c, n, d0):
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    big_b = -(b + c) + d0 * (2 * n - b + c)
    big_c = -c * d0 * (1 - d0)
    q = (-big_b + np.sqrt(np.maximum(big_b * big_b - 8 * n * big_c, 0.0))) / (4 * n)
    v = n * (2 * q + d0 * (1 - d0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, (b - c - n * d0) / np.sqrt(np.where(v > 0, v, 1.0)), 0.0)


def outcomes(n):
    b, c = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    keep = b + c <= n
    b, c = b[keep].astype(float), c[keep].astype(float)
    lt = gammaln(n + 1) - gammaln(b + 1) - gammaln(c + 1) - gammaln(n - b - c + 1)
    return b, c, lt


def log_probs(lt, b, c, n, theta, d0):
    q01, q10, r = theta + d0, theta, 1 - 2 * theta - d0
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = (
            lt
            + np.where(b > 0, b * np.log(max(q01, 1e-300)), 0.0)
            + np.where(c > 0, c * np.log(max(q10, 1e-300)), 0.0)
            + np.where(n - b - c > 0, (n - b - c) * np.log(max(r, 1e-300)), 0.0)
        )
    if q01 <= 0:
        lp = np.where(b > 0, -np.inf, lp)
    if q10 <= 0:
        lp = np.where(c > 0, -np.inf, lp)
    if r <= 0:
        lp = np.where(n - b - c > 0, -np.inf, lp)
    return lp


def p_upper(b0, c0, n, d0, grid=20_000):
    """sup over theta of P(Z(d0) <= z_obs) - dense grid plus local refinement."""
    b, c, lt = outcomes(n)
    z = score_z(b, c, n, d0)
    z_obs = float(score_z(b0, c0, n, d0))
    mask = z <= z_obs + 1e-12
    bm, cm, ltm = b[mask], c[mask], lt[mask]
    lo, hi = max(0.0, -d0), (1 - d0) / 2

    def f(theta):
        return float(np.exp(log_probs(ltm, bm, cm, n, theta, d0)).sum())

    thetas = np.linspace(lo, hi, grid + 1)
    values = np.array([f(t) for t in thetas])
    i = int(values.argmax())
    a, e = thetas[max(i - 1, 0)], thetas[min(i + 1, grid)]
    res = minimize_scalar(lambda t: -f(t), bounds=(a, e), method="bounded",
                          options={"xatol": 1e-13})
    return min(1.0, max(values[i], -res.fun))


def max_size_curve(n, delta, rejected, thetas):
    """Exact rejection probability on a theta grid for a boolean rejection set over outcomes."""
    b, c, lt = outcomes(n)
    return np.array([np.exp(log_probs(lt, b, c, n, t, delta))[rejected].sum() for t in thetas])


def asymptotic_and_exact_max_sizes(n, delta, alpha=0.05, k_p=1000, k_s=2001):
    """Reproduces the plan's section 1.9.1.6 binary table (upper boundary)."""
    b, c, lt = outcomes(n)
    z = score_z(b, c, n, delta)
    order = np.argsort(z, kind="stable")
    b, c, lt, z = b[order], c[order], lt[order], z[order]
    last = np.searchsorted(z, z + 1e-12, side="right") - 1
    psup = np.zeros(len(z))
    for theta in np.linspace(0, (1 - delta) / 2, k_p + 1):
        np.maximum(psup, np.cumsum(np.exp(log_probs(lt, b, c, n, theta, delta)))[last], out=psup)
    rej_exact = psup <= alpha
    rej_asym = norm.cdf(z) <= alpha
    size_e = size_a = 0.0
    for theta in np.linspace(0, (1 - delta) / 2, k_s):
        prob = np.exp(log_probs(lt, b, c, n, theta, delta))
        size_e = max(size_e, prob[rej_exact].sum())
        size_a = max(size_a, prob[rej_asym].sum())
    return float(size_a), float(size_e)
