# ruff: noqa
# Frozen calibration protocol, kept verbatim from the recorded run (plan section 1.9.1.6);
# do not restyle or change it, because any edit would alter the pre-specified protocol.
"""Frozen protocol (plan section 1.9.1.6): reproducible calibration of the null-imposed bootstrap-t one-sided tests at the equivalence boundaries.

RNG: NumPy PCG64. One Generator per cell, seeded with
  seed = int.from_bytes(sha256(f"openbench-calibration-v1|{scenario}|{boundary}|{delta}|{n}").digest()[:8], "big").
Within a cell the generator is used sequentially: data for replicate r, then its B bootstrap index draws.
Replicates per cell R = 4000 (MC s.e. at size 0.05 = 0.0034); bootstrap resamples B = 999 (alpha*(B+1) = 50 is an integer).
Acceptance (pre-specified in revision 3, unchanged): empirical one-sided size <= alpha + 2 s.e. and <= 0.06.
"""
import hashlib, json, sys, time
import numpy as np

ALPHA, R, B = 0.05, 4000, 999
TOL = 1e-12

def seed_for(*parts):
    key = "openbench-calibration-v1|" + "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")

# ---- data-generating processes. Each returns an (R, n) array of paired differences with E[d] exactly = boundary value.
def two_point(lo, plo, hi):
    def gen(rng, size, delta, sign):
        u = rng.random(size)
        eps = np.where(u < plo, lo, hi)          # mean zero by construction
        return sign * delta + eps
    return gen

def mirror(g):
    def gen(rng, size, delta, sign):
        return 2 * sign * delta - g(rng, size, delta, sign)   # reflect eps about the boundary
    return gen

def sym_trunc_normal(rng, size, delta, sign):
    eps = rng.normal(0, 0.15, size)
    eps = np.clip(eps, -0.6, 0.6)                 # symmetric clipping keeps mean zero
    return sign * delta + eps

def three_point_zero(rng, size, delta, sign):
    u = rng.random(size)
    eps = np.where(u < 0.3, -0.1, np.where(u < 0.9, 0.0, 0.3))   # mean -0.03 + 0.03 = 0, 60% tied at zero
    return sign * delta + eps

def per_item_mean(Rrep):
    def gen(rng, size, delta, sign):
        Rn, n = size
        if sign > 0:
            pa = rng.uniform(0.1, 0.8, size); pb = pa + delta
        else:
            pa = rng.uniform(0.2 + 0.0, 0.9, size); pb = pa - delta
        ya = rng.binomial(Rrep, pa) / Rrep; yb = rng.binomial(Rrep, pb) / Rrep
        return yb - ya                            # E[d] = pb - pa = sign*delta exactly; lattice 1/R
    return gen

def zero_inflated(rng, size, delta, sign):
    # near-bounds metric: both models score the same (d = 0) on most items; the rest differ by 0.5
    pi = 2 * delta
    u = rng.random(size)
    return np.where(u < pi, sign * 0.5, 0.0)      # E[d] = sign * 0.5 * 2 delta = sign*delta

SHAPE_ONLY = {   # location-invariant shapes: margin does not affect size; run at delta = 0.10
    "symmetric_trunc_normal": sym_trunc_normal,
    "two_point_70_30_right_skew": two_point(-0.06, 0.7, 0.14),
    "two_point_70_30_left_skew": mirror(two_point(-0.06, 0.7, 0.14)),
    "two_point_85_15_right_skew": two_point(-0.03, 0.85, 0.17),
    "two_point_85_15_left_skew": mirror(two_point(-0.03, 0.85, 0.17)),
    "three_point_60pct_tied_zero": three_point_zero,
}
MARGIN_DEPENDENT = {
    "per_item_mean_R3": per_item_mean(3),
    "per_item_mean_R5": per_item_mean(5),
    "zero_inflated_near_bounds": zero_inflated,
}

def one_cell(gen, n, delta, boundary, rng):
    sign = +1 if boundary == "upper" else -1
    D = gen(rng, (R, n), delta, sign)
    d_bar = D.mean(1); s = D.std(1, ddof=1); se = s / np.sqrt(n)
    rejected = 0; degenerate = 0; zero_var_resamples = 0
    for r in range(R):
        if s[r] <= TOL:
            degenerate += 1; continue               # not established -> counts as non-rejection
        c = D[r] - d_bar[r]
        c = np.where(np.abs(c) <= TOL * max(1.0, np.abs(D[r]).max()), 0.0, c)
        idx = rng.integers(0, n, size=(B, n))
        x = c[idx]; mu = x.mean(1); sd = x.std(1, ddof=1)
        mu = np.where(np.abs(mu) <= TOL, 0.0, mu)
        zv = sd <= TOL; zero_var_resamples += int(zv.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(zv, np.where(mu == 0, 0.0, np.sign(mu) * np.inf), mu / (sd / np.sqrt(n)))
        if boundary == "upper":                    # H02: Delta >= +delta, reject for small t2
            t_obs = (d_bar[r] - delta) / se[r]
            p = (1 + np.sum(t <= t_obs + TOL)) / (B + 1)
        else:                                      # H01: Delta <= -delta, reject for large t1
            t_obs = (d_bar[r] + delta) / se[r]
            p = (1 + np.sum(t >= t_obs - TOL)) / (B + 1)
        rejected += p <= ALPHA
    size = rejected / R
    mcse = np.sqrt(ALPHA * (1 - ALPHA) / R)
    return dict(size=size, mcse_at_alpha=mcse, mcse_at_estimate=float(np.sqrt(size * (1 - size) / R)),
                passes=bool(size <= ALPHA + 2 * mcse and size <= 0.06), degenerate_samples=degenerate,
                zero_variance_resample_fraction=zero_var_resamples / (R * B))


def run_grid():
    """Run every frozen cell and return the result rows (same order and seeds as recorded)."""
    ns = [30, 50, 100, 200, 500]
    jobs = [(name, g, 0.10, "upper") for name, g in SHAPE_ONLY.items()]
    for name, g in MARGIN_DEPENDENT.items():
        for delta in [0.02, 0.05, 0.10, 0.20]:
            for bd in ["lower", "upper"]:
                jobs.append((name, g, delta, bd))
    rows = []
    for name, g, delta, bd in jobs:
        for n in ns:
            rng = np.random.Generator(np.random.PCG64(seed_for(name, bd, delta, n)))
            res = one_cell(g, n, delta, bd, rng)
            res.update(scenario=name, boundary=bd, delta=delta, n=n, seed=seed_for(name, bd, delta, n))
            rows.append(res)
    return rows
