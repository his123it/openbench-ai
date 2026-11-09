"""Timing measurements that set the provisional U4 limits (run manually; not a test).

    python tests/reference/measure_timing.py
"""

from __future__ import annotations

import json
import platform
import random
import sys
import time

from openbench.stats import sign_flip_test, tost_paired_binary_exact
from openbench.stats.equivalence import binary_exact_p_upper


def _time(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def main() -> None:
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "sign_flip": {}, "binary": {}}
    rng = random.Random(0)
    for k in (16, 18, 20, 22):
        diffs = [round(rng.uniform(-0.5, 0.5), 3) or 0.001 for _ in range(k)]
        out["sign_flip"][k] = round(_time(lambda d=diffs, k=k: sign_flip_test(d, exact_max=k)), 3)
        print("sign_flip exact k =", k, out["sign_flip"][k], "s", flush=True)
    for n in (50, 100, 200, 500, 1000):
        b, c = int(round(0.09 * n)), int(round(0.05 * n))  # a typical near-boundary table
        p_time = _time(lambda b=b, c=c, n=n: binary_exact_p_upper(b, c, n, 0.10))
        full = _time(lambda b=b, c=c, n=n: tost_paired_binary_exact(b, c, n, 0.10))
        out["binary"][n] = {"b": b, "c": c, "one_p_value_s": round(p_time, 2), "tost_with_interval_s": round(full, 2)}
        print("binary n =", n, out["binary"][n], flush=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
