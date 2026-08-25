# -*- coding: utf-8 -*-
"""Uncertainty of the extrapolation: Monte Carlo calibration perturbation + breakeven hierarchy ratio.

## Why this module exists (the second link most simulation work is missing)

Calibration constants are not ground truth; they are measurements with spread. The
machine's run-to-run drift is itself measured (same benchmark, three runs,
t(2048) = 0.648/0.772/0.711, about ±9%; the alpha tier drifts ~20%).
**An extrapolation table without uncertainty bands leaves the reader unable to tell
"2.18x" from "1.9-2.5x"** -- and whether a conclusion holds depends precisely on the
band's ends, not the median.

## Perturbation conventions (each with provenance; guesses are explicitly labeled "assumption")

  alpha curve     x U[0.80, 1.20]   the whole curve times one factor (run-to-run drift
                                    ~20% measured; drift is machine state, the whole
                                    curve rises and falls together)
  beta (slow)     x U[0.90, 1.10]   same benchmark, three runs, ±9% measured, rounded to ±10%
  beta (fast)     x U[0.995,1.005]  intra-node tier run-to-run spread <0.3% measured
                                    (physics-endorsed tier)
  splits          x U[0.95, 1.05]   measured range 0.042-0.046
  arrival chain   x U[0.90, 1.10]   "assumption": a tensor-op chain should drift less
                                    than communication; ±10% is conservative

Fixed seed: anyone re-running gets bit-identical bands (`python -m sim.uncertainty`).

## Reading discipline

The band propagates **calibration uncertainty** only; it excludes model structure error
(the validation gates own that). When the two are combined, the wider one governs.
"""
from __future__ import annotations

import random

from .calibrate import synthetic
from .core import MoEGeometry, one_hop_call, two_hop_call

SEED = 20260825
N_DRAWS = 400

# (name, multiplicative perturbation lower bound, upper bound, provenance)
PERTURB = [
    ("alpha", 0.80, 1.20, "run-to-run drift ~20% (measured)"),
    ("beta_slow", 0.90, 1.10, "three runs ±9% (measured, rounded)"),
    ("beta_fast", 0.995, 1.005, "run-to-run spread <0.3% (measured, physics-endorsed tier)"),
    ("splits", 0.95, 1.05, "measured range 0.042-0.046"),
    ("chain", 0.90, 1.10, "assumption: tensor chain drifts less than communication"),
]


def _perturbed(ratio: float, chain: float, f: dict):
    """Build the perturbed synthetic cluster for one set of factors."""
    c = synthetic(ratio, chain_us_per_row=chain * f["chain"])
    # alpha: one factor for the whole curve; beta: separate fast/slow factors
    # (see the convention table in the module docstring)
    for lvl, bf in ((c.fast, f["beta_fast"]), (c.slow, f["beta_slow"]),
                    (c.flat, f["beta_slow"])):
        lvl.alpha_pts = [(w, a * f["alpha"]) for w, a in lvl.alpha_pts]
        lvl.beta_pts = [(x, b * bf) for x, b in lvl.beta_pts]
    c.splits_sync_ms *= f["splits"]
    return c


def _speedup(cluster, q: int, tok: int, k_base: int = 6) -> float:
    k = k_base if k_base % q == 0 and k_base >= q else q
    m = max(k // q, 1)
    g = MoEGeometry(name="mc", n_groups=16, R=cluster.R, k=k, M=m,
                    seq=tok, mbs=1, gbs=16 * cluster.R * tok)
    return one_hop_call(cluster, g) / two_hop_call(cluster, g)


def mc_band(ratio: float, chain: float, q: int = 3, tok: int = 4096,
            n: int = N_DRAWS, seed: int = SEED):
    """Return (p5, median, p95): quantiles of the one-hop/two-hop ratio under calibration perturbation."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        f = {name: rng.uniform(lo, hi) for name, lo, hi, _ in PERTURB}
        vals.append(_speedup(_perturbed(ratio, chain, f), q, tok))
    vals.sort()
    return vals[int(0.05 * n)], vals[n // 2], vals[int(0.95 * n)]


def breakeven_ratio(chain: float, q: int = 3, tok: int = 4096,
                    lo: float = 1.0, hi: float = 32.0) -> float:
    """Smallest hierarchy ratio at which two-hop starts winning (ratio=1); returns the boundary if there is no crossing inside the interval."""
    def s(r):
        return _speedup(synthetic(r, chain_us_per_row=chain), q, tok)
    if s(lo) >= 1.0:
        return lo
    if s(hi) < 1.0:
        return hi
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if s(mid) >= 1.0:
            hi = mid
        else:
            lo = mid
    return hi


def heatmap(chain: float, q: int = 3,
            ratios=(1.03, 1.5, 2, 3, 4.5, 6, 8, 11, 16),
            toks=(512, 1024, 2048, 4096, 8192, 16384)):
    """Ratio matrix over (ratios x toks), for figures. Rows = tok, columns = ratio."""
    return [[_speedup(synthetic(r, chain_us_per_row=chain), q, t)
             for r in ratios] for t in toks], list(ratios), list(toks)


def geometry_grid(chain: float, tok: int = 4096):
    """Geometry sensitivity: how the breakeven hierarchy ratio moves with (n_groups, R, k, M).

    (k, M) is enumerated explicitly -- each row's actual q = k/M differs; labels follow
    (k, M) and do not masquerade as one q (the first version labeled the k=4/8 rows as
    q=3, which was wrong).
    """
    rows = []
    for ng in (8, 16, 32):
        for R in (4, 8, 16):
            for k, m in ((4, 1), (4, 2), (6, 2), (6, 3), (8, 2), (8, 4)):
                if m > ng:
                    continue

                def s(r):
                    c = synthetic(r, R=R, chain_us_per_row=chain)
                    g = MoEGeometry(name="grid", n_groups=ng, R=R, k=k, M=m,
                                    seq=tok, mbs=1, gbs=ng * R * tok)
                    return one_hop_call(c, g) / two_hop_call(c, g)

                lo, hi = 1.0, 32.0
                if s(lo) >= 1.0:
                    be = lo
                elif s(hi) < 1.0:
                    be = hi
                else:
                    for _ in range(40):
                        mid = (lo + hi) / 2.0
                        if s(mid) >= 1.0:
                            hi = mid
                        else:
                            lo = mid
                    be = hi
                rows.append((ng, R, k, m, be))
    return rows


def main() -> None:
    from .sweep import CHAIN_SCENARIOS
    print("Monte Carlo uncertainty bands (%d draws, seed %d; perturbation conventions in the module docstring)" %
          (N_DRAWS, SEED))
    print("Geometry: 16 groups x 8, k=6/M=2 (q=3), T=4096; value = one-hop/two-hop (>1 = two-hop faster)")
    ratios = [1.03, 2.0, 3.2, 4.5, 8.0, 15.7]
    for name, chain in CHAIN_SCENARIOS:
        print("\n-- %s --" % name)
        print("%-10s %10s %18s" % ("ratio", "median", "[p5, p95]"))
        for r in ratios:
            p5, med, p95 = mc_band(r, chain)
            print("%-10.2f %10.2f       [%.2f, %.2f]" % (r, med, p5, p95))
    print("\nBreakeven hierarchy ratio (smallest hierarchy ratio with ratio=1, q=3, T=4096):")
    for name, chain in CHAIN_SCENARIOS:
        print("  %-24s %.2f" % (name, breakeven_ratio(chain)))
    print("\nTwo robustness anchors (whether a conclusion holds depends on the band's ends):")
    p5, _, p95 = mc_band(1.03, CHAIN_SCENARIOS[2][1])   # zero overhead + flat
    print("  flat column, most favorable case (zero implementation overhead) p95 = %.2f -> %s" %
          (p95, "≤1, the negative verdict is robust to calibration error" if p95 <= 1.0
           else "!! crosses the line -- tighten the written conclusion"))
    p5b, _, _ = mc_band(8.0, CHAIN_SCENARIOS[0][1])     # PyTorch chain + 8x
    print("  8x column, least favorable case (PyTorch chain) p5 = %.2f -> %s" %
          (p5b, "still >1, the direction conclusion is robust" if p5b > 1.0
           else "<1: the 8x column's win depends on the implementation tier; report the fused tier and above only"))


if __name__ == "__main__":
    main()
