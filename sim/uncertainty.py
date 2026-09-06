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
    ("beta_slow", 0.90, 1.10, "three runs +-9% (measured, rounded)"),
    ("beta_fast", 0.995, 1.005, "run-to-run spread <0.3% (measured, physics-endorsed tier)"),
    ("splits", 0.95, 1.05, "measured range 0.042-0.046"),
    ("chain", 0.90, 1.10, "assumption: tensor chain drifts less than communication"),
    ("x_half", 30.0 / 54.0, 87.0 / 54.0,
     "half-performance size: bootstrap 90% interval [30, 87] KiB around 54 (measured)"),
]


def _perturbed(ratio: float, chain: float, f: dict):
    """Build the perturbed synthetic cluster for one set of factors."""
    from .calibrate import X_HALF_FLAT
    c = synthetic(ratio, chain_us_per_row=chain * f["chain"],
                  x_half=X_HALF_FLAT * f["x_half"])
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


def breakeven_vs_hidden_width(widths=None, k: int = 6, M: int = 2,
                              tok: int = 4096, n_groups: int = 16, R: int = 8):
    """Effective breakeven hierarchy ratio at each measured hidden width.

    The threshold is not flat in H, and H has to move in both places at once: it
    widens the payload every collective carries and it widens the gather inside the
    arrival chain, and the two do not scale alike. Over the fourfold range from 2048
    to 8192 the payload rises by four while the measured chain rises by about 1.5, so
    the chain's share of two-hop falls and the threshold falls with it.

    Quoting the reference threshold at one hidden width and applying it at another is
    the error this exists to prevent, and the direction matters: every contemporary
    MoE model sits above the reference width, where a threshold stated at 2048
    over-prices the arrival chain -- a cost only two-hop pays. That bias runs against
    the method this repository proposes, not in its favour.

    Returns [(H, breakeven), ...] at the widths the chain sweep measured.
    """
    from .machine import CHAIN_H_SWEEP_MS, chain_us_per_row_at
    ws = sorted(CHAIN_H_SWEEP_MS) if widths is None else list(widths)
    out = []
    for H in ws:
        chain = chain_us_per_row_at(H)

        def s(r, H=H, chain=chain):
            c = synthetic(r, chain_us_per_row=chain)
            g = MoEGeometry(name="H%d" % H, n_groups=n_groups, R=R, k=k, M=M, H=H,
                            seq=tok, mbs=1, gbs=n_groups * R * tok)
            return one_hop_call(c, g) / two_hop_call(c, g)

        lo, hi = 1.0, 32.0
        if s(lo) >= 1.0:
            out.append((H, lo))
            continue
        if s(hi) < 1.0:
            out.append((H, hi))
            continue
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if s(mid) >= 1.0:
                hi = mid
            else:
                lo = mid
        out.append((H, hi))
    return out


# ---------------------------------------------------------------------------
# The scale axis, as one named construction
# ---------------------------------------------------------------------------

#: The four defensible treatments of the two alpha entries no corpus constrains.
#: alpha is direct-measured at worlds 8, 16 and 128. Nothing in the corpus measures
#: 256 or 512 except the one dataset sitting ~5x below every other in absolute
#: bandwidth and fitting worst, so past 128 ranks the answer is a band over these
#: four treatments rather than a line.
ALPHA_TREATMENTS = {
    "same-corpus refit": {256: 0.425, 512: 2.888},
    "borrowed points (previously shipped)": {256: 0.735, 512: 1.859},
    "no growth past 128": {256: 0.378, 512: 0.378},
    "linear in peers past 128": {256: 0.378 + 0.0107 * 128,
                                 512: 0.378 + 0.0107 * 384},
}

#: Hierarchy ratio held fixed while the cluster grows, and the worlds whose alpha
#: was measured directly. Below 128 every treatment agrees to the digit, which is
#: what makes those three points quotable and the ones past them not.
SCALE_HIERARCHY_RATIO = 3.2
MEASURED_WORLDS = (32, 64, 128)
EXTRAP_WORLDS = (128, 256, 512)


def scale_ratio(w: int, alpha_pts=None, ratio: float = SCALE_HIERARCHY_RATIO,
                chain: float = None) -> float:
    """one-hop/two-hop at world ``w`` on the fused tier, hierarchy ratio held fixed.

    This is the single construction behind the scale figure (F10), the scale claims in
    docs/05 and the honesty test in tests/test_sim.py. It used to be copied into each
    of them separately, and the prose copy silently went stale through two
    recalibrations while the generated figure stayed correct; see the 2026-09-05
    correction note in docs/05.
    """
    from .calibrate import ALPHA_PTS
    if chain is None:
        from .sweep import CHAIN_SCENARIOS
        chain = CHAIN_SCENARIOS[1][1]          # hypothetical fused target
    c = synthetic(ratio, chain_us_per_row=chain)
    pts = sorted(dict(ALPHA_PTS).items()) if alpha_pts is None else alpha_pts
    for lvl in (c.fast, c.slow, c.flat):
        lvl.alpha_pts = pts
    g = MoEGeometry(name="scale", n_groups=w // 8, R=8, k=6, M=2,
                    seq=4096, mbs=1, gbs=w * 4096)
    return one_hop_call(c, g) / two_hop_call(c, g)


def scale_treatment_curves(worlds=EXTRAP_WORLDS) -> dict:
    """{treatment label: [ratio at each world]}, one curve per alpha treatment."""
    from .calibrate import ALPHA_PTS
    base = dict(ALPHA_PTS)
    return {lab: [scale_ratio(w, sorted({**base, **ov}.items())) for w in worlds]
            for lab, ov in ALPHA_TREATMENTS.items()}


def scale_band(w: int = 512) -> tuple:
    """(lo, hi) of the ratio at world ``w`` across the four alpha treatments.

    The spread is the whole reason this repository claims nothing past 128 ranks: it
    is produced by the choice between four treatments of one unmeasured constant, not
    by anything measured.
    """
    vals = [c[list(EXTRAP_WORLDS).index(w)] for c in scale_treatment_curves().values()]
    return min(vals), max(vals)


def geometry_breakeven_spread(chain: float) -> float:
    """Largest breakeven displacement (k, M) produces at a fixed (group count, R).

    The geometry axis of the three-axis ranking in docs/05. Taken at fixed world so it
    measures the routing shape alone, with the scale axis held out of it.
    """
    rows = geometry_grid(chain)
    worst = 0.0
    for ng in (8, 16, 32):
        for R in (4, 8, 16):
            sub = [be for (a, b, _k, _m, be) in rows if a == ng and b == R]
            if sub:
                worst = max(worst, max(sub) - min(sub))
    return worst


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
          (p95, "<=1, the negative verdict is robust to calibration error" if p95 <= 1.0
           else "!! crosses the line -- tighten the written conclusion"))
    p5b, _, _ = mc_band(8.0, CHAIN_SCENARIOS[0][1])     # PyTorch chain + 8x
    print("  8x column, least favorable case (PyTorch chain) p5 = %.2f -> %s" %
          (p5b, "still >1, the direction conclusion is robust" if p5b > 1.0
           else "<1: the 8x column's win depends on the implementation tier; report the fused tier and above only"))


if __name__ == "__main__":
    main()
