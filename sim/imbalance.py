# -*- coding: utf-8 -*-
"""Routing skew: an all-to-all finishes when the busiest peer finishes.

The cost model in `core.py` spreads payload evenly over peers. Real MoE routing does
not: expert load has a measured coefficient of variation, and a variable-length
all-to-all completes on its slowest link, not its average one. This module prices
that gap.

## Why it is not a small correction, and why it is asymmetric

Take a rank's incoming load as roughly normal with coefficient of variation
$CV_{rank}$. The collective waits for the maximum over its peers, and the expected
maximum of $n$ standard normals grows like $\\sqrt{2 \\ln n}$, so

    inflation(n) = 1 + CV_rank * sqrt(2 ln n)

The number of peers differs by strategy, and that is the whole point:

    one-hop   n = EP        (128 peers, inflation over the widest sample)
    Hop A     n = N_g       (16 peers)
    Hop B     n = R         (8 peers)

Fewer peers means a smaller expected maximum, so **skew penalises one-hop more than
two-hop**. Hop A is doubly favoured: its payload is one row per (token, group), an
aggregate over the $q$ experts a token picks inside the group, and aggregation
divides the coefficient of variation by $\\sqrt{q}$ before the maximum is taken.

This is a real term the cost model was missing, and it points the opposite way from
the arrival chain: implementation overhead hurts two-hop, routing skew helps it.

## The measured input

Expert-level load CV on the ablation testbed: 16 samples, median **0.136**, range
0.118 to 0.151. That is steady-state, with the usual load-balancing loss active.

## Where this must NOT be applied

The Tier-1 targets come from a microbenchmark that divides its buffer **equally**
across peers. It is balanced by construction, so applying an inflation factor there
would be modelling an effect the measurement does not contain, and the gate would
correctly reject it. Skew belongs to end-to-end prediction, which is Tier-2
territory and currently locked. This module therefore ships as an analysis tool with
its own entry point, and `core.py` does not call it.
"""
from __future__ import annotations

import math

# Expert-level load coefficient of variation, steady state, 16 measured samples.
CV_EXPERT_MEDIAN = 0.136
CV_EXPERT_RANGE = (0.118, 0.151)


def expected_max_z(n_peers: int) -> float:
    """Expected maximum of n standard normals, the usual sqrt(2 ln n) estimate.

    Crude for small n, which is the regime Hop B lives in, so the two-hop side of
    every comparison below is if anything over-penalised. That is the conservative
    direction for a module whose headline is that skew favours two-hop.
    """
    n = max(int(n_peers), 2)
    return math.sqrt(2.0 * math.log(n))


def inflation(n_peers: int, cv_expert: float = CV_EXPERT_MEDIAN,
              aggregate_over: int = 1) -> float:
    """Factor by which the busiest peer exceeds the mean.

    aggregate_over: how many independent expert loads are summed before the maximum
    is taken. Hop A carries one row per (token, group), an aggregate over the q
    experts chosen inside that group, so its effective CV is smaller by sqrt(q).
    """
    cv = cv_expert / math.sqrt(max(aggregate_over, 1))
    return 1.0 + cv * expected_max_z(n_peers)


def strategy_inflation(ep: int, n_groups: int, R: int, q: int,
                       cv_expert: float = CV_EXPERT_MEDIAN) -> dict:
    """Inflation factors for the three collectives the two strategies use."""
    return {"one_hop": inflation(ep, cv_expert),
            "hop_a": inflation(n_groups, cv_expert, aggregate_over=q),
            "hop_b": inflation(R, cv_expert)}


def adjusted_ratio(cluster, geom, cv_expert: float = CV_EXPERT_MEDIAN) -> dict:
    """One-hop over two-hop, with and without skew, so the effect is visible.

    The byte terms scale by their inflation factors; alpha, the splits sync and the
    arrival chain do not, since none of them depends on how much payload arrives.
    """
    from .core import _a2a_ms

    f = strategy_inflation(geom.ep, geom.n_groups, geom.R, geom.q, cv_expert)

    def one_hop(scale):
        return _a2a_ms(cluster.flat, geom.ep,
                       geom.rows_one_hop() * scale, geom.row_bytes(),
                       1.0 / geom.ep)

    def two_hop(scale_a, scale_b):
        a = _a2a_ms(cluster.slow, geom.n_groups, geom.rows_hop_a() * scale_a,
                    geom.row_bytes(),
                    geom.M / geom.n_groups if geom.n_groups > 1 else 1.0)
        b = _a2a_ms(cluster.fast, geom.R, geom.rows_hop_b() * scale_b,
                    geom.row_bytes(), 1.0 / geom.R)
        chain = cluster.chain_us_per_row * geom.rows_hop_b() / 1000.0
        return a + b + cluster.splits_sync_ms + chain

    balanced = one_hop(1.0) / two_hop(1.0, 1.0)
    skewed = (one_hop(f["one_hop"])
              / two_hop(f["hop_a"], f["hop_b"]))
    return {"factors": f, "ratio_balanced": balanced, "ratio_skewed": skewed,
            "shift": skewed - balanced}


def main() -> None:
    from .calibrate import synthetic
    from .core import MoEGeometry
    from .sweep import CHAIN_SCENARIOS

    g = MoEGeometry(name="skew", n_groups=16, R=8, k=6, M=2, seq=4096, mbs=1,
                    gbs=16 * 8 * 4096)
    f = strategy_inflation(g.ep, g.n_groups, g.R, g.q)
    print("Measured expert load CV %.3f (range %.3f-%.3f, 16 samples)"
          % (CV_EXPERT_MEDIAN, *CV_EXPERT_RANGE))
    print("Busiest-peer inflation, by collective:")
    print("  one-hop  %3d peers            %.3f" % (g.ep, f["one_hop"]))
    print("  Hop A    %3d peers, agg q=%d   %.3f" % (g.n_groups, g.q, f["hop_a"]))
    print("  Hop B    %3d peers            %.3f" % (g.R, f["hop_b"]))
    print("\nEffect on the one-hop / two-hop ratio, fused arrival chain:")
    print("  %-8s %11s %11s %9s" % ("ratio", "balanced", "with skew", "shift"))
    for r in (1.03, 2.0, 3.2, 8.0):
        c = synthetic(r, chain_us_per_row=CHAIN_SCENARIOS[1][1])
        out = adjusted_ratio(c, g)
        print("  %-8.2f %11.3f %11.3f %+9.3f"
              % (r, out["ratio_balanced"], out["ratio_skewed"], out["shift"]))
    print("\nSkew favours two-hop, because the maximum is taken over fewer peers")
    print("(16 and 8 against 128) and Hop A additionally aggregates q experts per")
    print("message. This is the one modelled effect that points the opposite way")
    print("from the arrival chain, and core.py deliberately does not apply it: the")
    print("Tier-1 microbenchmark divides its buffer equally, so the calibration")
    print("contains no skew to reproduce.")


if __name__ == "__main__":
    main()
