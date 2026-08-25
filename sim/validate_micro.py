# -*- coding: utf-8 -*-
"""Tier-1 validation gate: the communication micro level -- model vs same-machine direct measurements of both strategies.

## Why the gate targets pooled multi-run medians, not single runs

The same benchmark run back-to-back on the same machine drifts ~20% run to run (example:
two-hop @2048 tok, three runs at 0.648 / 0.772 / 0.711 ms). A single-run target has
resolution below the machine drift, so the gate's red/green becomes a dice roll -- we got
burned by this (the first version happened to use one "convenient" run and passed; swap
in another run and it fails). **The instrument's resolution must beat the effect being
measured** -- that holds for validation gates too.

## Preregistered gate (thresholds frozen before the target values)

  Relative error: median ≤ 20%, worst ≤ 35%;
  crossover (where one-hop/two-hop flips from <1 to >1 and back) falls inside the
  measured window [2048, 16384].

Tier-1 pass = **communication-level** extrapolation allowed. Step-level extrapolation is
guarded separately by Tier-2 (sim/validate.py).

## Target values (pooled medians over 4 runs; benchmark is the pure-communication form: fixed-length equal splits, no counts exchange, no arrival chain)

External users on a different machine: re-measure with the same bench convention, replace
this table, re-run the gate.
"""
from __future__ import annotations

import dataclasses

from .core import MoEGeometry, one_hop_call, two_hop_call

# (token/rank, one-hop ms, two-hop ms, number of runs pooled)
# Convention: 16 groups x 8 cards, k=6/M=2 row structure; the tiers below 256 tokens
# have run-to-run spread 1.8-2.4 and are not targets
MICRO_TARGETS = [
    (256, 0.394, 0.340, 4),
    (1024, 0.639, 0.510, 2),
    (2048, 0.856, 0.670, 4),
    (4096, 1.366, 1.217, 4),
    (8192, 2.119, 2.360, 2),
]


def validate_micro(cluster, verbose: bool = True):
    micro = dataclasses.replace(cluster, splits_sync_ms=0.0, chain_us_per_row=0.0)
    rows, rel = [], []
    for tok, mv, mt, n_runs in MICRO_TARGETS:
        g = MoEGeometry(name="micro", n_groups=16, R=8, k=6, M=2,
                        seq=tok, mbs=1, gbs=16 * 8 * tok)
        pv, pt = one_hop_call(micro, g), two_hop_call(micro, g)
        ev, et = (pv - mv) / mv, (pt - mt) / mt
        rel += [abs(ev), abs(et)]
        rows.append((tok, mv, pv, ev, mt, pt, et, n_runs))
    rel_sorted = sorted(rel)
    med = rel_sorted[len(rel_sorted) // 2]
    worst = rel_sorted[-1]

    cross_lo = cross_hi = None
    prev = None
    for tok in (1024, 2048, 4096, 8192, 16384):
        g = MoEGeometry(name="x", n_groups=16, R=8, k=6, M=2,
                        seq=tok, mbs=1, gbs=16 * 8 * tok)
        ratio = one_hop_call(micro, g) / two_hop_call(micro, g)
        if prev is not None and (prev[1] - 1.0) * (ratio - 1.0) < 0:
            cross_lo, cross_hi = prev[0], tok
        prev = (tok, ratio)
    cross_ok = cross_lo is not None and cross_lo >= 2048 and cross_hi <= 16384

    ok = (med <= 0.20) and (worst <= 0.35) and cross_ok
    if verbose:
        print("Tier-1 (communication micro level) vs pooled multi-run medians")
        print("%6s %4s  %8s %8s %7s   %8s %8s %7s" %
              ("tok", "runs", "1hop ms", "pred", "err", "2hop ms", "pred", "err"))
        for tok, mv, pv, ev, mt, pt, et, n in rows:
            print("%6d %4d  %8.3f %8.3f %+6.1f%%   %8.3f %8.3f %+6.1f%%" %
                  (tok, n, mv, pv, ev * 100, mt, pt, et * 100))
        print("relative error: median %.1f%% (gate 20), worst %.1f%% (gate 35); crossover %s (gate [2048,16384])"
              % (med * 100, worst * 100,
                 "%s-%s" % (cross_lo, cross_hi) if cross_lo else "no crossing"))
        print("**Tier-1 %s**" % ("PASS -- communication-level extrapolation allowed" if ok else "FAIL"))
    return ok, {"median": med, "worst": worst,
                "cross": (cross_lo, cross_hi), "rows": rows}


if __name__ == "__main__":
    from .calibrate import flat_supernode
    validate_micro(flat_supernode())
