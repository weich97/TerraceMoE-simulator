# -*- coding: utf-8 -*-
"""Expert-FFN compute, from a measured GEMM roofline -- reported as a bracket.

## What this module is, and what it refuses to be

`sim/` models communication, guarded by two gates. This adds the other half of an
MoE step, the expert matmuls, and takes it exactly as far as the data supports --
which turns out to be further for wide experts than for narrow ones, and in
neither case as far as a step time.

  we have     a measured bf16 GEMM roofline (achieved TFLOPS vs size, 256 samples
              per size, two independent campaigns) and measured step times
  we lack     any operator-level timeline: no run ever recorded dispatch, combine
              and compute as separate spans
  we also lack any measurement of a **non-square** GEMM, which is the shape every
              expert matmul actually has

Two consequences, both load-bearing:

1. Compute time and communication time **cannot be added**. They overlap, by an
   amount nothing on hand resolves; that is the same gap that keeps Tier-2 locked
   (docs/07). `comm_share_upper_bound` is therefore a bound, not a prediction.
2. Compute time itself is a **range, not a value**, whenever experts are narrow.
   The roofline is square; an expert matmul is (rows x hidden x d_expert) with
   rows usually far larger than d_expert. Three defensible ways to index a square
   curve by a non-square shape -- smallest dimension, geometric mean, row count --
   agree to 1.19x when d_expert = hidden, and diverge to **2.25x** by the time
   d_expert = hidden/2. This module returns all three and calls the spread what it
   is: the point at which the data stops deciding.

The FLOP count is exact. The efficiency is bracketed. Anyone who wants a single
compute number from this hardware has to measure tall-skinny GEMMs; we did not.

## The roofline

Achieved bf16 TFLOPS for a square GEMM, medians over 256 measurements per size:

    1024 -> 141.6    2048 -> 271.1    4096 -> 327.4
    8192 -> 296.7   12288 -> 291.4   16384 -> 319.2

Not monotone: efficiency peaks at 4096 and dips through 8192-12288. That is
measured, reproduced across both campaigns, and kept rather than smoothed -- a
fitted monotone curve would erase a real feature. The headline it carries for MoE:
at 1024 the machine delivers **43% of its own peak**, so narrow experts do not
merely do less work, they do it less efficiently.
"""
from __future__ import annotations

import bisect

# (square GEMM size, achieved bf16 TFLOPS) -- medians of 256 measurements each
GEMM_TFLOPS = [(1024, 141.6), (2048, 271.1), (4096, 327.4),
               (8192, 296.7), (12288, 291.4), (16384, 319.2)]
PEAK_TFLOPS = max(t for _, t in GEMM_TFLOPS)
INDEX_RULES = ("min_dim", "geometric_mean", "rows")


def achieved_tflops(dim: float) -> float:
    """Achieved TFLOPS at a given problem size; clamped outside the measured range.

    Clamped rather than extrapolated: below 1024 the curve falls steeply and
    guessing how far would invent the very effect this module exists to report.
    Anything under 1024 reads as 141.6, which understates the penalty -- the
    conservative direction for a lower bound on efficiency.
    """
    xs = [x for x, _ in GEMM_TFLOPS]
    ys = [y for _, y in GEMM_TFLOPS]
    if dim <= xs[0]:
        return ys[0]
    if dim >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, dim)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (dim - x0) / (x1 - x0)


def index_dims(rows: int, hidden: int, d_expert: int) -> dict:
    """The three defensible ways to index a square roofline by a non-square shape."""
    return {"min_dim": float(min(rows, hidden, d_expert)),
            "geometric_mean": (rows * hidden * d_expert) ** (1.0 / 3.0),
            "rows": float(rows)}


def efficiency_bracket(rows: int, hidden: int, d_expert: int) -> dict:
    """Achieved TFLOPS under each rule, plus the spread that says how much it matters.

    A spread near 1 means the shape is close enough to square that the data
    decides. A large spread means the answer is assumption-dominated and the
    compute time must be quoted as a range.
    """
    per_rule = {k: achieved_tflops(v)
                for k, v in index_dims(rows, hidden, d_expert).items()}
    lo, hi = min(per_rule.values()), max(per_rule.values())
    return {"per_rule": per_rule, "lo": lo, "hi": hi, "spread": hi / lo}


def expert_ffn(tokens_per_rank: int, k: int, n_experts: int, ep: int,
               hidden: int, d_expert: int, n_mats: int = 2,
               backward: bool = False) -> dict:
    """Expert-FFN cost for one MoE layer on one rank: exact FLOPs, bracketed time.

    tokens_per_rank x k rows arrive at a rank and spread over its n_experts/ep
    local experts, so each expert's matmul gets shorter as the expert count grows.

    n_mats: matrices per expert (2 for up/down, 3 for a gated FFN).
    backward: charged at 2x forward FLOPs, the standard accounting; not measured.
    """
    experts_local = max(n_experts // ep, 1)
    rows_per_expert = max(tokens_per_rank * k // experts_local, 1)
    flops = (2.0 * rows_per_expert * hidden * d_expert
             * n_mats * experts_local * (2.0 if backward else 1.0))
    br = efficiency_bracket(rows_per_expert, hidden, d_expert)
    return {"flops": flops,
            "rows_per_expert": rows_per_expert,
            "experts_local": experts_local,
            "ms_fast": flops / (br["hi"] * 1e12) * 1e3,
            "ms_slow": flops / (br["lo"] * 1e12) * 1e3,
            "tflops_bracket": (br["lo"], br["hi"]),
            "assumption_spread": br["spread"],
            "efficiency_vs_peak": (br["lo"] / PEAK_TFLOPS, br["hi"] / PEAK_TFLOPS)}


def comm_share_upper_bound(comm_ms: float, compute_ms: float) -> float:
    """Communication's share of a step **if nothing overlaps** -- an upper bound.

    Real steps overlap communication with compute, so the exposed share is lower
    by an amount no data on hand can pin down (docs/07). Use it to rank geometries
    and to spot regimes where communication cannot possibly dominate; never quote
    it as a predicted share.
    """
    total = comm_ms + compute_ms
    return comm_ms / total if total > 0 else float("nan")


def main() -> None:
    print("Measured bf16 GEMM roofline (256 samples per size):")
    for n, t in GEMM_TFLOPS:
        print("  %6d  %6.1f TFLOPS  %3.0f%% of peak" % (n, t, 100 * t / PEAK_TFLOPS))

    print("\nExpert-FFN forward, one layer, one rank: T=4096, k=6, E=128, EP=128,")
    print("hidden=2048, varying expert width -- where the data stops deciding:")
    print("  %-9s %-13s %-19s %-9s" %
          ("d_expert", "rows/expert", "TFLOPS bracket", "spread"))
    for de in (2048, 1024, 512, 256):
        r = expert_ffn(4096, 6, 128, 128, 2048, de)
        print("  %-9d %-13d %6.1f - %-10.1f %-9.2fx"
              % (de, r["rows_per_expert"], r["tflops_bracket"][0],
                 r["tflops_bracket"][1], r["assumption_spread"]))
    print("\n  Wide experts (d_expert = hidden): the three index rules agree to 1.19x,")
    print("  so compute time is usable. Narrow experts: 2.25x, and the number is")
    print("  assumption-dominated -- report the range or measure tall-skinny GEMMs.")
    print("\nCompute is never added to communication here: the two overlap by an")
    print("amount no measurement on hand resolves (docs/07).")


if __name__ == "__main__":
    main()
