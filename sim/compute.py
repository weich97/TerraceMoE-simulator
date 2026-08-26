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
  we now have  measurements of **non-square** GEMMs at expert-FFN shapes, which is
              what this module used to say it lacked (see below)

Two consequences. The first still stands, the second has been settled by
measurement:

1. Compute time and communication time **cannot be added**. They overlap, by an
   amount nothing on hand resolves; that is the same gap that keeps Tier-2 locked
   (docs/07). `comm_share_upper_bound` is therefore a bound, not a prediction.
2. Compute time used to be a range, not a value, because the roofline is square
   and an expert matmul is (rows x hidden x d_expert). Three defensible ways to
   index a square curve by a non-square shape -- smallest dimension, geometric
   mean, row count -- diverged by up to 2.25x, and this module returned all three
   and refused to choose. **Eight expert-FFN shapes were then measured directly**,
   on two nodes, and they choose: see NONSQUARE_TFLOPS.

The FLOP count is exact. The efficiency is now a value with a measured error bar
rather than an assumption-dominated bracket.

## The roofline

Achieved bf16 TFLOPS for a square GEMM, medians over 256 measurements per size:

    1024 -> 141.6    2048 -> 271.1    4096 -> 327.4
    8192 -> 296.7   12288 -> 291.4   16384 -> 319.2

Not monotone: efficiency peaks at 4096 and dips through 8192-12288. That is
measured, reproduced across both campaigns, and kept rather than smoothed -- a
fitted monotone curve would erase a real feature.

## What the square curve does *not* say, and we used to think it did

An earlier version of this module read the 141.6 at size 1024 as a statement about
narrow experts: 43% of peak, so narrow experts do less work and do it less
efficiently. **The non-square measurements refute that reading.** A GEMM whose
smallest dimension is 1536 runs at 310 to 328 TFLOPS, not at 206. The penalty at
square-1024 is a *total-work* effect, not a *shape* effect: a square 1024 matmul is
2.1 GFLOP and never fills the machine, while an expert FFN of 1536 x 2048 x 5504 is
34.6 GFLOP and does. Real expert matmuls are large enough to escape the dip even
when one of their dimensions is small.

The correct statement is narrower and still useful: efficiency falls when the
*total* work per call is small, which happens at high expert counts and small
micro-batches, and not merely because d_expert is small.
"""
from __future__ import annotations

import bisect

# (square GEMM size, achieved bf16 TFLOPS) -- medians of 256 measurements each
GEMM_TFLOPS = [(1024, 141.6), (2048, 271.1), (4096, 327.4),
               (8192, 296.7), (12288, 291.4), (16384, 319.2)]
PEAK_TFLOPS = max(t for _, t in GEMM_TFLOPS)
INDEX_RULES = ("min_dim", "geometric_mean", "rows")

# Measured expert-FFN GEMMs, bf16, (rows, hidden, d_expert) -> achieved TFLOPS.
# Mean of one run on each of two nodes; the two agree to 0.0-3.0% on every shape.
# Same session also re-measured the square references at 328.7 (4096) and 296.6
# (8192), against the 327.4 and 296.7 in the table above, so the curve itself is
# reproduced and these shapes can be read against it directly.
NONSQUARE_TFLOPS = {
    (1536, 2048, 5504): 309.7,
    (1536, 4096, 11008): 326.7,
    (1536, 7168, 2048): 328.3,
    (3072, 2048, 5504): 324.8,
    (3072, 4096, 11008): 335.9,
    (3072, 7168, 2048): 338.0,
    (6144, 2048, 5504): 331.4,
    (6144, 4096, 11008): 330.7,
}

# Scored against those eight shapes. Every rule under-predicts, because the square
# curve is the wrong prior for a large non-square GEMM, but they are not equally
# wrong. The geometric mean is the rule this module uses; the other two are kept so
# the comparison stays auditable and so efficiency_bracket still means something.
#
#   min_dim          median |err| 19.0%   bias -19.0%   worst 37.1%
#   rows             median |err| 11.2%   bias -11.2%   worst 37.1%
#   geometric_mean   median |err|  6.4%   bias  -6.4%   worst 10.9%
#
# The residual bias is left uncorrected. Eight points on one machine is not enough
# to fit a correction factor onto, and the direction is the safe one: predicting
# compute a little slow understates communication's share, which is the quantity
# comm_share_upper_bound is supposed to bound from above.
INDEX_RULE = "geometric_mean"
INDEX_RULE_ERROR = {"median_abs": 0.064, "bias": -0.064, "worst": 0.109}


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
    tflops = br["per_rule"][INDEX_RULE]
    ms = flops / (tflops * 1e12) * 1e3
    err = INDEX_RULE_ERROR["worst"]
    return {"flops": flops,
            "rows_per_expert": rows_per_expert,
            "experts_local": experts_local,
            # The value: the geometric-mean rule, which the measured non-square
            # shapes selected, with its measured worst-case error as the band.
            "tflops": tflops,
            "ms": ms,
            "ms_lo": ms * (1.0 - err),
            "ms_hi": ms * (1.0 + err),
            "rule": INDEX_RULE,
            "rule_error": dict(INDEX_RULE_ERROR),
            # Kept for auditing: what the two rejected rules would have said, and
            # how far apart the three are on this shape.
            "ms_fast": flops / (br["hi"] * 1e12) * 1e3,
            "ms_slow": flops / (br["lo"] * 1e12) * 1e3,
            "tflops_bracket": (br["lo"], br["hi"]),
            "assumption_spread": br["spread"],
            "efficiency_vs_peak": tflops / PEAK_TFLOPS}


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

    print("\nMeasured expert-FFN shapes, and what each index rule would have said:")
    print("  %-22s %9s %8s %8s %8s" %
          ("rows x hidden x d_exp", "measured", "geo-mean", "min-dim", "rows"))
    for shape, meas in sorted(NONSQUARE_TFLOPS.items()):
        per = efficiency_bracket(*shape)["per_rule"]
        print("  %-22s %9.1f %8.0f %8.0f %8.0f"
              % ("%d x %d x %d" % shape, meas, per["geometric_mean"],
                 per["min_dim"], per["rows"]))
    print("  geometric mean is the rule in use: %.1f%% median error, worst %.1f%%"
          % (100 * INDEX_RULE_ERROR["median_abs"], 100 * INDEX_RULE_ERROR["worst"]))

    print("\nExpert-FFN forward, one layer, one rank: T=4096, k=6, E=128, EP=128,")
    print("hidden=2048, varying expert width:")
    print("  %-9s %-13s %-10s %-18s" %
          ("d_expert", "rows/expert", "TFLOPS", "ms (measured band)"))
    for de in (2048, 1024, 512, 256):
        r = expert_ffn(4096, 6, 128, 128, 2048, de)
        print("  %-9d %-13d %-10.1f %.3f - %.3f"
              % (de, r["rows_per_expert"], r["tflops"], r["ms_lo"], r["ms_hi"]))
    print("\n  The band is the index rule's measured worst-case error, not a spread")
    print("  between assumptions: eight non-square shapes settled which rule to use.")
    print("\nCompute is never added to communication here: the two overlap by an")
    print("amount no measurement on hand resolves (docs/07).")


if __name__ == "__main__":
    main()
