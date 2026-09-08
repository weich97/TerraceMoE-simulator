# -*- coding: utf-8 -*-
"""The constant everything hinges on over-charged by 2.8x, and the cause was a denominator.

## What this module said this morning, and why it was wrong

On 2026-09-08 this file reported that re-measuring the arrival chain gave a number three
times below the shipped `calibrate.CHAIN_US_PER_ROW`, that two August readings agreed
with each other and two September readings agreed with each other, and that a factor of
three between them was "a step, not noise". It declined to adopt the newer pair on the
grounds that choosing by date is not choosing by evidence. It then named the two things
that would settle it: run the calibration's own instrument on today's stack, or recover
the software versions August was measured under.

Neither was needed. The instrument's **archived output** was still on the shared
filesystem, and it settles the question against this module's own framing: the two dates
never disagreed. **The factor of three was a unit collision inside the calibration.**

## The collision

`bench/machine/dispatch_oplevel` (not in this repository) is parameterised by an
**input-row** count `R`. It builds `rx` as `[R, H]` and `rslot` as `[R, quota]`, and the
first thing it does -- before any of the five stages it times -- is expand those into
`R * quota` (row, slot) pairs. The sort, the histogram, the `[pairs, H]` gather and both
index gathers all run over pairs. So every number that instrument reports is a cost over
`R * quota` pairs, and `R` alone tells you nothing about the work.

The sweeps behind the calibration ran at `R = 24576` with `quota = 3`: **73728 pairs**.

The reference geometry is a different point. 4096 tokens per rank at k = 6 and M = 2 is
`R = 8192` input rows and **24576 pairs**. `core.py` charges the chain over
`rows_hop_b()`, which is tokens x k -- pairs, correctly.

So the numeral 24576 named the sweep's input rows *and* the reference geometry's pairs,
three times apart, and `CHAIN_US_PER_ROW = 2.15 * 1000 / 24576` divided a 73728-pair
measurement by 24576. The comment beside it -- "the reference geometry sits at 24576
rows, inside that range" -- is the collision written down.

## Three independent confirmations

1. **The instrument's source says so.** `bench_arrival(R, quota, ...)`, `rx = randn(R, H)`,
   `rslot` of shape `[R, quota]`, `_expand_arrival_quota` -> pairs, and the gather is
   labelled `[pairs, H]` in the instrument's own output.

2. **The published sweep is reproduced exactly at 73728 pairs.** `machine.CHAIN_H_SWEEP_MS`
   ships {1024: 2.37, 2048: 2.51, 4096: 2.85, 8192: 3.79}. The archived two-node means at
   `R = 24576`, `quota = 3` are 2.368, 2.510, 2.850 and 3.787. Nothing else in the archive
   matches those four numbers.

3. **Six readings at the reference geometry agree, across two dates and three
   instruments.** All at 24576 pairs and H = 2048 on one idle card: 0.7810, 0.7721 and
   0.7758 ms from the calibration's own instrument on 2026-08-23 and 08-24; 0.7446 and
   0.7295 ms from the 2026-08-26 row sweep's `R = 8192` point on two nodes; 0.7150 ms from
   this repository's re-measurement on 2026-09-08. A 9% spread. **The model was charging
   2.15 ms.**

## Two real effects that are not the unit error

**The histogram changed on 2026-08-23**, and the instrument caught it mid-flight. Runs at
00:41 and 01:01 time `bincount(owner)` at 0.818 and 0.784 ms and the whole chain at 1.19
ms. From 08:11 the output carries a second entry, a fixed-length `scatter_add`, at 0.327
ms, and the whole chain drops to 0.78. That is a genuine 1.5x from a code change, it is
in the live path today (`terrace.ta2a_fwd.fixed_hist`), and the adopted level is measured
after it.

**Load costs about 1.4x.** The readings above are one idle card. Timed in situ during the
two-hop verdict, with all eight cards of the node working and the slowest rank taken, the
same chain at the same geometry is 1.043 ms. That is the level a real dispatch pays and
therefore the level that ships -- the less flattering of the two.

## Why no test here could have caught it

`validate.predict_g` charges the chain in the dispatch arm and subtracts it in the combine
arm, where the back-solved `combine_extra_ms` is rescaled by `rows_hop_b() / 24576` -- the
same row count. The two cancel **identically**, at every holdout geometry, for any value
of the constant. `chain_cancels_in_validation()` below demonstrates it: the Tier-2 gate
returns the same MAE to machine precision whether the chain costs 0.0875, 0.0424 or
nothing at all.

So the repository's only end-to-end validation is structurally blind to the one constant
its headline threshold is most sensitive to. That is how the error survived two
sweeps, eighteen measurements and a documented re-measurement. It is also why correcting
it disturbs no validated prediction.

## What changed

    CHAIN_US_PER_ROW          0.0875 -> 0.0424 us per pair (in situ, 8 cards)
    INDEX_NS_PER_ROW            85.8 -> 28.6 ns per pair
    GATHER_GBPS_MEASURED         490 -> 1469 GB/s
    CHAIN_LEVEL_CALIBRATION_MS  2.15 -> 1.042 ms at the reference geometry
    CHAIN_LINEAR_MIN_ROWS       8192 -> 24576 (it was input rows, now pairs)
    effective breakeven ratio   3.98 -> 2.49

The threshold this repository publishes falls by a third, which moves it *toward* the
method: a machine needs a hierarchy ratio of 2.49 rather than 3.98 before two-hop pays
with the arrival chain that exists today. Both measured supernode boundaries here, 5.10
and 7.51, clear it either way -- and did before, which is why the measured two-hop win
stands unaffected.

The fused-chain figures move the other way. On the corrected denominator a fused kernel
should reach 2.8 to 5.6 ns per pair, so the 0.012 us/row design target this repository
quotes is conservative by two- to fourfold, and its 1.49 threshold is a ceiling on what
fusing would buy rather than an estimate of it.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# The two denominators. Every number in this module is stated in pairs.
# ---------------------------------------------------------------------------

#: The reference geometry: 4096 tokens per rank, k = 6, M = 2.
REFERENCE_PAIRS = 24576        # tokens x k, and what core.rows_hop_b() counts
REFERENCE_INPUT_ROWS = 8192    # tokens x M, what the instrument is parameterised by
REFERENCE_H = 2048

#: The sweeps behind the calibration. Same numeral, different quantity.
SWEEP_INPUT_ROWS = 24576
SWEEP_QUOTA = 3
SWEEP_PAIRS = SWEEP_INPUT_ROWS * SWEEP_QUOTA   # 73728

# ---------------------------------------------------------------------------
# The calibration's own instrument, from its archived output on the shared
# filesystem: experiments/k1-bitcheck/oplevel_*.json in the private repository.
# All at the reference geometry, one card, no collectives, 20 iterations.
# ---------------------------------------------------------------------------

#: (timestamp, histogram implementation, whole-chain ms). The 2026-08-23 change of
#: bincount for a fixed-length scatter_add is visible between the second row and the
#: third: 2.4x on that step, 1.5x on the whole chain.
CALIBRATION_INSTRUMENT = [
    ("2026-08-23 00:41", "bincount", 1.1924),
    ("2026-08-23 01:01", "bincount", 1.1811),
    ("2026-08-23 08:11", "fixed_hist", 0.7810),
    ("2026-08-23 12:21", "fixed_hist", 0.7721),
    ("2026-08-24 07:06", "fixed_hist", 0.7758),
]

#: The histogram step either side of the change, from the same runs: (bincount ms,
#: fixed_hist ms) at the reference geometry.
HISTOGRAM_CHANGE = (0.7939, 0.3270)

#: The 2026-08-26 row sweep, on the denominator the instrument actually uses:
#: (input rows, pairs, node 14 ms, node 15 ms). The reference geometry is the
#: 8192-input-row row, not the 24576 one.
AUGUST_ROW_SWEEP = [
    (1024, 3072, 0.2418, 0.2541),
    (2048, 6144, 0.2812, 0.2808),
    (4096, 12288, 0.4300, 0.4210),
    (8192, 24576, 0.7446, 0.7295),
    (16384, 49152, 1.4368, 1.3963),
    (24576, 73728, 2.5573, 2.4464),
    (32768, 98304, 3.2951, 3.3382),
    (49152, 147456, 4.9724, 4.9216),
    (65536, 196608, 6.5339, 6.4078),
]

#: The 2026-08-26 hidden-width sweep, all at SWEEP_PAIRS: (H, node 14 ms, node 15 ms).
#: Their means are machine.CHAIN_H_SWEEP_MS, which is what pins the denominator.
AUGUST_WIDTH_SWEEP = [
    (1024, 2.3586, 2.3781),
    (2048, 2.5521, 2.4672),
    (4096, 2.8834, 2.8165),
    (8192, 3.8183, 3.7550),
]

# ---------------------------------------------------------------------------
# This repository's 2026-09-08 re-measurement. Kept because it is the reading that
# raised the alarm, and because it is an independent instrument.
# ---------------------------------------------------------------------------

#: Row sweep, H = 2048: (output pairs, ms). One card, op level, median of 7 repeats of
#: 15 calls. bench/chain_sweep.py. Note this sweep is indexed by *pairs* directly, and
#: at quota 6 rather than 3, so its shape is not comparable to AUGUST_ROW_SWEEP point
#: for point -- only its levels at equal pair counts are.
ROW_SWEEP = [
    (1024, 0.3420), (2048, 0.3562), (4096, 0.3889), (8192, 0.4482),
    (12288, 0.4984), (16384, 0.5809), (24576, 0.7150), (32768, 0.8494),
    (49152, 1.1942), (65536, 1.7768),
]

#: Width sweep at 24576 pairs: (hidden width, ms).
WIDTH_SWEEP = [(1024, 0.7168), (2048, 0.7235), (4096, 0.7739), (8192, 1.0025)]

#: The live arrival sequence against the reference, same shapes, same card:
#: (pairs, H, reference ms, live ms). The live path is 5-16% dearer, which is why the
#: reference chain is treated as representative of what a dispatch actually pays.
LIVE_VS_REFERENCE = [
    (8192, 2048, 0.4685, 0.4900), (8192, 8192, 0.4854, 0.5156),
    (24576, 2048, 0.6855, 0.7752), (24576, 8192, 0.9955, 1.0520),
    (65536, 2048, 1.7767, 2.0674), (65536, 8192, 2.7319, 3.0258),
]

#: The whole chain at the reference geometry and H = 2048, every reading there is, in
#: pairs: (label, ms). The first five are one idle card; the last is in situ.
REFERENCE_READINGS = [
    ("2026-08-23 instrument", 0.7810),
    ("2026-08-23 instrument", 0.7721),
    ("2026-08-24 instrument", 0.7758),
    ("2026-08-26 sweep, node 14", 0.7446),
    ("2026-08-26 sweep, node 15", 0.7295),
    ("2026-09-08 re-measurement", 0.7150),
]

#: What the model charged before the correction, at the same geometry.
PRE_CORRECTION_LEVEL_MS = 2.15

#: Per-pair cost at the reference shape, by reading.
US_PER_ROW = {
    "shipped before 2026-09-08": 0.0875,
    "one idle card, August": 0.0303,
    "one idle card, September": 0.0291,
    "in situ, 8 cards (shipped)": 0.0424,
    "fused design target": 0.0120,
}


def us_per_row_here(pairs: int = REFERENCE_PAIRS) -> float:
    """Reference-shape per-pair cost from this repository's row sweep."""
    ms = dict(ROW_SWEEP)[pairs]
    return ms * 1000.0 / pairs


def live_overhead() -> tuple:
    """(min, max) of live/reference across the comparison. Both near 1, which is why
    the reference is treated as representative of what a dispatch actually pays."""
    r = [live / ref for _p, _h, ref, live in LIVE_VS_REFERENCE]
    return min(r), max(r)


def august_width_means() -> dict:
    """{H: two-node mean ms}. These are machine.CHAIN_H_SWEEP_MS, to two decimals,
    which is what proves that sweep was taken over SWEEP_PAIRS."""
    return {H: (a + b) / 2.0 for H, a, b in AUGUST_WIDTH_SWEEP}


def published_sweep_is_at_pairs() -> bool:
    """The shipped hidden-width sweep is the archived two-node mean at SWEEP_PAIRS."""
    from .machine import CHAIN_H_SWEEP_MS
    return all(abs(CHAIN_H_SWEEP_MS[H] - m) < 0.005
               for H, m in august_width_means().items())


def ratio_to_calibration() -> dict:
    """{hidden width: calibration per-pair cost over this repository's, at that width}.

    Before the denominator was corrected this returned 3.31 to 3.78 and was read as the
    machine having changed. Put on one denominator it is 1.10 to 1.26: the September
    reading is 10 to 26 percent cheaper than the August one, which is drift, and the
    trend with H is that the August sweep carries three times the pairs and so sits
    further from the launch floor.
    """
    from .machine import CHAIN_H_SWEEP_MS
    return {H: (CHAIN_H_SWEEP_MS[H] / SWEEP_PAIRS) / (ms / REFERENCE_PAIRS)
            for H, ms in WIDTH_SWEEP}


def reference_spread() -> tuple:
    """(min, max) of the six reference-geometry readings [ms]. They span 9 percent."""
    v = [ms for _lab, ms in REFERENCE_READINGS]
    return min(v), max(v)


def overcharge() -> float:
    """How many times the pre-correction model over-charged the chain, against the
    median of every reading at the geometry it was charging."""
    v = sorted(ms for _lab, ms in REFERENCE_READINGS)
    median = (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2.0
    return PRE_CORRECTION_LEVEL_MS / median


def histogram_speedup() -> float:
    """What the 2026-08-23 bincount -> fixed_hist change bought, on that step alone."""
    old, new = HISTOGRAM_CHANGE
    return old / new


def chain_cancels_in_validation(values=(0.0875, 0.0424, 0.0303, 0.0)) -> dict:
    """The Tier-2 gate's verdict is identical for every chain constant. Proof, not claim.

    Returns {constant: (MAE, tuple of predicted G)}. All entries are equal to machine
    precision, because ``validate.predict_g`` adds the chain in the dispatch arm and
    the back-solved combine constant -- rescaled by the same row count -- removes it.
    """
    import dataclasses

    from .calibrate import flat_supernode
    from .validate import HOLDOUTS, calibrate_combine, predict_g

    base = flat_supernode()
    out = {}
    for v in values:
        c = dataclasses.replace(base, chain_us_per_row=v)
        ce = calibrate_combine(c)
        gs = tuple(predict_g(c, h, ce) for h in HOLDOUTS)
        errs = [abs(predict_g(c, h, ce) - h.g_measured)
                for h in HOLDOUTS if h.role == "holdout"]
        out[v] = (sum(errs) / len(errs), gs)
    return out


def breakevens() -> dict:
    """The effective breakeven each reading implies, at the reference geometry."""
    from .uncertainty import breakeven_ratio
    return {k: breakeven_ratio(v) for k, v in US_PER_ROW.items()}


def main() -> None:
    from .machine import CHAIN_H_SWEEP_MS
    print("The arrival chain constant: the factor of three was a denominator.")
    print()
    print("The instrument expands each input row into quota pairs before the first")
    print("stage it times. The sweeps ran at %d input rows with quota %d, so they"
          % (SWEEP_INPUT_ROWS, SWEEP_QUOTA))
    print("measure the chain over %d pairs. The reference geometry has %d input"
          % (SWEEP_PAIRS, REFERENCE_INPUT_ROWS))
    print("rows and %d pairs. One numeral, two quantities, three times apart."
          % REFERENCE_PAIRS)
    print()
    print("Proof the shipped sweep is at %d pairs -- archived two-node means:" % SWEEP_PAIRS)
    print("%-8s %10s %10s %10s %10s" % ("H", "node 14", "node 15", "mean", "shipped"))
    for H, a, b in AUGUST_WIDTH_SWEEP:
        print("%-8d %10.4f %10.4f %10.4f %10.2f" % (H, a, b, (a + b) / 2, CHAIN_H_SWEEP_MS[H]))
    print()
    print("Every reading at the reference geometry (%d pairs, H = %d), one card:"
          % (REFERENCE_PAIRS, REFERENCE_H))
    for lab, ms in REFERENCE_READINGS:
        print("   %-30s %7.4f ms   %.4f us/pair" % (lab, ms, ms * 1000 / REFERENCE_PAIRS))
    lo, hi = reference_spread()
    print("   spread %.0f%% across two dates and three instruments."
          % (100 * (hi - lo) / lo))
    print("   the model charged %.2f ms here: %.2fx over." % (PRE_CORRECTION_LEVEL_MS,
                                                              overcharge()))
    print()
    print("Two real effects that are not the unit error:")
    print("   the 2026-08-23 bincount -> fixed_hist change bought %.2fx on that step,"
          % histogram_speedup())
    print("   and load costs 1.4x on top, which is the level that ships (1.043 ms).")
    print()
    print("%-30s %9s %10s" % ("reading", "us/pair", "breakeven"))
    for k, v in US_PER_ROW.items():
        print("%-30s %9.4f %10.2f" % (k, v, breakevens()[k]))
    print()
    print("Why nothing caught it: the chain cancels identically out of the Tier-2 gate.")
    r = chain_cancels_in_validation()
    for v, (mae, _gs) in r.items():
        print("   chain = %.4f us/pair -> holdout MAE %.10f" % (v, mae))
    print("   so the gate is blind to it, and correcting it disturbs no validated")
    print("   prediction. The published threshold moves from 3.98 to 2.49.")


if __name__ == "__main__":
    main()
