# -*- coding: utf-8 -*-
"""The constant everything hinges on, re-measured, and it moved by a factor of three.

## Why this matters more than any other constant here

`calibrate.CHAIN_US_PER_ROW` is 0.0875 microseconds per Hop-B row, and that file says of
it: "This is the constant everything hinges on -- it alone moves the breakeven ratio from
1.10 to 3.98." It sets the threshold this repository tells an adopter to measure their
machine against, so a factor of three in it is a factor of three in who the method is
for.

Two readings stood behind it and they agreed: 2.15 ms at 24576 rows from a single
measurement, and a nine-row-count sweep across two nodes on 2026-08-26 giving 0.0987.

## What was measured

The two-hop comparison on 2026-09-08 timed the real chain rather than charging a per-row
estimate, and came out at half the shipped constant. That is too load-bearing to leave
as a by-product, so both of the calibration's own sweeps were reproduced at the
calibration's own convention -- op level, single card, output pairs as the denominator:

    row sweep     1024 to 65536 output pairs at the reference hidden width
    width sweep   hidden width 1024 to 8192 at 24576 pairs

**Every point is about three times cheaper than the calibration records**, and the gap
is not a uniform shift. At 1024 pairs the chain is *slower* than the calibration's floor,
0.342 ms against 0.248. Above that it falls away: at 24576 pairs and H = 2048 it is
0.7235 ms against 2.51, and the ratio grows with hidden width, 3.31 at H = 1024 to 3.78
at 8192. Least squares over the width sweep gives 0.6455 ms of index work plus 0.0423 ms
per 1024 of hidden width, against the calibration's 2.109 and 0.2056 -- the index term
3.3 times smaller, the gather term 4.9 times.

## It is not the reference implementation being cheaper than the live one

That was the obvious objection, since these timings run `terrace.ops.k1_arrival_ref`
rather than the dispatch itself. The reference's docstring names the only difference,
`r_idx` by arithmetic rather than by table lookup, and its sort primitive turns out to
*be* the live chain's `_stable_argsort_small`. Timing the live sequence explicitly --
`_expand_arrival_quota`, `_stable_argsort_small`, `fixed_hist`, and the two gathers, the
exact lines the reference says it replicates -- puts it **5 to 16 percent** above the
reference across three sizes and two widths. Not a factor of three.

So the reference is representative, and the gap is a real change in what this machine
delivers, not an artefact of what was timed.

## What it would change, and why nothing is changed

    chain cost                       us/row    effective breakeven
    shipped calibration              0.0875                   3.98
    2026-08-26 re-measurement        0.0987                   4.34
    live chain today, one card       0.0315                   2.13
    real chain under load, 8 cards   0.0424                   2.49
    fused design target              0.0120                   1.49

The threshold this repository publishes would fall from 3.98 to about 2.5, and the case
for fusing the chain would weaken correspondingly: the fused target buys 2.49 to 1.49
rather than 3.98 to 1.49.

**The constant does not move here.** Two readings from August agree with each other, two
from September agree with each other, and they are a factor of three apart -- far outside
the 20 percent run-to-run drift this machine documents. That is a step, not noise, and
adopting the newer pair would be choosing by date rather than by evidence. What is more,
the shape differs as well as the level, and a constant whose *shape* changed is not the
same quantity being re-read.

What would settle it, in order of cost: re-run the calibration's own instrument
(`bench/machine/dispatch_oplevel`, which is not in this repository) on today's stack, so
the two dates are compared through one benchmark; failing that, record the CANN and
torch versions the August measurements were taken under, since a stack upgrade between
the dates is the leading explanation and is checkable from records rather than from
machines.

Until then this module is the second reading, stated at the same volume as the first.
"""
from __future__ import annotations

#: Row sweep, H = 2048: (output pairs, ms). One card, op level, median of 7 repeats of
#: 15 calls. 2026-09-08, bench/chain_sweep.py.
ROW_SWEEP = [
    (1024, 0.3420), (2048, 0.3562), (4096, 0.3889), (8192, 0.4482),
    (12288, 0.4984), (16384, 0.5809), (24576, 0.7150), (32768, 0.8494),
    (49152, 1.1942), (65536, 1.7768),
]

#: Width sweep at 24576 pairs: (hidden width, ms), against calibration.CHAIN_H_SWEEP_MS.
WIDTH_SWEEP = [(1024, 0.7168), (2048, 0.7235), (4096, 0.7739), (8192, 1.0025)]

#: The live arrival sequence against the reference, same shapes, same card:
#: (pairs, H, reference ms, live ms). The live path is 5-16% dearer, not 3x.
LIVE_VS_REFERENCE = [
    (8192, 2048, 0.4685, 0.4900), (8192, 8192, 0.4854, 0.5156),
    (24576, 2048, 0.6855, 0.7752), (24576, 8192, 0.9955, 1.0520),
    (65536, 2048, 1.7767, 2.0674), (65536, 8192, 2.7319, 3.0258),
]

REFERENCE_PAIRS = 24576
REFERENCE_H = 2048

#: Per-row cost at the reference shape, by reading.
US_PER_ROW = {
    "shipped calibration": 0.0875,
    "2026-08-26 re-measurement": 0.0987,
    "live chain today, one card": 0.0315,
    "real chain under load, 8 cards": 0.0424,
    "fused design target": 0.0120,
}


def us_per_row_here(pairs: int = REFERENCE_PAIRS) -> float:
    """Reference-shape per-row cost from the row sweep."""
    ms = dict(ROW_SWEEP)[pairs]
    return ms * 1000.0 / pairs


def live_overhead() -> tuple:
    """(min, max) of live/reference across the comparison. Both near 1, which is why
    the reference is treated as representative of what a dispatch actually pays."""
    r = [live / ref for _p, _h, ref, live in LIVE_VS_REFERENCE]
    return min(r), max(r)


def ratio_to_calibration() -> dict:
    """{hidden width: how many times cheaper than the calibration's own sweep}."""
    from .machine import CHAIN_H_SWEEP_MS
    return {H: CHAIN_H_SWEEP_MS[H] / ms for H, ms in WIDTH_SWEEP}


def breakevens() -> dict:
    """The effective breakeven each reading implies, at the reference geometry."""
    from .uncertainty import breakeven_ratio
    return {k: breakeven_ratio(v) for k, v in US_PER_ROW.items()}


def main() -> None:
    from .machine import CHAIN_H_SWEEP_MS
    print("The arrival chain, re-measured at the calibration's own shape and")
    print("convention: op level, one card, output pairs as the denominator.")
    print()
    print("%-8s %12s %14s %8s" % ("H", "measured ms", "calibration ms", "cheaper"))
    for H, ms in WIDTH_SWEEP:
        print("%-8d %12.4f %14.2f %7.2fx" % (H, ms, CHAIN_H_SWEEP_MS[H],
                                             CHAIN_H_SWEEP_MS[H] / ms))
    print()
    lo, hi = live_overhead()
    print("The live sequence costs %.0f%% to %.0f%% more than the reference, so the"
          % (100 * (lo - 1), 100 * (hi - 1)))
    print("gap above is the machine, not the implementation that was timed.")
    print()
    print("%-34s %9s %10s" % ("reading", "us/row", "breakeven"))
    for k, v in US_PER_ROW.items():
        print("%-34s %9.4f %10.2f" % (k, v, breakevens()[k]))
    print()
    print("Nothing is adopted. Two August readings agree, two September readings")
    print("agree, and they are three times apart -- a step, not drift, and the shape")
    print("moved as well as the level. Taking the newer pair would be choosing by")
    print("date. The module docstring names what would settle it.")


if __name__ == "__main__":
    main()
