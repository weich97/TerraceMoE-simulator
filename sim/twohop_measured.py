# -*- coding: utf-8 -*-
"""Two-hop measured against one-hop across a supernode boundary. The first win.

## Why this file is the one that mattered

Until 2026-09-08 this repository had never measured two-hop dispatch beating one-hop.
Its seven end-to-end geometries all ran inside a single supernode, where the hierarchy
ratio is 1.03 and there is nothing to buy, and six of the seven lost; the one positive,
+3.6% at the base point, is attributed in docs/03 to a micro-level window rather than to
hierarchy. Every statement that two-hop pays somewhere has been a model output, and
docs/03 says plainly why that is the weak kind of evidence: a negative verdict is never
audited, because nobody builds what was rejected.

Both supernode boundaries are now measured (`sim/hierarchy.py`), and at this geometry
the model made a specific falsifiable prediction with the arrival chain that actually
exists -- no fused kernel, no hypothetical tier. This is that prediction measured.

## What was run

Sixteen idle nodes, eight in each of two supernodes, 128 ranks. The comparison mirrors
`sim/core.py` term for term so the numbers are commensurable with the model:

    one hop     one all-to-all over all 128 ranks, T*k rows
    two hop     Hop A across the boundary carrying T*M rows, half of them local;
                the arrival chain; Hop B inside the supernode carrying T*k rows;
                and the splits readback a variable-length exchange cannot avoid

Groups are supernodes, so n_groups is 2 and R is one supernode's 64 cards. Hop A
therefore has exactly one cross-boundary peer per card, the narrowest case there is --
and the peer sweep in `sim/tiers.py` had already shown that costs nothing.

The arrival chain is **the repository's own** `terrace.ops.k1_arrival_ref`, which its
docstring calls the executable spec of the fused kernel and which the test suite holds
bit-identical to the live chain. Not a per-row estimate: the real gather, the real
stable bucket sort, the real histogram, on the device at the real shapes.

Each phase is timed alone with the ranks aligned and the host waiting, and the two-hop
total is their sum -- which is what `sim/core.py` assumes and what the call-count scan
independently established, since collectives on this machine do not pipeline.

## The result

**Two-hop wins in every configuration measured**, by 1.43x to 2.23x:

    H = 2048  M = 1  q = 6    one hop  7.381 ms    two hop  4.181 ms    G = 1.77
    H = 2048  M = 2  q = 3    one hop  7.324 ms    two hop  5.122 ms    G = 1.43
    H = 4096  M = 1  q = 6    one hop 13.637 ms    two hop  6.105 ms    G = 2.23
    H = 4096  M = 2  q = 3    one hop 13.627 ms    two hop  8.235 ms    G = 1.65

## Where the model was right and where it was wrong

It called the verdict correctly four times out of four and the magnitude to 14%. Term by
term it is right for partly compensating reasons, and the decomposition is the useful
part:

    one hop      accurate to 1-3%
    Hop A        under-priced by 2.2 to 2.5x
    chain        over-priced by 2.06x: 0.0424 us per row measured
                 against the 0.0875 the calibration shipped
    Hop B        under-priced by 13 to 32%

The Hop A error is mine and it is instructive. The `slow` level was given the bandwidth
of a world-128 all-to-all spanning the boundary, but that collective sends only half its
wire bytes across; the other half stays inside a supernode. Hop A sends **only** across.
So the level was fed a mixture and used as a pure tier. Re-parameterising with the pure
cross-boundary tier of 7.9 GB/s takes the median error on G from 14.0% to 9.0% and turns
an over-prediction into a slight under-prediction. Adopting the measured chain cost as
well pushes it back to 15.1%, because that removes an error that was compensating.

So the model's aggregate accuracy here rests partly on two errors cancelling, and this
module records both rather than quietly fixing one.

> **Update (2026-09-08, later the same day).** One of the two has now been fixed, and
> not by choice: the chain was not "over-priced on one run" but wrong by construction.
> The calibration divided a 73728-pair measurement by 24576, and the 0.0424 measured
> here -- taken on the pair count `core.py` actually charges -- is the correct level.
> It is now `calibrate.CHAIN_US_PER_ROW`; see [chain_remeasured](chain_remeasured.py).
>
> That removes the compensating error. The shipped model is now the mixture slow level
> with the measured chain, a fourth combination not among the three below, and by the
> mechanism the third row demonstrates it must score **worse** than 0.140 here. That is
> the right trade: the model is now wrong in one term instead of wrong in two that
> happened to cancel, and the remaining error has a named cause and a known fix. Hop A
> is still fed a mixture, and closing this gap means fixing that.

## What this is not

Not a step-time claim. This is the communication-call level, which is the tier Tier-1
unlocks; Tier-2 fails and forbids any statement about training throughput. A real
implementation also overlaps differently from a sum of separately timed phases.

Measured on the pool110/pool12 boundary, which is the **deeper** of the two at a ratio
of 7.51. The other boundary is 5.10 and the model puts G there at 1.33 for the first row
above -- still a win, unmeasured.

And it needs the routing constraint: M = 1 or 2 with two supernodes. What that costs in
quality is measured at M = 4 with eight groups only, which is the gap
`docs/12-m-quality-experiment.md` exists to state.
"""
from __future__ import annotations

#: (hidden width, M, q, one-hop ms, Hop A ms, chain ms, Hop B ms, splits ms).
#: 128 ranks over two supernodes, T = 4096 tokens per rank, k = 6, medians of 5 repeats
#: of 7 calls, every time the slowest rank's. 2026-09-08, bench/twohop_verdict.py.
MEASURED = [
    (2048, 1, 6, 7.381, 1.562, 1.043, 1.477, 0.099),
    (2048, 2, 3, 7.324, 2.646, 1.026, 1.355, 0.094),
    (4096, 1, 6, 13.637, 2.638, 1.096, 2.270, 0.101),
    (4096, 2, 3, 13.627, 4.813, 1.113, 2.212, 0.096),
]

TOKENS = 4096
K = 6
N_GROUPS = 2
CARDS_PER_GROUP = 64
BOUNDARY = "pool110/pool12"

#: Per-pair cost of the real arrival chain at these shapes. Recorded here as "cheaper
#: than the calibration says by about half" and deliberately not adopted, on the grounds
#: that one run of one shape family should not displace a constant with a sweep behind
#: it. The sweep turned out to be in different units; this reading was right, and since
#: 2026-09-08 it *is* calibrate.CHAIN_US_PER_ROW.
CHAIN_US_PER_ROW_MEASURED = 0.0424

#: Median relative error of the model's G under three parameterisations, from the
#: comparison in the module docstring, all computed under the pre-2026-09-08 chain
#: constant. The middle one is the honest fix for the Hop A error; the third shows that
#: adopting the measured chain on top removes a compensating error and makes the total
#: worse. The chain has since been adopted for a reason unrelated to this comparison, so
#: none of the three is the shipped combination any more -- see the update above.
MODEL_ERROR = {"slow level fed a mixture": 0.140,
               "slow level as a pure tier": 0.090,
               "pure tier and measured chain": 0.151}

PURE_CROSS_BOUNDARY_GBPS = 7.9


def two_hop_ms(row) -> float:
    """Hop A plus the chain plus Hop B plus the splits readback, as core.py composes."""
    return row[4] + row[5] + row[6] + row[7]


def g(row) -> float:
    """One-hop time over two-hop time. Above 1 means two-hop is faster."""
    return row[3] / two_hop_ms(row)


def every_configuration_wins() -> bool:
    return all(g(r) > 1.0 for r in MEASURED)


def chain_share(row) -> float:
    """How much of two-hop the arrival chain is. It is the term the method fights."""
    return row[5] / two_hop_ms(row)


def main() -> None:
    print("One hop against two, measured across the %s supernode boundary." % BOUNDARY)
    print("128 ranks, 8 nodes in each supernode, T = %d tokens per rank, k = %d."
          % (TOKENS, K))
    print("The arrival chain is the repository's own reference chain, run on device.")
    print()
    print("%-6s %-4s %-4s %9s %8s %8s %8s %8s %9s %7s"
          % ("H", "M", "q", "one hop", "hop A", "chain", "hop B", "splits",
             "two hop", "G"))
    for r in MEASURED:
        print("%-6d %-4d %-4d %9.3f %8.3f %8.3f %8.3f %8.3f %9.3f %7.3f %s"
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], two_hop_ms(r), g(r),
                 "WIN" if g(r) > 1 else "loss"))
    print()
    print("Two-hop wins in %d of %d configurations, by %.2fx to %.2fx."
          % (sum(1 for r in MEASURED if g(r) > 1), len(MEASURED),
             min(g(r) for r in MEASURED), max(g(r) for r in MEASURED)))
    print()
    print("The model called all four correctly. Its median error on G is %.0f%% as the"
          % (100 * MODEL_ERROR["slow level fed a mixture"]))
    print("slow level was parameterised, and %.0f%% once that level is given the pure"
          % (100 * MODEL_ERROR["slow level as a pure tier"]))
    print("cross-boundary tier of %.1f GB/s instead of a mixture. The arrival chain"
          % PURE_CROSS_BOUNDARY_GBPS)
    print("measures %.4f us per pair against the %.4f the calibration shipped -- and"
          % (CHAIN_US_PER_ROW_MEASURED, 0.0875))
    print("that measurement is now the shipped constant: the 0.0875 divided a")
    print("73728-pair sweep by 24576. See sim/chain_remeasured.py.")
    print()
    print("Communication-call level only. Tier-2 fails, so nothing here is a claim")
    print("about training throughput, and the seven end-to-end geometries of docs/03")
    print("ran inside one supernode at ratio 1.03 and stand exactly as measured.")


if __name__ == "__main__":
    main()
