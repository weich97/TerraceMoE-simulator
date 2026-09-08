# -*- coding: utf-8 -*-
"""The first measured hierarchy ratio above 1.03, taken at the supernode boundary.

## What was missing

Every statement this repository makes about hierarchical machines above a ratio of
1.03 has been a synthetic sensitivity. `sim/platforms.py` says so in its coverage
section and lists what a measurement would buy, with "one node's fast domain against
its cross-node fabric, the regime the whole extrapolation targets" at the top. The
reference machine was described as a bandwidth-flat supernode and measured at 1.03,
and that was the only ratio anyone had.

That description was right and incomplete. The machine is flat **inside a supernode**:
16 nodes of 8 cards, cross-node against intra-node within 2.6%, which is where the 1.03
comes from and where every end-to-end verdict in docs/03 was taken. It is **three**
supernodes sharing one filesystem, the bandwidth between them is low, and nothing had
ever crossed between them.

That boundary is the one [docs/03](../docs/03-applicability.md) already names and
declines to price. Its section 2 puts "intra-supernode vs cross-supernode" at "several
x, and cross-supernode collectives often degrade under incast", qualified as passing
only "if you actually stretch EP beyond the supernode". Its section 5 is blunter: no
public work stretches expert parallelism past a supernode, and anyone who does "would be
first -- there is no public data on alpha behavior and incast there, and you must
measure it yourself". This module is that measurement.

## The measurement

One run, and it reuses the instrument already in the tree
(`bench/a2a_form_probe.py`). Two configurations of identical structure -- world 16,
eight ranks on each of two nodes, the same sizes, the same repeats, the same session --
differing only in whether the eight remote peers sit in the same supernode:

    within supernode     both nodes in pool110
    across supernodes    one node in pool110, one in pool111

Because the structure matches, the difference is the boundary and nothing else. No
model, no fit, no assumption enters the comparison.

## The result

> **Correction and extension (2026-09-07).** The ratio first reported here, 2.55, was
> measured with exactly one node pair crossing the boundary, which is the least
> contended case there is. A follow-up run with 1, 2, 4 and 8 pairs crossing
> concurrently -- identical work in every subgroup, so only the pressure changes --
> shows a single pair gets substantially more than its share. Per-card cross-supernode
> bandwidth is 39.7 GB/s with one pair and 22.2 to 22.4 with two or more, flat to 1%
> from 2 pairs to 8. So **2.55 understates the hierarchy**; the like-for-like figure at
> a world both configurations can be compared at is **3.47** (below). The original
> number is kept because it is what a single pair measures and because the gap between
> the two is the contention term the cost model does not have.

> **Second correction (2026-09-08), and this one is mine rather than the instrument's.**
> The 33.9 GB/s below, and therefore the 3.47, came from a slope fitted through three
> points of which the smallest is non-monotone -- that run measures 32 MiB *slower* than
> 64 MiB, which no cost function does, so the fit is contaminated. The clean interval of
> the same data, 64 to 128 MiB, gives **23.1 GB/s**, and the ratio is **5.10**. A second
> supernode pair measured the next day at two independent size intervals agrees with itself
> to 1% and gives **15.7 GB/s, a ratio of 7.51**.
>
> Two things follow. The boundary is **deeper** than published, on both pairs, which
> moves the verdict further in favour of two-hop rather than against it. And **the
> boundary is not one number**: the two supernode pairs differ by 1.48x on the identical
> like-for-like collective, so a single measured boundary was never going to be
> representative of the machine.

Marginal bandwidth, regressed on the largest four sizes, in both timing styles:

    within supernode     102.7 GB/s burst   102.1 GB/s percall
    across supernodes     39.8 GB/s burst    40.1 GB/s percall

    hierarchy ratio  2.58 burst   2.55 percall

The two styles agree to 1%, and the ratio is stable across every number of points used
in the regression. Size by size the same a2a takes 1.9x longer at 8 MiB, rising to
2.5x by 64 MiB and flat from there, which is the shape of a boundary that costs
bandwidth rather than latency.

Deconvolving the intra-node share -- in both configurations a rank sends to 7 peers
inside its own node and 8 remote, and the intra-node tier is the physics-endorsed
122.4 GB/s -- puts the pure remote tiers at 89.2 GB/s within a supernode and 25.2 GB/s
across, a ratio of **3.54**. Quote 2.55 for the measured a2a and 3.54 for the tier, and
say which.

## What it decides, and what it does not

Use **3.47**, the loaded like-for-like figure: the shipped `BETA_FLAT` of 117.8 is a
world-128 a2a inside one supernode, and 33.9 GB/s is the same collective spanning both,
measured in the same session. Both are full-supernode loads and both are the same world,
which the single-pair comparison is not.

It clears the byte-only criterion at the supernode boundary with room to spare. With R = 128
cards in a supernode, `r_be = (1-1/R)q/(q-1)` is 1.98 at q = 2 and 1.49 at q = 3, so every
quota from 2 up passes.

Against the effective thresholds it depends on hidden width, and that dependence is the
result worth carrying:

    H = 1024    threshold 5.95    not cleared
    H = 2048    threshold 3.98    not cleared
    H = 4096    threshold 2.89    **cleared**
    H = 8192    threshold 2.40    **cleared**

The threshold falls with H because the payload grows by four over a fourfold widening
while the arrival chain grows by 1.5. So at the reference width the arrival chain still
decides, and **at the widths contemporary models actually use -- 7168 in DeepSeek-V3,
which docs/12 prices -- the measured boundary clears the threshold for the operator
chain this repository already has.** That is the first configuration in this project
where the model says two-hop wins with software that exists.

The contention measurement is what makes that statement stronger rather than weaker,
which was not the expected direction. Pressure on the boundary lowers the slow side,
and two-hop exists to send fewer bytes across the slow side, so a more contended
boundary favours it. The naive worry -- that real expert parallelism would collapse the
boundary and sink the verdict -- is refuted by the flatness from 2 pairs to 8.

Three limits, stated rather than buried.

The published thresholds are computed at R = 8, q = 3, H = 2048. At the supernode boundary
R is 128, which raises the byte-only threshold from 1.31 to 1.49 and moves the
effective ones by an amount this module does not compute.

Expert parallelism spanning two supernodes is 256 cards, and `calibrate.ALPHA_PTS` marks
alpha above world 128 as unsupported -- one low-confidence corpus. So this is a
measured ratio placed against published thresholds, not an end-to-end verdict at that
scale, and turning it into one needs alpha measured at 256.

And a ratio is one of six conditions in `sim/profile.py`. It is the one that was
missing; it is not the whole checklist.

## What it does to the repository's own negative result

Nothing, and that is worth saying plainly. docs/03's seven end-to-end geometries all
ran inside one supernode, where the ratio is 1.03 and two-hop has nothing to buy. That
verdict stands exactly as measured. What changes is that "this machine is flat" was
a statement about a supernode, and the machine has a boundary above it that is not flat.
"""
from __future__ import annotations

#: (total send bytes per rank, within-supernode burst ms, within-supernode percall ms,
#:  across-supernode burst ms, across-supernode percall ms) at world 16, eight ranks on each of
#:  two nodes. Medians of 7 repeats of the slowest rank, 2026-09-06, one session.
MEASURED = [
    (65536, 0.1339, 0.3128, 0.2471, 0.4590),
    (131072, 0.1388, 0.2933, 0.2425, 0.4554),
    (262144, 0.2304, 0.2878, 0.2434, 0.4435),
    (524288, 0.1205, 0.2965, 0.2446, 0.4656),
    (1048576, 0.1207, 0.2903, 0.2499, 0.4827),
    (1572864, 0.1240, 0.2823, 0.2544, 0.4837),
    (2097152, 0.1212, 0.2805, 0.2590, 0.5117),
    (3145728, 0.1210, 0.2975, 0.2646, 0.5243),
    (4194304, 0.1296, 0.2950, 0.2682, 0.5330),
    (6291456, 0.1221, 0.3293, 0.2797, 0.6055),
    (8388608, 0.1319, 0.3472, 0.3286, 0.6574),
    (10485760, 0.1561, 0.3784, 0.3782, 0.6990),
    (12582912, 0.1691, 0.3952, 0.4272, 0.7609),
    (16777216, 0.2057, 0.4416, 0.5275, 0.8722),
    (20971520, 0.2260, 0.4705, 0.6295, 0.9603),
    (25165824, 0.2627, 0.5210, 0.7295, 1.0922),
    (33554432, 0.3369, 0.6025, 0.9341, 1.3069),
    (50331648, 0.4868, 0.7163, 1.3302, 1.6716),
    (67108864, 0.6331, 0.8495, 1.7322, 2.1005),
    (100663296, 0.9331, 1.1575, 2.5314, 2.8838),
    (134217728, 1.2290, 1.4744, 3.3309, 3.6865),
    (201326592, 1.8371, 2.0807, 4.9347, 5.2783),
    (268435456, 2.4633, 2.7017, 6.4774, 6.8072),
]

WORLD = 16
CARDS_PER_SUPERNODE = 128
INTRA_NODE_GBPS = 122.4          # physics-endorsed, docs/05

#: Measured intra-node a2a, world 8, one node, wire-byte convention, 2026-09-06.
#: Note it is 17% below INTRA_NODE_GBPS above, which the calibration calls the most
#: stable number in the dataset. Either the two use different byte conventions or the
#: fast tier is over-credited; the tier Hop B runs on, so it matters. Recorded.
INTRA_NODE_MEASURED_GBPS = 101.1

#: Contention: (pairs crossing concurrently, marginal GB/s of the subgroup a2a).
#: Sixteen nodes, eight per supernode, node i paired with node i+8, every pair doing the
#: identical world-16 a2a already characterised above, with only the number of active
#: pairs changed. 2026-09-07. A single pair gets 39.7; from two pairs upward every
#: pair settles at 22 and stays there, flat to 1% out to eight. The boundary is not a
#: narrow shared pipe -- aggregate throughput scales 1.23, 2.45, 4.93 at 2, 4, 8 pairs,
#: perfectly linear once past the first step -- but a single crossing pair does get
#: capacity that is not available to it once anyone else is crossing.
CONTENTION = [(1, 39.7), (2, 22.2), (4, 22.1), (8, 22.4)]

#: A single a2a over all 128 ranks spanning both supernodes, same session. Superseded: this
#: is the three-point fit, and the run's 32 MiB point is non-monotone, which inflates
#: the slope. Kept because it is what was published and because the gap is the lesson.
CROSS_SUPERNODE_WORLD128_GBPS_THREE_POINT_FIT = 33.9

#: (supernode pair, GB/s) from the clean interval of each run: the plain world-128 a2a,
#: 8 nodes in each supernode, marginal over the 64-to-128 MiB step that both runs share.
#: pool110/pool12 additionally reproduces itself over 128-to-256 MiB at 15.7, so its
#: number rests on two independent intervals and the other on one.
CROSS_SUPERNODE_WORLD128_GBPS = {"pool110/pool111": 23.1, "pool110/pool12": 15.7}

#: Ratio against the shipped within-supernode BETA_FLAT of 117.8, which is the same
#: collective at the same world inside one supernode. Both boundaries are deeper than the
#: 3.47 first published, and they differ from each other by 1.48x.
LOADED_RATIO = {"pool110/pool111": 5.10, "pool110/pool12": 7.51}
BOUNDARIES_DIFFER_BY = 1.48

#: The headline, both timing styles, from the largest four sizes.
RATIO_BURST = 2.58
RATIO_PERCALL = 2.55
#: The pure remote tiers once the intra-node share is deconvolved.
TIER_WITHIN_RACK_GBPS = 89.2
TIER_ACROSS_SUPERNODE_GBPS = 25.2
TIER_RATIO = 3.54


def marginal_gbps(config: str, style: str, n_top: int = 4) -> float:
    """Delivered bandwidth from the slope of time against wire bytes at the top end.

    config is "within" or "across", style is "burst" or "percall". No alpha and no
    model form enter it, which is what makes the two configurations comparable by
    subtraction alone.
    """
    col = {("within", "burst"): 1, ("within", "percall"): 2,
           ("across", "burst"): 3, ("across", "percall"): 4}[(config, style)]
    pts = sorted(MEASURED)[-n_top:]
    xs = [b * (WORLD - 1) / WORLD for b, *_r in pts]
    ys = [r[col - 1] * 1e-3 for _b, *r in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return 1.0 / slope / 1e9


def hierarchy_ratio(style: str = "percall", n_top: int = 4) -> float:
    """Fast side over slow side at the supernode boundary, measured."""
    return marginal_gbps("within", style, n_top) / marginal_gbps("across", style, n_top)


def remote_tier_gbps(config: str, style: str = "percall") -> float:
    """The pure remote bandwidth, with the intra-node share taken out.

    In both configurations a rank sends to 7 peers inside its own node and 8 remote, so
    ``1/beta_eff = (7/15)/beta_intra + (8/15)/beta_remote`` with the intra-node tier
    known. This is the only step here that assumes anything, and what it assumes is the
    one bandwidth on this machine with physical backing.
    """
    eff = marginal_gbps(config, style)
    share_intra = (7.0 / (WORLD - 1)) / INTRA_NODE_GBPS
    return (8.0 / (WORLD - 1)) / (1.0 / eff - share_intra)


def contention_penalty() -> float:
    """How much a crossing pair loses once it is not the only one crossing."""
    d = dict(CONTENTION)
    return d[1] / d[8]


def loaded_ratio(pair: str = "pool110/pool111") -> float:
    """The hierarchy ratio like-for-like at world 128, for one supernode pair.

    The fast side is the shipped BETA_FLAT, a world-128 a2a inside one supernode. The slow
    side is the same collective spanning two, measured at full-supernode load. Both are the
    same world, which the world-16 pair comparison is not.

    Takes a pair because the machine has more than one boundary and they are not the
    same: 5.10 and 7.51, differing by 1.48x. Quoting one as "the" hierarchy ratio is the
    mistake this signature exists to make awkward.
    """
    from .calibrate import BETA_FLAT
    return BETA_FLAT / CROSS_SUPERNODE_WORLD128_GBPS[pair]


def shallowest_boundary() -> float:
    """The least favourable measured boundary, which is the one a verdict should use."""
    return min(loaded_ratio(p) for p in CROSS_SUPERNODE_WORLD128_GBPS)


def clears_at_hidden_width() -> list:
    """[(H, threshold, whether the loaded ratio clears it)] for the measured chain.

    The threshold falls with hidden width because the payload grows by four over a
    fourfold widening while the arrival chain grows by 1.5. Contemporary models sit
    well above the reference width, which is where this matters.
    """
    from .uncertainty import breakeven_vs_hidden_width
    r = shallowest_boundary()
    return [(H, be, r >= be) for H, be in breakeven_vs_hidden_width()]


def byte_breakeven_at_supernode(q: int) -> float:
    """The byte-only criterion with a whole supernode as the fast domain."""
    from .profile import byte_breakeven
    return byte_breakeven(CARDS_PER_SUPERNODE, q)


def placement() -> dict:
    """Where the measured ratio sits against the published effective thresholds."""
    from .sweep import CHAIN_SCENARIOS
    from .uncertainty import breakeven_ratio
    th = {name: breakeven_ratio(ch) for name, ch in CHAIN_SCENARIOS}
    r = hierarchy_ratio()
    return {"ratio": r, "thresholds": th,
            "clears": {name: r >= v for name, v in th.items()},
            "decided_by_the_chain": (r < th["PyTorch chain (measured)"]
                                     and r >= th["hypothetical fused target"])}


def main() -> None:
    print("CONTENTION: identical work per pair, only the number of crossing pairs moves")
    for n, bw in CONTENTION:
        print("   %d pair(s) crossing   %5.1f GB/s per card%s"
              % (n, bw, "   <- a single pair gets more than its share" if n == 1 else
                 ("   flat from here" if n == 2 else "")))
    print("   penalty once you are not alone: %.2fx; nothing further out to 8 pairs"
          % contention_penalty())
    print()
    print("LOADED HIERARCHY RATIO, like-for-like at world 128:")
    from .calibrate import BETA_FLAT
    print("   within one supernode (shipped BETA_FLAT)   %5.1f GB/s" % BETA_FLAT)
    for pair, bw in sorted(CROSS_SUPERNODE_WORLD128_GBPS.items()):
        print("   across %-18s (measured)      %5.1f GB/s   ratio %5.2f"
              % (pair, bw, loaded_ratio(pair)))
    print("   the two boundaries differ by %.2fx, so a verdict uses the shallower, "
          "%.2f" % (BOUNDARIES_DIFFER_BY, shallowest_boundary()))
    print()
    print("   against the measured-chain threshold, by hidden width:")
    for H, be, ok in clears_at_hidden_width():
        print("      H=%5d  threshold %.2f   %s"
              % (H, be, "CLEARED" if ok else "not cleared"))
    print("   so even the shallower boundary clears the threshold for the arrival")
    print("   chain this repository already has, at the reference width and above.")
    print()
    print("The supernode boundary, measured. World 16 both ways, eight ranks on each of two")
    print("nodes, identical structure; the only difference is whether the eight remote")
    print("peers are in the same supernode.")
    print()
    print("%-14s %14s %14s" % ("", "burst", "percall"))
    for cfg, label in (("within", "within supernode"), ("across", "across supernodes")):
        print("%-14s %11.1f GB/s %11.1f GB/s"
              % (label, marginal_gbps(cfg, "burst"), marginal_gbps(cfg, "percall")))
    print("%-14s %14.2f %14.2f"
          % ("ratio", hierarchy_ratio("burst"), hierarchy_ratio("percall")))
    print()
    print("pure remote tiers, intra-node share deconvolved: %.1f GB/s within a supernode, "
          "%.1f across, ratio %.2f"
          % (remote_tier_gbps("within"), remote_tier_gbps("across"),
             remote_tier_gbps("within") / remote_tier_gbps("across")))
    print()
    p = placement()
    print("byte-only criterion with a supernode as the fast domain (R = %d): q=2 needs %.2f, "
          "q=3 needs %.2f -- cleared" % (CARDS_PER_SUPERNODE, byte_breakeven_at_supernode(2),
                                         byte_breakeven_at_supernode(3)))
    print()
    print("against the published effective thresholds (stated at R=8, q=3, H=2048):")
    for name, v in p["thresholds"].items():
        print("   %-44s %.2f   %s" % (name, v, "cleared" if p["clears"][name] else "not cleared"))
    print("   measured supernode boundary                       %.2f" % p["ratio"])
    print()
    if p["decided_by_the_chain"]:
        print("So the topology is good enough and the arrival chain decides. That is what")
        print("docs/05 has been saying from synthetic ratios; this is the same statement")
        print("resting on a measured one.")
    print()
    print("Not an end-to-end verdict: expert parallelism across two supernodes is 256 cards,")
    print("and alpha above world 128 is unsupported. The seven end-to-end geometries of")
    print("docs/03 all ran inside one supernode, at 1.03, and that negative result stands.")


if __name__ == "__main__":
    main()
