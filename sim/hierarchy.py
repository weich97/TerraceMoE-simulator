# -*- coding: utf-8 -*-
"""The first measured hierarchy ratio above 1.03, taken at the rack boundary.

## What was missing

Every statement this repository makes about hierarchical machines above a ratio of
1.03 has been a synthetic sensitivity. `sim/platforms.py` says so in its coverage
section and lists what a measurement would buy, with "one node's fast domain against
its cross-node fabric, the regime the whole extrapolation targets" at the top. The
reference machine was described as a bandwidth-flat supernode and measured at 1.03,
and that was the only ratio anyone had.

That description was right and incomplete. The machine is flat **inside a rack**: 16
nodes of 8 cards, cross-node against intra-node within 2.6%, which is where the 1.03
comes from and where every end-to-end verdict in docs/03 was taken. It is two racks,
and nothing had ever crossed between them.

## The measurement

One run, and it reuses the instrument already in the tree
(`bench/a2a_form_probe.py`). Two configurations of identical structure -- world 16,
eight ranks on each of two nodes, the same sizes, the same repeats, the same session --
differing only in whether the eight remote peers sit in the same rack:

    within rack     both nodes in rack 110
    across racks    one node in rack 110, one in rack 111

Because the structure matches, the difference is the boundary and nothing else. No
model, no fit, no assumption enters the comparison.

## The result

Marginal bandwidth, regressed on the largest four sizes, in both timing styles:

    within rack     102.7 GB/s burst   102.1 GB/s percall
    across racks     39.8 GB/s burst    40.1 GB/s percall

    hierarchy ratio  2.58 burst   2.55 percall

The two styles agree to 1%, and the ratio is stable across every number of points used
in the regression. Size by size the same a2a takes 1.9x longer at 8 MiB, rising to
2.5x by 64 MiB and flat from there, which is the shape of a boundary that costs
bandwidth rather than latency.

Deconvolving the intra-node share -- in both configurations a rank sends to 7 peers
inside its own node and 8 remote, and the intra-node tier is the physics-endorsed
122.4 GB/s -- puts the pure remote tiers at 89.2 GB/s within a rack and 25.2 GB/s
across, a ratio of **3.54**. Quote 2.55 for the measured a2a and 3.54 for the tier, and
say which.

## What it decides, and what it does not

2.55 clears the byte-only criterion at the rack boundary comfortably. With R = 128
cards in a rack, `r_be = (1-1/R)q/(q-1)` is 1.98 at q = 2 and 1.49 at q = 3, so every
quota from 2 up passes, the loosest by 136%.

It then lands **between the two implementation thresholds this repository publishes**:
3.98 for the measured PyTorch arrival chain and 1.49 for the hypothetical fused one.
That is the whole point. On this boundary the topology is good enough and the arrival
chain is what decides, which is the thing docs/05 has been saying from synthetic
numbers -- fix the implementation first, then talk topology -- now resting on a
measured ratio.

Three limits, stated rather than buried.

The published thresholds are computed at R = 8, q = 3, H = 2048. At the rack boundary
R is 128, which raises the byte-only threshold from 1.31 to 1.49 and moves the
effective ones by an amount this module does not compute.

Expert parallelism spanning two racks is 256 cards, and `calibrate.ALPHA_PTS` marks
alpha above world 128 as unsupported -- one low-confidence corpus. So this is a
measured ratio placed against published thresholds, not an end-to-end verdict at that
scale, and turning it into one needs alpha measured at 256.

And a ratio is one of six conditions in `sim/profile.py`. It is the one that was
missing; it is not the whole checklist.

## What it does to the repository's own negative result

Nothing, and that is worth saying plainly. docs/03's seven end-to-end geometries all
ran inside one rack, where the ratio is 1.03 and two-hop has nothing to buy. That
verdict stands exactly as measured. What changes is that "this machine is flat" was
a statement about a rack, and the machine has a boundary above it that is not flat.
"""
from __future__ import annotations

#: (total send bytes per rank, within-rack burst ms, within-rack percall ms,
#:  across-rack burst ms, across-rack percall ms) at world 16, eight ranks on each of
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
CARDS_PER_RACK = 128
INTRA_NODE_GBPS = 122.4          # physics-endorsed, docs/05

#: The headline, both timing styles, from the largest four sizes.
RATIO_BURST = 2.58
RATIO_PERCALL = 2.55
#: The pure remote tiers once the intra-node share is deconvolved.
TIER_WITHIN_RACK_GBPS = 89.2
TIER_ACROSS_RACK_GBPS = 25.2
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
    """Fast side over slow side at the rack boundary, measured."""
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


def byte_breakeven_at_rack(q: int) -> float:
    """The byte-only criterion with a whole rack as the fast domain."""
    from .profile import byte_breakeven
    return byte_breakeven(CARDS_PER_RACK, q)


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
    print("The rack boundary, measured. World 16 both ways, eight ranks on each of two")
    print("nodes, identical structure; the only difference is whether the eight remote")
    print("peers are in the same rack.")
    print()
    print("%-14s %14s %14s" % ("", "burst", "percall"))
    for cfg, label in (("within", "within rack"), ("across", "across racks")):
        print("%-14s %11.1f GB/s %11.1f GB/s"
              % (label, marginal_gbps(cfg, "burst"), marginal_gbps(cfg, "percall")))
    print("%-14s %14.2f %14.2f"
          % ("ratio", hierarchy_ratio("burst"), hierarchy_ratio("percall")))
    print()
    print("pure remote tiers, intra-node share deconvolved: %.1f GB/s within a rack, "
          "%.1f across, ratio %.2f"
          % (remote_tier_gbps("within"), remote_tier_gbps("across"),
             remote_tier_gbps("within") / remote_tier_gbps("across")))
    print()
    p = placement()
    print("byte-only criterion with a rack as the fast domain (R = %d): q=2 needs %.2f, "
          "q=3 needs %.2f -- cleared" % (CARDS_PER_RACK, byte_breakeven_at_rack(2),
                                         byte_breakeven_at_rack(3)))
    print()
    print("against the published effective thresholds (stated at R=8, q=3, H=2048):")
    for name, v in p["thresholds"].items():
        print("   %-44s %.2f   %s" % (name, v, "cleared" if p["clears"][name] else "not cleared"))
    print("   measured rack boundary                       %.2f" % p["ratio"])
    print()
    if p["decided_by_the_chain"]:
        print("So the topology is good enough and the arrival chain decides. That is what")
        print("docs/05 has been saying from synthetic ratios; this is the same statement")
        print("resting on a measured one.")
    print()
    print("Not an end-to-end verdict: expert parallelism across two racks is 256 cards,")
    print("and alpha above world 128 is unsupported. The seven end-to-end geometries of")
    print("docs/03 all ran inside one rack, at 1.03, and that negative result stands.")


if __name__ == "__main__":
    main()
