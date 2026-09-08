# -*- coding: utf-8 -*-
"""A structural improvement to the bandwidth model, tested and only partly adopted.

## The proposal

`sim/core.py` gives each named level -- fast, slow, flat -- one bandwidth, calibrated
per configuration. That is a fiction, and it is a fiction in the load-bearing place: the
comparison this repository exists to make prices one hop over the whole fabric against a
Hop A on the pure slow tier and a Hop B on the pure fast tier. Those are three different
mixtures of physical links, and each gets a beta calibrated on a fourth.

The obvious fix is to price a collective from the topology. Count how many of a rank's
peers sit at each tier -- inside its node, elsewhere in its supernode, in another supernode -- and
compose the transfer from per-tier link bandwidths:

    1 / beta_eff  =  sum over tiers of  (peers at tier / total peers) / beta_tier

Three constants for a two-supernode machine instead of one per configuration, and adding a
boundary becomes one more constant rather than a new kind of level.

> **Refuted, and by the measurement this module asked for (2026-09-08).** The
> peer-count effect described below was inferred from deconvolved all-to-alls, and the
> module said the thing to do was measure the cross-supernode tier directly at one, two and
> four peers per card under load. That was run on a second supernode pair, with split sizes
> arranged so **every card sends only across the boundary** -- no intra-supernode traffic, so
> nothing to deconvolve. Per-card cross-supernode bandwidth is **flat**: 7.82 GB/s at one
> peer rising to 8.03 at sixty-four, 2.6% over a sixty-fourfold change in spread, with
> most repeats agreeing to under 1%.
>
> So the peer-count dependence is an artefact of the deconvolution, not a property of
> the fabric, and the two consequences point in opposite directions. **The charge
> against two-hop is withdrawn**: Hop A running at one cross-supernode peer gets the same
> per-card bandwidth as an all-to-all spread over sixty-four, so the coarsest hierarchy
> costs it nothing, and the open risk against the supernode-boundary verdict is closed. But
> **the -31% out-of-sample failure below still stands and now has no explanation**; what
> it indicts is the deconvolution that produced the tier numbers, which is the one step
> in this module that assumes anything.
>
> The sections below are kept as written, because the reasoning they record is how the
> measurement got specified, and because a hypothesis that survived two tiers of
> indirect evidence and died on the first direct one is worth leaving visible.

## It works within a world and fails across worlds

Calibrated on the world-16 configurations, the composition predicts **23.4 GB/s** for a
world-128 all-to-all spanning both supernodes. The measurement is **33.9**. Thirty-one
percent out, in the direction of predicting the machine slower than it is.

So a fixed bandwidth per tier does not compose either, and the reason is visible once
the implied tier bandwidth is read back out of each configuration:

    cross-node   103.0 GB/s at   8 peers      118.9 GB/s at 120 peers
    cross-supernode    13.3 GB/s at   8 peers       20.0 GB/s at  64 peers

**A tier delivers more per card when the collective spreads its bytes over more peers
at that tier**, and the pattern holds on both tiers independently. Physically this is
path diversity and link filling: eight flows do not occupy a fabric that sixty-four
flows do.

A second variable is separable in the same data. Holding peers fixed at eight and
varying only how many nodes cross the boundary concurrently, cross-supernode falls from
25.9 GB/s alone to 13.3 under an eightfold load (`sim/hierarchy.py`). So the delivered
bandwidth at a tier depends on **how widely the bytes are spread and how much company
they have**, and the model has a term for neither.

## What is adopted, and what is not

The topology and the peer counting are adopted: they are arithmetic, they are exact,
and they let any configuration state which links it actually traverses instead of
naming a level. `peer_counts` is the useful part of the proposal and it is right.

The composition is **not** adopted as a replacement for the shipped betas, because a
model that misses an out-of-sample configuration by 31% is not an improvement over one
calibrated per configuration -- it is the same error moved somewhere less honest. What
it is good for is stating precisely what is missing, which the numbers above now do.

Fitting the peer-count dependence is possible -- both tiers fit
`beta = beta_inf * n/(n + n_half)`, the same saturating shape the model already uses for
message size, with n_half near 5 for cross-supernode and near 1 for cross-node -- but on two
points per tier that is an interpolation with no degrees of freedom left to test it. It
is written down as a hypothesis with the measurement that would falsify it, not shipped
as a calibration.

## Why this matters more to two-hop than to one-hop

Hop A runs at world = number of groups, so it spreads the slow tier's bytes over the
fewest peers of any collective in the scheme. One hop over the full fabric spreads the
same tier over the most. The peer-count effect therefore charges two-hop and credits
one-hop, and it does so hardest exactly where the hierarchy is coarsest: a two-supernode
machine gives Hop A a single cross-supernode peer.

That was not priced anywhere, it pointed against the method, and it was named as the
first thing to measure before the supernode-boundary verdict in `sim/hierarchy.py` could be
treated as settled. **It has now been measured and it is not there**: see the correction
at the top. Hop A pays nothing for running at one peer.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Implied pure-tier bandwidth by (tier, peers at that tier), read out of the measured
#: configurations by deconvolving the tiers whose value is already known. The intra-node
#: anchor is the world-8 one-node measurement, 101.1 GB/s over 7 peers.
#: All wire-byte convention, 2026-09-06 and 2026-09-07, one machine.
TIER_BY_PEERS = {
    ("intra_node", 7): 101.1,
    ("cross_node", 8): 103.0,
    ("cross_node", 120): 118.9,
    ("cross_supernode", 8): 13.3,      # under an eightfold crossing load
    ("cross_supernode", 64): 20.0,     # under full load, world 128 across both supernodes
}

#: The same tier at the same peer count, alone rather than in company. The pair with
#: ("cross_supernode", 8) above is what isolates concurrent pressure from peer spread.
CROSS_SUPERNODE_8_PEERS_UNCONTENDED = 25.9

#: The direct test, 2026-09-08, on the pool110/pool12 boundary: (peers per card, total
#: send bytes per rank, ms). Split sizes are arranged so a card sends ONLY across the
#: boundary, to exactly this many peers, and receives from as many; each card sends
#: bytes/2 across at every peer count, so the load on the boundary is identical along
#: the sweep and the only variable is how widely it is spread. Nothing is deconvolved.
#: Produced by bench/xsupernode_peers.py.
PEER_SWEEP = [
    (1, 67108864, 4.7749), (2, 67108864, 4.7243), (4, 67108864, 4.7307),
    (8, 67108864, 4.7510), (16, 67108864, 4.7711), (32, 67108864, 4.8775),
    (64, 67108864, 5.1810),
    (1, 134217728, 9.0566), (2, 134217728, 8.9636), (4, 134217728, 8.9986),
    (8, 134217728, 8.9874), (16, 134217728, 9.0236), (32, 134217728, 9.1272),
    (64, 134217728, 9.3504),
    (1, 268435456, 17.6363), (2, 268435456, 17.4453), (4, 268435456, 17.4877),
    (8, 268435456, 17.4413), (16, 268435456, 17.4614), (32, 268435456, 17.5515),
    (64, 268435456, 17.7128),
]


def measured_cross_supernode_gbps(peers: int) -> float:
    """Per-card cross-supernode bandwidth at a peer count, marginal between the two largest
    sizes so the fixed cost drops out. Flat to 2.6% over the whole sweep."""
    a = dict(((n, b), t) for n, b, t in PEER_SWEEP)
    lo, hi = 134217728, 268435456
    return (hi - lo) / 2.0 / ((a[(peers, hi)] - a[(peers, lo)]) * 1e-3) / 1e9


def peer_effect_span() -> float:
    """Ratio of the best to the worst cross-supernode bandwidth across the peer sweep.

    If a peer-count effect existed this would be large; it is 1.026. Kept as a function
    so a test can assert the flatness rather than a stored number.
    """
    vals = [measured_cross_supernode_gbps(n) for n in (1, 2, 4, 8, 16, 32, 64)]
    return max(vals) / min(vals)

#: What the composition predicted for the configuration it had not seen, and what that
#: configuration measured. Kept as data because the gap is the finding.
OUT_OF_SAMPLE = {"config": "world 128 across both supernodes",
                 "predicted_gbps": 23.4, "measured_gbps": 33.9}


@dataclass(frozen=True)
class Topology:
    """Enough of a machine to say which links a collective crosses."""
    cards_per_node: int = 8
    nodes_per_supernode: int = 16
    supernodes: int = 2

    @property
    def cards_per_rack(self) -> int:
        return self.cards_per_node * self.nodes_per_supernode


def peer_counts(topo: Topology, world: int, nodes_per_supernode_used: int = None) -> dict:
    """How many of a rank's ``world - 1`` peers sit at each tier.

    Exact arithmetic, no calibration. ``nodes_per_supernode_used`` says how the world is laid
    out when it does not fill the machine: a world of 16 can be two nodes of one supernode or
    one node in each of two, and those traverse entirely different links. Default packs
    supernodes in turn, which is what a scheduler does unless told otherwise.
    """
    n = topo.cards_per_node
    per_rack = (nodes_per_supernode_used if nodes_per_supernode_used is not None
                else min(topo.nodes_per_supernode, max(1, world // n)))
    cards_here = min(world, per_rack * n)
    intra = min(n, world) - 1
    cross_node = max(0, cards_here - n) if world > n else 0
    cross_supernode = max(0, world - cards_here)
    out = {"intra_node": intra}
    if cross_node:
        out["cross_node"] = cross_node
    if cross_supernode:
        out["cross_supernode"] = cross_supernode
    assert sum(out.values()) == world - 1, (out, world)
    return out


def compose(peers: dict, tier_gbps: dict) -> float:
    """Effective bandwidth from per-tier bandwidths, weighted by peer share.

    The serial-egress reading: a card's port carries every tier's bytes in turn, so the
    inverse bandwidths add in proportion. The parallel reading, taking the slowest tier
    alone, was also tried and misses the same out-of-sample configuration by a similar
    margin, so the choice between them is not what this model gets wrong.
    """
    total = sum(peers.values())
    return 1.0 / sum((k / total) / tier_gbps[t] for t, k in peers.items())


def tier_at(tier: str, peers: int) -> float:
    """Measured tier bandwidth at a peer count, interpolated in log peers where needed.

    Refuses to extrapolate below the smallest measured peer count, which is exactly the
    regime Hop A runs in and exactly where the model has no evidence.
    """
    pts = sorted((p, b) for (t, p), b in TIER_BY_PEERS.items() if t == tier)
    if not pts:
        raise ValueError("no measurement for tier %r" % tier)
    if peers < pts[0][0]:
        raise ValueError(
            "%s is measured down to %d peers; %d is the regime Hop A runs in and this "
            "model has no evidence there. See the module docstring."
            % (tier, pts[0][0], peers))
    if peers >= pts[-1][0]:
        return pts[-1][1]
    import math
    for (p0, b0), (p1, b1) in zip(pts, pts[1:]):
        if p0 <= peers <= p1:
            t = (math.log(peers) - math.log(p0)) / (math.log(p1) - math.log(p0))
            return b0 + t * (b1 - b0)
    return pts[-1][1]


def hop_a_peer_count(n_groups: int) -> int:
    """Peers Hop A spreads the slow tier over. One less than the group count."""
    return max(n_groups - 1, 1)


def main() -> None:
    topo = Topology()
    print("Peer counts are exact arithmetic; the bandwidths are not.")
    print()
    print("%-34s %s" % ("configuration", "peers by tier"))
    for label, world, npr in (("world 8, one node", 8, 1),
                              ("world 16, two nodes one supernode", 16, 2),
                              ("world 16, one node per supernode", 16, 1),
                              ("world 128, 8 nodes per supernode", 128, 8),
                              ("world 128, one full supernode", 128, 16)):
        print("%-34s %s" % (label, peer_counts(topo, world, npr)))
    print()
    print("A fixed bandwidth per tier, calibrated on world 16 and asked about world 128:")
    print("   predicted %.1f GB/s   measured %.1f   %+.0f%%"
          % (OUT_OF_SAMPLE["predicted_gbps"], OUT_OF_SAMPLE["measured_gbps"],
             100 * (OUT_OF_SAMPLE["predicted_gbps"] / OUT_OF_SAMPLE["measured_gbps"] - 1)))
    print()
    print("because a tier delivers more when more peers use it:")
    for tier in ("cross_node", "cross_supernode"):
        pts = sorted((p, b) for (t, p), b in TIER_BY_PEERS.items() if t == tier)
        print("   %-12s %s" % (tier, "   ".join("%6.1f GB/s at %3d peers" % (b, p)
                                                for p, b in pts)))
    print("   cross_supernode at 8 peers is %.1f alone against %.1f in company, so peer"
          % (CROSS_SUPERNODE_8_PEERS_UNCONTENDED, TIER_BY_PEERS[("cross_supernode", 8)]))
    print("   spread and concurrent load are separate effects and both are unmodelled.")
    print()
    print("What it costs two-hop: Hop A runs at world = group count, so it spreads the")
    print("slow tier over the fewest peers of anything in the scheme.")
    for ng in (2, 4, 16):
        print("   %2d groups -> Hop A has %2d peer(s) at the slow tier%s"
              % (ng, hop_a_peer_count(ng),
                 "   below anything measured" if hop_a_peer_count(ng) < 8 else ""))
    print("One hop spreads the same tier over every rank in the world, so this effect")
    print("charges two-hop and credits one-hop. It is not priced anywhere.")


if __name__ == "__main__":
    main()
