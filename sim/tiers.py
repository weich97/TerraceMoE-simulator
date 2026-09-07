# -*- coding: utf-8 -*-
"""A structural improvement to the bandwidth model, tested and only partly adopted.

## The proposal

`sim/core.py` gives each named level -- fast, slow, flat -- one bandwidth, calibrated
per configuration. That is a fiction, and it is a fiction in the load-bearing place: the
comparison this repository exists to make prices one hop over the whole fabric against a
Hop A on the pure slow tier and a Hop B on the pure fast tier. Those are three different
mixtures of physical links, and each gets a beta calibrated on a fourth.

The obvious fix is to price a collective from the topology. Count how many of a rank's
peers sit at each tier -- inside its node, elsewhere in its rack, in another rack -- and
compose the transfer from per-tier link bandwidths:

    1 / beta_eff  =  sum over tiers of  (peers at tier / total peers) / beta_tier

Three constants for a two-rack machine instead of one per configuration, and adding a
boundary becomes one more constant rather than a new kind of level.

## It works within a world and fails across worlds

Calibrated on the world-16 configurations, the composition predicts **23.4 GB/s** for a
world-128 all-to-all spanning both racks. The measurement is **33.9**. Thirty-one
percent out, in the direction of predicting the machine slower than it is.

So a fixed bandwidth per tier does not compose either, and the reason is visible once
the implied tier bandwidth is read back out of each configuration:

    cross-node   103.0 GB/s at   8 peers      118.9 GB/s at 120 peers
    cross-rack    13.3 GB/s at   8 peers       20.0 GB/s at  64 peers

**A tier delivers more per card when the collective spreads its bytes over more peers
at that tier**, and the pattern holds on both tiers independently. Physically this is
path diversity and link filling: eight flows do not occupy a fabric that sixty-four
flows do.

A second variable is separable in the same data. Holding peers fixed at eight and
varying only how many nodes cross the boundary concurrently, cross-rack falls from
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
message size, with n_half near 5 for cross-rack and near 1 for cross-node -- but on two
points per tier that is an interpolation with no degrees of freedom left to test it. It
is written down as a hypothesis with the measurement that would falsify it, not shipped
as a calibration.

## Why this matters more to two-hop than to one-hop

Hop A runs at world = number of groups, so it spreads the slow tier's bytes over the
fewest peers of any collective in the scheme. One hop over the full fabric spreads the
same tier over the most. The peer-count effect therefore charges two-hop and credits
one-hop, and it does so hardest exactly where the hierarchy is coarsest: a two-rack
machine gives Hop A a single cross-rack peer.

That is not priced anywhere, it points against the method, and it is the first thing to
measure before the rack-boundary verdict in `sim/hierarchy.py` is treated as settled.
The measurement is cheap and specific: the cross-rack tier at one, two and four peers
per card under full load.
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
    ("cross_rack", 8): 13.3,      # under an eightfold crossing load
    ("cross_rack", 64): 20.0,     # under full load, world 128 across both racks
}

#: The same tier at the same peer count, alone rather than in company. The pair with
#: ("cross_rack", 8) above is what isolates concurrent pressure from peer spread.
CROSS_RACK_8_PEERS_UNCONTENDED = 25.9

#: What the composition predicted for the configuration it had not seen, and what that
#: configuration measured. Kept as data because the gap is the finding.
OUT_OF_SAMPLE = {"config": "world 128 across both racks",
                 "predicted_gbps": 23.4, "measured_gbps": 33.9}


@dataclass(frozen=True)
class Topology:
    """Enough of a machine to say which links a collective crosses."""
    cards_per_node: int = 8
    nodes_per_rack: int = 16
    racks: int = 2

    @property
    def cards_per_rack(self) -> int:
        return self.cards_per_node * self.nodes_per_rack


def peer_counts(topo: Topology, world: int, nodes_per_rack_used: int = None) -> dict:
    """How many of a rank's ``world - 1`` peers sit at each tier.

    Exact arithmetic, no calibration. ``nodes_per_rack_used`` says how the world is laid
    out when it does not fill the machine: a world of 16 can be two nodes of one rack or
    one node in each of two, and those traverse entirely different links. Default packs
    racks in turn, which is what a scheduler does unless told otherwise.
    """
    n = topo.cards_per_node
    per_rack = (nodes_per_rack_used if nodes_per_rack_used is not None
                else min(topo.nodes_per_rack, max(1, world // n)))
    cards_here = min(world, per_rack * n)
    intra = min(n, world) - 1
    cross_node = max(0, cards_here - n) if world > n else 0
    cross_rack = max(0, world - cards_here)
    out = {"intra_node": intra}
    if cross_node:
        out["cross_node"] = cross_node
    if cross_rack:
        out["cross_rack"] = cross_rack
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
                              ("world 16, two nodes one rack", 16, 2),
                              ("world 16, one node per rack", 16, 1),
                              ("world 128, 8 nodes per rack", 128, 8),
                              ("world 128, one full rack", 128, 16)):
        print("%-34s %s" % (label, peer_counts(topo, world, npr)))
    print()
    print("A fixed bandwidth per tier, calibrated on world 16 and asked about world 128:")
    print("   predicted %.1f GB/s   measured %.1f   %+.0f%%"
          % (OUT_OF_SAMPLE["predicted_gbps"], OUT_OF_SAMPLE["measured_gbps"],
             100 * (OUT_OF_SAMPLE["predicted_gbps"] / OUT_OF_SAMPLE["measured_gbps"] - 1)))
    print()
    print("because a tier delivers more when more peers use it:")
    for tier in ("cross_node", "cross_rack"):
        pts = sorted((p, b) for (t, p), b in TIER_BY_PEERS.items() if t == tier)
        print("   %-12s %s" % (tier, "   ".join("%6.1f GB/s at %3d peers" % (b, p)
                                                for p, b in pts)))
    print("   cross_rack at 8 peers is %.1f alone against %.1f in company, so peer"
          % (CROSS_RACK_8_PEERS_UNCONTENDED, TIER_BY_PEERS[("cross_rack", 8)]))
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
