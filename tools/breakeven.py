# -*- coding: utf-8 -*-
"""Breakeven criterion for two-hop (hierarchical) all-to-all: whether your cluster should use this method at all.

This is the single most important file in this repository. T-Route + T-A2A are
designed **for clusters with a pronounced hierarchy**; on a bandwidth-flat
interconnect they do not pay off — we measured a negative return ourselves on a
flat supernode (see docs/03-applicability.md). **Run this criterion first, then
decide whether to adopt.**

------------------------------------------------------------------------------
Model

Per token, per destination group (group = server / rack / supernode, depending
on where your hierarchy boundary sits), with R accelerator dies in the group
and the token hitting q experts inside that group (q = top-k / group-count cap M):

    One-hop (flat a2a)   all q payload rows cross the **slow** link          slow side q
    Two-hop (T-A2A)      1 row crosses the slow link to a delegate die,
                         then fans out on the **fast** side                  slow side 1 + fast side q(1-1/R)

    Slow-side saved = q - 1          fast-side added = q(1 - 1/R)

    **Breakeven ratio  r_be = (1 - 1/R) * q / (q - 1)**

Your cluster's fast/slow bandwidth ratio (beta_fast / beta_slow) must be
**greater than** r_be for two-hop to come out ahead on bytes. At R=8, r_be
decreases monotonically in q with limit 0.875:

    q=2 -> 1.750    q=3 -> 1.3125    q=4 -> 1.167    q=6 -> 1.050    q=8 -> 1.000

Two caveats:
  * The formula assumes both sides send payload per (token, expert) row —
    consistent with mainstream implementations. If the two-hop side additionally
    dedupes at die level (a token routed to multiple experts on the same die
    sends only one row), r_be drops further (see --dedup).
  * Bytes are only half the ledger: two-hop pays the fixed overhead of one
    extra collective (launch/sync). For small messages and deep pipelines,
    audit the alpha side separately; passing the criterion only means "worth
    an experiment", not "guaranteed faster".

------------------------------------------------------------------------------
Public reference points (order of magnitude, sources in docs/03; trust your own
measurements over these):

    Intra-NVLink domain vs cross-node IB/RoCE    ~3-18x   — criterion passes easily; methods in
                                                            this family have been validated by
                                                            several public works
    Intra-server HCCS vs cross-server RoCE       ~8x      — same as above
    Inside a flat supernode (uniform switching)  ~1.0     — **criterion fails, do not use two-hop**

Usage:
    python tools/breakeven.py --ratio 8          # your cluster's fast/slow bandwidth ratio
    python tools/breakeven.py --ratio 8 --dies 8 --dedup
"""
from __future__ import annotations

import argparse
from math import comb


def D_of(q: int, R: int, E: int) -> float:
    """Expected number of distinct dies a token's q experts land on. Used only for die-level dedup.

    S = R*E experts are uniformly selectable inside the group; the probability
    that a given die has none of its experts chosen is C(S-E, q)/C(S, q); take
    the complement and multiply by R to get the expected number of dies hit.
    """
    S = R * E
    if q <= 0 or q > S:
        return 0.0
    return R * (1.0 - comb(S - E, q) / comb(S, q))


def breakeven(q: int, R: int, dedup: bool = False, E: int = 4) -> float:
    """Lower bound on beta_fast/beta_slow for two-hop to win on bytes. At q=1 two-hop is a pure loss (saves 0); returns inf."""
    if q <= 1:
        return float("inf")
    added = (D_of(q, R, E) if dedup else q) * (1.0 - 1.0 / R)
    return added / (q - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ratio", type=float, default=None,
                    help="your cluster's fast-side / slow-side bandwidth (measure it yourself, do not use nameplate)")
    ap.add_argument("--dies", type=int, default=8, help="number of dies R per group (fast-side domain)")
    ap.add_argument("--epr", type=int, default=4, help="experts per die (only used with --dedup)")
    ap.add_argument("--dedup", action="store_true",
                    help="looser criterion when the two-hop side dedupes at die level")
    args = ap.parse_args()
    R, E = args.dies, args.epr

    print("Breakeven bandwidth ratio for two-hop all-to-all (R=%d dies per group)" % R)
    print()
    print("  q (experts per group)   r_be%s" % (" (die-level dedup)" if args.dedup else ""))
    rows = []
    for q in range(2, 13):
        r = breakeven(q, R, args.dedup, E)
        rows.append((q, r))
        mark = ""
        if args.ratio is not None:
            mark = "   <- your %.2f is %s" % (args.ratio,
                                           "**enough**" if args.ratio > r else "not enough")
        print("  %2d               %7.4f%s" % (q, r, mark))
    print()
    if args.ratio is None:
        print("Add --ratio <your measured ratio> to get a verdict. **Use measured values, not nameplate**:")
        print("nameplate bandwidth and effective collective bandwidth often differ by more than 2x.")
        return
    good = [q for q, r in rows if args.ratio > r]
    if good:
        # **Marginal-case warning**: when the byte-side margin is this thin, the
        # alpha side (the fixed overhead of the extra collective that two-hop pays)
        # will almost certainly eat it. A flat interconnect (ratio ~1.0) can "pass
        # mathematically" at large q (the limit of r_be is 1-1/R < 1), but that
        # 1-3% margin is not a signal engineering can cash in.
        margin = max(args.ratio / r - 1.0 for q, r in rows if q in good)
        if margin < 0.10:
            print("Ratio %.2f only scrapes past the line at q ∈ %s, max margin %.1f%% — **marginal case, "
                  "almost certain to be eaten by the alpha side; do not adopt**." % (args.ratio, good, margin * 100))
            print("This method wants a hierarchy with multiples of margin (see docs/03 §2), not a percent-level squeak past the line.")
        else:
            print("At ratio %.2f, two-hop comes out ahead on bytes at q ∈ %s (max margin %.0f%%) — "
                  "worth an integration experiment." % (args.ratio, good, margin * 100))
            print("Next step: audit the alpha side (the fixed overhead of two-hop's extra collective), see docs/03 §4.")
    else:
        print("Ratio %.2f does not clear the breakeven line at any quota — **your interconnect is too flat "
              "for this method, do not use two-hop**. The routing-constraint part of T-Route (load "
              "balancing) can still be evaluated independently."
              % args.ratio)


if __name__ == "__main__":
    main()
