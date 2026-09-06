# -*- coding: utf-8 -*-
"""The measurement that decided whether a collective's fixed cost adds or overlaps.

## The question this settles

`sim/core.py` prices one all-to-all as `alpha(world) + wire/beta`: the fixed cost is
paid in series with the bytes. Recalibrating the whole model under the alternative
rule, the two combining in quadrature, fit this repository's size-sweep corpora three
times better at identical parameter count and fit the Tier-1 corpus twice as badly
(docs/05). Two corpora, two benchmark styles, opposite answers, and nothing offline
could say whether the machine or the instrument was speaking. `sim/fit.py` had recorded
the two families as disagreeing about the *level* of the same machine; the
recalibration showed they disagree about its *shape* as well.

So both styles were run on the same machine, at the same worlds, over the same sizes,
in the same session. That is the whole experiment, and it is cheap: two idle nodes and
about twenty minutes.

    burst    N collectives back to back, one synchronisation at the end, cost = total/N.
             The host runs ahead and never observes an individual call. This is the
             regime `calibrate.ALPHA_PTS` belongs to, and it had only ever been run at
             tiny payloads.
    percall  ranks aligned on a barrier, then one collective timed alone, the host
             waiting for it. This is the regime the size sweeps belong to, and it had
             only ever been run at large payloads.

The two rules are indistinguishable at both ends of the size axis -- at tiny sizes both
give alpha, at huge sizes both give wire/beta -- and differ most where the two terms
are comparable, which on this machine is a buffer near 13 MB. The grid is dense there
for that reason.

## The answer

**The two styles want opposite rules, and with the exponent left free the data picks
them out unprompted.** Fitting `t = (alpha^p + T^p)^(1/p)` with p free gives p = 0.99
on the percall data, which is addition, and drives p to the bound on the burst data,
which is a hard maximum. Held at the two named values:

    burst     additive 12.5%   quadrature  4.3%     -> overlap
    percall   additive  2.3%   quadrature  3.9%     -> addition

The verdict does not depend on how aggressively the noisy small-size burst points are
trimmed; it holds at every cut, and at no cut at all.

Two independent checks agree. Forcing the additive rule onto the burst data needs
135 GB/s, **above the 122.4 GB/s per-card aggregate egress** docs/05 endorses as
physics, while the overlap rule fits at 106.9 and stays inside it. And the alpha the
overlap rule recovers from the burst data, 0.114 ms at world 8 and 0.120 at world 16,
lands near the independent call-count scan's 0.129 and 0.134 -- a benchmark that took
no part in this fit.

## What it means

The rule is a property of the **host regime**, not of the machine. When the host runs
ahead, the next call is set up while the previous one is still moving bytes, so the
fixed cost hides behind the transfer and the call costs the larger of the two. When the
host observes each call, the two serialise and the costs add. Both readings were right
about their own measurement, which is why neither corpus could be dismissed.

**The shipped model stays additive, and now for a reason rather than by inheritance.**
An MoE dispatch is host-exposed by construction: a variable-length all-to-all cannot be
issued until its per-peer counts have come back to the host, which `core.py` already
prices as `splits_sync_ms`. That is the percall regime, and the percall regime adds.

One shipped constant is called into question by this, and it is recorded rather than
changed: see `ALPHA16_NOTE` below.
"""
from __future__ import annotations

#: Per-card aggregate egress, the physical ceiling any all-to-all is bounded by:
#: (6 x intra-node link 112.1 + 1 x in-package direct 185)/7, endorsed by an intra-node
#: a2a measuring 122.6 GB/s. See docs/05.
PHYSICAL_CEILING_GBPS = 122.4

#: (world, total send bytes per rank, burst ms, percall ms, burst spread, percall
#: spread). Medians of 7 repeats of the slowest rank; the spreads are (max-min)/min
#: across those repeats, and exist so a reader can see which points carry weight.
#: Measured 2026-09-06 on two idle nodes of the reference machine, 8 cards each,
#: bf16 buffers, aligned sizes. Produced by bench/a2a_form_probe.py.
MEASURED = [
    (8, 65536, 0.1278, 0.2378, 0.300, 0.042),
    (8, 131072, 0.1259, 0.2604, 0.226, 0.168),
    (8, 262144, 0.1202, 0.2403, 0.066, 0.051),
    (8, 524288, 0.1206, 0.2326, 5.416, 0.041),
    (8, 1048576, 0.1179, 0.2418, 0.073, 0.030),
    (8, 1572864, 0.1206, 0.2440, 0.049, 0.041),
    (8, 2097152, 0.1217, 0.2498, 0.037, 0.050),
    (8, 3145728, 0.1221, 0.2658, 0.089, 0.032),
    (8, 4194304, 0.1248, 0.2699, 0.062, 0.032),
    (8, 6291456, 0.1238, 0.2837, 0.058, 0.033),
    (8, 8388608, 0.1285, 0.2923, 1.850, 0.081),
    (8, 10485760, 0.1131, 0.3326, 0.026, 0.029),
    (8, 12582912, 0.1237, 0.3974, 0.006, 0.149),
    (8, 16777216, 0.1599, 0.3954, 0.012, 0.025),
    (8, 20971520, 0.1945, 0.4360, 0.002, 0.010),
    (8, 25165824, 0.2283, 0.4284, 0.031, 0.022),
    (8, 33554432, 0.2934, 0.5085, 0.003, 0.062),
    (8, 50331648, 0.4275, 0.6389, 0.003, 0.013),
    (8, 67108864, 0.5597, 0.7687, 0.002, 0.075),
    (8, 100663296, 0.8273, 1.0491, 0.004, 0.031),
    (8, 134217728, 1.1109, 1.3283, 0.002, 0.032),
    (8, 201326592, 1.6563, 1.9367, 0.003, 0.023),
    (8, 268435456, 2.2259, 2.4922, 0.004, 0.027),
    (16, 65536, 0.1339, 0.3128, 0.461, 0.020),
    (16, 131072, 0.1388, 0.2933, 0.757, 0.137),
    (16, 262144, 0.2304, 0.2878, 2.986, 0.016),
    (16, 524288, 0.1205, 0.2965, 0.062, 0.057),
    (16, 1048576, 0.1207, 0.2903, 0.242, 0.041),
    (16, 1572864, 0.1240, 0.2823, 1.211, 0.057),
    (16, 2097152, 0.1212, 0.2805, 0.207, 0.020),
    (16, 3145728, 0.1210, 0.2975, 0.124, 0.035),
    (16, 4194304, 0.1296, 0.2950, 6.747, 0.037),
    (16, 6291456, 0.1221, 0.3293, 3.445, 0.034),
    (16, 8388608, 0.1319, 0.3472, 0.115, 0.067),
    (16, 10485760, 0.1561, 0.3784, 3.139, 0.030),
    (16, 12582912, 0.1691, 0.3952, 0.019, 0.035),
    (16, 16777216, 0.2057, 0.4416, 0.189, 0.044),
    (16, 20971520, 0.2260, 0.4705, 0.023, 0.020),
    (16, 25165824, 0.2627, 0.5210, 0.011, 0.017),
    (16, 33554432, 0.3369, 0.6025, 0.007, 0.011),
    (16, 50331648, 0.4868, 0.7163, 0.229, 0.020),
    (16, 67108864, 0.6331, 0.8495, 0.058, 0.020),
    (16, 100663296, 0.9331, 1.1575, 0.031, 0.006),
    (16, 134217728, 1.2290, 1.4744, 0.011, 0.022),
    (16, 201326592, 1.8371, 2.0807, 0.002, 0.019),
    (16, 268435456, 2.4633, 2.7017, 0.011, 0.041),
]

STYLES = ("burst", "percall")

#: The plateau each style sits at before the transfer takes over, read straight off the
#: table as the median of every size at or below 4 MiB, by (style, world). The host
#: running ahead sits at 0.12 ms at both worlds; the host watching each call sits at
#: twice that. The factor is the host exposure docs/09 measured independently at 129
#: against 255 microseconds, reproduced here without being looked for -- and note that
#: the burst plateau barely moves from world 8 to world 16, which is the same flatness
#: across the node boundary that ALPHA16_NOTE is about.
PLATEAU_MS = {("burst", 8): 0.1217, ("burst", 16): 0.1240,
              ("percall", 8): 0.2440, ("percall", 16): 0.2933}


def plateau(style: str, world: int, max_bytes: int = 4 * 1024 * 1024) -> float:
    """Median time over the sizes small enough that the transfer has not taken over."""
    i = 2 if style == "burst" else 3
    vals = sorted(r[i] for r in MEASURED if r[0] == world and r[1] <= max_bytes)
    return vals[len(vals) // 2]

#: What this says about a shipped constant, recorded and not acted on.
#:
#: Fitting the burst data under the rule it turns out to want gives alpha(8) = 0.114
#: and alpha(16) = 0.120. The shipped table carries 0.111 and 0.157. The world-8 entry
#: is confirmed to three digits; the world-16 entry is 30% high, and the 41% step the
#: table puts between the two worlds is not there -- alpha barely moves crossing to a
#: second node, which is what a bandwidth-flat supernode should do and exactly what the
#: independent call-count scan reported (0.129 to 0.134) when docs/05 could not
#: adjudicate it.
#:
#: Three benchmarks now agree the step is not real and only the table says it is. The
#: constant still does not move here, because changing a calibration constant on the
#: strength of a benchmark it was not fitted against is the mixing error calibrate.py
#: warns about, and because the corpus that would be greened by it (D) and the corpus
#: that would be reddened (C) are the same pair the repository already refuses to
#: choose between. What has changed is that this is no longer a disagreement between
#: two readings: it is two readings against one table entry.
ALPHA16_NOTE = ("alpha(16) = 0.157 is contradicted by two independent benchmarks, "
                "both of which put it near alpha(8); recorded, not changed")


def rows(style: str, max_spread: float = 0.30):
    """(world, bytes, ms) for one timing style, dropping repeats that disagree.

    The burst measurement is noisy at small sizes, where a single call is a few tens of
    microseconds and the timer is resolving a plateau rather than a transfer. The
    verdict does not depend on the cut -- ``compare_styles`` checks that -- but a fit
    should not be asked to chase points whose own repeats span a factor of five.
    """
    if style not in STYLES:
        raise ValueError("style must be one of %s" % (STYLES,))
    i = 2 if style == "burst" else 3
    s = 4 if style == "burst" else 5
    return [(w, float(b), r[i]) for r in MEASURED
            for w, b in [(r[0], r[1])] if r[s] <= max_spread]


def compare_styles(max_spread: float = 0.30, exponents=(1.0, 2.0)) -> dict:
    """Fit both rules to both styles. Returns {style: {rule label: fit}}.

    Delegates to ``fit.compare_forms``, so this is scored by the same code that scored
    the shipped corpora and nothing here is a special case.
    """
    from .fit import compare_forms
    return {st: compare_forms(rows(st, max_spread), exponents=exponents)
            for st in STYLES}


def preferred_rule(style: str, max_spread: float = 0.30) -> str:
    res = compare_styles(max_spread)[style]
    return min(res, key=lambda k: res[k]["median"])


def main() -> None:
    print("Both timing styles, same machine, same worlds, same sizes, same session.")
    print("46 sizes from 64 KiB to 256 MiB at worlds 8 (one node) and 16 (two nodes).")
    print()
    res = compare_styles()
    print("%-9s %-20s %9s %10s %9s %8s" %
          ("style", "rule", "median", "beta GB/s", "alpha(8)", "alpha(16)"))
    for st in STYLES:
        for label, r in res[st].items():
            flag = "  <- above the %.1f GB/s ceiling" % PHYSICAL_CEILING_GBPS \
                if r["beta_inf"] > PHYSICAL_CEILING_GBPS else ""
            print("%-9s %-20s %8.2f%% %10.1f %9.4f %8.4f%s" %
                  (st, label, 100 * r["median"], r["beta_inf"],
                   r["alpha"][8], r["alpha"][16], flag))
        print("   -> %s wants %s" % (st, preferred_rule(st)))
    print()
    print("The rule is a property of the host regime, not of the machine. Running")
    print("ahead, the next call is set up while the previous one still moves bytes, so")
    print("the fixed cost hides behind the transfer. Watching each call, the two")
    print("serialise and add. An MoE dispatch cannot be issued before its per-peer")
    print("counts reach the host, so it is in the second regime -- which is the rule")
    print("sim/core.py ships.")
    print()
    print("Note on a shipped constant: %s." % ALPHA16_NOTE)


if __name__ == "__main__":
    main()
