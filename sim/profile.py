# -*- coding: utf-8 -*-
"""What a machine has to look like for hierarchical MoE communication to pay.

A hierarchy ratio alone does not decide it. Two-hop dispatch trades slow-side bytes
for fast-side bytes, one extra collective, and a local permutation — so a machine
qualifies only if several conditions hold at once, and a single failure sinks the
whole thing regardless of how good the ratio looks.

This module turns the cost model into that checklist. Every condition below is
derived from `sim/core.py` and the calibration, not asserted: point it at a machine
description and it tells you which conditions pass, which fails first, and by how
much. It is the form of the answer someone deciding whether to adopt actually needs.

## The conditions, and where each comes from

1. **A hierarchy exists, past the implementation-adjusted breakeven.**
   The byte account alone gives r_be = (1−1/R)·q/(q−1) — 1.31 at R=8, q=3. But the
   arrival chain moves the real threshold: 3.87 with a PyTorch op chain, 1.45 with
   a fused kernel, 1.07 at zero overhead. So the honest threshold depends on the
   software as much as the fabric, and this check reports it against the tier you
   declare.

2. **Expert parallelism spans more than one fast domain.**
   If every expert fits inside one NVLink/HCCS domain, there is no slow hop to
   deduplicate and the method has nothing to do — the right move is to keep EP
   inside the domain, not to optimise a hop that need not exist. Rack-scale
   domains make this failure mode common, not rare.

3. **q = k/M ≥ 2.**
   The saving is (q−1)/q of the slow-side payload. At q=1 (M=k) each token already
   sends one row per group and there is nothing to deduplicate.

4. **The fixed-cost account does not eat the gain.**
   Two hops pay α(N_g) + α(R) where one hop pays α(EP). On a machine where α barely
   grows with world size, that swap is a loss before any bytes move. Platform A is
   exactly this case — α(16)+α(8) ≈ α(128) — and it is a machine property, not a
   topology property, so it must be measured per machine.

5. **Messages are large enough to be bandwidth-bound.**
   Below the half-performance size the collective is latency- and
   saturation-dominated, and byte savings do not convert into time. Per-peer bytes
   should sit comfortably above x_half.

6. **The arrival chain's price, reported separately (advisory).**
   Condition 1 already prices it — that is the whole gap between the byte breakeven
   and the effective one — so this is not a second gate; making it one would let the
   checklist contradict the model, which it must never do. It gets its own line
   because it is the only item entirely under the adopter's control, and fusing it
   moves the threshold further than most topology changes would.

**The checklist and the model must agree.** Conditions are necessary conditions
derived from the same cost model that produces the ratio; a machine the checklist
clears is a machine the model scores above 1. `tests/test_sim.py` pins that, because
a checklist that can disagree with its own model is worse than no checklist.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core import MoEGeometry, one_hop_call, two_hop_call


@dataclass
class Condition:
    name: str
    passed: bool
    detail: str
    margin: float = float("nan")     # >0 = pass, magnitude = headroom


def byte_breakeven(R: int, q: int) -> float:
    """r_be = (1-1/R)*q/(q-1): the pure byte account, no implementation cost."""
    if q <= 1:
        return float("inf")
    return (1.0 - 1.0 / R) * q / (q - 1)


def check(ratio: float, R: int, k: int, M: int, ep: int,
          tokens_per_rank: int, hidden: int, chain_us_per_row: float,
          alpha_of_world=None, x_half_bytes: float = 54 * 1024,
          bytes_per_elem: int = 2) -> list:
    """Run the checklist for one machine + geometry. Returns Conditions in order.

    alpha_of_world: callable world -> ms. Supply the machine's own measured curve;
    without it, condition 4 is reported as unknown rather than guessed.
    """
    out = []
    q = k // M if M else 0
    n_groups = max(ep // R, 1)

    # 3 first: it gates the meaning of everything else
    out.append(Condition(
        "q = k/M >= 2", q >= 2,
        "q=%d (k=%d, M=%d); the slow-side saving is (q-1)/q = %.0f%%"
        % (q, k, M, 100 * (q - 1) / q if q else 0),
        margin=q - 2))

    # The gate that actually decides, and the only one allowed to disagree with
    # nothing: the effective breakeven is computed from the same cost model that
    # produces the ratio, with this machine's own arrival-chain cost folded in.
    # The pure byte breakeven is reported beside it because the gap between the two
    # *is* the implementation's price.
    from .uncertainty import breakeven_ratio
    r_be_bytes = byte_breakeven(R, q) if q >= 2 else float("inf")
    r_be_eff = (breakeven_ratio(chain_us_per_row, q=q, tok=tokens_per_rank)
                if q >= 2 else float("inf"))
    out.append(Condition(
        "hierarchy ratio beats the effective breakeven", ratio >= r_be_eff,
        "ratio %.2f vs effective breakeven %.2f (bytes alone would need %.2f; "
        "the gap is what the arrival chain costs)" % (ratio, r_be_eff, r_be_bytes),
        margin=ratio - r_be_eff))

    out.append(Condition(
        "EP spans more than one fast domain", n_groups >= 2,
        "EP=%d over domains of %d cards = %d group(s)%s"
        % (ep, R, n_groups,
           "" if n_groups >= 2 else " -- keep EP inside the domain instead"),
        margin=n_groups - 2))

    per_peer = tokens_per_rank * k * hidden * bytes_per_elem / max(ep, 1)
    out.append(Condition(
        "messages are bandwidth-bound", per_peer >= 4 * x_half_bytes,
        "one-hop per-peer %.0f KiB vs half-performance size %.0f KiB (want >=4x)"
        % (per_peer / 1024, x_half_bytes / 1024),
        margin=per_peer / x_half_bytes - 4))

    if alpha_of_world is None:
        out.append(Condition(
            "fixed cost: two small collectives beat one big one", False,
            "UNKNOWN -- supply the machine's measured alpha(world); this is a "
            "machine property and cannot be assumed"))
    else:
        two = alpha_of_world(n_groups) + alpha_of_world(R)
        one = alpha_of_world(ep)
        out.append(Condition(
            "fixed cost: two small collectives beat one big one", two < one,
            "alpha(%d)+alpha(%d) = %.3f ms vs alpha(%d) = %.3f ms"
            % (n_groups, R, two, ep, one),
            margin=one - two))

    # Advisory, not a gate: the chain's cost is already priced into the effective
    # breakeven above. This line says how much of that threshold is self-inflicted,
    # because it is the one item entirely under the adopter's control.
    fused = chain_us_per_row <= 0.02
    r_be_fused = breakeven_ratio(0.012, q=q, tok=tokens_per_rank) if q >= 2 else float("inf")
    out.append(Condition(
        "arrival chain cost (advisory, already priced above)", True,
        "%.4f us/row -- %s" % (
            chain_us_per_row,
            "fused tier; the threshold is as low as this model goes" if fused else
            "an op chain: it raises the breakeven from %.2f (fused) to %.2f, so "
            "fusing it is worth more than most topology changes"
            % (r_be_fused, r_be_eff)),
        margin=0.02 - chain_us_per_row))
    return out


# Per-call launch cost, measured on the paths we have instrumented:
#   device-initiated kernel put   36-57 us   (one-sided bench floor, 7 serial puts)
#   host-initiated point-to-point 257-268 us (PyTorch cross-device copy_)
#   host readback of split sizes  44 us      (already a separate term in core.py)
LAUNCH_DEVICE_INITIATED_MS = (0.036, 0.057)
LAUNCH_HOST_P2P_MS = (0.257, 0.268)

# The call-count scan that docs/09 asked for has now been run: N back-to-back tiny
# collectives at fixed world, N over 1..256, four payloads from 256 B to 16 KiB,
# two ops, seven repetitions, worlds 8 and 16. It gives the per-call fixed cost
# directly instead of leaving it inside alpha, and it turns out to depend less on
# the world than on **who is waiting**.
#
#   deep queue    the host runs ahead and the device never idles. The per-call
#                 cost stops falling past N ~ 16 and holds flat to N = 256, so
#                 collectives do not pipeline: two cost twice one. This is the
#                 regime the shipped alpha corresponds to.
#   host exposed  the host observes each call before issuing the next, which is
#                 what happens wherever a device value has to come back (a splits
#                 readback, a dynamic shape, Python control flow). Per-call cost
#                 roughly doubles.
#
# Payload independence holds across the whole scan: 256 B and 16 KiB give the same
# per-call cost to within the run-to-run spread, as they must, since the wire term
# at 16 KiB is 0.12 us.
# World 8 is the mean of two independent single-node scans (133 and 124 us) and a
# third that pushed N to 1024 on one of them and landed at 117, so the whole set
# spans 117-133 for a nominal 128. Worlds 2 and 4 are two nodes each. World 16 spans
# both nodes and has one scan; a later rerun at three payloads put it at 134.
#
# The worlds 2 and 4 entries are the instrument's own calibration, and they are the
# reason the world-8 disagreement is worth taking seriously: against a table of
# 90 / 97 / 111 / 157 the scan reads 87 / 98 / 129 / 134, so it reproduces the two
# worlds where nothing is in dispute to within 3% and then departs at 8 and 16.
#
# It departs in *opposite directions*, and that is what keeps the table where it is.
# Setting alpha(8) to the measured 129 makes every validation corpus pass, including
# the world-8 one that currently fails. Setting alpha(16) to the measured 134 --
# the same scan, the same day, the same convention -- breaks the world-16 corpus,
# from 8.0% to 15.5%. Adopting the half of a measurement that greens a gate and
# discarding the half that reddens another is choosing by outcome, so neither is
# adopted. What the scan does establish is the *shape*: alpha climbs while a node
# fills, 87 to 129 from world 2 to 8, and then barely moves crossing to a second
# node, 129 to 134, which is what a bandwidth-flat supernode should do and is not
# what the tabulated 111 to 157 says.
PER_CALL_DEEP_QUEUE_MS = {2: 0.087, 4: 0.098, 8: 0.129, 16: 0.134}  # a2a, plateau N>=64
PER_CALL_DEEP_QUEUE_RANGE_MS = {8: (0.117, 0.133)}  # three runs, two nodes
PER_CALL_HOST_EXPOSED_MS = {8: 0.255, 16: 0.283}    # a2a, host waits per call (w8: 247 and 263)
PER_CALL_SPREAD = 0.15                              # across payloads, same session
HOST_SUBMIT_MS = {2: 0.039, 4: 0.043, 8: 0.058, 16: 0.076}   # host CPU per enqueue

# The gap between the two regimes is what a device-initiated stack removes. It is
# world-independent to within the spread, 126 us at world 8 and 149 at world 16,
# of which roughly half is submission and the rest is completion detection.
HOST_EXPOSURE_MS = 0.130

# What this does and does not settle for the shipped alpha table. Read against
# ALPHA_PTS, the deep-queue floor lands within the documented 20% run-to-run drift
# of both entries it can reach -- 129 us measured against alpha(8) = 111, and 134
# against alpha(16) = 157 -- from a different benchmark months later. It does not
# confirm the *step* between them: alpha rises 41% from world 8 to 16 where the
# direct measurement rises 4%. Nothing here
# changes the table; the two conclusions are that alpha is corroborated at the
# level and that it belongs to the deep-queue regime.
LAUNCH_UPPER_BOUND_MS = 0.111    # retained: pre-measurement bound, still the value
                                 # launch_sensitivity's old sweep was quoted against


def launch_sensitivity(geom, chain_us_per_row: float,
                       deltas_ms=(0.0, 0.04, 0.08, 0.111, 0.130, 0.16, 0.25)) -> list:
    """How much the launch split moves the verdict.

    Two-hop issues one more collective than one-hop, so charging an extra fixed
    cost per call shifts the comparison by exactly one of them. The sweep was
    written before the cost was measured; it now has a measured point in it.

    At the measured host exposure of HOST_EXPOSURE_MS = 0.130, the breakeven on
    the reference geometry moves from 3.87 to 4.04 -- under 5%, because the
    arrival chain dominates everything else on this machine. Remove the chain and
    the same 0.130 moves the breakeven from 1.07 to 1.24, which is 16%. So the
    ordering is unchanged and worth stating plainly: fuse the chain first, and
    only then does host exposure become the next thing worth paying for.

    The sweep still does **not** answer whether to adopt a device-initiated stack.
    Removing the host from the critical path also changes overlap, and overlap is
    the Tier-2 gap this repository cannot price.
    """
    from .calibrate import synthetic
    from .core import one_hop_call, two_hop_call

    out = []
    for d in deltas_ms:
        lo, hi = 1.0, 32.0

        def ratio(r):
            c = synthetic(r, chain_us_per_row=chain_us_per_row)
            return one_hop_call(c, geom) / (two_hop_call(c, geom) + d)

        if ratio(lo) >= 1.0:
            be = lo
        elif ratio(hi) < 1.0:
            be = hi
        else:
            for _ in range(40):
                mid = (lo + hi) / 2.0
                if ratio(mid) >= 1.0:
                    hi = mid
                else:
                    lo = mid
            be = hi
        out.append({"extra_launch_ms": d, "breakeven": be,
                    "ratio_at_3.2": ratio(3.2), "ratio_at_8": ratio(8.0)})
    return out


def verdict(conditions) -> dict:
    failed = [c for c in conditions if not c.passed]
    return {"qualifies": not failed,
            "n_failed": len(failed),
            "first_failure": failed[0].name if failed else None}


def profile_from_spec(cluster, geom: MoEGeometry, ratio: float = None) -> dict:
    """Checklist for a ClusterSpec + geometry, plus the model's own verdict.

    The model's ratio (one-hop time / two-hop time) is reported alongside the
    checklist as a cross-check: the checklist is a set of necessary conditions, the
    ratio is what the full cost model says once all of them interact.
    """
    r = ratio if ratio is not None else cluster.ratio()
    conds = check(ratio=r, R=geom.R, k=geom.k, M=geom.M, ep=geom.ep,
                  tokens_per_rank=geom.tokens_per_rank, hidden=geom.H,
                  chain_us_per_row=cluster.chain_us_per_row,
                  alpha_of_world=cluster.flat.alpha_ms)
    return {"conditions": conds, "verdict": verdict(conds),
            "model_ratio": one_hop_call(cluster, geom) / two_hop_call(cluster, geom)}


def main() -> None:
    from .calibrate import flat_supernode, synthetic
    from .sweep import CHAIN_SCENARIOS

    g = MoEGeometry(name="op", n_groups=16, R=8, k=6, M=2, seq=4096, mbs=1,
                    gbs=16 * 8 * 4096)
    cases = [("platform A, as measured", flat_supernode(), None),
             ("NVLink+IB class, PyTorch chain",
              synthetic(9.0, chain_us_per_row=CHAIN_SCENARIOS[0][1]), 9.0),
             ("NVLink+IB class, fused chain",
              synthetic(9.0, chain_us_per_row=CHAIN_SCENARIOS[1][1]), 9.0)]
    for label, spec, ratio in cases:
        r = profile_from_spec(spec, g, ratio)
        print("=== %s ===" % label)
        for c in r["conditions"]:
            print("  [%s] %-48s %s" % ("PASS" if c.passed else "FAIL",
                                       c.name, c.detail))
        v = r["verdict"]
        print("  -> %s;  full model says one-hop/two-hop = %.2f\n"
              % ("qualifies" if v["qualifies"]
                 else "does not qualify (first failure: %s)" % v["first_failure"],
                 r["model_ratio"]))
    print("The checklist is necessary conditions; the model ratio is what they add")
    print("up to. When they disagree, read the condition that fails -- it names the")
    print("reason, which a single ratio never does.")


if __name__ == "__main__":
    main()
