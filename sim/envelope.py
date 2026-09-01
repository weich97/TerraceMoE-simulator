# -*- coding: utf-8 -*-
"""What size of model a cluster runs well, computed rather than assumed.

## Why a capacity target cannot be a free input

Searching architectures at a capacity someone picked reports the cheapest way to deliver
that capacity, which is useful only if the capacity was the right one. It usually is not
free to choose. A cluster has an upper edge, where the parameters stop fitting, and a
lower edge, where there is too little work per accelerator to cover the communication and
utilisation falls away. Between them is the band the cluster is good at.

The upper edge is arithmetic. The lower edge is what this repository's cost model is for,
so it is computed here rather than taken from anyone's rule of thumb:

    utilisation ceiling = gemm efficiency  x  compute / (compute + communication)

with communication from ``core``, which passes the Tier-1 gate at 4.1 percent median error, and compute
from ``compute``'s measured roofline. The product is a ceiling because it assumes nothing
overlaps. Real overlap raises it, by an amount the Tier-2 gate says this repository cannot
predict, so the ceiling is reported as a ceiling and the band it implies is conservative
at the lower edge in the direction that matters: a shape this says is communication bound
is communication bound at best.

## Scope

Only the mixture-of-experts layers are priced. Attention, embeddings and the dense layers
are absent, so the number below is not the MFU a training run would report; it is the
utilisation ceiling of the part of the step this repository models. Comparing it against a
published end-to-end MFU compares two different quantities, and the published figure will
be the lower of the two whenever attention is a large share of the step.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .codesign import (UNMEASURED, Machine, MemoryProfile, MoEArch,
                       dispatch_breakdown, expert_matmul, residency)


@dataclass(frozen=True)
class Point:
    arch: MoEArch
    total_params: float
    active_params: float
    params_per_accel: float
    fits: bool
    resident_gib_per_accel: float
    compute_ms: float
    comm_ms: float
    strategy: str
    gemm_efficiency: float
    compute_share: float
    utilisation_ceiling: float
    rows_per_expert: int
    expert_bound_by: str


def total_params(arch: MoEArch) -> float:
    """Routed and shared expert parameters. Attention and embeddings are not counted."""
    return ((arch.n_experts + arch.n_shared_experts)
            * arch.n_moe_layers * arch.params_per_expert())


def active_params(arch: MoEArch) -> float:
    return ((arch.k + arch.n_shared_experts)
            * arch.n_moe_layers * arch.params_per_expert())


def evaluate(arch: MoEArch, machine: Machine,
             mem: MemoryProfile = UNMEASURED) -> Point:
    from .compute import PEAK_TFLOPS

    ep = machine.fabric.ep
    mm = expert_matmul(arch, machine, ep)
    d = dispatch_breakdown(arch, machine)

    two_ok = d["domain_condition"] and d["fanout_condition"]
    if two_ok and d["two_hop_ms"] < d["one_hop_ms"]:
        comm, strat = d["two_hop_ms"], "two-hop"
    else:
        comm, strat = d["one_hop_ms"], "one-hop"

    # dispatch and combine both cross the fabric, and both are paid every layer
    comm *= 2.0

    tot = total_params(arch)
    res = residency(arch, machine, ep, mem)

    peak = machine.peak_tflops
    achieved = (mm["flops"] / (mm["ms_compute_bound"] * 1e-3) / 1e12
                if mm["ms_compute_bound"] > 0 else 0.0)
    eff = achieved / peak if peak else 0.0

    share = mm["ms"] / (mm["ms"] + comm) if (mm["ms"] + comm) > 0 else 0.0

    return Point(
        arch=arch, total_params=tot, active_params=active_params(arch),
        params_per_accel=tot / ep,
        fits=res["fits"],
        resident_gib_per_accel=res["total_bytes"] / 2 ** 30,
        compute_ms=mm["ms"], comm_ms=comm, strategy=strat,
        gemm_efficiency=eff, compute_share=share,
        utilisation_ceiling=eff * share,
        rows_per_expert=mm["rows_per_expert"],
        expert_bound_by=mm["bound_by"])


def scale_family(base: MoEArch, factors) -> list:
    """The same shape at several sizes, grown by expert count alone.

    Growing the expert count leaves every other term of the architecture alone, so the
    family isolates model size. It also raises total parameters without raising active
    parameters, which is the direction sparsity exists to exploit.
    """
    out = []
    for f in factors:
        E = int(round(base.n_experts * f))
        if E < base.k:
            continue
        out.append(replace(base, name="%s-E%d" % (base.name, E), n_experts=E))
    return out


def envelope(base: MoEArch, machine: Machine,
             factors=(0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
             floor: float = 0.5, mem: MemoryProfile = UNMEASURED) -> dict:
    """The band of model sizes this cluster runs well, and both edges with reasons.

    floor is the compute share below which the shape is called communication bound. It is
    a reporting choice, not a measurement, and the returned band moves with it.
    """
    pts = [evaluate(a, machine, mem) for a in scale_family(base, factors)]
    ok = [p for p in pts if p.fits and p.compute_share >= floor]
    return {
        "points": pts,
        "band": (min((p.total_params for p in ok), default=None),
                 max((p.total_params for p in ok), default=None)),
        "upper_edge": "expert residency exceeds usable memory per accelerator",
        "lower_edge": ("compute share falls below %.0f%%, so the moe layer is "
                       "communication bound whatever overlaps" % (100 * floor)),
        "floor": floor,
        "accelerators": machine.fabric.ep,
        "caveat": ("mixture-of-experts layers only; attention and embeddings are not "
                   "priced, so this is not an end-to-end MFU"),
        "residency_measured": mem.measured,
    }


def table(env: dict) -> str:
    lines = ["%-9s %12s %12s %10s %9s %9s %9s %-9s"
             % ("experts", "total P", "P/accel", "rows/exp", "compute", "comm",
                "ceiling", "state"),
             "-" * 88]
    for p in env["points"]:
        state = ("does not fit" if not p.fits
                 else "comm bound" if p.compute_share < env["floor"] else "ok")
        lines.append("%-9d %12.3g %12.3g %10d %9.3f %9.3f %8.1f%% %-9s"
                     % (p.arch.n_experts, p.total_params, p.params_per_accel,
                        p.rows_per_expert, p.compute_ms, p.comm_ms,
                        100 * p.utilisation_ceiling, state))
    lo, hi = env["band"]
    lines.append("")
    lines.append("band on %d accelerators: %s to %s total expert parameters"
                 % (env["accelerators"],
                    ("%.3g" % lo) if lo else "none", ("%.3g" % hi) if hi else "none"))
    lines.append("upper edge: " + env["upper_edge"])
    lines.append("lower edge: " + env["lower_edge"])
    lines.append(env["caveat"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The architecture quantity the machine actually constrains
# ---------------------------------------------------------------------------

def first_order_threshold(machine) -> float:
    """The expert MLP width at which compute and slow-side traffic balance, to leading
    order.

    Per mixture-of-experts layer and rank, the expert matmul is ``6 * T * k * H * d``
    flops and the dispatch moves ``T * k * H * bytes`` across the slow level. Token count,
    top-k and hidden width appear in both and cancel, leaving

        compute / communication = 3 * d_expert * beta_slow / flops_per_second

    so the width alone decides which side an architecture sits on, and the machine enters
    only through its compute-to-network ratio. Expert count does not appear at all: making
    a model larger by adding experts changes what fits, not what binds.

    The full model puts the crossover above this, because the fixed cost of a collective
    and the arrival chain are paid by communication and are not in the leading term. Use
    ``compute_bound_threshold`` for the number to act on and this one for the reading.
    """
    return (machine.peak_tflops * 1e12
            / (3.0 * machine.fabric.beta_inter_gbps * 1e9))


def compute_bound_threshold(template: MoEArch, machine: Machine,
                            strategy: str = "best",
                            lo: int = 128, hi: int = 262144) -> int:
    """The smallest expert MLP width at which the expert matmul covers the dispatch.

    strategy is "one-hop", "two-hop", or "best". Everything in ``template`` except
    ``d_expert`` is held, and the search is a bisection on width, so the returned number
    is specific to that architecture and machine and not a universal constant.
    """
    def bound(d):
        a = replace(template, d_expert=int(d))
        mm = expert_matmul(a, machine, machine.fabric.ep)
        br = dispatch_breakdown(a, machine)
        if strategy == "one-hop":
            comm = br["one_hop_ms"]
        elif strategy == "two-hop":
            comm = br["two_hop_ms"]
        else:
            two_ok = br["domain_condition"] and br["fanout_condition"]
            comm = (min(br["one_hop_ms"], br["two_hop_ms"]) if two_ok
                    else br["one_hop_ms"])
        return mm["ms"] >= 2.0 * comm

    if bound(lo):
        return lo
    if not bound(hi):
        return -1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if bound(mid):
            hi = mid
        else:
            lo = mid
    return hi
