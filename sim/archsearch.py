# -*- coding: utf-8 -*-
"""Search the MoE architecture space at fixed capacity, ranked by what a machine costs.

## The question this answers

An unconstrained search over architectures reports that the smallest model is the
fastest, which is true and useless. The co-design question is the constrained one: given
a capacity target, which shape delivers it most cheaply on this machine.

Two quantities pin the target. Total routed expert parameters set what the model knows;
active parameters per token set what it spends per token, and stand in here for quality,
which nothing in this repository measures. With

    P = n_mats * hidden * d_expert          parameters in one expert, one layer
    total  = E * P * layers
    active = k * P * layers

holding both fixed leaves a one-parameter family. Scaling granularity by g takes
``E -> E*g``, ``d_expert -> d_expert/g`` and ``k -> k*g``, and both products are
unchanged. That family is the fine-grained expert decision, and it is exactly the axis
DeepSeek-V2 moved along; here it has a cost attached to each point.

The group cap M is free on top of it: any divisor of k gives a legal routing constraint,
and it moves the deduplication quota q = k/M, which is what the dispatch cost turns on.
M does not appear in either capacity product, so it is free of the target by
construction. It is not free of quality, and this module does not pretend otherwise; see
``docs/12-m-quality-experiment.md``.

## What the ranking is and is not

The Tier-2 gate fails, so communication and compute may not be added into a step time.
Every candidate therefore carries its components separately, and the composite offered
for sorting is the no-overlap sum, which ``compute.comm_share_upper_bound`` already
establishes as an upper bound rather than an estimate. Ranking by an upper bound is
defensible when the bound is loose in the same direction for every candidate; it is not
a step-time prediction and is labeled on every row.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .codesign import (UNMEASURED, Machine, MemoryProfile, MoEArch,
                       dispatch_breakdown, expert_matmul, min_ep_for_memory)


@dataclass(frozen=True)
class Candidate:
    arch: MoEArch
    granularity: float
    one_hop_ms: float
    two_hop_ms: float
    best_dispatch_ms: float
    strategy: str
    expert_ms: float
    expert_bound_by: str
    min_ep: int
    feasible: bool
    reason: str = ""

    @property
    def no_overlap_ms(self) -> float:
        """Dispatch plus expert with nothing hidden. An upper bound, not a step time."""
        return self.best_dispatch_ms + self.expert_ms


def _divisors(n: int):
    return [d for d in range(1, n + 1) if n % d == 0]


def granularity_family(base: MoEArch, factors=(0.5, 1, 2, 4, 8)):
    """Architectures with the same total and active parameter counts as ``base``."""
    out = []
    for g in factors:
        E = int(round(base.n_experts * g))
        d = int(round(base.d_expert / g))
        k = int(round(base.k * g))
        if E < 1 or d < 1 or k < 1 or k > E:
            continue
        # the two invariants must survive the rounding, or the point is not comparable
        if abs(E * d - base.n_experts * base.d_expert) > 0.01 * base.n_experts * base.d_expert:
            continue
        if abs(k * d - base.k * base.d_expert) > 0.01 * base.k * base.d_expert:
            continue
        out.append((g, replace(base, name="%s-g%g" % (base.name, g),
                               n_experts=E, d_expert=d, k=k)))
    return out


def evaluate(arch: MoEArch, machine: Machine,
             mem: MemoryProfile = UNMEASURED) -> Candidate:
    ep_floor = min_ep_for_memory(arch, machine, mem)["min_ep"]
    feasible = 0 < ep_floor <= machine.fabric.ep
    reason = "" if feasible else ("experts need EP %d, fabric has %d"
                                  % (ep_floor, machine.fabric.ep))

    mm = expert_matmul(arch, machine, machine.fabric.ep)
    d = dispatch_breakdown(arch, machine)

    two_ok = d["domain_condition"] and d["fanout_condition"]
    if two_ok and d["two_hop_ms"] < d["one_hop_ms"]:
        best, strat = d["two_hop_ms"], "two-hop"
    else:
        best, strat = d["one_hop_ms"], "one-hop"
        if not two_ok:
            reason = (reason + "; " if reason else "") + (
                "two-hop does not apply: "
                + ("expert parallelism fits one fast domain" if not d["domain_condition"]
                   else "q = k/M is 1, nothing to deduplicate"))

    return Candidate(arch=arch, granularity=0.0,
                     one_hop_ms=d["one_hop_ms"], two_hop_ms=d["two_hop_ms"],
                     best_dispatch_ms=best, strategy=strat,
                     expert_ms=mm["ms"], expert_bound_by=mm["bound_by"],
                     min_ep=ep_floor, feasible=feasible, reason=reason)


def search(base: MoEArch, machine: Machine,
           granularities=(0.5, 1, 2, 4, 8),
           hiddens=None, mem: MemoryProfile = UNMEASURED) -> list:
    """Every legal shape at ``base``'s capacity, each priced on ``machine``.

    hiddens is swept independently of the capacity family: changing hidden width at fixed
    total parameters means changing d_expert to match, which the family already does, so
    a hidden sweep here holds d_expert and moves capacity. Pass it only when the capacity
    target is meant to move with it, and read the two sweeps separately.
    """
    out = []
    for g, a in granularity_family(base, granularities):
        for M in _divisors(a.k):
            if M > machine.fabric.nodes:
                continue
            cand = evaluate(replace(a, name="%s-M%d" % (a.name, M), M=M),
                            machine, mem)
            out.append(replace(cand, granularity=g))
    if hiddens:
        for H in hiddens:
            for M in _divisors(base.k):
                if M > machine.fabric.nodes:
                    continue
                cand = evaluate(replace(base, name="%s-H%d-M%d" % (base.name, H, M),
                                        hidden=H, M=M), machine, mem)
                out.append(replace(cand, granularity=1.0))
    return out


def rank(cands, key="no_overlap_ms", feasible_only: bool = True) -> list:
    c = [x for x in cands if x.feasible] if feasible_only else list(cands)
    return sorted(c, key=lambda x: getattr(x, key) if key != "no_overlap_ms"
                  else x.no_overlap_ms)


def table(cands, top: int = 12) -> str:
    lines = ["%-6s %-7s %-7s %-4s %-4s %-4s %10s %10s %10s %-9s"
             % ("gran", "experts", "d_exp", "k", "M", "q", "dispatch", "expert",
                "sum(UB)", "strategy"),
             "-" * 88]
    for c in rank(cands)[:top]:
        a = c.arch
        lines.append("%-6g %-7d %-7d %-4d %-4d %-4d %10.3f %10.3f %10.3f %-9s"
                     % (c.granularity, a.n_experts, a.d_expert, a.k, a.M, a.q,
                        c.best_dispatch_ms, c.expert_ms, c.no_overlap_ms, c.strategy))
    lines.append("")
    lines.append("sum(UB) is dispatch plus expert with nothing overlapped, an upper "
                 "bound and not a step time: the step-level gate fails.")
    return "\n".join(lines)
