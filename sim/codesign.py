# -*- coding: utf-8 -*-
"""Architecture and machine as inputs, a step breakdown and a binding resource as output.

## What this is for

The rest of this repository answers one question: on a given machine, does a hierarchical
dispatch beat a flat one. That question takes the model architecture as given. Turned
around, the same cost terms answer a more useful question, which is which architectures a
given machine is good at running. Expert count, expert width and hidden width are chosen
by whoever designs the model, and each of them moves a different term:

  hidden width      moves the payload of every collective, and the gather inside the
                    arrival chain, and the arithmetic intensity of the expert matmul
  expert width      moves the expert matmul and the memory an expert occupies
  expert count      moves how many tokens reach each expert, which is what decides
                    whether the expert matmul is compute bound or memory bound
  top-k and M       move the traffic and the deduplication quota q = k/M
  expert parallelism moves everything, and is itself bounded below by memory capacity

The machine side is symmetric: intra-node and inter-node bandwidth, the fixed cost of a
collective, cards per node, memory bandwidth and memory capacity. Both sides are inputs
here, neither is a constant.

## What is validated and what is not

The communication terms compose from ``core.one_hop_call`` and ``core.two_hop_call``,
which pass the Tier-1 gate at 4.1 percent median error. The arrival chain comes from ``machine.py``,
which reproduces its own measured sweep to 4.2 percent. The compute-bound branch of the
expert matmul comes from ``compute.py``'s measured roofline and its eight measured
non-square shapes.

The memory-bound branch below is NOT measured. It is a roofline ceiling computed from
memory bandwidth, and it binds only where tokens per expert fall below the machine's
balance point, which is outside every shape ``compute.py`` measured. It is here because
leaving it out silently asserts that the expert matmul is always compute bound, which is
false for fine-grained experts at small micro-batches, and an unstated assumption is
worse than a labeled model. Every result that touches it is marked.

Step-level composition remains behind the Tier-2 gate, which fails. Nothing here returns
a step time; the breakdown returns per-term costs and says plainly that they do not add.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .compute import INDEX_RULE, INDEX_RULE_ERROR, PEAK_TFLOPS, efficiency_bracket
from .core import MoEGeometry, one_hop_call, two_hop_call
from .machine import Accelerator, ArrivalChain, PYTORCH_CHAIN


# ---------------------------------------------------------------------------
# The two sides
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MoEArch:
    """The architecture knobs that move a system cost.

    d_expert is the MLP width of one expert, the quantity papers call the MoE
    intermediate size. n_mats is 3 for the gated feed-forward every current model uses.
    """
    name: str
    hidden: int
    d_expert: int
    n_experts: int
    k: int
    M: int = 1
    n_shared_experts: int = 0
    n_moe_layers: int = 1
    seq: int = 4096
    mbs: int = 1
    gbs: int = 512
    n_mats: int = 3
    bytes_per_elem: int = 2
    optimizer_bytes_per_param: float = 6.0

    @property
    def q(self) -> int:
        if self.k % self.M:
            raise ValueError("%s: k=%d is not divisible by M=%d, so the per-group quota "
                             "k/M is not an integer" % (self.name, self.k, self.M))
        return self.k // self.M

    @property
    def tokens_per_rank(self) -> int:
        return self.seq * self.mbs

    def params_per_expert(self) -> int:
        return self.n_mats * self.hidden * self.d_expert

    def expert_bytes_all_layers(self, with_optimizer: bool = True) -> float:
        per = self.bytes_per_elem + (self.optimizer_bytes_per_param
                                     if with_optimizer else 0.0)
        experts = self.n_experts + self.n_shared_experts
        return experts * self.n_moe_layers * self.params_per_expert() * per


@dataclass(frozen=True)
class Fabric:
    """The machine's communication side, in the units a datasheet uses.

    alpha_ms is a curve over world size, not a scalar. A single fixed cost charges hop A
    and hop B the full-fabric figure and so buries two-hop before any byte moves, which
    is the fixed-cost condition of the machine profile being asserted rather than
    evaluated. Supply the measured points; the curve is interpolated between them and
    clamped outside.
    """
    name: str
    cards_per_node: int
    nodes: int
    beta_intra_gbps: float
    beta_inter_gbps: float
    alpha_ms: tuple                 # ((world, ms), ...), ascending in world
    x_half_bytes: float = 0.0       # per-peer message size at half of asymptotic bandwidth

    def alpha_at(self, world: int) -> float:
        pts = sorted(self.alpha_ms)
        if world <= pts[0][0]:
            return pts[0][1]
        if world >= pts[-1][0]:
            return pts[-1][1]
        for (w0, a0), (w1, a1) in zip(pts, pts[1:]):
            if w0 <= world <= w1:
                return a0 + (a1 - a0) * (world - w0) / float(w1 - w0)
        return pts[-1][1]

    def fixed_cost_condition(self) -> dict:
        """Two hops pay two fixed costs where one paid one."""
        two = self.alpha_at(self.nodes) + self.alpha_at(self.cards_per_node)
        one = self.alpha_at(self.ep)
        return {"two_hop_alpha_ms": two, "one_hop_alpha_ms": one,
                "clears": two < one, "margin_ms": one - two}

    @property
    def ep(self) -> int:
        return self.cards_per_node * self.nodes

    @property
    def ratio(self) -> float:
        return self.beta_intra_gbps / self.beta_inter_gbps


@dataclass(frozen=True)
class Machine:
    accel: Accelerator
    fabric: Fabric
    chain: ArrivalChain = PYTORCH_CHAIN

    @property
    def balance_flops_per_byte(self) -> float:
        """Tokens per expert below which the expert matmul is memory bound.

        The expert matmul's arithmetic intensity is the number of rows reaching one
        expert, because the weights are read once and the flops are proportional to the
        rows. So the machine balance point is read directly as a token count.
        """
        return self.peak_tflops * 1e12 / (self.accel.hbm_gbps * 1e9)

    @property
    def peak_tflops(self) -> float:
        """The accelerator's own figure when it has one, else the measured curve's peak.

        compute.py's roofline was measured on platform A. Borrowing its peak for another
        accelerator asserts they compute alike, so an Accelerator that states its own
        peak overrides it and one that does not is flagged by transfers_compute below.
        """
        return self.accel.peak_tflops or PEAK_TFLOPS

    @property
    def transfers_compute(self) -> bool:
        """False when the expert matmul is being priced on a borrowed roofline."""
        return bool(self.accel.peak_tflops)


# ---------------------------------------------------------------------------
# Expert matmul with both roofline branches
# ---------------------------------------------------------------------------

def expert_matmul(arch: MoEArch, machine: Machine, ep: int,
                  backward: bool = False) -> dict:
    """One MoE layer's expert cost on one rank, compute bound and memory bound.

    The memory-bound branch is a ceiling, not a measurement; see the module docstring.
    """
    experts_local = max(arch.n_experts // ep, 1)
    rows_per_expert = max(arch.tokens_per_rank * arch.k // experts_local, 1)

    flops = (2.0 * rows_per_expert * arch.hidden * arch.d_expert
             * arch.n_mats * experts_local * (2.0 if backward else 1.0))

    br = efficiency_bracket(rows_per_expert, arch.hidden, arch.d_expert)
    tflops = br["per_rule"][INDEX_RULE]
    if machine.accel.peak_tflops:
        tflops *= machine.accel.peak_tflops / PEAK_TFLOPS
    ms_compute = flops / (tflops * 1e12) * 1e3

    # Weights are read once per expert per pass whatever the row count; the activations
    # are the smaller term until the rows get large.
    w_bytes = (experts_local * arch.params_per_expert() * arch.bytes_per_elem
               * (2.0 if backward else 1.0))
    a_bytes = (rows_per_expert * experts_local
               * (arch.hidden + 2.0 * arch.d_expert) * arch.bytes_per_elem)
    ms_memory = (w_bytes + a_bytes) / (machine.accel.hbm_gbps * 1e9) * 1e3

    bound = "memory" if ms_memory > ms_compute else "compute"
    return {
        "rows_per_expert": rows_per_expert,
        "experts_local": experts_local,
        "flops": flops,
        "ms_compute_bound": ms_compute,
        "ms_memory_bound": ms_memory,
        "ms": max(ms_compute, ms_memory),
        "bound_by": bound,
        "arithmetic_intensity": rows_per_expert,
        "machine_balance": machine.balance_flops_per_byte,
        "compute_measured": (rows_per_expert >= 1536
                             and not machine.accel.peak_tflops),
        "roofline_borrowed": not machine.transfers_compute,
        "tflops_rule": INDEX_RULE,
        "tflops_worst_error": INDEX_RULE_ERROR["worst"],
    }


# ---------------------------------------------------------------------------
# Memory feasibility, which sets the floor on expert parallelism
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryProfile:
    """The residency terms that cannot be computed from an architecture.

    An earlier version of this module took a single ``reserve_frac`` of 0.35, meaning
    everything that is not an expert weight. That number was invented, and worse, it
    conflated two different kinds of quantity. Optimizer state and gradients follow from
    the parameter count and the optimizer, and are arithmetic. Activation residency
    follows from the recompute policy and can be computed once its per-token constant is
    known. Allocator fragmentation and framework and communication buffers cannot be
    derived at all: they are properties of a runtime on a device and have to be read off
    a running job.

    Only the last two are inputs here, and a profile whose ``measured`` flag is false
    marks every result computed from it, so an assumption cannot be mistaken for a
    reading further downstream.
    """
    name: str
    activation_bytes_per_token_layer: float
    overhead_frac: float
    recompute: str = "unknown"
    measured: bool = False
    source: str = ""


#: Placeholder. The two numbers are order-of-magnitude guesses and are here so the model
#: runs before a machine is available, not so anyone can quote them. Replace with a
#: profile from docs/11-residency-measurement.md.
UNMEASURED = MemoryProfile(
    name="unmeasured placeholder",
    activation_bytes_per_token_layer=0.0,
    overhead_frac=0.15,
    recompute="assumed full recompute",
    measured=False,
    source="invented; see docs/11-residency-measurement.md for what would replace it")


def residency(arch: MoEArch, machine: Machine, ep: int,
              profile: MemoryProfile = UNMEASURED,
              non_expert_params: int = 0) -> dict:
    """Bytes resident on one accelerator, by term, each labelled with where it came from."""
    expert = arch.expert_bytes_all_layers(with_optimizer=True) / ep
    per_param = arch.bytes_per_elem + arch.optimizer_bytes_per_param
    non_expert = non_expert_params * per_param
    act = (profile.activation_bytes_per_token_layer
           * arch.tokens_per_rank * arch.n_moe_layers)
    cap = machine.accel.hbm_capacity_gb * 1e9
    overhead = cap * profile.overhead_frac
    total = expert + non_expert + act + overhead
    return {
        "expert_bytes": expert,
        "non_expert_bytes": non_expert,
        "activation_bytes": act,
        "overhead_bytes": overhead,
        "total_bytes": total,
        "capacity_bytes": cap,
        "fits": total <= cap,
        "headroom_bytes": cap - total,
        "provenance": {
            "expert_bytes": "computed from the architecture and the optimizer",
            "non_expert_bytes": ("computed" if non_expert_params
                                 else "NOT SUPPLIED: attention and embeddings omitted"),
            "activation_bytes": ("measured: " + profile.source if profile.measured
                                 else "ASSUMED: " + profile.source),
            "overhead_bytes": ("measured: " + profile.source if profile.measured
                               else "ASSUMED: " + profile.source),
        },
        "measured": profile.measured,
        "profile": profile.name,
    }


def min_ep_for_memory(arch: MoEArch, machine: Machine,
                      profile: MemoryProfile = UNMEASURED,
                      non_expert_params: int = 0) -> dict:
    """Smallest power-of-two expert parallelism whose residency fits, and why.

    Returns a dict rather than an integer so the caller cannot use the number without
    also receiving whether the terms behind it were measured.
    """
    ep = 1
    while ep < arch.n_experts:
        r = residency(arch, machine, ep, profile, non_expert_params)
        if r["fits"]:
            return {"min_ep": ep, "residency": r, "measured": r["measured"]}
        ep *= 2
    r = residency(arch, machine, arch.n_experts, profile, non_expert_params)
    return {"min_ep": arch.n_experts if r["fits"] else -1,
            "residency": r, "measured": r["measured"]}


# ---------------------------------------------------------------------------
# The dispatch question, at this architecture on this machine
# ---------------------------------------------------------------------------

def dispatch_breakdown(arch: MoEArch, machine: Machine) -> dict:
    """One-hop against two-hop for one MoE layer's dispatch call.

    Returns the two call times and the terms behind them. This is a communication-level
    quantity: the step-level gate fails, so nothing here is a claim about step time.
    """
    from .calibrate import saturating_beta
    from .core import ClusterSpec, Level

    f = machine.fabric
    ep = f.ep
    rows_hop_b = arch.tokens_per_rank * arch.k
    chain_us = machine.chain.us_per_row(rows_hop_b, arch.hidden, machine.accel)

    ap = sorted(f.alpha_ms)
    cl = ClusterSpec(
        name=f.name, R=f.cards_per_node,
        fast=Level("fast", ap, saturating_beta(f.beta_intra_gbps, f.x_half_bytes)),
        slow=Level("slow", ap, saturating_beta(f.beta_inter_gbps, f.x_half_bytes)),
        flat=Level("flat", ap, saturating_beta(f.beta_inter_gbps, f.x_half_bytes)),
        chain_us_per_row=chain_us)

    g = MoEGeometry(name=arch.name, n_groups=f.nodes, R=f.cards_per_node,
                    k=arch.k, M=arch.M, H=arch.hidden, seq=arch.seq, mbs=arch.mbs,
                    gbs=arch.gbs, moe_layers=arch.n_moe_layers,
                    bytes_per_elem=arch.bytes_per_elem)

    one, two = one_hop_call(cl, g), two_hop_call(cl, g)
    return {
        "ep": ep, "q": arch.q, "ratio": f.ratio,
        "one_hop_ms": one, "two_hop_ms": two, "G": one / two,
        "chain_us_per_row": chain_us,
        "chain_ms": machine.chain.ms(rows_hop_b, arch.hidden, machine.accel),
        "chain_share_of_two_hop": machine.chain.ms(
            rows_hop_b, arch.hidden, machine.accel) / two,
        "domain_condition": ep > f.cards_per_node,
        "fanout_condition": arch.q >= 2,
    }


def profile(arch: MoEArch, machine: Machine,
            mem: MemoryProfile = UNMEASURED, non_expert_params: int = 0) -> dict:
    """Everything the two sides imply together, and which term binds."""
    mep = min_ep_for_memory(arch, machine, mem, non_expert_params)
    ep_floor = mep["min_ep"]
    fits = 0 < ep_floor <= machine.fabric.ep
    mm = expert_matmul(arch, machine, machine.fabric.ep)
    disp = dispatch_breakdown(arch, machine)
    return {
        "arch": arch.name, "machine": machine.fabric.name,
        "memory": {"min_ep": ep_floor, "configured_ep": machine.fabric.ep,
                   "fits": fits, "measured": mep["measured"],
                   "breakdown": mep["residency"],
                   "expert_gib": arch.expert_bytes_all_layers() / 2 ** 30},
        "expert_matmul": mm,
        "dispatch": disp,
        "binds": ("memory capacity" if not fits
                  else "expert matmul, memory bound" if mm["bound_by"] == "memory"
                  else "expert matmul, compute bound"),
        "caveat": ("step-level composition is behind a failing gate; the terms above "
                   "are per-call and per-layer and do not add to a step time"),
        "memory_caveat": (None if mep["measured"] else
                          "the residency verdict rests on an unmeasured memory profile; "
                          "see docs/11-residency-measurement.md"),
    }


def sweep(base: MoEArch, machine: Machine, knob: str, values,
          mem: MemoryProfile = UNMEASURED) -> list:
    """One architecture knob against the machine, everything else held."""
    out = []
    for v in values:
        a = replace(base, **{knob: v})
        try:
            out.append((v, profile(a, machine, mem)))
        except ValueError as e:
            out.append((v, {"error": str(e)}))
    return out
