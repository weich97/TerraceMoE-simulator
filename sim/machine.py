# -*- coding: utf-8 -*-
"""Machine and implementation descriptors, so a verdict can be computed from specs.

## Why this module exists

``calibrate.CHAIN_US_PER_ROW`` is, in that file's own words, "the constant everything
hinges on": it alone moves the corrected breakeven hierarchy ratio from 1.10 to 3.98. It is also a
single number measured on one machine at one hidden width, so it cannot transfer. Every
prediction this repository makes for an unmeasured cluster has to borrow it, which is why
those predictions are labeled extrapolation and why the reported threshold is stated only
at the reference operating point.

That constant is not irreducible. The arrival chain has five stages, and
``terrace/ta2a_dispatch.py`` names them where the K1 kernel replaces them: pair expansion,
owner stable bucket sort, ``i_send`` histogram, the ``[pairs, H]`` send gather, and the
gate flat gather. Exactly one of the five moves payload. Splitting the chain along that
line gives two terms that each depend on something a machine publishes:

    chain(rows, H) = n_ops * launch + rows * H * bytes * traffic / gather_bw

Three independent measurements agree on this split.

1. The hidden-width sweep (thirty iterations on each of two nodes, 24576 rows) measures
   2.37, 2.51, 2.85 and 3.79 ms at H of 1024, 2048, 4096 and 8192. Least squares gives an
   H-independent 2.109 ms and 0.2056 ms per 1024 of hidden width. The slope is
   ``24576 * 1024 * 2`` bytes read and written again in 0.2056 ms, which is 490 GB/s
   sustained: a gather running at memory bandwidth, not at anything to do with the fabric.

2. The row sweep (nine row counts on each of two nodes) refutes the first reading we
   took of the H-independent term. Read as a fixed launch cost it would be 2.109 ms
   whatever the row count, and 16.3 of the measured 0.129 ms world-8 launches, which is
   a tidy integer and was briefly convincing. The same sweep measures the whole chain at
   0.248 ms at 1024 rows, which a 2.109 ms floor cannot produce. The term is therefore
   linear in rows, not fixed: 2.109 ms over 24576 rows is 85.8 ns of index work per row,
   and 85.8 plus the 16.7 ns the gather costs at H=2048 is 102.5 ns per row against the
   86.5 to 101.8 ns the row sweep measures directly. The launch floor is real but small,
   and the row sweep sizes it: 0.248 ms is 1.9 of those same launches.

3. The K1 kernel collapses all five stages into one kernel, so it pays the gather and
   not the index work. That predicts 8.4 to 16.7 ns per row at H=2048, depending on
   whether it materialises the payload or writes straight into the send buffer. The
   figure this repository has been quoting for a fused chain, 0.012 us/row, is 12 ns and
   sits between them. It was entered as an estimate and is now bracketed by a
   measurement it was not derived from.

## What this buys

The implementation tier stops being a label and becomes ``n_ops``, which is countable from
source. The machine stops being a bandwidth ratio and becomes a small set of published
numbers. Hidden width stops being a reference point and becomes an argument, which matters
because the chain and the payload scale differently in it: quadrupling H multiplies the
payload by four and the chain by about 1.5, so the chain's share of two-hop falls and the
effective breakeven falls with it. Reporting a threshold at one hidden width and then
applying it at another is the error this module exists to prevent.

## What it does not buy

``gather_bw`` is measured on one machine here. On another accelerator it is
``gather_efficiency * hbm_bw``, and the efficiency of a row gather is itself a property of
the memory system, not a universal constant. The efficiency shipped below is the one
implied by machine A, and any use of it elsewhere is an assumption stated as such. Nothing
in this module has been validated against a second machine's arrival chain, because we
have not measured one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

BYTES_BF16 = 2

# ---------------------------------------------------------------------------
# Measured inputs. Provenance for each is in the module docstring.
# ---------------------------------------------------------------------------

CHAIN_H_SWEEP_MS = {1024: 2.37, 2048: 2.51, 4096: 2.85, 8192: 3.79}
CHAIN_H_SWEEP_ROWS = 24576

CHAIN_FIXED_MS = 2.109        # least squares intercept over CHAIN_H_SWEEP_MS
CHAIN_PER_1024H_MS = 0.2056   # least squares slope, ms per 1024 of hidden width

#: Sustained bandwidth of the [pairs, H] gather, counting the payload read and written
#: once each. Derived from CHAIN_PER_1024H_MS; not a vendor figure.
GATHER_GBPS_MEASURED = (CHAIN_H_SWEEP_ROWS * 1024 * BYTES_BF16 * 2
                        / (CHAIN_PER_1024H_MS * 1e-3) / 1e9)

#: Index, sort and histogram work in the unfused chain, per row, independent of hidden
#: width. CHAIN_FIXED_MS over CHAIN_H_SWEEP_ROWS. This is the term the fused kernel
#: removes and the only implementation-specific number in the chain model.
INDEX_NS_PER_ROW = CHAIN_FIXED_MS * 1e6 / CHAIN_H_SWEEP_ROWS

#: Whole-chain wall clock at 1024 rows, where the linear form no longer holds because the
#: work is too small to fill the launches. calibrate.CHAIN_FLOOR_MS carries the same
#: number; it is 1.9 of the measured world-8 launch cost.
LAUNCH_FLOOR_OPS = 2


@dataclass(frozen=True)
class Accelerator:
    """What a machine publishes, plus one number an afternoon of microbenchmarks gives.

    hbm_gbps and hbm_capacity_gb are vendor figures. launch_ms is measured, and is the
    per-call cost of getting an operator onto the device at the world size in use;
    profile.PER_CALL_DEEP_QUEUE_MS holds the measured curve for machine A. gather_eff is
    the fraction of hbm_gbps a row gather sustains, and is the one number here that has
    been measured on a single machine only.
    """
    name: str
    hbm_gbps: float
    hbm_capacity_gb: float
    launch_ms: float
    gather_eff: float = 0.0
    #: Achieved dense bf16 matmul throughput. Zero means unknown, and the expert matmul
    #: is then priced on platform A's measured roofline, which is a borrowed prior and
    #: is reported as such rather than silently applied.
    peak_tflops: float = 0.0
    notes: str = ""

    @property
    def gather_gbps(self) -> float:
        if self.gather_eff <= 0.0:
            raise ValueError(
                "%s has no measured gather efficiency; supply gather_eff, or use "
                "Accelerator.from_gather_bw when the sustained figure is known directly"
                % self.name)
        return self.hbm_gbps * self.gather_eff

    @classmethod
    def from_gather_bw(cls, name, gather_gbps, hbm_capacity_gb, launch_ms,
                       hbm_gbps=None, peak_tflops=0.0, notes=""):
        """For a machine whose sustained gather is measured but whose HBM figure is not
        being published. gather_eff is then whatever the two imply, or zero if unknown."""
        hb = hbm_gbps or gather_gbps
        return cls(name=name, hbm_gbps=hb, hbm_capacity_gb=hbm_capacity_gb,
                   launch_ms=launch_ms, gather_eff=gather_gbps / hb,
                   peak_tflops=peak_tflops, notes=notes)


@dataclass(frozen=True)
class ArrivalChain:
    """An implementation of the arrival chain, priced per row with a launch floor.

    index_ns_per_row is the expansion, sort and histogram work, which does not touch the
    payload and so does not scale with hidden width. A fused kernel removes it. traffic is
    how many times the payload crosses memory: 2 when it is read and written back, 1 when
    a fused path writes straight into the send buffer, 0 for the zero-overhead bound.
    floor_ops is how many launches the implementation cannot avoid, which sets the cost
    below the row count where the linear form stops holding.
    """
    name: str
    index_ns_per_row: float
    traffic: float = 2.0
    floor_ops: int = LAUNCH_FLOOR_OPS

    def gather_ns_per_row(self, H: int, accel: Accelerator,
                          bytes_per_elem: int = BYTES_BF16) -> float:
        if self.traffic == 0.0:
            return 0.0
        return H * bytes_per_elem * self.traffic / accel.gather_gbps

    def ns_per_row(self, H: int, accel: Accelerator,
                   bytes_per_elem: int = BYTES_BF16) -> float:
        return self.index_ns_per_row + self.gather_ns_per_row(H, accel, bytes_per_elem)

    def ms(self, rows: int, H: int, accel: Accelerator,
           bytes_per_elem: int = BYTES_BF16) -> float:
        linear = rows * self.ns_per_row(H, accel, bytes_per_elem) / 1e6
        return max(self.floor_ops * accel.launch_ms, linear)

    def us_per_row(self, rows: int, H: int, accel: Accelerator) -> float:
        return self.ms(rows, H, accel) * 1000.0 / rows


PYTORCH_CHAIN = ArrivalChain("PyTorch operator chain", INDEX_NS_PER_ROW, 2.0)
FUSED_CHAIN = ArrivalChain("fused kernel (K1)", 0.0, 1.0, floor_ops=1)
NO_CHAIN = ArrivalChain("zero implementation overhead", 0.0, 0.0, floor_ops=0)

TIERS = (PYTORCH_CHAIN, FUSED_CHAIN, NO_CHAIN)


# ---------------------------------------------------------------------------
# Memory capacity decides expert parallelism, and expert parallelism decides
# whether the criterion applies at all.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MoEModel:
    """Enough of a model to size its residency, in the units papers report."""
    name: str
    hidden: int
    experts: int
    expert_ffn_mult: float = 4.0      # intermediate size over hidden
    n_moe_layers: int = 1
    bytes_per_param: int = 2
    optimizer_bytes_per_param: float = 6.0   # fp32 master plus two moments, amortised
    #: Weight matrices per expert. SwiGLU, which every current MoE uses, has three:
    #: gate and up of hidden x intermediate, and down of intermediate x hidden. A plain
    #: two-matrix feed-forward is the older shape and costs a third less.
    matrices_per_expert: int = 3

    @property
    def params_per_expert(self) -> int:
        inter = int(self.hidden * self.expert_ffn_mult)
        return self.matrices_per_expert * self.hidden * inter * self.n_moe_layers

    def bytes_per_expert(self, with_optimizer: bool = True) -> float:
        per = self.bytes_per_param + (self.optimizer_bytes_per_param
                                      if with_optimizer else 0.0)
        return self.params_per_expert * per


def min_expert_parallelism(model: MoEModel, accel: Accelerator,
                           reserve_frac: float,
                           with_optimizer: bool = True) -> int:
    """Smallest EP whose per-rank expert residency fits, rounded up to a power of two.

    reserve_frac is everything resident before an expert is placed, as a fraction of
    capacity. It has no default on purpose. The value this function is most sensitive to
    is the one nobody has measured, and a default would let it be picked up silently;
    supplying it forces the caller to say where it came from.

    ``codesign.residency`` is the better entry point: it separates the part of residency
    that follows from the architecture, which is arithmetic, from the part that only a
    running job can report, and it labels each term. Use this one only for a quick bound
    with a reserve you can defend. See docs/11-residency-measurement.md.
    """
    usable = accel.hbm_capacity_gb * 1e9 * (1.0 - reserve_frac)
    total = model.experts * model.bytes_per_expert(with_optimizer)
    ep = 1
    while ep < model.experts and total / ep > usable:
        ep *= 2
    return ep


def spans_multiple_domains(ep: int, domain_size: int) -> bool:
    """The domain condition: with the whole group inside one fast domain there is no
    slow-side traffic to deduplicate, and the criterion returns nothing rather than no."""
    return ep > domain_size
