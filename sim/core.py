# -*- coding: utf-8 -*-
"""Simulation core: cluster spec, MoE geometry -> traffic, per-call cost of both strategies, step-level composition.

Every formula is annotated with its measured provenance (internal measurement records).
**All formulas are the 2026-08-24 red-team-corrected versions** (internal measurement
records): the byte ledger counts "both sides send every copy, no card-level dedup";
the Hop A copy destined for the sender's own node is a pure self-copy (_send_index
always equals the sender; it never crosses the link).
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# Cluster spec
# ----------------------------------------------------------------------------

@dataclass
class Level:
    """One communication domain level: a set of cards that can run a2a together.

    alpha_ms(world) and beta_gbps(per_peer_bytes) are both callables or interpolation
    tables -- the calibration layer (sim/calibrate.py) loads measured points into them;
    synthetic clusters supply analytic forms directly.
    """
    name: str
    # alpha(world) [ms]: fixed overhead per collective call. Measured shape: world>=16
    # roughly linear (another test frame, internal measurement records: slope
    # ~0.0107 ms/rank; rescaled wholesale to this machine)
    alpha_pts: list = field(default_factory=list)     # [(world, ms)]
    # beta effective bandwidth [GB/s], interpolated by per-peer bytes (aligned-size
    # convention; internal measurement records: flat at real row widths)
    beta_pts: list = field(default_factory=list)      # [(per_peer_bytes, GB/s)]

    def alpha_ms(self, world: int) -> float:
        return _interp(self.alpha_pts, float(world))

    def beta_gbps(self, per_peer_bytes: float) -> float:
        return _interp(self.beta_pts, per_peer_bytes, logx=True)


def _interp(pts, x, logx=False):
    """Piecewise-linear interpolation, clamped beyond the endpoints (extrapolation is calibration's job, not interpolation's)."""
    assert pts, "empty interpolation table -- calibration never ran"
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    if logx:
        t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    else:
        t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


@dataclass
class ClusterSpec:
    """One (possibly hypothetical) cluster.

    fast  = intra-group domain (the R cards inside a node)
    slow  = cross-group domain (between nodes; in cross-supernode scenarios = between supernodes)
    flat  = the "full fabric" a one-hop a2a actually crosses (world = n_groups*R;
            on a flat machine it runs at slow speed, on a hierarchical machine its
            effective bandwidth is set by the slow edge -- given explicitly at
            calibration/synthesis time)

    knobs (all with measured provenance; can be toggled per scenario):
      splits_sync_ms    host-side retrieval of splits for variable-length a2a, per call
                        (internal measurement records)
      chain_us_per_row  local tensor ops of the two-hop arrival chain, **per row**.
                        The first version treated it as a constant -- the validation gate
                        failed on the spot (load-axis MAE 0.084): the measured penalty
                        grows faster than bytes as T grows, and the arrival chain is
                        tensor ops, linear in row count.
                        Measured at 24576 rows (internal measurement records, k=6)
                        => 0.0875 us/row. Fused-kernel scenario ~0.012; perfect scenario 0.
    """
    name: str
    R: int
    fast: Level
    slow: Level
    flat: Level
    splits_sync_ms: float = 0.044
    chain_us_per_row: float = 2.15 * 1000.0 / 24576.0   # = 0.0875 us/row, internal measurement records/flag

    def ratio(self) -> float:
        """Fast/slow bandwidth ratio (taken at the 8 MB aligned operating point; for reporting)."""
        p = 8 * 2 ** 20
        return self.fast.beta_gbps(p) / self.slow.beta_gbps(p)


# ----------------------------------------------------------------------------
# MoE geometry -> per-call traffic
# ----------------------------------------------------------------------------

@dataclass
class MoEGeometry:
    """Parameterization of the control-testbed geometries (geometry table, internal measurement records)."""
    name: str
    n_groups: int          # number of groups = number of nodes (a group is the hierarchy boundary)
    R: int                 # cards per group
    k: int                 # top-k
    M: int                 # cap on groups per token
    H: int = 2048          # hidden dim
    seq: int = 4096
    mbs: int = 1
    gbs: int = 512
    moe_layers: int = 19   # 20 layers - 1 dense first layer
    bytes_per_elem: int = 2   # bf16

    @property
    def q(self) -> int:
        assert self.k % self.M == 0
        return self.k // self.M

    @property
    def ep(self) -> int:
        return self.n_groups * self.R

    @property
    def tokens_per_rank(self) -> int:
        return self.seq * self.mbs

    @property
    def microbatches(self) -> int:
        # DP = EP (always true on the control testbed); GBS spread over DP, then cut by MBS
        assert self.gbs % (self.ep * self.mbs) == 0, "GBS not divisible; the testbed would reject this outright"
        return self.gbs // (self.ep * self.mbs)

    def calls_per_step_fwd(self) -> int:
        # internal measurement records: forward dispatch calls per step
        # = MoE layers x microbatches x 2 (recompute replay)
        return self.moe_layers * self.microbatches * 2

    def calls_per_step_bwd(self) -> int:
        return self.moe_layers * self.microbatches

    def row_bytes(self) -> int:
        return self.H * self.bytes_per_elem

    # ---- rows sent per rank per call (corrected ledger, internal measurement records; both sides send every copy) ----
    def rows_one_hop(self) -> int:
        return self.tokens_per_rank * self.k

    def rows_hop_a(self) -> int:
        # 1 row per (token, selected group); the copy going to the sender's own group is
        # a pure self-copy that never crosses the link, but it still occupies a row in
        # the buffer (_send_index layout). The wire ledger is deducted on the cost side.
        return self.tokens_per_rank * self.M

    def rows_hop_b(self) -> int:
        return self.tokens_per_rank * self.k


# ----------------------------------------------------------------------------
# Strategy costs
# ----------------------------------------------------------------------------

def _a2a_ms(level: Level, world: int, total_rows: int, row_bytes: int,
            self_fraction: float) -> float:
    """Wall clock of one a2a [ms].

    Of total_rows, self_fraction is self-copy (never crosses the link); the rest spreads
    uniformly per peer, time = alpha(world) + wire bytes / beta(per-peer bytes).
    beta is looked up by **per-peer bytes** -- internal measurement records/internal
    measurement records: bandwidth is sensitive to per-peer message size, flat under the
    aligned convention (integer multiples of the real row width), and the calibration
    table uses exactly that convention.
    """
    wire_rows = total_rows * (1.0 - self_fraction)
    wire_bytes = wire_rows * row_bytes
    per_peer = wire_bytes / max(world - 1, 1)
    beta = level.beta_gbps(per_peer)                       # GB/s
    return level.alpha_ms(world) + wire_bytes / (beta * 1e6)   # bytes/(GB/s)=ns*... -> ms


def one_hop_call(c: ClusterSpec, g: MoEGeometry) -> float:
    """Vendor one-hop: a single full-fabric a2a. Self-copy share = 1/EP (uniform-routing expectation)."""
    return _a2a_ms(c.flat, g.ep, g.rows_one_hop(), g.row_bytes(),
                   self_fraction=1.0 / g.ep)


def hop_a_self_fraction(g: MoEGeometry) -> float:
    """Fraction of Hop-A rows that stay in the source group.

    A token emits ``M`` Hop-A rows, one for each selected group.  Under the
    uniform-routing assumption used by the balanced calibration, ``M/N_g`` of
    those rows are expected to target the source group *per token*.  Dividing by
    the ``M`` emitted rows gives the self-copy fraction ``1/N_g``.  Keeping this
    calculation separate prevents the expected self-row count from being mistaken
    for a fraction again.
    """
    return 1.0 / g.n_groups if g.n_groups > 1 else 1.0


def two_hop_call(c: ClusterSpec, g: MoEGeometry) -> float:
    """T-A2A two-hop (serial, matching the current implementation; internal measurement records: no pipelining).

    Hop A: cross-group domain, world = n_groups; a token has M/N_g expected local
           rows among M emitted rows, so the self-copy share is 1/N_g (that copy's
           _send_index always equals the sender -- purely local; internal measurement
           records).
    Hop B: intra-group domain, world = R; self-copy 1/R.
    Extra: one splits host sync (two-hop does one more variable-length exchange than
           one-hop) + the local arrival chain (per row).
    """
    a = _a2a_ms(c.slow, g.n_groups, g.rows_hop_a(), g.row_bytes(),
                self_fraction=hop_a_self_fraction(g))
    b = _a2a_ms(c.fast, g.R, g.rows_hop_b(), g.row_bytes(),
                self_fraction=1.0 / g.R)
    chain = c.chain_us_per_row * g.rows_hop_b() / 1000.0
    return a + b + c.splits_sync_ms + chain


def step_delta(c: ClusterSpec, g: MoEGeometry) -> dict:
    """Per-step communication delta (on - off) [ms] plus the components the G prediction needs.

    dispatch and combine mirror each other (same bytes, same structure; internal
    measurement records); backward replays the forward splits (internal measurement
    records) at half the call count. G is composed on the validate side (it needs the
    off arm's measured step time).
    """
    per_call = two_hop_call(c, g) - one_hop_call(c, g)
    calls = 2 * (g.calls_per_step_fwd() + g.calls_per_step_bwd())   # x2: dispatch+combine
    return {
        "per_call_on_ms": two_hop_call(c, g),
        "per_call_off_ms": one_hop_call(c, g),
        "per_call_delta_ms": per_call,
        "calls_per_step": calls,
        "step_delta_ms": per_call * calls,
    }
