# -*- coding: utf-8 -*-
"""Phase-decomposed step model: dispatch / expert compute / combine, with overlap.

This is the repository's **second** model. The first (`core.py` + `calibrate.py`)
prices one collective call and is validated by Tier-1 and Tier-1b. This one prices
a whole training step by composing phases, and it is **deliberately shipped
uncalibrated**: its structure is fixed, its free parameters are named, and the
measurement that would fill them is specified below. Nothing here may be used to
predict a step time until that measurement exists and the gate turns green.

## Why a second model instead of extending the first

Adding phases to the communication model would quietly change what its gates mean.
Keeping them apart lets the communication model stay a validated instrument while
this one carries the risk.

## Structure

Per MoE layer, per microbatch, the forward chain is a strict dependency:

    router -> [splits sync] -> Hop A a2a -> arrival chain -> Hop B a2a
           -> expert FFN
           -> combine a2a -> chain

Nothing inside that chain can overlap anything else inside it. Overlap comes from
*other* work being in flight -- other microbatches, other layers, the attention and
dense parts of the model. So the exposed cost of a communication phase is

    exposed = span - min(span, concurrent_compute_available)

and `concurrent_compute_available` is what the pipeline structure supplies. This is
a structural account, not a fitted fraction, and it is why the six single-parameter
overlap families all failed (docs/07): they modelled exposure as a property of the
model, when the data says it is a property of the *schedule*.

The measured exposure ratios support that reading -- they track microbatch count,
not geometry size:

    mbs=1  ratio -0.15      mbs=2  ratio +0.42      mbs=4  ratio +0.62

With one microbatch there is little else in flight to hide behind; with four there
is a queue of independent work.

## Free parameters (all UNCALIBRATED -- this is the honest part)

    OVERLAP_EFFICIENCY   how much of the theoretically-concurrent compute actually
                         hides communication. 1.0 = perfect scheduler, 0.0 = none.
                         Depends on stream assignment and launch order, so it is a
                         property of the framework build, not of the hardware.
    NON_MOE_COMPUTE_MS   attention + dense + optimizer per step. Derivable from
                         measured step time minus everything else, but only once
                         the phases are separable.
    CHAIN_EXPOSURE       whether the arrival chain (local tensor ops) lands on the
                         compute stream and therefore competes with expert GEMMs,
                         or interleaves. Currently unknown; it changes the sign of
                         some predictions.

## The measurement that fills them

Specified in docs/09-phase-model.md. Summary: profile >=3 complete steps per arm
with **stream attribution**, compute per-op exposure as
`span - (span intersect compute-stream-busy)`, and check the two arms' exposure
difference reconciles with their measured step-time difference to within +-10%
before fitting anything. Without that self-consistency check, any fit is decoration.

Minimum viable run: one geometry, two arms, 4 nodes. That geometry already has
step-time ground truth (two-hop slower by 1833 ms), so the model can be scored
against a number nobody can retune.
"""
from __future__ import annotations

from dataclasses import dataclass, field

UNCALIBRATED = None


@dataclass
class PhaseSpans:
    """Per-microbatch, per-layer durations in ms, before any overlap is applied."""
    splits_sync: float = 0.0
    hop_a: float = 0.0
    arrival_chain: float = 0.0
    hop_b: float = 0.0
    expert_ffn_lo: float = 0.0      # compute is bracketed; see sim/compute.py
    expert_ffn_hi: float = 0.0
    combine: float = 0.0

    @property
    def comm_ms(self) -> float:
        return self.splits_sync + self.hop_a + self.hop_b + self.combine

    @property
    def local_ms(self) -> float:
        return self.arrival_chain

    def compute_bracket(self):
        return (self.expert_ffn_lo, self.expert_ffn_hi)


@dataclass
class OverlapModel:
    """The schedule-dependent half. Every field here is currently unmeasured."""
    overlap_efficiency: float = UNCALIBRATED
    non_moe_compute_ms: float = UNCALIBRATED
    chain_on_compute_stream: bool = UNCALIBRATED

    def is_calibrated(self) -> bool:
        return (self.overlap_efficiency is not None
                and self.non_moe_compute_ms is not None
                and self.chain_on_compute_stream is not None)

    def missing(self) -> list:
        return [n for n, v in (("overlap_efficiency", self.overlap_efficiency),
                               ("non_moe_compute_ms", self.non_moe_compute_ms),
                               ("chain_on_compute_stream",
                                self.chain_on_compute_stream)) if v is None]


def spans_for(cluster, geom, expert_cfg: dict) -> PhaseSpans:
    """Assemble one microbatch-layer's phase spans from the two validated models.

    Communication spans come from the Tier-1/Tier-1b-validated call model; the
    expert-FFN span comes from the measured roofline as a bracket. Both are used
    here exactly as they are elsewhere -- this module adds no new primitive, only a
    composition rule.
    """
    from .compute import expert_ffn
    from .core import _a2a_ms

    hop_a = _a2a_ms(cluster.slow, geom.n_groups, geom.rows_hop_a(),
                    geom.row_bytes(),
                    geom.M / geom.n_groups if geom.n_groups > 1 else 1.0)
    hop_b = _a2a_ms(cluster.fast, geom.R, geom.rows_hop_b(), geom.row_bytes(),
                    1.0 / geom.R)
    chain = cluster.chain_us_per_row * geom.rows_hop_b() / 1000.0
    ffn = expert_ffn(geom.tokens_per_rank, geom.k,
                     expert_cfg["n_experts"], geom.ep,
                     geom.H, expert_cfg["d_expert"],
                     expert_cfg.get("n_mats", 2))
    return PhaseSpans(splits_sync=cluster.splits_sync_ms, hop_a=hop_a,
                      arrival_chain=chain, hop_b=hop_b,
                      expert_ffn_lo=ffn["ms_fast"], expert_ffn_hi=ffn["ms_slow"],
                      combine=hop_a + hop_b)


def step_ms(spans: PhaseSpans, overlap: OverlapModel, microbatches: int,
            moe_layers: int):
    """Predicted step time. Refuses to answer while the overlap model is unmeasured.

    Raising rather than returning a plausible number is the whole point: a
    phase-composed step time that silently assumes an overlap efficiency is
    indistinguishable from a measured one in a plot, and that is how a simulator
    starts lying.
    """
    if not overlap.is_calibrated():
        raise NotImplementedError(
            "phase model is uncalibrated; missing %s. Run the protocol in "
            "docs/09-phase-model.md -- no step-time prediction until then."
            % ", ".join(overlap.missing()))
    per_layer_comm = spans.comm_ms
    concurrent = overlap.non_moe_compute_ms / max(moe_layers, 1)
    if microbatches > 1:
        concurrent += spans.expert_ffn_lo * (microbatches - 1) / microbatches
    hidden = min(per_layer_comm, concurrent * overlap.overlap_efficiency)
    exposed_comm = per_layer_comm - hidden
    local = spans.arrival_chain if overlap.chain_on_compute_stream else 0.0
    return (exposed_comm + local + spans.expert_ffn_lo) * moe_layers * microbatches


def report(cluster, geom, expert_cfg: dict, overlap: OverlapModel = None) -> dict:
    """What the phase model can say today: spans, bracket, and the missing pieces."""
    overlap = overlap or OverlapModel()
    s = spans_for(cluster, geom, expert_cfg)
    return {"spans": s, "comm_ms": s.comm_ms,
            "compute_bracket": s.compute_bracket(),
            "calibrated": overlap.is_calibrated(),
            "missing": overlap.missing()}


def main() -> None:
    from .calibrate import flat_supernode
    from .core import MoEGeometry
    g = MoEGeometry(name="n4", n_groups=4, R=8, k=6, M=2, seq=4096, mbs=1,
                    gbs=4 * 8 * 4096)
    r = report(flat_supernode(), g, {"n_experts": 128, "d_expert": 2048})
    s = r["spans"]
    print("Phase spans, one MoE layer, one microbatch (ms), geometry n4:")
    print("  splits sync    %7.3f" % s.splits_sync)
    print("  Hop A a2a      %7.3f" % s.hop_a)
    print("  arrival chain  %7.3f" % s.arrival_chain)
    print("  Hop B a2a      %7.3f" % s.hop_b)
    print("  combine        %7.3f" % s.combine)
    print("  ---- communication total %7.3f" % s.comm_ms)
    print("  expert FFN     %7.3f - %.3f  (bracketed, see sim/compute.py)"
          % s.compute_bracket())
    print()
    print("Step-time prediction: **unavailable**. Missing: %s" % ", ".join(r["missing"]))
    print("Fill them with the protocol in docs/09-phase-model.md; until then this")
    print("model reports spans only, and sim/core.py remains the validated one.")


if __name__ == "__main__":
    main()
