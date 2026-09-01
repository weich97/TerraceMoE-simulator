# -*- coding: utf-8 -*-
"""The one controlled comparison the published record contains, priced by the co-design
layer.

## Why this module is this small

docs/13 gathered 93 published runs to test the envelope's lower edge and reached a
negative verdict: the record does not locate it. Model sizes at a fixed cluster size
spread 16x to 99x, no published sweep reaches a knee, and the throughput metrics are
mutually incomparable. Nothing in that verdict can validate ``sim/envelope.py``, and this
module does not try.

The record does contain exactly one controlled granularity experiment. MoE Parallel
Folding (arXiv:2504.14960, verified in docs/13a) trains Mixtral 8x22B and a fine-grained
reparameterisation of it -- 64 experts at one eighth the width, top-8, the paper's
"G8T8" -- on the same Eos H100 cluster, the same stack, the same global batch, and
reports MFU for both across four cluster sizes. Same backbone, same total expert
parameters, only the expert shape moves. That is the axis ``archsearch`` prices, so this
is the one place the record can check the co-design layer's direction, and the paper
even names the mechanism the model uses: "the smaller hidden sizes decrease GEMM
efficiency."

## What the check can and cannot mean

The model side is the mixture-of-experts utilisation ceiling from ``envelope.evaluate``,
at the expert parallelism the model itself prefers (swept here; the paper does not state
its EP degree). The record side is end-to-end MFU. These are different quantities: the
ceiling prices only the MoE layers on a synthetic H100 whose chain, alpha curve and
GEMM-efficiency shape are borrowed from platform A. So the check is directional only:

  1. the model must put G8T8's ceiling below Mixtral's at every cluster size, and
  2. neither ceiling may rise from the largest per-accelerator token load to the
     smallest -- the endpoint direction of every published fixed-batch sweep. Strict
     monotonicity is not required: the GEMM-efficiency term is a bracket lookup over
     measured shapes, and inside its interpolation error the ceiling may wobble.

The size of the fixed-batch slope is deliberately not a check. The record's own causal
statements (docs/13, 3b) attribute that slope to pipeline bubbles and per-DP-group
batch, which are step-level schedule quantities behind the failed Tier-2 gate; a
MoE-layer ceiling that moves less than the published MFU does is consistent with the
record, not contradicted by it.

The absolute levels are not a check in either direction, and a residency verdict is out
of scope entirely: under pure expert parallelism neither model's weights-plus-optimizer
fit, because the real runs shard them over pipeline stages and ZeRO, which the co-design
layer does not price. ``fits`` from the evaluation is therefore ignored here, and said
so, rather than silently.

Status of the inputs: shapes and MFU values are verifier-confirmed against the paper and
the released configs (docs/13a), with the Table 4 sweeps at global batch 1024 -- the
corrected figure, not the 256 of the optimal-config comparison.
"""
from __future__ import annotations

from dataclasses import replace

from .codesign import Machine, MoEArch, synthetic_dgx_h100
from .envelope import evaluate

# ---------------------------------------------------------------------------
# The pair, as published. Sources verified in docs/13a.
# ---------------------------------------------------------------------------

MPF_SEQ = 4096
MPF_GLOBAL_BATCH_SEQS = 1024   # Table 4 scaling sweeps (verifier correction: not 256)
MPF_CLUSTERS = (128, 256, 512, 1024)

#: hidden 6144, 56 MoE layers, 8 experts of intermediate 16384, top-2 (released config).
MIXTRAL_8X22B = MoEArch(name="Mixtral 8x22B", hidden=6144, d_expert=16384,
                        n_experts=8, k=2, M=2, n_moe_layers=56,
                        seq=MPF_SEQ, mbs=1, gbs=MPF_GLOBAL_BATCH_SEQS)

#: The paper's fine-grained upcycling of the same model: "64 experts and 8 active
#: experts per token ... each expert possessing a hidden size that is one-eighth of the
#: original". Total expert parameters unchanged; active halved. M = k because MPF runs
#: unconstrained top-k -- q = 1, so two-hop never applies and the model prices what the
#: paper ran.
MIXTRAL_G8T8 = MoEArch(name="Mixtral-8x22B-G8T8", hidden=6144, d_expert=2048,
                       n_experts=64, k=8, M=8, n_moe_layers=56,
                       seq=MPF_SEQ, mbs=1, gbs=MPF_GLOBAL_BATCH_SEQS)

#: MFU percent by cluster size, MoE Parallel Folding Table 4 (MCore w/ Folding rows).
MPF_MFU = {
    "Mixtral 8x22B":      {128: 52.2, 256: 50.7, 512: 48.9, 1024: 44.9},
    "Mixtral-8x22B-G8T8": {128: 30.0, 256: 29.3, 512: 26.7, 1024: 25.5},
}


# ---------------------------------------------------------------------------
# Pricing the pair
# ---------------------------------------------------------------------------

def eos_like(ep_ranks: int) -> Machine:
    """The synthetic H100 fabric sized to an expert-parallel group of ``ep_ranks``.

    When the whole group sits inside one 8-card NVLink domain the flat a2a never
    touches InfiniBand, so the flat level is priced on the fast link. The hierarchy
    question is then withdrawn, which is the model's own domain condition doing its job.
    """
    nodes = max(1, ep_ranks // 8)
    m = synthetic_dgx_h100(nodes=nodes)
    if nodes == 1:
        f = replace(m.fabric, beta_inter_gbps=m.fabric.beta_intra_gbps)
        m = Machine(m.accel, f, m.chain)
    return m


def tokens_per_accel(cluster: int) -> int:
    return MPF_GLOBAL_BATCH_SEQS * MPF_SEQ // cluster


def best_point(arch: MoEArch, cluster: int):
    """The evaluation at the expert parallelism the model prefers for this shape.

    The paper does not state its EP degree, so the model is allowed to pick the one it
    scores cheapest, exactly as its own doctrine tells a designer to. Only powers of two
    up to the expert count and the cluster are considered.
    """
    toks = tokens_per_accel(cluster)
    assert toks % MPF_SEQ == 0, "tokens per accelerator not a whole sequence count"
    a = replace(arch, mbs=toks // MPF_SEQ)
    best = None
    for ep in (8, 16, 32, 64, 128):
        if ep > arch.n_experts or ep > cluster:
            continue
        p = evaluate(a, eos_like(ep))
        if best is None or p.utilisation_ceiling > best[1].utilisation_ceiling:
            best = (ep, p)
    return best


def comparison() -> list:
    """Rows of (cluster, tokens/accel, name, chosen ep, ceiling %, observed MFU %)."""
    out = []
    for cluster in MPF_CLUSTERS:
        for arch in (MIXTRAL_8X22B, MIXTRAL_G8T8):
            ep, p = best_point(arch, cluster)
            out.append((cluster, tokens_per_accel(cluster), arch.name, ep,
                        100.0 * p.utilisation_ceiling, MPF_MFU[arch.name][cluster]))
    return out


def checks(rows=None) -> dict:
    """The two directional checks stated in the module docstring."""
    rows = rows or comparison()
    by = {}
    for cluster, _, name, _, ceil, mfu in rows:
        by.setdefault(name, {})[cluster] = ceil
    ordering = all(by["Mixtral-8x22B-G8T8"][c] < by["Mixtral 8x22B"][c]
                   for c in MPF_CLUSTERS)
    first, last = MPF_CLUSTERS[0], MPF_CLUSTERS[-1]
    endpoint_decline = all(by[n][last] <= by[n][first] for n in by)
    return {"granularity_ordering_holds": ordering,
            "endpoint_decline_holds": endpoint_decline}


if __name__ == "__main__":
    print("MoE Parallel Folding's controlled granularity pair, against the co-design "
          "layer's MoE-layer ceiling")
    print("(synthetic H100 construction, borrowed constants labeled in "
          "sim.codesign.synthetic_dgx_h100; ceilings are MoE-layer-only and are not "
          "MFU predictions)")
    print()
    print("%-8s %-11s %-22s %-6s %12s %14s" % (
        "cluster", "tokens/acc", "model", "EP*", "ceiling %", "published MFU"))
    print("-" * 78)
    for cluster, toks, name, ep, ceil, mfu in comparison():
        print("%-8d %-11d %-22s %-6d %12.1f %14.1f" % (
            cluster, toks, name, ep, ceil, mfu))
    print()
    ck = checks()
    print("granularity ordering (G8T8 ceiling below Mixtral at every size): %s"
          % ("holds" if ck["granularity_ordering_holds"] else "FAILS"))
    print("endpoint decline (ceiling no higher at 4k tokens/acc than at 32k): %s"
          % ("holds" if ck["endpoint_decline_holds"] else "FAILS"))
    print()
    print("The ceiling moves less than the published MFU across the sweep; the papers "
          "themselves attribute that slope to pipeline bubbles and per-DP-group batch "
          "(docs/13, 3b), which are step-level quantities behind the failed Tier-2 "
          "gate.")
    print("EP* is the expert parallelism the model scores cheapest; the paper does not "
          "state its own. Residency is out of scope: pure-EP weights+optimizer do not "
          "fit and the real runs shard over PP/ZeRO, which the model does not price.")
