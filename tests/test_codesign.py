# -*- coding: utf-8 -*-
"""Pins for the co-design layer: machine.py, codesign.py, envelope.py, archsearch.py,
record.py.

Every numeric claim those modules' docstrings make is asserted here, so it cannot
drift silently: the arrival-chain model against its own measured sweeps, the docs/12
group-cap table cell by cell, the provenance flags that mark unmeasured residency, and
the two directional checks the published record permits (sim/record.py).
"""
import pytest

from sim.machine import (BYTES_BF16, CHAIN_H_SWEEP_MS, CHAIN_H_SWEEP_ROWS,
                         FUSED_CHAIN, GATHER_GBPS_MEASURED, INDEX_NS_PER_ROW,
                         NO_CHAIN, PYTORCH_CHAIN, Accelerator)
from sim.codesign import (REFERENCE_ARCH, UNMEASURED, MemoryProfile, MoEArch,
                          dispatch_breakdown, expert_matmul, m_table,
                          min_ep_for_memory, profile, residency,
                          synthetic_dgx_h100)
from sim import archsearch, envelope, record

LAUNCH_MS_A = 0.129  # measured world-8 deep-queue per-call cost, platform A


def machine_a_accel():
    """Platform A's arrival-chain side, from its two measured constants."""
    return Accelerator.from_gather_bw("A-chain", GATHER_GBPS_MEASURED,
                                      hbm_capacity_gb=64.0, launch_ms=LAUNCH_MS_A)


# ---------------------------------------------------------------------------
# machine.py: the chain model against the sweeps it claims to reproduce
# ---------------------------------------------------------------------------

def test_chain_model_reproduces_hidden_width_sweep_to_3_percent():
    a = machine_a_accel()
    worst = 0.0
    for H, measured in CHAIN_H_SWEEP_MS.items():
        model = PYTORCH_CHAIN.ms(CHAIN_H_SWEEP_ROWS, H, a)
        err = abs(model - measured) / measured
        worst = max(worst, err)
        assert err < 0.03, "H=%d: model %.3f vs measured %.3f (%.1f%%)" % (
            H, model, measured, 100 * err)
    # the docstring's own figure: 2.9 percent worst case, at H=4096
    assert 0.025 < worst < 0.03


def test_chain_floor_binds_at_small_row_counts():
    # The row sweep measured the whole chain at 0.248 ms at 1024 rows; the model's
    # floor is two launches, 0.258 ms, within 5 percent of the measurement.
    a = machine_a_accel()
    model = PYTORCH_CHAIN.ms(1024, 2048, a)
    assert model == pytest.approx(2 * LAUNCH_MS_A)
    assert abs(model - 0.248) / 0.248 < 0.05
    # and the floor is genuinely the max, not an addition
    assert PYTORCH_CHAIN.ms(CHAIN_H_SWEEP_ROWS, 2048, a) > model


def test_chain_per_row_cost_sits_at_the_row_sweep_bracket():
    # 85.8 ns of index work plus 16.7 ns of gather at H=2048 is 102.5 ns per row,
    # against the 86.5 to 101.8 ns the row sweep measures directly.
    a = machine_a_accel()
    ns = PYTORCH_CHAIN.ns_per_row(2048, a)
    assert ns == pytest.approx(102.5, abs=0.5)
    assert INDEX_NS_PER_ROW == pytest.approx(85.8, abs=0.1)
    assert GATHER_GBPS_MEASURED == pytest.approx(490, abs=2)


def test_fused_chain_brackets_the_design_target():
    # K1 pays the gather and not the index work: 8.4 ns (write straight to the send
    # buffer) to 16.7 ns (materialise the payload) at H=2048. The 0.012 us/row design
    # target quoted throughout the repository is 12 ns and must sit inside.
    a = machine_a_accel()
    lo = FUSED_CHAIN.gather_ns_per_row(2048, a)          # traffic = 1
    hi = PYTORCH_CHAIN.gather_ns_per_row(2048, a)        # traffic = 2
    assert lo == pytest.approx(8.4, abs=0.1)
    assert hi == pytest.approx(16.7, abs=0.1)
    assert lo < 12.0 < hi
    assert NO_CHAIN.ns_per_row(2048, a) == 0.0


def test_chain_cost_at_the_reference_width_is_the_shipped_constant():
    """The hidden-width shape must not move the level every other figure is stated at.

    Two measurements of the chain disagree by the documented run-to-run drift: 2.15 ms
    at 24576 rows in the calibration, 2.51 ms at the same rows in the hidden-width
    sweep. The sweep is the only one that resolves the shape in H, so the shape is
    taken from it and the level from the calibration. If that ever stops holding, the
    reference threshold of 3.98 silently becomes 4.46 -- a defensible reading of the
    same data, but a different one, and not something to arrive at by accident.
    """
    from sim.calibrate import CHAIN_US_PER_ROW
    from sim.machine import CHAIN_H_SWEEP_MS, chain_us_per_row_at
    assert chain_us_per_row_at(2048) == pytest.approx(CHAIN_US_PER_ROW, rel=1e-12)

    costs = [chain_us_per_row_at(H) for H in sorted(CHAIN_H_SWEEP_MS)]
    assert costs == sorted(costs), "the chain cannot fall as hidden width grows"
    # over the fourfold range the payload rises by 4 and the chain by 1.51, which is
    # the whole reason the threshold falls with H
    assert costs[3] / costs[1] == pytest.approx(1.51, abs=0.005)

    with pytest.raises(ValueError):
        chain_us_per_row_at(3072)   # between measured points: refuses to interpolate


def test_breakeven_falls_with_hidden_width():
    """The effective threshold is not flat in H, and it falls rather than rises.

    An earlier revision of this analysis scaled the chain with H while holding the
    payload at the reference width, which has the sign of the effect backwards. The
    limit settles the direction without reference to any constant: every wire term
    scales exactly with H while alpha and the splits exchange do not, so at large H
    the comparison approaches the byte-only one and the threshold falls toward it.

    The direction matters because it runs against this repository's own proposal.
    Contemporary MoE models all sit above the reference width, so a threshold quoted
    at 2048 over-prices the arrival chain, and only two-hop pays it.
    """
    from sim.uncertainty import breakeven_vs_hidden_width
    rows = breakeven_vs_hidden_width()
    assert [H for H, _ in rows] == [1024, 2048, 4096, 8192]
    bes = [b for _H, b in rows]
    assert bes == sorted(bes, reverse=True), (
        "the threshold must fall as hidden width grows, got %s" % bes)
    for (H, got), doc in zip(rows, (5.95, 3.98, 2.89, 2.40)):
        assert abs(got - doc) <= 0.005, (
            "H=%d breakeven %.4f deviates from the docs/05 table's %.2f; correct one "
            "side or the other, never leave them apart" % (H, got, doc))


def test_accelerator_refuses_unmeasured_gather():
    bare = Accelerator("no-gather", hbm_gbps=3350, hbm_capacity_gb=80,
                       launch_ms=0.1)
    with pytest.raises(ValueError):
        bare.gather_gbps


# ---------------------------------------------------------------------------
# codesign.py: quota arithmetic, provenance flags, the docs/12 table
# ---------------------------------------------------------------------------

def test_quota_must_divide():
    bad = MoEArch(name="bad", hidden=1024, d_expert=1024, n_experts=8, k=8, M=3)
    with pytest.raises(ValueError):
        bad.q


def test_unmeasured_residency_marks_every_downstream_result():
    m = synthetic_dgx_h100()
    r = residency(REFERENCE_ARCH, m, ep=m.fabric.ep)
    assert r["measured"] is False
    assert r["provenance"]["activation_bytes"].startswith("ASSUMED")
    assert r["provenance"]["overhead_bytes"].startswith("ASSUMED")
    assert "NOT SUPPLIED" in r["provenance"]["non_expert_bytes"]

    mep = min_ep_for_memory(REFERENCE_ARCH, m)
    assert set(mep) == {"min_ep", "residency", "measured"}
    assert mep["measured"] is False

    p = profile(REFERENCE_ARCH, m)
    assert "docs/11" in p["memory_caveat"]

    measured = MemoryProfile(name="measured", activation_bytes_per_token_layer=1e3,
                             overhead_frac=0.10, measured=True, source="docs/11 run")
    r2 = residency(REFERENCE_ARCH, m, ep=m.fabric.ep, profile=measured)
    assert r2["measured"] is True
    assert r2["provenance"]["activation_bytes"].startswith("measured")
    assert profile(REFERENCE_ARCH, m, mem=measured)["memory_caveat"] is None


def test_docs12_group_cap_table_cell_by_cell():
    expected = {  # docs/12-m-quality-experiment.md, corrected 2026-09-01
        4: (0.948, 1.490),
        2: (1.204, 2.234),
        1: (1.391, 2.979),
    }
    for M, q, g16, gfu in m_table():
        e16, efu = expected[M]
        assert q == REFERENCE_ARCH.k // M
        assert g16 == pytest.approx(e16, abs=1e-3), "M=%d PyTorch chain" % M
        assert gfu == pytest.approx(efu, abs=1e-3), "M=%d fused" % M


def test_group_cap_direction_and_chain_ordering():
    rows = m_table()
    g16 = [r[2] for r in rows]
    gfu = [r[3] for r in rows]
    # tightening M (4 -> 2 -> 1) raises G in both tiers
    assert g16 == sorted(g16) and gfu == sorted(gfu)
    # the fused chain beats the operator chain at every cap
    assert all(f > p for p, f in zip(g16, gfu))


def test_dispatch_conditions_withdraw_rather_than_score():
    d = dispatch_breakdown(REFERENCE_ARCH, synthetic_dgx_h100(nodes=1))
    assert d["domain_condition"] is False  # one node: no slow hop exists
    unconstrained = MoEArch(name="u", hidden=7168, d_expert=2048, n_experts=256,
                            k=8, M=8, seq=4096)
    d2 = dispatch_breakdown(unconstrained, synthetic_dgx_h100())
    assert d2["fanout_condition"] is False  # q = 1, nothing to deduplicate


def test_expert_matmul_flags_the_memory_bound_regime():
    m = synthetic_dgx_h100()
    few_rows = MoEArch(name="tiny", hidden=7168, d_expert=2048, n_experts=128,
                       k=1, M=1, seq=128, mbs=1)
    mm = expert_matmul(few_rows, m, ep=m.fabric.ep)
    assert mm["arithmetic_intensity"] < mm["machine_balance"]
    assert mm["bound_by"] == "memory"
    assert mm["ms"] == mm["ms_memory_bound"] >= mm["ms_compute_bound"]

    big = expert_matmul(REFERENCE_ARCH, m, ep=m.fabric.ep)
    assert big["arithmetic_intensity"] > big["machine_balance"]
    assert big["bound_by"] == "compute"


# ---------------------------------------------------------------------------
# envelope.py: both edges, and the width thresholds
# ---------------------------------------------------------------------------

def test_envelope_upper_edge_is_memory():
    m = synthetic_dgx_h100()
    env = envelope.envelope(REFERENCE_ARCH, m, factors=(1, 128))
    small, huge = env["points"]
    assert small.fits and not huge.fits
    assert huge.params_per_accel > small.params_per_accel


def test_envelope_lower_edge_moves_with_expert_width_not_expert_count():
    m = synthetic_dgx_h100()
    from dataclasses import replace
    narrow = envelope.evaluate(REFERENCE_ARCH, m)
    wide = envelope.evaluate(replace(REFERENCE_ARCH, d_expert=16384), m)
    assert narrow.compute_share < 0.5 < wide.compute_share
    # expert count moves what fits, not what binds: share barely moves with E
    more_experts = envelope.evaluate(replace(REFERENCE_ARCH, n_experts=1024), m)
    assert abs(more_experts.compute_share - narrow.compute_share) < 0.1


def test_width_thresholds_are_ordered():
    m = synthetic_dgx_h100()
    first = envelope.first_order_threshold(m)
    full = envelope.compute_bound_threshold(REFERENCE_ARCH, m)
    # the full model prices fixed call costs and the chain on top of the leading
    # term, so its crossover sits above the first-order one
    assert first == pytest.approx(6593, abs=5)
    assert full > first
    assert REFERENCE_ARCH.d_expert < first  # the reference shape is comm bound


# ---------------------------------------------------------------------------
# archsearch.py: the capacity-preserving family and the ranking
# ---------------------------------------------------------------------------

def test_granularity_family_preserves_both_capacity_products():
    base = REFERENCE_ARCH
    fam = archsearch.granularity_family(base, factors=(0.5, 1, 2, 4, 8))
    assert any(g == 1 for g, _ in fam)
    for g, a in fam:
        assert a.n_experts * a.d_expert == pytest.approx(
            base.n_experts * base.d_expert, rel=0.01)
        assert a.k * a.d_expert == pytest.approx(
            base.k * base.d_expert, rel=0.01)


def test_search_respects_divisor_and_node_limits():
    m = synthetic_dgx_h100()
    cands = archsearch.search(REFERENCE_ARCH, m, granularities=(1, 2))
    assert cands
    for c in cands:
        assert c.arch.k % c.arch.M == 0
        assert c.arch.M <= m.fabric.nodes


def test_rank_sorts_by_the_no_overlap_upper_bound():
    m = synthetic_dgx_h100()
    ranked = archsearch.rank(archsearch.search(REFERENCE_ARCH, m))
    sums = [c.no_overlap_ms for c in ranked]
    assert sums == sorted(sums)
    assert all(c.feasible for c in ranked)


# ---------------------------------------------------------------------------
# record.py: the two directional checks the published record permits
# ---------------------------------------------------------------------------

def test_record_directional_checks_hold():
    ck = record.checks()
    assert ck["granularity_ordering_holds"] is True
    assert ck["endpoint_decline_holds"] is True


def test_record_model_prefers_the_nvlink_domain():
    # For both shapes at every cluster size the model keeps the expert-parallel
    # group inside one fast domain, which is the repository's own domain doctrine.
    for cluster in record.MPF_CLUSTERS:
        for arch in (record.MIXTRAL_8X22B, record.MIXTRAL_G8T8):
            ep, _ = record.best_point(arch, cluster)
            assert ep == 8, "%s at %d chose EP=%d" % (arch.name, cluster, ep)


def test_record_ceilings_pinned():
    rows = {(c, n): ceil for c, _, n, _, ceil, _ in record.comparison()}
    assert rows[(128, "Mixtral 8x22B")] == pytest.approx(90.1, abs=0.2)
    assert rows[(128, "Mixtral-8x22B-G8T8")] == pytest.approx(58.1, abs=0.2)
    assert rows[(1024, "Mixtral 8x22B")] == pytest.approx(81.2, abs=0.2)
    assert rows[(1024, "Mixtral-8x22B-G8T8")] == pytest.approx(57.2, abs=0.2)


def test_record_capacity_is_actually_controlled():
    a, b = record.MIXTRAL_8X22B, record.MIXTRAL_G8T8
    assert a.n_experts * a.d_expert == b.n_experts * b.d_expert
    assert a.hidden == b.hidden and a.n_moe_layers == b.n_moe_layers
    assert b.k * b.d_expert * 2 == a.k * a.d_expert  # active halved, as published
