# -*- coding: utf-8 -*-
"""Contract tests for terrace-sim: the geometry ledger, calibration consistency, and the current state of the two validation gate tiers."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.core import MoEGeometry, _interp, one_hop_call, two_hop_call  # noqa: E402
from sim.calibrate import aug_flat, synthetic                          # noqa: E402


def test_geometry_call_counts_match_ledger():
    """The flag geometry's per-step call counts must equal the ledger-pinned 152/76 (internal measurement records)."""
    g = MoEGeometry(name="flag", n_groups=16, R=8, k=6, M=2, mbs=1)
    assert g.microbatches == 4
    assert g.calls_per_step_fwd() == 152
    assert g.calls_per_step_bwd() == 76
    # n8: EP=64 -> microbatches double
    n8 = MoEGeometry(name="n8", n_groups=8, R=8, k=6, M=2, mbs=1)
    assert n8.microbatches == 8


def test_interp_clamps_do_not_extrapolate():
    assert _interp([(1, 10.0), (2, 20.0)], 0.5) == 10.0
    assert _interp([(1, 10.0), (2, 20.0)], 99) == 20.0


def test_calibration_matches_independent_ledger_numbers():
    """The fitted alpha(128)/beta must land near the numbers the ledger recorded independently -- calibration self-check.

    Internal measurement records: this machine's alpha(128)≈0.47 (back-solved from the
    64→192MB slope); beta≈118 GB/s. The fit takes a different route (least squares over
    33 sweeps, all sizes); the two are independent.
    """
    c = aug_flat()
    a128 = c.flat.alpha_ms(128)
    b = c.flat.beta_gbps(8e6)
    assert 0.35 <= a128 <= 0.60, "alpha(128)=%.3f deviates from the ledger's ~0.47" % a128
    assert 100 <= b <= 140, "beta=%.1f deviates from the ledger's ~118" % b


def test_breakeven_consistency_with_analysis():
    """The simulation core and the analytic criterion (analysis/hier_breakeven) must point the same way:

    ratio 8, q=3, zero implementation overhead -> two-hop wins; flat (1.03), q=3,
    PyTorch chain -> two-hop loses. These are two independent implementations of the
    same ledger; if the directions twist apart, one side computed wrong.
    """
    g = MoEGeometry(name="x", n_groups=16, R=8, k=6, M=2, mbs=1)
    hier = synthetic(8.0, chain_us_per_row=0.0)
    assert one_hop_call(hier, g) > two_hop_call(hier, g)
    flat = synthetic(1.03, chain_us_per_row=2.15 * 1000 / 24576)
    assert one_hop_call(flat, g) < two_hop_call(flat, g)


def test_tier1_micro_gate_passes():
    """Tier-1 (communication micro level) must currently pass -- it is the precondition for any extrapolation."""
    from sim.validate_micro import validate_micro
    ok, info = validate_micro(aug_flat(), verbose=False)
    assert ok, "Tier-1 dropped: median=%.3f worst=%.3f cross=%s" % (
        info["median"], info["worst"], info["cross"])


def test_tier2_step_gate_currently_fails_documented():
    """Tier-2 (step level) must currently **fail** -- this is the documented known state, not an aspiration.

    It is blocked by the phase-ledger/step-ledger contradiction in the internal
    measurement records (phase delta x call count ≈ 5x the step-level delta; the two
    arms overlap differently on dual streams, so phase spans do not add up to step time).
    **If this test ever turns red (the gate suddenly passes), good news does not follow
    automatically** -- it means someone changed the model or the data; a human must check
    whether the internal-measurement-records contradiction was truly resolved or the gate
    was merely loosened.
    """
    from sim.validate import validate
    ok, info = validate(aug_flat(), verbose=False)
    assert not ok, (
        "Tier-2 suddenly passed (MAE=%.4f). Hold the celebration: check whether the "
        "internal-measurement-records contradiction was truly resolved (overlap-aware "
        "composition + its own holdout points), or the gate was loosened." % info["mae"])


def test_sweep_internal_external_consistency():
    """The extrapolation's two anchors: the flat column reproduces the negative verdict (≤1), the 8x column reproduces the public positive results (>1)."""
    from sim.sweep import comm_speedup
    flat = synthetic(1.03)
    hier = synthetic(8.0)
    assert comm_speedup(flat, 3, 4096) <= 1.0
    assert comm_speedup(hier, 3, 4096) > 1.0


# ---------------------------------------------------------------- overlap model families


def test_overlap_families_all_fail_documented():
    """All six single-parameter overlap model families must currently **fail the gate** -- the documented negative result (docs/07).

    Same logic as the Tier-2 gate: a family suddenly passing retrodiction one day does
    not unlock anything -- serially evaluating six families on the same holdout set is
    itself model selection; unlocking only recognizes fresh holdout points collected per
    the docs/07 protocol plus human review. This test turning red = someone changed the
    model/data; investigate first.
    """
    from sim.overlap import evaluate
    from sim.calibrate import flat_supernode
    res = evaluate(flat_supernode(), verbose=False)
    assert len(res) == 6
    passed = [k for k, r in res.items() if r["gate"]]
    assert not passed, (
        "Families %s suddenly passed retrodiction. This is not an unlock: passing "
        "retrodiction = model selection on seven points; collect fresh holdout points "
        "per the docs/07 protocol, and only after human review does the Tier-2 gate move." % passed)


def test_overlap_family_structure_pins():
    """Pin the two structural facts behind the negative result (the reading basis for docs/07 §1):

    - M4 (hide ∝ compute) gets every direction right (5/5) but misses the MAE gate by about 2x;
    - M2 (hide per call) is closest in magnitude yet flips sign on the scale axis (signs <5).
    If these two numbers change = the calibration or the model moved; the docs/07 table
    must be re-issued in step.
    """
    from sim.overlap import evaluate
    from sim.calibrate import flat_supernode
    res = evaluate(flat_supernode(), verbose=False)
    assert res["M4"]["signs_ok"] == 5 and 0.035 <= res["M4"]["mae"] <= 0.07
    assert res["M2"]["signs_ok"] < 5 and res["M2"]["mae"] <= 0.08
    assert res["M0"]["mae"] >= 0.10   # failure magnitude of the naive baseline (~0.14)
    # MAE snapshot pin for the docs/07 §1 table (±0.005): any move in the calibration
    # constants turns this red, a reminder to re-issue the docs/07 table in step
    # (review found a loose pin failed to catch a 20% drift in the arrival-chain constant)
    for fam, doc in (("M0", 0.141), ("M1", 0.149), ("M2", 0.065),
                     ("M3", 0.140), ("M4", 0.048), ("M5", 0.087)):
        assert abs(res[fam]["mae"] - doc) <= 0.005,             "%s MAE=%.4f deviates from the docs/07 snapshot %.3f" % (fam, res[fam]["mae"], doc)


# ---------------------------------------------------------------- uncertainty


def test_mc_bands_reproducible_and_anchor_robust():
    """Fixed-seed Monte Carlo must be bit-for-bit reproducible; both robustness anchors must hold:

    - flat column, most favorable case (zero implementation overhead), p95 ≤ 1: the
      negative verdict is robust to calibration error;
    - 8x column, least favorable case (PyTorch chain), p5 > 1: the hierarchical-machine
      direction is robust.
    A broken anchor = the calibration constants or perturbation conventions changed;
    the conclusion sentences in docs/05 must change in step.
    """
    from sim.uncertainty import mc_band
    from sim.sweep import CHAIN_SCENARIOS
    a = mc_band(8.0, CHAIN_SCENARIOS[0][1])
    b = mc_band(8.0, CHAIN_SCENARIOS[0][1])
    assert a == b, "same seed, different results -- reproducibility is broken"
    _, _, flat_p95 = mc_band(1.03, CHAIN_SCENARIOS[2][1])
    assert flat_p95 <= 1.0, "flat column p95=%.3f > 1: the negative verdict is no longer robust" % flat_p95
    hier_p5, _, _ = mc_band(8.0, CHAIN_SCENARIOS[0][1])
    assert hier_p5 > 1.0, "8x column p5=%.3f ≤ 1: the direction conclusion is no longer robust" % hier_p5


def test_breakeven_ordering_and_snapshot():
    """The breakeven hierarchy ratio must fall monotonically with implementation tier and match the docs/05 snapshot (±0.1)."""
    from sim.uncertainty import breakeven_ratio
    from sim.sweep import CHAIN_SCENARIOS
    bes = [breakeven_ratio(chain) for _, chain in CHAIN_SCENARIOS]
    assert bes[0] > bes[1] > bes[2] >= 1.0
    for got, doc in zip(bes, (4.20, 1.57, 1.16)):
        assert abs(got - doc) <= 0.1, "breakeven %.2f deviates from the docs/05 snapshot %.2f" % (got, doc)


def test_heatmap_monotone_in_ratio():
    """At a fixed token tier, a larger hierarchy ratio favors two-hop more -- the ratio is monotonically non-decreasing in ratio."""
    from sim.uncertainty import heatmap
    from sim.sweep import CHAIN_SCENARIOS
    mat, ratios, toks = heatmap(CHAIN_SCENARIOS[1][1])
    for row in mat:
        assert all(row[j + 1] >= row[j] - 1e-9 for j in range(len(row) - 1))


def test_geometry_grid_bounds():
    """All 54 geometry-grid breakevens fall in [1, 32] with no absurd values (>3)."""
    from sim.uncertainty import geometry_grid
    from sim.sweep import CHAIN_SCENARIOS
    rows = geometry_grid(CHAIN_SCENARIOS[1][1])
    assert len(rows) == 54
    assert all(1.0 <= be <= 3.0 for *_, be in rows),         "some geometry's breakeven is out of bounds: %s" % [r for r in rows if not 1.0 <= r[-1] <= 3.0]
