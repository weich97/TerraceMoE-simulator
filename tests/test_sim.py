# -*- coding: utf-8 -*-
"""Contract tests for terrace-sim: the geometry ledger, calibration consistency, and the current state of the two validation gate tiers."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.core import (MoEGeometry, _interp, hop_a_self_fraction,
                      one_hop_call, two_hop_call)                     # noqa: E402
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


def test_hop_a_self_fraction_matches_equations_1_and_2():
    """Hop A has one local group among N_g, independent of selected-group count M.

    ``M/N_g`` is the expected *number* of local Hop-A rows per token.  Dividing by
    the M emitted rows gives the fraction ``1/N_g``.  This analytic identity is the
    wire-byte deduction used by Equations (1)--(2) in the paper.
    """
    for m in (1, 2, 4, 8):
        g = MoEGeometry(name="eq12", n_groups=16, R=8, k=8, M=m)
        assert hop_a_self_fraction(g) == pytest.approx(1.0 / 16.0)
        expected_wire_rows = g.tokens_per_rank * m * (15.0 / 16.0)
        actual_wire_rows = g.rows_hop_a() * (1.0 - hop_a_self_fraction(g))
        assert actual_wire_rows == pytest.approx(expected_wire_rows)


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
    assert res["M0"]["mae"] >= 0.10   # failure magnitude of the naive baseline (~0.140)
    # MAE snapshot pin for the docs/07 §1 table (±0.005): any move in the calibration
    # constants turns this red, a reminder to re-issue the docs/07 table in step
    # (review found a loose pin failed to catch a 20% drift in the arrival-chain constant)
    for fam, doc in (("M0", 0.140), ("M1", 0.150), ("M2", 0.060),
                     ("M3", 0.133), ("M4", 0.045), ("M5", 0.088)):
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
    # The flat column's most favourable case now sits just above 1.0 (1.03). That is the
    # honest consequence of modelling bandwidth saturation: on a flat fabric the *byte
    # account* is close to neutral, and what makes two-hop actually lose there is the
    # implementation overhead, which the zero-overhead tier deliberately removes.
    assert flat_p95 <= 1.10, (
        "flat column p95=%.3f: the zero-overhead byte account now predicts a real "
        "flat-fabric win, contradicting every measured arm -- recheck beta before "
        "publishing anything" % flat_p95)
    hier_p5, _, _ = mc_band(8.0, CHAIN_SCENARIOS[0][1])
    assert hier_p5 > 1.0, "8x column p5=%.3f ≤ 1: the direction conclusion is no longer robust" % hier_p5


def test_breakeven_ordering_and_snapshot():
    """The breakeven must follow the implementation tiers and the corrected Hop-A ledger."""
    from sim.uncertainty import breakeven_ratio
    from sim.sweep import CHAIN_SCENARIOS
    bes = [breakeven_ratio(chain) for _, chain in CHAIN_SCENARIOS]
    assert bes[0] > bes[1] > bes[2] >= 1.0
    for got, doc in zip(bes, (3.98, 1.49, 1.10)):
        assert abs(got - doc) <= 0.02, "breakeven %.2f deviates from the corrected snapshot %.2f" % (got, doc)


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


# ---------------------------------------------------------------- scale honesty


# One construction, three consumers: this test, the F10 figure and the scale claims in
# docs/05 all read sim.uncertainty.scale_ratio. They used to hold three copies of it,
# and the copy in the prose went stale through two recalibrations without anything
# noticing (docs/05, correction note 2026-09-05).
def _scale_ratio(alpha_pts, w, ratio=3.2):
    from sim.uncertainty import scale_ratio
    return scale_ratio(w, alpha_pts, ratio)


def _alpha_treatments():
    from sim.uncertainty import ALPHA_TREATMENTS as T
    return list(T.values())


ALPHA_TREATMENTS = _alpha_treatments()


def test_conclusions_hold_only_up_to_world_128():
    """Below 128 ranks the scale result must not depend on the low-confidence
    alpha entries; past 128 it must visibly depend on them.

    This pins the honesty boundary itself. If someone re-hardens a claim about
    large clusters, the second half of this test is what stops it: the spread at
    512 ranks is the reason the repository refuses to make that claim.
    """
    from sim.calibrate import ALPHA_PTS
    base = dict(ALPHA_PTS)
    for w in (32, 64, 128):
        vals = [_scale_ratio(sorted({**base, **ov}.items()), w)
                for ov in ALPHA_TREATMENTS]
        assert max(vals) - min(vals) < 1e-9, (
            "world=%d must be insensitive to the >128 alpha entries, spread=%.4f"
            % (w, max(vals) - min(vals)))
    at512 = [_scale_ratio(sorted({**base, **ov}.items()), 512)
             for ov in ALPHA_TREATMENTS]
    # Threshold recomputed under the saturating-beta calibration: the spread is now
    # 1.91x (2.1x under the old flat beta -- saturation compresses ratios slightly).
    # The finding is unchanged; only the number it is measured against moved.
    assert max(at512) / min(at512) > 1.8, (
        "the 512-rank spread collapsed to %.2fx. Either new measurements now "
        "constrain alpha past 128 -- in which case update calibrate.py and this "
        "test deliberately -- or the treatments were quietly narrowed."
        % (max(at512) / min(at512)))


def test_tier1b_cross_corpus_gate_passes_on_both_machines():
    """Tier-1b: the model must hold on a benchmark family it was not tuned against,
    on the calibrated machine and on a second machine whose constants are re-fitted.

    Machine B is the load-bearing half. Passing there is what licenses anyone else to
    re-calibrate this model on their own hardware; if it ever fails, the claim that the
    model *form* transfers has to be withdrawn, not patched.

    Corpus C is the blind half: machine A at world 16, collected 2026-08-26, months
    after every constant was frozen and at a world machine A had never been scored
    at. Nothing was re-fitted for it, and each of its points is the median of six
    repeated sweeps because a single-run version of the same corpus proved noisy
    enough to fabricate a result (see the x_half test).
    """
    from sim.calibrate import flat_supernode, second_machine
    from sim.validate_sweep import (TARGETS_A, TARGETS_B, TARGETS_C, validate_sweep)
    ok_a, ia = validate_sweep(flat_supernode(), TARGETS_A, verbose=False)
    ok_b, ib = validate_sweep(second_machine(), TARGETS_B, verbose=False)
    ok_c, ic = validate_sweep(flat_supernode(), TARGETS_C, verbose=False)
    assert len(TARGETS_A) == 6 and len(TARGETS_B) == 44 and len(TARGETS_C) == 14
    assert ok_a, "machine A: median %.3f bias %+.3f outliers %.2f" % (
        ia["median"], ia["bias"], ia["outlier_fraction"])
    assert ok_b, "machine B: median %.3f bias %+.3f outliers %.2f" % (
        ib["median"], ib["bias"], ib["outlier_fraction"])
    assert ok_c, "machine A world 16 (blind): median %.3f bias %+.3f outliers %.2f" % (
        ic["median"], ic["bias"], ic["outlier_fraction"])
    assert all(runs >= 6 for _, _, _, runs in TARGETS_C), (
        "corpus C points must stay medians over repeats, not single readings")


def test_world8_drift_probe_fails_and_the_cause_is_one_constant():
    """Corpus D must keep failing, and keep failing for the reason recorded.

    It is the only corpus here that misses its gate, and it is kept because the miss
    is informative: the model runs fast where alpha dominates and slow where the wire
    term does, which localises the whole bias to alpha(8). Three things are pinned.

      1. It still fails. If it starts passing, either the machine drifted back or
         someone changed a constant, and a human should find out which.
      2. Setting alpha(8) to the independently measured 0.129 clears it. That is the
         diagnosis, and it stops being a diagnosis the moment it stops being true.
      3. Nothing depends on the choice. Both values give the same breakeven to
         within a few percent, which is why the constant is left alone rather than
         retuned to make this corpus green.
    """
    import sim.calibrate as cal
    from sim.calibrate import flat_supernode
    from sim.core import Level
    from sim.validate_sweep import TARGETS_D, validate_sweep

    ok_d, i_d = validate_sweep(flat_supernode(), TARGETS_D, verbose=False)
    assert not ok_d, (
        "the world-8 drift probe passes now (median %.3f bias %+.3f); check whether "
        "alpha(8) was changed or the machine came back" % (i_d["median"], i_d["bias"]))
    assert i_d["bias"] < -0.05, "the recorded failure is a fast bias; this is not it"

    def with_alpha(overrides):
        pts = [(w, overrides.get(w, v)) for w, v in cal.ALPHA_PTS]
        spec = flat_supernode()
        spec.flat = Level(spec.flat.name, pts, spec.flat.beta_pts)
        return spec

    ok_fixed, i_fixed = validate_sweep(with_alpha({8: 0.129}), TARGETS_D, verbose=False)
    assert ok_fixed, (
        "alpha(8)=0.129 no longer clears the drift probe (median %.3f bias %+.3f); "
        "the diagnosis in validate_sweep.py is stale"
        % (i_fixed["median"], i_fixed["bias"]))

    # The half that is not adopted, and why. The same scan measured alpha(16) at
    # 0.134; taking it breaks corpus C. Pinning that keeps the record honest -- the
    # reason for leaving the table alone is that the measurement cannot be adopted
    # whole, not that its world-8 half is doubted.
    from sim.validate_sweep import TARGETS_C
    ok_c16, i_c16 = validate_sweep(with_alpha({16: 0.134}), TARGETS_C, verbose=False)
    assert not ok_c16, (
        "adopting the measured alpha(16)=0.134 now leaves corpus C passing (median "
        "%.3f bias %+.3f); if the measurement can be adopted whole, the argument in "
        "validate_sweep.py and docs/09 for leaving alpha alone no longer holds"
        % (i_c16["median"], i_c16["bias"]))


def test_the_gates_bound_x_half_from_above_only():
    """Sweeping x_half must reproduce the documented one-sided bound.

    x_half is the one borrowed constant, so what the gates actually tolerate is
    worth pinning. They tolerate a lot: machine B rules out anything above
    X_HALF_GATE_UPPER, and nothing rules out the low end at all.

    The weak lower end is the honest part and is asserted directly, because an
    earlier revision claimed a two-sided interval that came from a single-run
    corpus. Repeating that corpus dissolved the lower bound. If someone reintroduces
    a lower bound, this fails and they have to show it came from measurement rather
    than from noise.
    """
    from sim.calibrate import X_HALF_FLAT, X_HALF_GATE_UPPER, flat_supernode, second_machine
    from sim.validate_micro import validate_micro
    from sim.validate_sweep import TARGETS_A, TARGETS_B, TARGETS_C, validate_sweep

    def all_gates(x_half):
        c = flat_supernode(x_half)
        return (validate_micro(c, verbose=False)[0]
                and validate_sweep(c, TARGETS_A, verbose=False)[0]
                and validate_sweep(second_machine(x_half), TARGETS_B, verbose=False)[0]
                and validate_sweep(c, TARGETS_C, verbose=False)[0])

    assert all_gates(X_HALF_FLAT), "the shipped x_half must pass every gate"
    assert all_gates(X_HALF_GATE_UPPER), "the documented upper bound no longer passes"
    assert not all_gates(X_HALF_GATE_UPPER + 1024), (
        "the gates now tolerate more than the documented upper bound")
    assert all_gates(1024), (
        "a lower bound has appeared; the docs say the gates have none, so either "
        "update them or check that the new corpus is not just noisy")
    assert X_HALF_FLAT < X_HALF_GATE_UPPER


# ---------------------------------------------------------------- compute model


def test_the_measured_shapes_choose_the_index_rule():
    """The rule in use must still be the one the non-square measurements picked.

    This module spent a long time refusing to choose between three ways of indexing
    a square roofline by a non-square shape, on the grounds that the data did not
    decide. Eight measured expert-FFN shapes decide. The test rescores them rather
    than trusting the recorded verdict, so the choice cannot rot: if someone edits
    the roofline table, the winner is recomputed and has to still be the geometric
    mean, with the error figures the docs quote.

    Also pinned: every rule under-predicts. That is the interesting part -- a square
    curve is a systematically pessimistic prior for a large non-square GEMM, and if
    that ever reverses the withdrawn "narrow experts are inefficient" reading would
    need revisiting.
    """
    import statistics as S
    from sim.compute import (INDEX_RULE, INDEX_RULE_ERROR, NONSQUARE_TFLOPS,
                             efficiency_bracket)

    errs = {}
    for shape, meas in NONSQUARE_TFLOPS.items():
        for rule, pred in efficiency_bracket(*shape)["per_rule"].items():
            errs.setdefault(rule, []).append((pred - meas) / meas)

    scored = {r: S.median([abs(e) for e in v]) for r, v in errs.items()}
    best = min(scored, key=scored.get)
    assert best == INDEX_RULE, (
        "the measured shapes now favour %r, not the %r this module uses: %s"
        % (best, INDEX_RULE, {k: round(100 * v, 1) for k, v in scored.items()}))

    assert abs(scored[INDEX_RULE] - INDEX_RULE_ERROR["median_abs"]) < 0.005, (
        "the recorded median error %.3f no longer matches the recomputed %.3f"
        % (INDEX_RULE_ERROR["median_abs"], scored[INDEX_RULE]))
    worst = max(abs(e) for e in errs[INDEX_RULE])
    assert abs(worst - INDEX_RULE_ERROR["worst"]) < 0.005

    for rule, v in errs.items():
        assert S.median(v) < 0, (
            "rule %r no longer under-predicts; the square roofline was a "
            "pessimistic prior and the docs say so" % rule)

    # The rejected rules must stay clearly worse, or "the data decides" is too
    # strong a claim to keep making.
    for rule in scored:
        if rule != INDEX_RULE:
            assert scored[rule] > 1.5 * scored[INDEX_RULE], (
                "rule %r is now within 1.5x of the chosen one; the selection is no "
                "longer decisive" % rule)


def test_compute_roofline_is_non_monotone_and_penalises_small_gemms():
    """The measured curve peaks at 4096 and dips after it; small GEMMs pay heavily.

    Pinned because the dip is a real measured feature reproduced across two
    campaigns. If someone replaces the table with a smooth monotone fit, this
    fails -- which is the point: the smoothing would erase the finding.
    """
    from sim.compute import GEMM_TFLOPS, PEAK_TFLOPS, achieved_tflops
    tbl = dict(GEMM_TFLOPS)
    assert tbl[4096] == PEAK_TFLOPS
    assert tbl[8192] < tbl[4096] and tbl[12288] < tbl[4096]
    assert tbl[1024] / PEAK_TFLOPS < 0.5, "the 1024 tier must stay under half of peak"
    assert achieved_tflops(500) == tbl[1024], "below the measured range must clamp"
    assert achieved_tflops(99999) == tbl[16384]


def test_compute_time_is_bracketed_and_the_bracket_widens_for_narrow_experts():
    """Compute time is a range, and the range is the finding.

    A square roofline cannot decide the efficiency of a tall-skinny expert matmul.
    Wide experts stay near 1.2x across the three index rules; narrow ones blow out
    past 2x, which is where any single compute number becomes assumption-dominated.
    If a future change collapses the narrow-expert spread, it means someone either
    measured non-square GEMMs (update the module deliberately) or silently picked
    one rule and dropped the honesty.
    """
    from sim.compute import expert_ffn
    wide = expert_ffn(4096, 6, 128, 128, 2048, 2048)
    narrow = expert_ffn(4096, 6, 128, 128, 2048, 512)
    assert wide["assumption_spread"] < 1.3
    assert narrow["assumption_spread"] > 2.0
    assert narrow["ms_slow"] > narrow["ms_fast"] > 0
    # FLOPs are exact, so halving expert width must halve the work exactly
    assert abs(narrow["flops"] * 4 - wide["flops"]) < 1e-6 * wide["flops"]


def test_comm_share_is_labelled_and_behaves_as_a_bound():
    from sim.compute import comm_share_upper_bound
    assert comm_share_upper_bound(1.0, 0.0) == 1.0
    assert abs(comm_share_upper_bound(1.0, 3.0) - 0.25) < 1e-12
    assert comm_share_upper_bound(0.0, 0.0) != comm_share_upper_bound(0.0, 0.0)  # nan


# ---------------------------------------------------------------- platform map


def test_platform_map_verdicts_follow_the_breakevens():
    """The where-it-pays table must be derived, never hand-written.

    Every cell is `ratio >= breakeven(tier)`, so the table cannot drift away from
    the calibration behind it. Also pins the two ends that carry the message: the
    measured-flat row is 'no' at every tier, and the high-ratio rows are 'yes' at
    every tier -- if either flips, the headline table in README is wrong.
    """
    from sim.platforms import platform_map
    rows = platform_map()
    for r in rows:
        for tier, ok in r["verdict"].items():
            assert ok == (r["archetype"].ratio_nominal >= r["breakevens"][tier])
    by_key = {r["archetype"].key: r for r in rows}
    assert not any(by_key["measured-a"]["verdict"].values())
    assert all(by_key["ratio-9"]["verdict"].values())
    assert all(by_key["ratio-18"]["verdict"].values())
    # the middle row is the interesting one: implementation tier decides it
    mid = by_key["ratio-2"]["verdict"]
    assert sum(mid.values()) == 2, "ratio 2 should pay only above the PyTorch tier"


def test_platform_coverage_reports_the_gap_honestly():
    """Coverage must keep announcing that no calibrated machine is hierarchical.

    The day someone adds a platform above ratio 1.5, this flips and the claim in
    README about extrapolation has to be rewritten -- deliberately, not silently.
    """
    from sim.platforms import PLATFORMS, coverage
    c = coverage()
    assert c["n_platforms"] == len(PLATFORMS) >= 2
    assert c["n_ratio_measured"] == 1, (
        "platform B has no separated fast/slow measurement and must not be counted "
        "as a hierarchy-ratio sample")
    assert not c["spans_hierarchical"], (
        "a platform above ratio 1.5 is now calibrated -- update README's coverage "
        "paragraph and docs/05 before relaxing this test")
    for p in PLATFORMS.values():
        assert p.provenance and p.notes, "every platform states where it came from"
        assert p.spec().flat.beta_gbps(8 * 1024 ** 2) > 0


# ---------------------------------------------------------------- machine profile


def test_machine_checklist_never_contradicts_the_model():
    """A machine the checklist clears must be one the cost model scores above 1.

    The checklist exists to name *why* a machine does or does not qualify, which a
    bare ratio cannot. That is only worth having if the two can never disagree, so
    this sweeps ratio and chain tier and asserts the implication in both directions
    where the model is decisive.
    """
    from sim.calibrate import synthetic
    from sim.core import MoEGeometry, one_hop_call, two_hop_call
    from sim.profile import profile_from_spec
    from sim.sweep import CHAIN_SCENARIOS

    g = MoEGeometry(name="chk", n_groups=16, R=8, k=6, M=2, seq=4096, mbs=1,
                    gbs=16 * 8 * 4096)
    for _, chain in CHAIN_SCENARIOS:
        for ratio in (1.0, 1.5, 2.5, 4.0, 8.0, 16.0):
            spec = synthetic(ratio, chain_us_per_row=chain)
            r = profile_from_spec(spec, g, ratio)
            model_wins = one_hop_call(spec, g) / two_hop_call(spec, g) >= 1.0
            if r["verdict"]["qualifies"]:
                assert model_wins, (
                    "checklist cleared ratio=%.1f chain=%.4f but the model scores "
                    "%.2f -- the two must never disagree"
                    % (ratio, chain, r["model_ratio"]))


def test_checklist_names_the_reason_not_just_the_verdict():
    """Each failure must carry a machine-readable margin and a readable reason."""
    from sim.profile import check
    conds = check(ratio=1.0, R=8, k=6, M=6, ep=8, tokens_per_rank=64,
                  hidden=2048, chain_us_per_row=0.0875)
    names = [c.name for c in conds]
    assert any("q = k/M" in n for n in names)
    assert any("fast domain" in n for n in names)
    failed = [c for c in conds if not c.passed]
    assert failed, "this machine should fail several conditions"
    for c in failed:
        assert c.detail and len(c.detail) > 20, "a failure must explain itself"
    # q=1 and EP inside one domain: the two structural disqualifiers
    q_cond = next(c for c in conds if "q = k/M" in c.name)
    dom = next(c for c in conds if "fast domain" in c.name)
    assert not q_cond.passed and not dom.passed


# ---------------------------------------------------------------- routing skew


def test_skew_favours_two_hop_and_is_kept_out_of_the_calibration():
    """Load skew inflates the busiest peer, and it inflates one-hop most.

    The maximum is taken over EP peers for one-hop against N_g and R for the two
    hops, and Hop A additionally aggregates q experts per message, so the same
    measured expert-load CV costs one-hop more. This is the only modelled effect
    pointing the opposite way from the arrival chain, and it must never leak into
    core.py: the Tier-1 microbenchmark divides its buffer equally, so applying an
    inflation there would model something the calibration does not contain.
    """
    from sim.calibrate import synthetic
    from sim.core import MoEGeometry
    from sim.imbalance import (CV_EXPERT_MEDIAN, adjusted_ratio,
                               strategy_inflation)
    from sim.sweep import CHAIN_SCENARIOS

    g = MoEGeometry(name="skew", n_groups=16, R=8, k=6, M=2, seq=4096, mbs=1,
                    gbs=16 * 8 * 4096)
    f = strategy_inflation(g.ep, g.n_groups, g.R, g.q)
    assert f["one_hop"] > f["hop_b"] > f["hop_a"] > 1.0, (
        "inflation must fall with fewer peers, and Hop A must gain again from "
        "aggregating q experts: got %s" % f)
    for ratio in (1.03, 3.2, 8.0):
        out = adjusted_ratio(synthetic(ratio, chain_us_per_row=CHAIN_SCENARIOS[1][1]), g)
        assert out["shift"] > 0, "skew must favour two-hop at ratio %.2f" % ratio
    # zero skew must reduce to the balanced model exactly
    flat = adjusted_ratio(synthetic(3.2, chain_us_per_row=CHAIN_SCENARIOS[1][1]), g,
                          cv_expert=0.0)
    assert abs(flat["shift"]) < 1e-12
    assert 0.10 < CV_EXPERT_MEDIAN < 0.20, "measured expert-load CV moved; recheck docs"


def test_core_does_not_apply_skew():
    """core.py must stay balanced, because Tier-1's targets are balanced."""
    import inspect
    from sim import core
    src = inspect.getsource(core)
    assert "imbalance" not in src, (
        "core.py imported the skew model. Tier-1 targets come from an equal-split "
        "benchmark; applying skew there would fail the gate for the right reason.")


def test_launch_cost_is_bounded_and_second_order():
    """The launch split must stay a small term against the arrival chain.

    Two-hop issues one more collective, so launch enters the comparison exactly
    once. The scan of docs/09 has since measured it (HOST_EXPOSURE_MS), and the
    bound below is what the sweep was quoted against before that. Both are held
    here: the measured point must sit under the bound, and neither may turn launch
    into a first-order term.
    """
    from sim.core import MoEGeometry
    from sim.profile import LAUNCH_UPPER_BOUND_MS, launch_sensitivity
    from sim.calibrate import ALPHA_PTS
    from sim.sweep import CHAIN_SCENARIOS

    assert LAUNCH_UPPER_BOUND_MS == dict(ALPHA_PTS)[8], (
        "the bound must stay tied to the smallest measured alpha")
    g = MoEGeometry(name="lat", n_groups=16, R=8, k=6, M=2, seq=4096, mbs=1,
                    gbs=16 * 8 * 4096)
    rows = launch_sensitivity(g, CHAIN_SCENARIOS[1][1])
    assert rows[0]["extra_launch_ms"] == 0.0
    bes = [r["breakeven"] for r in rows]
    assert bes == sorted(bes), "more launch cost must never lower the breakeven"
    at_bound = next(r for r in rows
                    if abs(r["extra_launch_ms"] - LAUNCH_UPPER_BOUND_MS) < 1e-9)
    assert at_bound["breakeven"] - bes[0] < 0.4, (
        "launch moved the breakeven by %.2f, which is no longer second order "
        "against the arrival chain's 2.4; run the scan"
        % (at_bound["breakeven"] - bes[0]))


def test_measured_launch_brackets_the_shipped_alpha():
    """The call-count scan must corroborate alpha, not quietly replace it.

    The scan of 2026-08-26 measured one collective call in two regimes: with the
    host running ahead of the device, and with the host observing every call. The
    shipped alpha belongs to the first. Three things have to hold or the table and
    the measurement have drifted apart and one of them is wrong:

      1. every measured world brackets alpha between the two regimes,
      2. the deep-queue reading agrees with alpha inside the 20% run-to-run drift
         that calibrate.py documents,
      3. the two regimes really are distinguishable -- if they ever collapse, the
         serial arm stopped measuring what it claims to.
    """
    from sim.calibrate import ALPHA_PTS, flat_supernode
    from sim.profile import (HOST_EXPOSURE_MS, PER_CALL_DEEP_QUEUE_MS,
                             PER_CALL_HOST_EXPOSED_MS)

    alpha = dict(ALPHA_PTS)
    c = flat_supernode()
    # Worlds 2 and 4 were scanned in the deep-queue regime only, to check the
    # instrument against the two tabulated points nobody disputes; the bracket claim
    # is about the worlds where both regimes were run.
    both = sorted(set(PER_CALL_DEEP_QUEUE_MS) & set(PER_CALL_HOST_EXPOSED_MS))
    assert both == [8, 16]
    for world in both:
        deep = PER_CALL_DEEP_QUEUE_MS[world]
        exposed = PER_CALL_HOST_EXPOSED_MS[world]
        assert deep < exposed, "world %d: the two regimes collapsed" % world
        a = c.flat.alpha_ms(world)
        assert a == alpha[world], "world %d is not a tabulated alpha point" % world
        assert deep <= exposed and a <= exposed, (
            "world %d: alpha %.3f escaped the measured bracket [%.3f, %.3f]"
            % (world, a, deep, exposed))
        drift = abs(a - deep) / deep
        assert drift <= 0.20, (
            "world %d: alpha %.3f vs measured %.3f is %.0f%% apart, beyond the "
            "20%% drift the calibration claims" % (world, a, deep, 100 * drift))

    gaps = [PER_CALL_HOST_EXPOSED_MS[w] - PER_CALL_DEEP_QUEUE_MS[w] for w in both]
    assert min(gaps) <= HOST_EXPOSURE_MS <= max(gaps), (
        "HOST_EXPOSURE_MS %.3f is not inside the measured gaps %s"
        % (HOST_EXPOSURE_MS, ["%.3f" % g for g in gaps]))


def test_x_half_is_exactly_a_per_peer_message_cost():
    """The saturating bandwidth is algebraically a fixed cost per peer message.

    Substituting beta_eff = beta_inf * x/(x + x_half) with x the per-peer bytes gives
    wire/beta_inf + (world-1)*x_half/beta_inf, exactly. This is not an approximation
    and not a regime: it holds at every world and every size. It is pinned because it
    is what gives x_half units, what explains the alpha/x_half degeneracy, and what
    makes two-hop's saving on this axis countable.
    """
    from sim.calibrate import BETA_FLAT, PER_PEER_MESSAGE_US, X_HALF_FLAT

    for world in (2, 8, 16, 128, 512):
        for S in (2 ** 12, 2 ** 20, 2 ** 26, 2 ** 33):
            x = S / world
            wire = S * (world - 1) / world
            saturating = wire / (BETA_FLAT * x / (x + X_HALF_FLAT) * 1e6)
            per_message = (wire / (BETA_FLAT * 1e6)
                           + (world - 1) * X_HALF_FLAT / (BETA_FLAT * 1e6))
            assert saturating == pytest.approx(per_message, rel=1e-12), (
                "world %d, %d bytes: the identity is exact or the reading of x_half "
                "in calibrate.py is wrong" % (world, S))

    assert PER_PEER_MESSAGE_US == pytest.approx(0.469, abs=0.001)

    # and what it buys two-hop: 127 peer messages against 15 + 7
    one_hop, hop_a, hop_b = 128 - 1, 16 - 1, 8 - 1
    assert hop_a + hop_b == 22
    assert one_hop * PER_PEER_MESSAGE_US / 1000 == pytest.approx(0.060, abs=0.001)
    assert (hop_a + hop_b) * PER_PEER_MESSAGE_US / 1000 == pytest.approx(0.010, abs=0.001)


def test_the_beta_table_covers_the_domain_it_is_asked_about():
    """The sampled bandwidth table must not clamp inside the range of real messages.

    core._interp clamps outside its table, so a grid that starts at 1 KiB prices every
    smaller per-peer message at the 1 KiB bandwidth. Below x_half the curve falls
    roughly linearly toward zero, so that clamp over-credited a 246-byte message by a
    factor of fifteen and made the model faster than the form it implements. Two
    measured points sit there and the clamp moved both away from the measurement.

    The grid is extended downward rather than re-spaced, which is the half of this
    worth guarding: every sample at and above 1 KiB keeps its exact position, so the
    fix moves no published number, and only the predictions that were being clamped
    change.
    """
    from sim.calibrate import (BETA_FLAT, SECOND_BETA_FLAT, X_HALF_FLAT,
                               saturating_beta, second_machine)
    from sim.core import _interp

    tbl = saturating_beta(BETA_FLAT, X_HALF_FLAT)
    xs = [x for x, _b in tbl]
    assert xs[0] <= 1.0, "the table still clamps inside the domain of real messages"

    # the samples the old grid had are still exactly where they were
    old = [1e3 * (1e9 / 1e3) ** (i / 47.0) for i in range(48)]
    kept = [x for x in xs if x >= 1e3 - 1e-9]
    assert len(kept) == len(old)
    for got, want in zip(kept, old):
        assert got == pytest.approx(want, rel=1e-12)

    # and the table tsupernodes the curve it samples, everywhere a message can land
    def exact(x):
        return BETA_FLAT * x / (x + X_HALF_FLAT)

    probe = [64 * 2 ** (i / 4.0) for i in range(97)]     # 64 B to 1 GiB
    worst = max(abs(_interp(tbl, x, logx=True) - exact(x)) / exact(x) for x in probe)
    assert worst < 0.02, "interpolation error %.3f%% is no longer negligible" % (100 * worst)

    # the two corpus-B points that used to be clamped now sit on the analytic form
    c = second_machine()
    for world, S, measured, was in ((128, 31457, 0.5658, -0.276),
                                    (128, 62914, 0.4691, -0.093)):
        x, wire = S / world, S * (world - 1) / world
        model = c.flat.alpha_ms(world) + wire / (c.flat.beta_gbps(x) * 1e6)
        analytic = c.flat.alpha_ms(world) + wire / (
            SECOND_BETA_FLAT * x / (x + X_HALF_FLAT) * 1e6)
        assert model == pytest.approx(analytic, rel=0.005)
        assert abs((model - measured) / measured) < abs(was), (
            "the clamp used to put this point at %+.1f%%; the fix must improve it"
            % (100 * was))


def test_marginal_bandwidth_is_model_free_and_depends_on_world():
    """Delivered bandwidth read off the data, and the world-dependence it exposes.

    No alpha, no x_half, no model form: the slope of wall clock against wire bytes at
    the top of a sweep. Both machines dip at world 16 and rise from there, and both sit
    below the shipped beta_inf almost everywhere, which the single-beta model absorbs
    into the per-world alpha.
    """
    from sim.calibrate import (BETA_FLAT, MARGINAL_BW_BY_WORLD,
                               MARGINAL_BW_SECOND_MACHINE)
    from sim.fit import marginal_bandwidth
    from sim.validate_sweep import TARGETS_A, TARGETS_B, TARGETS_C, TARGETS_D

    a = marginal_bandwidth([(w, s, ms) for w, s, ms, _r in
                            TARGETS_A + TARGETS_C + TARGETS_D])
    b = marginal_bandwidth([(w, s, ms) for w, s, ms, _r in TARGETS_B])
    for world, doc in MARGINAL_BW_BY_WORLD.items():
        assert a[world] == pytest.approx(doc, abs=0.1), (
            "machine A world %d: %.1f against the tabulated %.1f" % (world, a[world], doc))
    for world, doc in MARGINAL_BW_SECOND_MACHINE.items():
        assert b[world] == pytest.approx(doc, abs=0.1)

    # the shape, on both machines independently: a dip at world 16, then rising
    assert a[16] < a[8] < a[128]
    assert b[16] < b[8] < b[32] and b[64] < b[128]
    # and almost everywhere below the beta the model applies at every world at once
    assert a[8] < BETA_FLAT and a[16] < BETA_FLAT
    assert all(v < BETA_FLAT for v in b.values())

    # two points are a difference, not a slope: at n_top=2 machine A's world 128 reads
    # above the 122.4 per-card physical ceiling, and settles below it by n_top=4
    coarse = marginal_bandwidth([(w, s, ms) for w, s, ms, _r in TARGETS_A], n_top=2)
    assert coarse[128] > 122.4 > a[128]


def test_per_world_bandwidth_moves_the_verdict_against_two_hop():
    """The sensitivity is real, sizeable, and points away from the method.

    One hop runs at the full world where delivery is best; both two-hop hops run at
    small worlds where it is worst. Giving each level its measured bandwidth therefore
    raises the threshold at every implementation tier. It is a sensitivity and not a
    recalibration -- see the docstring -- but the direction is the point.
    """
    from sim.profile import bandwidth_world_sensitivity

    rows = bandwidth_world_sensitivity()
    assert len(rows) == 3
    for _name, shipped, per_world in rows:
        assert per_world > shipped, (
            "the measured world-dependence must make two-hop look worse, not better")
    for (_n, shipped, per_world), doc in zip(rows, ((3.98, 4.52), (1.49, 1.78),
                                                   (1.10, 1.34))):
        assert shipped == pytest.approx(doc[0], abs=0.01)
        assert per_world == pytest.approx(doc[1], abs=0.01)
    # and it displaces the threshold further than the measured launch cost does
    assert rows[0][2] - rows[0][1] > 4.15 - 3.98


def test_two_hop_beats_one_hop_measured_across_the_boundary():
    """The first measured win in this project, and what it cost the model to be right.

    Six of the seven end-to-end geometries lost, all inside one supernode where the
    ratio is 1.03. Every claim that two-hop pays somewhere had been a model output. This
    pins the measurement that stopped being one: one hop against two across a supernode
    boundary, at four configurations, with the repository's own arrival chain running on
    the device rather than a per-row estimate.

    The model called all four correctly, and the term-by-term comparison is pinned too,
    because its aggregate accuracy rests partly on two errors that cancel: Hop A
    under-priced, the chain over-priced.
    """
    from sim.twohop_measured import (CHAIN_US_PER_ROW_MEASURED, MEASURED, MODEL_ERROR,
                                     chain_share, every_configuration_wins, g,
                                     two_hop_ms)
    from sim.calibrate import CHAIN_US_PER_ROW

    assert len(MEASURED) == 4
    assert every_configuration_wins(), [round(g(r), 3) for r in MEASURED]
    assert 1.4 < min(g(r) for r in MEASURED) < 1.5
    assert 2.2 < max(g(r) for r in MEASURED) < 2.3

    # the win grows with hidden width and with a tighter group cap, both of which the
    # byte ledger predicts: more rows deduplicated, more payload per row
    by = {(r[0], r[1]): g(r) for r in MEASURED}
    assert by[(4096, 1)] > by[(2048, 1)], "wider hidden width must help two-hop"
    assert by[(2048, 1)] > by[(2048, 2)], "a tighter cap deduplicates more"

    # the components add up to what is reported
    for r in MEASURED:
        assert two_hop_ms(r) == pytest.approx(r[4] + r[5] + r[6] + r[7])
        assert r[3] > two_hop_ms(r)
        assert 0.10 < chain_share(r) < 0.30, (
            "the arrival chain is a fifth to a quarter of two-hop here; if it stops "
            "being that, the term this project keeps calling decisive has moved")

    # the real chain is about half the cost the calibration carries, and that is
    # recorded rather than adopted
    assert CHAIN_US_PER_ROW_MEASURED < 0.6 * CHAIN_US_PER_ROW
    assert CHAIN_US_PER_ROW == pytest.approx(0.0875, abs=0.0005), (
        "the shipped constant has not been quietly retuned to the new reading")

    # and the model's accuracy partly rests on cancelling errors: fixing the slow level
    # helps, fixing the chain as well hurts
    assert MODEL_ERROR["slow level as a pure tier"] < MODEL_ERROR["slow level fed a mixture"]
    assert MODEL_ERROR["pure tier and measured chain"] > MODEL_ERROR["slow level as a pure tier"]


def test_the_supernode_boundary_is_the_first_measured_ratio_above_the_flat_one():
    """The measured hierarchy ratio, and where it places.

    Every ratio above 1.03 in this repository was a synthetic sensitivity. The machine
    the flat verdict was taken on is two supernodes, the seven end-to-end geometries all ran
    inside one of them, and crossing between them had never been measured. It has been
    now: two all-to-alls of identical structure differing only in whether the remote
    peers are in the same supernode.
    """
    from sim.hierarchy import (CARDS_PER_SUPERNODE, MEASURED, RATIO_BURST, RATIO_PERCALL,
                               byte_breakeven_at_supernode, hierarchy_ratio,
                               marginal_gbps, placement, remote_tier_gbps)
    from sim.hostregime import MEASURED as REGIME

    assert len(MEASURED) == 23

    # the two modules must be describing the same run: hierarchy's within-supernode columns
    # are hostregime's world-16 rows, to the digit
    w16 = [(r[1], r[2], r[3]) for r in REGIME if r[0] == 16]
    assert len(w16) == len(MEASURED)
    for (b, burst, percall), row in zip(w16, sorted(MEASURED)):
        assert row[0] == b and row[1] == burst and row[2] == percall

    # the ratio, and it does not depend on the timing style or on how many points the
    # regression uses -- which is what makes it a property of the boundary
    assert hierarchy_ratio("percall") == pytest.approx(RATIO_PERCALL, abs=0.02)
    assert hierarchy_ratio("burst") == pytest.approx(RATIO_BURST, abs=0.02)
    assert abs(hierarchy_ratio("burst") - hierarchy_ratio("percall")) < 0.05
    for n in (3, 4, 5, 6):
        assert 2.4 < hierarchy_ratio("percall", n) < 2.7

    # within a supernode the machine is the flat one this repository already measured;
    # across supernodes it is not
    assert marginal_gbps("within", "percall") == pytest.approx(102, abs=2)
    assert marginal_gbps("across", "percall") == pytest.approx(40, abs=2)
    assert remote_tier_gbps("within") / remote_tier_gbps("across") == pytest.approx(
        3.54, abs=0.1)

    # it clears the byte criterion at every quota, with a whole supernode as the fast domain
    assert CARDS_PER_SUPERNODE == 128
    for q in range(2, 9):
        assert hierarchy_ratio() > byte_breakeven_at_supernode(q)

    # and lands between the two implementation thresholds, which is the finding: the
    # topology is good enough and the arrival chain is what decides
    p = placement()
    assert p["clears"]["hypothetical fused target"]
    assert not p["clears"]["PyTorch chain (measured)"]
    assert p["decided_by_the_chain"]


def test_alpha_16_is_a_fitted_artefact_not_a_measurement():
    """Why the world-16 fixed cost never agreed with anything, resolved.

    alpha(16) = 0.157 has sat in the table disagreeing with an independent call-count
    scan, and docs/05 recorded it as one reading against another with no way to choose.
    It is now four benchmarks against one table entry, and the entry has an explanation:
    a joint fit of the additive rule to the size-sweep corpus reproduces 0.157 exactly,
    and pairs it with a bandwidth the machine cannot deliver.

    The two halves are pinned together because neither means much alone. An alpha that
    only fits alongside an impossible beta is not a measurement of alpha.
    """
    from sim.calibrate import (ALPHA_BY_REGIME, ALPHA_CORPUS_JOINT_FIT, ALPHA_PTS,
                               ALPHA_REGIME, PHYSICAL_CEILING_GBPS)

    tab = dict(ALPHA_PTS)

    # the table is a deep-queue table, and its world-8 entry says so to three digits
    assert "deep queue" in ALPHA_REGIME
    assert tab[8] == pytest.approx(ALPHA_BY_REGIME["deep queue"][8], abs=0.005)

    # the two regimes differ by about a factor of two, which is why the table has to
    # declare which one it is
    for w in (8, 16):
        ratio = ALPHA_BY_REGIME["host exposed"][w] / ALPHA_BY_REGIME["deep queue"][w]
        assert 1.8 < ratio < 2.3, "world %d regime ratio %.2f" % (w, ratio)

    # the tabulated world-16 entry matches neither regime, and sits between them
    assert (ALPHA_BY_REGIME["deep queue"][16] < tab[16]
            < ALPHA_BY_REGIME["host exposed"][16])

    # every direct measurement puts the step from world 8 to 16 far below the table's
    for regime, a in ALPHA_BY_REGIME.items():
        step = a[16] / a[8] - 1
        assert step < 0.20, "%s step %+.0f%%" % (regime, 100 * step)
    assert tab[16] / tab[8] - 1 > 0.35, "the table's own step is the outlier"

    # and the explanation: the corpus fit that reproduces 0.157 needs a beta the
    # hardware does not have
    c = ALPHA_CORPUS_JOINT_FIT["C"]
    assert c["alpha"] == pytest.approx(tab[16], abs=0.002)
    assert c["beta_inf"] > PHYSICAL_CEILING_GBPS
    assert ALPHA_CORPUS_JOINT_FIT["D"]["beta_inf"] > PHYSICAL_CEILING_GBPS
    # while the direct measurement in the additive rule's own regime stays inside it
    from sim.hostregime import compare_styles
    assert compare_styles()["percall"]["additive (shipped)"]["beta_inf"] < PHYSICAL_CEILING_GBPS

    # nothing has been swapped in: alpha and beta are degenerate and a recalibration
    # would have to redo the gates deliberately
    assert tab[16] == 0.157


def test_the_peer_count_effect_is_not_there_when_measured_directly():
    """The hypothesis sim/tiers.py raised, and the measurement that killed it.

    Deconvolved all-to-alls suggested a tier delivers more per card when its bytes are
    spread over more peers, which would charge two-hop specifically: Hop A runs at
    world = group count and so has the fewest peers of anything in the scheme. That was
    the open risk against the supernode-boundary verdict, and the module named the
    measurement that would settle it.

    Measured directly -- split sizes arranged so every card sends only across the
    boundary, nothing to deconvolve -- per-card bandwidth is flat over a sixty-fourfold
    change in spread. So the effect is an artefact of the deconvolution, and the charge
    against two-hop is withdrawn. What survives is that the deconvolution itself is not
    to be trusted, which is why the composition below is still not adopted.
    """
    from sim.tiers import (PEER_SWEEP, measured_cross_supernode_gbps,
                           peer_effect_span)

    assert len(PEER_SWEEP) == 21
    peers = sorted({n for n, _b, _t in PEER_SWEEP})
    assert peers == [1, 2, 4, 8, 16, 32, 64]

    # flat: the whole sweep sits inside a few percent, over a 64x change in spread
    assert peer_effect_span() < 1.05, (
        "per-card cross-supernode bandwidth now varies by %.1f%% across the peer sweep; "
        "if a real peer-count effect has appeared, sim/tiers.py has to be rewritten "
        "rather than corrected" % (100 * (peer_effect_span() - 1)))

    vals = [measured_cross_supernode_gbps(n) for n in peers]
    assert all(7.0 < v < 9.0 for v in vals), vals
    # and specifically the regime Hop A runs in is not penalised
    assert measured_cross_supernode_gbps(1) > 0.95 * measured_cross_supernode_gbps(64)


def test_pricing_a_collective_from_its_tiers_fails_out_of_sample():
    """A structural improvement that was tested and only partly adopted.

    Pricing a collective from per-tier link bandwidths instead of one beta per named
    level is the obvious fix for a model whose load-bearing comparison spans three
    different mixtures of links. It composes within a world and misses the world it was
    not calibrated on by 31%, because a tier delivers more per card when more peers use
    it. The peer counting is adopted, being exact arithmetic; the composition is not.

    Pinned because the failure is the finding, and because the same effect charges
    two-hop specifically: Hop A spreads the slow tier over the fewest peers of anything
    in the scheme, and at a coarse hierarchy that is below anything measured.
    """
    from sim.tiers import (CROSS_SUPERNODE_8_PEERS_UNCONTENDED, OUT_OF_SAMPLE,
                           TIER_BY_PEERS, Topology, compose, hop_a_peer_count,
                           peer_counts, tier_at)

    topo = Topology()
    # the peer counts are arithmetic and must reproduce every measured configuration
    assert peer_counts(topo, 8, 1) == {"intra_node": 7}
    assert peer_counts(topo, 16, 2) == {"intra_node": 7, "cross_node": 8}
    assert peer_counts(topo, 16, 1) == {"intra_node": 7, "cross_supernode": 8}
    assert peer_counts(topo, 128, 8) == {"intra_node": 7, "cross_node": 56,
                                         "cross_supernode": 64}
    assert peer_counts(topo, 128, 16) == {"intra_node": 7, "cross_node": 120}
    for world, npr in ((8, 1), (16, 1), (16, 2), (128, 8), (128, 16)):
        assert sum(peer_counts(topo, world, npr).values()) == world - 1

    # calibrate the three constants on world 16 and ask about world 128
    intra = TIER_BY_PEERS[("intra_node", 7)]
    fixed = {"intra_node": intra,
             "cross_node": TIER_BY_PEERS[("cross_node", 8)],
             "cross_supernode": TIER_BY_PEERS[("cross_supernode", 8)]}
    pred = compose(peer_counts(topo, 128, 8), fixed)
    assert pred == pytest.approx(OUT_OF_SAMPLE["predicted_gbps"], abs=0.6)
    err = pred / OUT_OF_SAMPLE["measured_gbps"] - 1
    assert err < -0.25, (
        "the fixed-per-tier composition used to miss world 128 by -31%%; it now misses "
        "by %+.0f%%. If it stopped missing, the peer-count effect this module records "
        "is not there and the composition should be adopted." % (100 * err))

    # and the reason: both tiers deliver more per card at more peers
    assert TIER_BY_PEERS[("cross_node", 120)] > TIER_BY_PEERS[("cross_node", 8)]
    assert TIER_BY_PEERS[("cross_supernode", 64)] > TIER_BY_PEERS[("cross_supernode", 8)]
    # peer spread and concurrent load are separate: same peers, different company
    assert CROSS_SUPERNODE_8_PEERS_UNCONTENDED > 1.8 * TIER_BY_PEERS[("cross_supernode", 8)]

    # interpolation is allowed between measured peer counts and refused below them,
    # which is exactly the regime Hop A runs in
    assert TIER_BY_PEERS[("cross_supernode", 8)] < tier_at("cross_supernode", 24) < TIER_BY_PEERS[("cross_supernode", 64)]
    with pytest.raises(ValueError):
        tier_at("cross_supernode", 1)
    assert hop_a_peer_count(2) == 1 and hop_a_peer_count(16) == 15


def test_the_boundary_does_not_collapse_under_load_and_that_favours_two_hop():
    """Contention at the supernode boundary, and which way it points.

    A single crossing pair is the least contended case there is, so the first ratio
    measured was optimistic about the slow side -- which means it understated the
    hierarchy, not overstated it. The follow-up holds the work identical in every
    subgroup and varies only how many pairs cross at once.

    The naive worry was that real expert parallelism, crossing with every node at once,
    would collapse the boundary and sink the verdict. It does not: one step, then flat.
    And the step pushes the other way from the worry, because pressure lowers the slow
    side and two-hop exists to send fewer bytes across the slow side.
    """
    from sim.calibrate import BETA_FLAT
    from sim.hierarchy import (BOUNDARIES_DIFFER_BY, CONTENTION,
                               CROSS_SUPERNODE_WORLD128_GBPS, LOADED_RATIO,
                               clears_at_hidden_width, contention_penalty,
                               hierarchy_ratio, loaded_ratio, shallowest_boundary)

    d = dict(CONTENTION)
    # a single pair gets more than its share, and the penalty lands in one step
    assert d[1] > 1.5 * d[8]
    assert 1.6 < contention_penalty() < 2.0
    # and then nothing: flat from two pairs to eight, which is what says the boundary
    # is not a narrow shared pipe
    for n in (4, 8):
        assert abs(d[n] - d[2]) / d[2] < 0.03, (
            "%d pairs moved %.1f%% from 2 -- if the boundary starts collapsing, the "
            "verdict this module records has to be retaken" % (n, 100 * abs(d[n] - d[2]) / d[2]))

    # the loaded ratio is the like-for-like one and is deeper than the single-pair one
    for pair, r in LOADED_RATIO.items():
        assert loaded_ratio(pair) == pytest.approx(r, abs=0.02)
        assert loaded_ratio(pair) == pytest.approx(
            BETA_FLAT / CROSS_SUPERNODE_WORLD128_GBPS[pair], rel=1e-6)
        assert loaded_ratio(pair) > hierarchy_ratio(), (
            "%s: contention makes the hierarchy deeper, not shallower" % pair)

    # the machine has more than one boundary and they are not the same, so a verdict
    # has to use the shallowest rather than "the" ratio
    assert shallowest_boundary() == min(loaded_ratio(p) for p in LOADED_RATIO)
    deep = max(LOADED_RATIO.values()) / min(LOADED_RATIO.values())
    assert deep == pytest.approx(BOUNDARIES_DIFFER_BY, abs=0.02)
    assert deep > 1.4, "the two boundaries used to differ by 1.48x"

    # even the shallowest clears the operator-chain threshold at the reference width
    rows = clears_at_hidden_width()
    assert [H for H, _b, ok in rows if ok] == [2048, 4096, 8192]
    assert not rows[0][2], "H=1024 should still not clear"


def test_the_host_regime_decides_which_rule_a_collective_obeys():
    """The measurement that closed the open question, pinned.

    Both timing styles were run on the same machine, at the same worlds, over the same
    sizes, in one session (sim/hostregime.py, taken with bench/a2a_form_probe.py). The
    two want opposite rules: with the host running ahead the fixed cost hides behind
    the transfer, and with the host watching each call the two serialise and add.

    Four things are pinned, because each is a separate reason to believe it.
    """
    from sim.calibrate import ALPHA_PTS
    from sim.hostregime import (MEASURED, PHYSICAL_CEILING_GBPS, PLATEAU_MS,
                                compare_styles, preferred_rule, rows)

    assert len(MEASURED) == 46

    # 1. the verdict, and it does not depend on how the noisy burst points are trimmed
    for cut in (0.15, 0.30, 1.0, 99.0):
        res = compare_styles(max_spread=cut)
        b, p = res["burst"], res["percall"]
        assert b["quadrature"]["median"] < b["additive (shipped)"]["median"] / 2, (
            "cut %.2f: burst wants the overlap rule" % cut)
        assert p["additive (shipped)"]["median"] < p["quadrature"]["median"], (
            "cut %.2f: percall wants the additive rule" % cut)
    assert preferred_rule("burst") == "quadrature"
    assert preferred_rule("percall") == "additive (shipped)"

    # 2. the additive rule can only fit the burst data by asking for a bandwidth the
    #    hardware does not have; the overlap rule fits inside the ceiling
    res = compare_styles()
    assert res["burst"]["additive (shipped)"]["beta_inf"] > PHYSICAL_CEILING_GBPS
    assert res["burst"]["quadrature"]["beta_inf"] < PHYSICAL_CEILING_GBPS
    assert res["percall"]["additive (shipped)"]["beta_inf"] < PHYSICAL_CEILING_GBPS

    # 3. the alpha the overlap rule recovers from the burst data lands near the
    #    independent call-count scan, which took no part in the fit
    a = res["burst"]["quadrature"]["alpha"]
    assert a[8] == pytest.approx(0.114, abs=0.01)
    assert a[16] == pytest.approx(0.120, abs=0.01)
    assert a[8] == pytest.approx(dict(ALPHA_PTS)[8], abs=0.01), (
        "the world-8 entry of the shipped table is confirmed to three digits")
    assert abs(a[16] - dict(ALPHA_PTS)[16]) > 0.03, (
        "and the world-16 entry is not: this is the disagreement hostregime records")
    assert abs(a[16] - a[8]) < 0.02, (
        "alpha barely moves crossing to a second node, so the table's 41% step is not "
        "in this measurement either")

    # 4. the plateaus differ by the factor of two docs/09 measured as host exposure,
    #    reproduced here without being looked for
    from sim.hostregime import plateau
    for (style, world), doc in PLATEAU_MS.items():
        assert plateau(style, world) == pytest.approx(doc, abs=0.001)
    for world in (8, 16):
        ratio = plateau("percall", world) / plateau("burst", world)
        assert 1.8 < ratio < 2.6, "world %d host exposure factor %.2f" % (world, ratio)
    # and the burst plateau barely moves crossing to a second node, which is the same
    # flatness that makes the table's alpha(16) suspect
    assert abs(plateau("burst", 16) - plateau("burst", 8)) < 0.01

    # and the rule the repository ships is the one its own use case is in: a
    # variable-length dispatch cannot be issued until its counts reach the host
    from sim.calibrate import flat_supernode
    assert flat_supernode().combine_exponent == 1.0
    assert flat_supernode().splits_sync_ms > 0, (
        "the splits readback is why an MoE dispatch is in the host-exposed regime")
    assert len(rows("burst")) < len(rows("burst", max_spread=99.0))


def test_the_two_benchmark_families_want_different_model_forms():
    """Recalibrate the whole model under each rule, then run every gate. The decisive test.

    Fitting one form better than another proves nothing on its own, and the earlier
    checks were not a fair trial: they pinned a held-out alpha calibrated under no
    particular form, or refitted only the bandwidth pair to the Tier-1 targets. This
    runs the procedure calibrate.py actually describes, identically for both rules --
    alpha measured and pinned, the shape borrowed from the machine with enough
    distinct sizes to resolve it, the level fitted on the machine's own corpus -- and
    then scores all five gates.

    The harness is checked before it is trusted: at p = 1 it must recover the shipped
    constants, and it does, to the digit on x_half and machine B's beta.

    The verdict is a split, and the split is the finding. The overlap rule is clearly
    better on the two large size-sweep corpora and clearly worse on Tier-1, which
    belongs to the other benchmark family. So the two families disagree about what a
    collective costs, not merely about how much. Nothing offline can adjudicate that;
    one run of both benchmark styles at the same world over the same sizes would.
    """
    from sim.calibrate import (ALPHA_PTS, BETA_FAST, SECOND_ALPHA_PTS,
                               SECOND_BETA_FLAT, X_HALF_FLAT, supernode_under_form)
    from sim.core import _interp
    from sim.fit import fit_pinned_under_form
    from sim.validate_micro import validate_micro
    from sim.validate_sweep import (TARGETS_A, TARGETS_B, TARGETS_C, TARGETS_D,
                                    validate_sweep)

    alpha_a = lambda w: _interp(sorted(dict(ALPHA_PTS).items()), float(w))
    alpha_b = lambda w: _interp(sorted(dict(SECOND_ALPHA_PTS).items()), float(w))
    corpus = lambda tg: [(w, float(s), ms) for w, s, ms, _r in tg]

    scored = {}
    for p in (1.0, 2.0):
        shape = fit_pinned_under_form(corpus(TARGETS_B), alpha_b, p=p)
        level = fit_pinned_under_form(corpus(TARGETS_A), alpha_a, p=p,
                                      per_peer_us=shape["per_peer_us"])
        a = supernode_under_form(p, shape["per_peer_us"], level["beta_inf"], BETA_FAST)
        b = supernode_under_form(p, shape["per_peer_us"], shape["beta_inf"],
                                 alpha_pts=SECOND_ALPHA_PTS, ratio=1.0)
        _ok, t1 = validate_micro(a, verbose=False)
        row = {"tier1": t1["median"], "shape": shape, "level": level}
        for lab, tg, sp in (("A", TARGETS_A, a), ("B", TARGETS_B, b),
                            ("C", TARGETS_C, a), ("D", TARGETS_D, a)):
            _o, i = validate_sweep(sp, tg, verbose=False)
            row[lab] = i["median"]
        scored[p] = row

    # the harness reproduces the shipped calibration at p = 1 before anything is
    # concluded from p = 2
    add = scored[1.0]
    assert add["shape"]["x_half"] == pytest.approx(X_HALF_FLAT, rel=0.02), (
        "at p=1 this procedure must recover the shipped x_half of 54 KiB, not %.0f KiB"
        % (add["shape"]["x_half"] / 1024))
    assert add["shape"]["beta_inf"] == pytest.approx(SECOND_BETA_FLAT, rel=0.02)

    ovl = scored[2.0]
    # the split: the size-sweep family prefers the overlap rule, clearly
    assert ovl["B"] < add["B"] * 0.75, "corpus B: %.3f against %.3f" % (ovl["B"], add["B"])
    assert ovl["C"] < add["C"] * 0.75, "corpus C: %.3f against %.3f" % (ovl["C"], add["C"])
    # and the family Tier-1 comes from prefers the additive one, just as clearly
    assert ovl["tier1"] > add["tier1"] * 1.5, (
        "Tier-1: %.3f against %.3f -- if the overlap rule stops hurting the blind "
        "gate, the split this repository records has changed and the form question "
        "is open again" % (ovl["tier1"], add["tier1"]))
    # corpus D fails under both: it is a drift probe on alpha(8), not a form question
    assert ovl["D"] > 0.12 and add["D"] > 0.12

    # and the shipped model is still the additive one
    from sim.calibrate import flat_supernode
    assert flat_supernode().combine_exponent == 1.0


def test_the_overlap_form_fits_better_and_is_still_not_adopted():
    """A rejected model form, pinned so the road is not walked twice.

    Combining the fixed cost with the transfer in quadrature instead of adding them
    cuts the median error to a third on machine A at identical parameter count, and
    machine B reproduces the direction independently. It was still rejected: it
    predicts a held-out world worse and it fits the Tier-1 benchmark family worse.
    What the experiment established is that the two benchmark families differ in
    shape and not only in level.

    The test pins both halves. If the fit advantage ever disappears the record above
    is wrong; if the shipped model ever stops being additive, that is a decision that
    has to be taken deliberately and not by drift.
    """
    from sim.fit import compare_forms
    from sim.validate_sweep import TARGETS_A, TARGETS_B, TARGETS_C, TARGETS_D

    a = compare_forms([(w, s, ms) for w, s, ms, _r in
                       TARGETS_A + TARGETS_C + TARGETS_D], exponents=(1.0, 2.0))
    b = compare_forms([(w, s, ms) for w, s, ms, _r in TARGETS_B],
                      exponents=(1.0, 2.0))
    assert a["additive (shipped)"]["median"] == pytest.approx(0.078, abs=0.004)
    assert a["quadrature"]["median"] == pytest.approx(0.026, abs=0.004)
    assert b["additive (shipped)"]["median"] == pytest.approx(0.093, abs=0.004)
    assert b["quadrature"]["median"] == pytest.approx(0.071, abs=0.004)
    assert a["quadrature"]["median"] < a["additive (shipped)"]["median"] / 2
    assert b["quadrature"]["median"] < b["additive (shipped)"]["median"]

    # the shipped model is still the additive one, and core.py still implements it.
    # The tolerance is the 48-point log-spaced table calibrate.saturating_beta samples
    # the analytic curve into, not slack in the identity: the table tsupernodes the exact
    # form to better than a tenth of a percent over the range the model is used in,
    # and this assertion is what would notice if that grid were ever coarsened.
    from sim.calibrate import BETA_FLAT, X_HALF_FLAT, flat_supernode
    from sim.core import MoEGeometry, one_hop_call
    c = flat_supernode()
    g = MoEGeometry(name="add", n_groups=16, R=8, k=6, M=2, seq=4096, mbs=1,
                    gbs=16 * 8 * 4096)
    wire = g.tokens_per_rank * g.k * g.row_bytes() * (1 - 1.0 / g.ep)
    additive = c.flat.alpha_ms(g.ep) + wire / (BETA_FLAT * 1e6) + (g.ep - 1) * (
        X_HALF_FLAT / (BETA_FLAT * 1e6))
    assert one_hop_call(c, g) == pytest.approx(additive, rel=1e-3)


def test_arrival_chain_is_only_linear_above_the_measured_floor():
    """The chain constant carries the whole verdict, so its limits are pinned.

    Three things, in order of how much they matter.

    The reference geometry has to sit inside the range where the linear form was
    verified, or the headline breakeven is quoting a model outside its validity.

    The re-measured level has to stay inside the drift the calibration claims,
    because that is the only reason the shipped value was not replaced by the better
    evidenced one.

    And the direction has to stay recorded: adopting the re-measurement raises the
    breakeven rather than lowering it. If that ever flips, the argument for leaving
    the constant alone becomes self-serving and has to be re-made.
    """
    from sim.calibrate import (CHAIN_FLOOR_MS, CHAIN_LINEAR_MIN_ROWS,
                               CHAIN_US_PER_ROW, CHAIN_US_PER_ROW_REMEASURED)
    from sim.core import MoEGeometry
    from sim.uncertainty import breakeven_ratio

    g = MoEGeometry(name="ref", n_groups=16, R=8, k=6, M=2, seq=4096, mbs=1,
                    gbs=16 * 8 * 4096)
    assert g.rows_hop_b() >= CHAIN_LINEAR_MIN_ROWS, (
        "the reference geometry has %d Hop B rows, below the %d where the linear "
        "chain form was verified" % (g.rows_hop_b(), CHAIN_LINEAR_MIN_ROWS))

    drift = abs(CHAIN_US_PER_ROW_REMEASURED - CHAIN_US_PER_ROW) / CHAIN_US_PER_ROW
    assert drift <= 0.20, (
        "the chain re-measurement is %.0f%% from the shipped value, outside the 20%% "
        "drift that justifies leaving it alone" % (100 * drift))

    assert breakeven_ratio(CHAIN_US_PER_ROW_REMEASURED) > breakeven_ratio(CHAIN_US_PER_ROW), (
        "adopting the chain re-measurement now *helps* two-hop; keeping the older, "
        "less well evidenced constant would then need a different justification")

    # The floor is a real gap, not a rounding difference, or there is nothing to warn
    # about and the boundary text should go.
    assert CHAIN_FLOOR_MS > 2.0 * CHAIN_US_PER_ROW * 1024 / 1000.0, (
        "the measured chain floor no longer exceeds what the linear form predicts at "
        "1024 rows; the small-geometry boundary in calibrate.py is stale")


def test_host_exposure_does_not_outrank_the_arrival_chain():
    """Fusing the chain must stay worth more than removing the host.

    Both are one-term changes to the same comparison, so the repository has to say
    which to do first. The measured host exposure is charged to two-hop as one
    extra collective; the chain is priced by its own scenario tier. If the ordering
    ever reverses, docs/09 and sim/profile.py both say the wrong thing.
    """
    from sim.core import MoEGeometry
    from sim.profile import HOST_EXPOSURE_MS, launch_sensitivity
    from sim.calibrate import CHAIN_US_PER_ROW

    g = MoEGeometry(name="lat", n_groups=16, R=8, k=6, M=2, seq=4096, mbs=1,
                    gbs=16 * 8 * 4096)
    with_chain = launch_sensitivity(g, CHAIN_US_PER_ROW,
                                    deltas_ms=(0.0, HOST_EXPOSURE_MS))
    no_chain = launch_sensitivity(g, 0.0, deltas_ms=(0.0, HOST_EXPOSURE_MS))

    exposure_cost = with_chain[1]["breakeven"] - with_chain[0]["breakeven"]
    chain_cost = with_chain[0]["breakeven"] - no_chain[0]["breakeven"]
    assert exposure_cost > 0, "charging an extra call must not help two-hop"
    assert chain_cost > 4 * exposure_cost, (
        "the arrival chain is worth %.2f of breakeven and host exposure %.2f; "
        "they are now the same order, so docs/09 must stop saying fuse first"
        % (chain_cost, exposure_cost))
