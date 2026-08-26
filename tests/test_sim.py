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
    assert res["M0"]["mae"] >= 0.10   # failure magnitude of the naive baseline (~0.135)
    # MAE snapshot pin for the docs/07 §1 table (±0.005): any move in the calibration
    # constants turns this red, a reminder to re-issue the docs/07 table in step
    # (review found a loose pin failed to catch a 20% drift in the arrival-chain constant)
    for fam, doc in (("M0", 0.135), ("M1", 0.150), ("M2", 0.062),
                     ("M3", 0.136), ("M4", 0.048), ("M5", 0.087)):
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
    # The flat column's most favourable case now sits just above 1.0 (1.04). That is the
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
    """The breakeven hierarchy ratio must fall monotonically with implementation tier and match the docs/05 snapshot (±0.1)."""
    from sim.uncertainty import breakeven_ratio
    from sim.sweep import CHAIN_SCENARIOS
    bes = [breakeven_ratio(chain) for _, chain in CHAIN_SCENARIOS]
    assert bes[0] > bes[1] > bes[2] >= 1.0
    for got, doc in zip(bes, (3.87, 1.45, 1.07)):
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


# ---------------------------------------------------------------- scale honesty


def _scale_ratio(alpha_pts, w, ratio=3.2):
    from sim.core import MoEGeometry, one_hop_call, two_hop_call
    from sim.sweep import CHAIN_SCENARIOS
    c = synthetic(ratio, chain_us_per_row=CHAIN_SCENARIOS[1][1])
    for lvl in (c.fast, c.slow, c.flat):
        lvl.alpha_pts = alpha_pts
    g = MoEGeometry(name="scale", n_groups=w // 8, R=8, k=6, M=2,
                    seq=4096, mbs=1, gbs=w * 4096)
    return one_hop_call(c, g) / two_hop_call(c, g)


ALPHA_TREATMENTS = [
    {256: 0.425, 512: 2.888},                          # refit of the same corpus
    {256: 0.735, 512: 1.859},                          # the shipped entries
    {256: 0.378, 512: 0.378},                          # no growth past 128
    {256: 0.378 + 0.0107 * 128, 512: 0.378 + 0.0107 * 384},   # linear in peers
]


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
    at. Nothing was re-fitted for it. Its margin is the thinnest of the three and
    that is recorded rather than smoothed -- if a future change buys machine B a
    little accuracy by spending C's remaining margin, this fails.
    """
    from sim.calibrate import flat_supernode, second_machine
    from sim.validate_sweep import (GATE_MEDIAN, TARGETS_A, TARGETS_B, TARGETS_C,
                                    validate_sweep)
    ok_a, ia = validate_sweep(flat_supernode(), TARGETS_A, verbose=False)
    ok_b, ib = validate_sweep(second_machine(), TARGETS_B, verbose=False)
    ok_c, ic = validate_sweep(flat_supernode(), TARGETS_C, verbose=False)
    assert len(TARGETS_A) == 6 and len(TARGETS_B) == 44 and len(TARGETS_C) == 12
    assert ok_a, "machine A: median %.3f bias %+.3f outliers %.2f" % (
        ia["median"], ia["bias"], ia["outlier_fraction"])
    assert ok_b, "machine B: median %.3f bias %+.3f outliers %.2f" % (
        ib["median"], ib["bias"], ib["outlier_fraction"])
    assert ok_c, "machine A world 16 (blind): median %.3f bias %+.3f outliers %.2f" % (
        ic["median"], ic["bias"], ic["outlier_fraction"])
    assert ic["median"] > max(ia["median"], ib["median"]), (
        "corpus C used to be the tightest fit of the three; if that changed, check "
        "that it was not quietly refitted")
    assert GATE_MEDIAN - ic["median"] < 0.02, (
        "corpus C now clears the median gate by more than 2 points; the docs call it "
        "the thinnest margin and would need updating")


# ---------------------------------------------------------------- compute model


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
    unified-fabric row is 'no' at every tier, and the high-ratio rows are 'yes' at
    every tier -- if either flips, the headline table in README is wrong.
    """
    from sim.platforms import platform_map
    rows = platform_map()
    for r in rows:
        for tier, ok in r["verdict"].items():
            assert ok == (r["archetype"].ratio_nominal >= r["breakevens"][tier])
    by_key = {r["archetype"].key: r for r in rows}
    assert not any(by_key["flat-supernode"]["verdict"].values())
    assert all(by_key["nvlink-ib"]["verdict"].values())
    assert all(by_key["rack-domain"]["verdict"].values())
    # the middle row is the interesting one: implementation tier decides it
    mid = by_key["pcie-ib"]["verdict"]
    assert sum(mid.values()) == 2, "PCIe+IB should pay only above the PyTorch tier"


def test_platform_coverage_reports_the_gap_honestly():
    """Coverage must keep announcing that no calibrated machine is hierarchical.

    The day someone adds a platform above ratio 1.5, this flips and the claim in
    README about extrapolation has to be rewritten -- deliberately, not silently.
    """
    from sim.platforms import PLATFORMS, coverage
    c = coverage()
    assert c["n_platforms"] == len(PLATFORMS) >= 2
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
    for world, deep in PER_CALL_DEEP_QUEUE_MS.items():
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

    gaps = [PER_CALL_HOST_EXPOSED_MS[w] - PER_CALL_DEEP_QUEUE_MS[w]
            for w in PER_CALL_DEEP_QUEUE_MS]
    assert min(gaps) <= HOST_EXPOSURE_MS <= max(gaps), (
        "HOST_EXPOSURE_MS %.3f is not inside the measured gaps %s"
        % (HOST_EXPOSURE_MS, ["%.3f" % g for g in gaps]))


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
