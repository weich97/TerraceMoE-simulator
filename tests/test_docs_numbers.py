# -*- coding: utf-8 -*-
"""Read the numbers back out of the prose, and recompute each one from the code.

## Why this file exists

The repository's discipline is that every number in the documentation reproduces from
a named code construction and is pinned by a test. Until now the pinning half was done
against *constants inside the code*, which catches a constant moving but not a sentence
going stale. Twice it failed to catch exactly that:

  - the docs/12 group-cap table was produced by a construction nobody recorded, and
    could not be reproduced from the shipped code at all (corrected 2026-09-01);
  - the docs/05 scale numbers sat through two recalibrations untouched while the figure
    printed beside them, being generated, tracked the code the whole time. The prose
    said the ratio fell 1.68 -> 1.52 -> 1.47 and that the direction of the trend flipped
    past 128 dies; the code said 1.50 / 1.47 / 1.51 and no flip at all, and the second
    of those was a stated reason for a published refusal (corrected 2026-09-05).

Both are one failure mode: prose is not executable, so nothing re-derives it. This file
makes it executable. Each claim below names a document, quotes the sentence the number
lives in, and recomputes it from the module that owns it. Editing either side alone
turns this red, which is the point -- a number and its construction have to move
together or not at all.

## What belongs here

A claim belongs here when the number is **derived**: computed by code in this
repository from the calibration. Measured constants do not -- they belong in
tests/test_sim.py next to the measurement whose provenance they carry, because
recomputing a measurement from itself proves nothing.

## How a claim is written

Whitespace is collapsed before matching, so a template is written as one line however
the document happens to wrap it, and the unicode minus is normalised to ASCII. Write
the sentence exactly as it appears and put ``{n}`` where each number goes.
"""
import io
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

NUM = r"([-+]?\d+(?:\.\d+)?)"


def _read(rel):
    """Document text with whitespace collapsed and the unicode minus normalised."""
    with io.open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        text = fh.read()
    return re.sub(r"\s+", " ", text.replace("−", "-"))


def _pattern(template):
    """Literal prose with ``{n}`` standing in for each number."""
    parts = re.sub(r"\s+", " ", template.replace("−", "-")).split("{n}")
    return re.compile(NUM.join(re.escape(p) for p in parts))


# ---------------------------------------------------------------------------
# The live values, each from the module that owns the number
# ---------------------------------------------------------------------------


def _tier1():
    from sim.calibrate import flat_supernode
    from sim.validate_micro import validate_micro
    _ok, info = validate_micro(flat_supernode(), verbose=False)
    return info


def _tier1_at_x_half(kib):
    from sim.calibrate import flat_supernode
    from sim.validate_micro import validate_micro
    _ok, info = validate_micro(flat_supernode(x_half=kib * 1024), verbose=False)
    return info


def _sweep_row(chain_index):
    from sim.calibrate import synthetic
    from sim.sweep import CHAIN_SCENARIOS, comm_speedup
    chain = CHAIN_SCENARIOS[chain_index][1]
    return [comm_speedup(synthetic(r, chain_us_per_row=chain), 3, 4096)
            for r in (1.03, 3.2, 8.0, 15.7)]


def _breakevens():
    from sim.sweep import CHAIN_SCENARIOS
    from sim.uncertainty import breakeven_ratio
    return [breakeven_ratio(chain) for _name, chain in CHAIN_SCENARIOS]


def _launch_breakeven(chain_index, extra_ms):
    from sim.core import MoEGeometry
    from sim.profile import launch_sensitivity
    from sim.sweep import CHAIN_SCENARIOS
    g = MoEGeometry(name="ref", n_groups=16, R=8, k=6, M=2, H=2048, seq=4096,
                    mbs=1, gbs=16 * 8 * 4096)
    for row in launch_sensitivity(g, CHAIN_SCENARIOS[chain_index][1]):
        if abs(row["extra_launch_ms"] - extra_ms) < 1e-9:
            return row["breakeven"]
    raise AssertionError("launch_sensitivity does not sweep %s ms" % extra_ms)


def _overlap():
    from sim.calibrate import flat_supernode
    from sim.overlap import evaluate
    return evaluate(flat_supernode(), verbose=False)


# ---------------------------------------------------------------------------
# The claims
# ---------------------------------------------------------------------------
# (document, template or compiled regex, callable returning the live values, tolerance)

CLAIMS = []


def claim(doc, template, values, tol, regex=None):
    CLAIMS.append((doc, regex or _pattern(template), values, tol, template))


# -- docs/05: the validation gates ------------------------------------------

claim("docs/05-simulator.md",
      "**Current: pass** ({n}% / {n}% / 4096",
      lambda: (100 * _tier1()["median"], 100 * _tier1()["worst"]), 0.05)

claim("docs/05-simulator.md",
      "the entire bootstrap interval of x_half passes it (30 KiB → {n}%, 87 KiB → {n}%",
      lambda: (100 * _tier1_at_x_half(30)["median"],
               100 * _tier1_at_x_half(87)["median"]), 0.05)

# -- docs/05: the two headline tables ---------------------------------------

claim("docs/05-simulator.md",
      "| PyTorch arrival chain (measured 0.0875 µs/row) | {n} | {n} | {n} | {n} |",
      lambda: _sweep_row(0), 0.005)

claim("docs/05-simulator.md",
      "| Hypothetical fused target (0.012) | {n} | {n} | {n} | {n} |",
      lambda: _sweep_row(1), 0.005)

claim("docs/05-simulator.md",
      "| Zero implementation overhead (upper bound) | {n} | {n} | {n} | {n} |",
      lambda: _sweep_row(2), 0.005)

# The three breakeven rows are matched as one block: taken singly, each row's opening
# is also the opening of a row in the sweep table above it.
claim("docs/05-simulator.md",
      "| PyTorch arrival chain (measured 0.0875 µs/row) | **{n}** | "
      "| Hypothetical fused target (0.012) | **{n}** | "
      "| Zero implementation overhead (upper bound) | {n} |",
      _breakevens, 0.005)

# -- docs/05: the Monte Carlo anchors ---------------------------------------

claim("docs/05-simulator.md",
      "zero implementation overhead): p95 = **{n}**",
      lambda: [_mc(1.03, 2)[2]], 0.005)

claim("docs/05-simulator.md",
      "least favorable case (PyTorch chain): p5 = **{n} > 1**",
      lambda: [_mc(8.0, 0)[0]], 0.005)

# -- docs/05: the scale axis, the claims that went stale --------------------

claim("docs/05-simulator.md",
      "answer is solid: {n} → {n} → {n} at 32 / 64 / 128",
      lambda: [_scale(w) for w in (32, 64, 128)], 0.005)

claim("docs/05-simulator.md",
      "lands anywhere in **{n} – {n}**",
      lambda: list(_band(512)), 0.005)

claim("docs/05-simulator.md",
      "geometry axis ((k,M) moves the breakeven by up to {n} at fixed world)",
      lambda: [_geom(1)], 0.005)

claim("docs/05-simulator.md",
      "tier the geometry axis widens to {n}",
      lambda: [_geom(0)], 0.005)

# -- docs/05: what x_half is, and the two experiments run against the model -----

claim("docs/05-simulator.md",
      "that cost: **{n} microseconds** here.",
      lambda: [_per_peer_us()], 0.0005)

claim("docs/05-simulator.md",
      "the full fabric sends **{n}** peer messages; Hop A sends {n} and Hop B {n}, "
      "so the swap pays **{n}** of them instead of {n}.",
      lambda: [127, 15, 7, 22, 127], 0.0)

claim("docs/05-simulator.md",
      "that is {n} ms against {n} ms per call.",
      lambda: [127 * _per_peer_us() / 1000, 22 * _per_peer_us() / 1000], 0.0005)

claim("docs/05-simulator.md",
      "| platform A, GB/s | {n} | {n} | — | — | {n} |",
      lambda: _marginal("A"), 0.5)

claim("docs/05-simulator.md",
      "| platform B, GB/s | {n} | {n} | {n} | {n} | {n} |",
      lambda: _marginal("B"), 0.5)

claim("docs/05-simulator.md",
      "| PyTorch chain (measured) | {n} | **{n}** | "
      "| hypothetical fused target | {n} | {n} | "
      "| zero implementation overhead | {n} | {n} |",
      lambda: [v for row in _bw_sensitivity() for v in row], 0.005)

claim("docs/05-simulator.md",
      "| additive (shipped) | {n}% | {n}% | baseline | "
      "| quadrature | **{n}%** | **{n}%** | same as shipped |",
      lambda: [100 * v for v in _form_medians()], 0.4)

claim("docs/05-simulator.md",
      "| p = 1, additive, o = {n} us | {n}% | {n}% | {n}% | {n}% | {n}% | "
      "| p = 2, overlap, o = {n} us | {n}% | {n}% | {n}% | {n}% | {n}% |",
      lambda: _recalibration_table(), 0.006)

claim("docs/05-simulator.md",
      "| **burst**, host runs ahead | {n}% | **{n}%** | {n} GB/s | "
      "| **percall**, host waits per call | **{n}%** | {n}% | {n} GB/s |",
      lambda: _host_regime_table(), 0.06)

claim("docs/05-simulator.md",
      "burst data, 0.114 ms at world 8 and {n} at world 16, lands near",
      lambda: [_regime_alpha()[16]], 0.001)

# -- docs/03: the measured supernode boundaries -----------------------------

claim("docs/03-applicability.md",
      "| pool110 / pool111 | {n} | **{n}** | | pool110 / pool12 | {n} | **{n}** |",
      lambda: _boundaries(), 0.02)

claim("docs/03-applicability.md",
      "the two pairs differ by {n}×, so any verdict should use the shallower, {n}.",
      lambda: _differ(), 0.02)

claim("docs/03-applicability.md",
      "{n} GB/s per card with one pair, then {n}, {n} and {n} at two, four and eight",
      lambda: [bw for _n, bw in _contention()], 0.05)

claim("docs/03-applicability.md",
      "per-card bandwidth is {n} GB/s at one peer and {n} at sixty-four",
      lambda: _peer_ends(), 0.02)

claim("docs/05-simulator.md",
      "| cross-node | {n} GB/s | {n} GB/s at 120 | "
      "| cross-supernode | {n} GB/s | {n} GB/s at 64 |",
      lambda: _tier_table(), 0.05)

claim("docs/05-simulator.md",
      "cross-supernode bandwidth is {n} GB/s at one peer and {n} at sixty-four",
      lambda: _peer_ends(), 0.02)

claim("docs/05-simulator.md",
      "it predicts {n} GB/s for a world-128 all-to-all spanning both supernodes; the "
      "measurement is {n}.",
      lambda: _oos(), 0.05)

claim("docs/05-simulator.md",
      "it is {n} ms at world 8 and {n} at world 16 with the host running ahead, and "
      "{n} and {n} with the host observing each call",
      lambda: _alpha_regimes(), 0.0005)

claim("docs/05-simulator.md",
      "it returns α = {n} exactly, paired with β∞ = **{n} GB/s**",
      lambda: _corpusC_fit(), 0.05)

claim("docs/03-applicability.md",
      "| 2048 | 1 | 6 | {n} ms | {n} ms | **{n}** | | 2048 | 2 | 3 | {n} ms | {n} ms | "
      "**{n}** | | 4096 | 1 | 6 | {n} ms | {n} ms | **{n}** | | 4096 | 2 | 3 | {n} ms | "
      "{n} ms | **{n}** |",
      lambda: _verdict_rows(), 0.006)

claim("README.md",
      "**Two-hop wins in every configuration, by {n}x to {n}x**",
      lambda: _verdict_span(), 0.006)

# -- docs/07: the overlap family table and its cross-references -------------

for _fam in ("M0", "M1", "M2", "M3", "M4", "M5"):
    claim("docs/07-tier2-overlap.md", None,
          (lambda f: (lambda: [_overlap()[f]["mae"]]))(_fam), 0.0005,
          regex=re.compile(r"\| %s \|[^|]*\|[^|]*\| \**%s\** \|" % (_fam, NUM)))

claim("docs/07-tier2-overlap.md",
      "(docs/05, Tier-1 median error {n}%)",
      lambda: [100 * _tier1()["median"]], 0.05)

claim("docs/07-tier2-overlap.md",
      "that is exactly M0's mistake (MAE {n},",
      lambda: [_overlap()["M0"]["mae"]], 0.0005)

# -- docs/09: what the measured launch cost moves ---------------------------

claim("docs/09-phase-model.md",
      "moves the breakeven hierarchy ratio on the reference geometry from {n} to {n}.",
      lambda: [_launch_breakeven(0, 0.0), _launch_breakeven(0, 0.130)], 0.005)

claim("docs/09-phase-model.md",
      "Remove the arrival chain and the same 130 microseconds moves it from {n} to {n}.",
      lambda: [_launch_breakeven(2, 0.0), _launch_breakeven(2, 0.130)], 0.005)

# -- docs/12: the group-cap table the correction was written for ------------

for _M, _q in ((4, 2), (2, 4), (1, 8)):
    claim("docs/12-m-quality-experiment.md", None,
          (lambda m: (lambda: [r[2] for r in _mtable() if r[0] == m]
                              + [r[3] for r in _mtable() if r[0] == m]))(_M),
          0.0005,
          regex=re.compile(r"\| %d \| %d \| %s \| %s \|" % (_M, _q, NUM, NUM)))

# -- README: the numbers it repeats from docs/05 ----------------------------

claim("README.md",
      "**pass**, {n}% median error, {n}% worst",
      lambda: (100 * _tier1()["median"], 100 * _tier1()["worst"]), 0.05)

claim("README.md",
      "the shipped code: it is {n} to {n}, and all four treatments rise.",
      lambda: list(_band(512)), 0.005)

claim("README.md",
      "Least squares over the shipped sweep gives {n}, {n} and {n}%.",
      lambda: _chain_fit() + [_chain_shares()[0]], 0.5)

claim("README.md",
      "{n} and {n} against {n} and {n} as `sim.uncertainty.breakeven_vs_hidden_width` "
      "computes them.",
      lambda: [5.96, 2.90, _hidden_width_series()[0], _hidden_width_series()[2]], 0.005)

claim("README.md",
      "At 512 ranks they span {n} to {n},",
      lambda: list(_band(512)), 0.005)

claim("README.md",
      "the corrected reference thresholds are {n} for the measured PyTorch chain and {n} for the hypothetical fused target",
      lambda: _breakevens()[:2], 0.005)

# -- docs/05: the threshold against hidden width ----------------------------

claim("docs/05-simulator.md",
      "measures {n}, {n}, {n} and {n} ms at H of 1024, 2048, 4096 and 8192.",
      lambda: _chain_sweep_ms(), 0.005)

claim("docs/05-simulator.md",
      "the chain grows by {n} while the payload every collective carries grows by four",
      lambda: _chain_growth(), 0.005)

claim("docs/05-simulator.md",
      "| effective breakeven, measured chain | {n} | **{n}** | {n} | {n} |",
      lambda: _hidden_width_series(), 0.005)

claim("docs/05-simulator.md",
      "H = 2048 point is {n} ms against the calibration's {n}",
      lambda: _chain_levels(), 0.005)

claim("docs/05-simulator.md",
      "adopting the sweep's level instead would put the reference threshold at {n}.",
      lambda: [_sweep_level_breakeven()], 0.005)


def _per_peer_us():
    from sim.calibrate import PER_PEER_MESSAGE_US
    return PER_PEER_MESSAGE_US


def _marginal(machine):
    from sim.fit import marginal_bandwidth
    from sim.validate_sweep import TARGETS_A, TARGETS_B, TARGETS_C, TARGETS_D
    tg = (TARGETS_A + TARGETS_C + TARGETS_D) if machine == "A" else TARGETS_B
    bw = marginal_bandwidth([(w, s, ms) for w, s, ms, _r in tg])
    return [bw[w] for w in sorted(bw)]


def _bw_sensitivity():
    from sim.profile import bandwidth_world_sensitivity
    return [(shipped, per_world)
            for _name, shipped, per_world in bandwidth_world_sensitivity()]


def _form_medians():
    """additive and quadrature medians on machine A then machine B, in table order."""
    from sim.fit import compare_forms
    from sim.validate_sweep import TARGETS_A, TARGETS_B, TARGETS_C, TARGETS_D
    a = compare_forms([(w, s, ms) for w, s, ms, _r in
                       TARGETS_A + TARGETS_C + TARGETS_D], exponents=(1.0, 2.0))
    b = compare_forms([(w, s, ms) for w, s, ms, _r in TARGETS_B],
                      exponents=(1.0, 2.0))
    return [a["additive (shipped)"]["median"], b["additive (shipped)"]["median"],
            a["quadrature"]["median"], b["quadrature"]["median"]]


def _recalibration_table():
    """The two calibrations of docs/05's decisive table, in the order it prints them."""
    from sim.calibrate import (ALPHA_PTS, BETA_FAST, SECOND_ALPHA_PTS,
                               supernode_under_form)
    from sim.core import _interp
    from sim.fit import fit_pinned_under_form
    from sim.validate_micro import validate_micro
    from sim.validate_sweep import (TARGETS_A, TARGETS_B, TARGETS_C, TARGETS_D,
                                    validate_sweep)
    aA = lambda w: _interp(sorted(dict(ALPHA_PTS).items()), float(w))
    aB = lambda w: _interp(sorted(dict(SECOND_ALPHA_PTS).items()), float(w))
    co = lambda tg: [(w, float(x), ms) for w, x, ms, _r in tg]
    out = []
    for p in (1.0, 2.0):
        sh = fit_pinned_under_form(co(TARGETS_B), aB, p=p)
        lv = fit_pinned_under_form(co(TARGETS_A), aA, p=p,
                                   per_peer_us=sh["per_peer_us"])
        a = supernode_under_form(p, sh["per_peer_us"], lv["beta_inf"], BETA_FAST)
        b = supernode_under_form(p, sh["per_peer_us"], sh["beta_inf"],
                                 alpha_pts=SECOND_ALPHA_PTS, ratio=1.0)
        out.append(sh["per_peer_us"])
        out.append(100 * validate_micro(a, verbose=False)[1]["median"])
        for tg, sp in ((TARGETS_A, a), (TARGETS_B, b), (TARGETS_C, a), (TARGETS_D, a)):
            out.append(100 * validate_sweep(sp, tg, verbose=False)[1]["median"])
    return out


def _host_regime_table():
    from sim.hostregime import compare_styles
    r = compare_styles()
    out = []
    for st in ("burst", "percall"):
        out += [100 * r[st]["additive (shipped)"]["median"],
                100 * r[st]["quadrature"]["median"],
                r[st]["additive (shipped)"]["beta_inf"]]
    return out


def _regime_alpha():
    from sim.hostregime import compare_styles
    return {w: round(v, 3)
            for w, v in compare_styles()["burst"]["quadrature"]["alpha"].items()}


def _alpha_regimes():
    from sim.calibrate import ALPHA_BY_REGIME as A
    return [A["deep queue"][8], A["deep queue"][16],
            A["host exposed"][8], A["host exposed"][16]]


def _corpusC_fit():
    from sim.calibrate import ALPHA_CORPUS_JOINT_FIT as F
    return [F["C"]["alpha"], F["C"]["beta_inf"]]


def _tier_table():
    from sim.tiers import TIER_BY_PEERS as T
    return [T[("cross_node", 8)], T[("cross_node", 120)],
            T[("cross_supernode", 8)], T[("cross_supernode", 64)]]


def _oos():
    from sim.tiers import OUT_OF_SAMPLE as O
    return [O["predicted_gbps"], O["measured_gbps"]]


def _contention():
    from sim.hierarchy import CONTENTION
    return CONTENTION


def _verdict_rows():
    from sim.twohop_measured import MEASURED, g, two_hop_ms
    out = []
    for r in MEASURED:
        out += [r[3], two_hop_ms(r), g(r)]
    return out


def _verdict_span():
    from sim.twohop_measured import MEASURED, g
    return [min(g(r) for r in MEASURED), max(g(r) for r in MEASURED)]


def _boundaries():
    from sim.hierarchy import CROSS_SUPERNODE_WORLD128_GBPS as G, loaded_ratio
    return [G["pool110/pool111"], loaded_ratio("pool110/pool111"),
            G["pool110/pool12"], loaded_ratio("pool110/pool12")]


def _differ():
    from sim.hierarchy import BOUNDARIES_DIFFER_BY, shallowest_boundary
    return [BOUNDARIES_DIFFER_BY, shallowest_boundary()]


def _peer_ends():
    from sim.tiers import measured_cross_supernode_gbps as f
    return [f(1), f(64)]


def _chain_sweep_ms():
    from sim.machine import CHAIN_H_SWEEP_MS
    return [CHAIN_H_SWEEP_MS[H] for H in sorted(CHAIN_H_SWEEP_MS)]


def _chain_levels():
    """The sweep's own H = 2048 reading, and the level the calibration ships."""
    from sim.machine import CHAIN_H_SWEEP_MS, CHAIN_LEVEL_CALIBRATION_MS
    return [CHAIN_H_SWEEP_MS[2048], CHAIN_LEVEL_CALIBRATION_MS]


def _hidden_width_series():
    from sim.uncertainty import breakeven_vs_hidden_width
    return [b for _H, b in breakeven_vs_hidden_width()]


def _sweep_level_breakeven():
    """The threshold under the sweep's own arrival-chain level rather than the
    calibration's. docs/05 reports both, because the gap between the two is the
    run-to-run drift already documented for that constant."""
    from sim.machine import CHAIN_H_SWEEP_MS, CHAIN_H_SWEEP_ROWS
    from sim.uncertainty import breakeven_ratio
    return breakeven_ratio(CHAIN_H_SWEEP_MS[2048] * 1000.0 / CHAIN_H_SWEEP_ROWS)


def _chain_fit():
    from sim.machine import CHAIN_FIXED_MS, CHAIN_PER_1024H_MS
    return [CHAIN_FIXED_MS, CHAIN_PER_1024H_MS]


def _chain_shares():
    """Gather against index work at the reference hidden width, as percentages."""
    from sim.machine import CHAIN_FIXED_MS, CHAIN_PER_1024H_MS
    gather = CHAIN_PER_1024H_MS * 2.0            # H = 2048, in units of 1024
    total = CHAIN_FIXED_MS + gather
    return [100.0 * gather / total, 100.0 * CHAIN_FIXED_MS / total]


def _chain_growth():
    from sim.machine import CHAIN_H_SWEEP_MS
    return [CHAIN_H_SWEEP_MS[8192] / CHAIN_H_SWEEP_MS[2048]]


def _mc(ratio, chain_index):
    from sim.sweep import CHAIN_SCENARIOS
    from sim.uncertainty import mc_band
    return mc_band(ratio, CHAIN_SCENARIOS[chain_index][1])


def _scale(w):
    from sim.uncertainty import scale_ratio
    return scale_ratio(w)


def _band(w):
    from sim.uncertainty import scale_band
    return scale_band(w)


def _geom(chain_index):
    from sim.sweep import CHAIN_SCENARIOS
    from sim.uncertainty import geometry_breakeven_spread
    return geometry_breakeven_spread(CHAIN_SCENARIOS[chain_index][1])


def _mtable():
    from sim.codesign import m_table
    return m_table()


def _load_local_claims():
    """Let a local, gitignored file add claims about documents this repo does not ship.

    Some of what this project writes lives outside the repository and is not published
    with it. Quoting that text here would put it in a public repository by the back
    door, so the claims that check it live in ``tests/local_claims.py``, which is
    gitignored and absent from a clean checkout. It is executed with this module's
    namespace, so it calls ``claim`` and reuses the helpers above exactly as the claims
    in this file do.
    """
    path = os.path.join(HERE, "local_claims.py")
    if not os.path.exists(path):
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location("_terrace_local_claims", path)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__.update({k: v for k, v in globals().items()
                         if not k.startswith("__")})
    spec.loader.exec_module(mod)


_load_local_claims()


# ---------------------------------------------------------------------------


@pytest.mark.parametrize("doc,pattern,values,tol,label",
                         CLAIMS,
                         ids=["%s::%s" % (c[0].split("-")[0],
                                          (c[4] or c[1].pattern)[:48])
                              for c in CLAIMS])
def test_documented_number_reproduces(doc, pattern, values, tol, label):
    """The sentence must still be there, and its numbers must still be the code's."""
    if not os.path.exists(os.path.join(ROOT, doc)):
        pytest.skip("%s is not in this checkout, so its claims cannot be checked "
                    "here. Documents this repository does not ship are registered "
                    "from tests/local_claims.py, which is gitignored." % doc)
    text = _read(doc)
    found = pattern.findall(text)
    assert found, (
        "%s no longer contains this claim, so nothing checks its numbers any more:\n"
        "  %s\n"
        "If the sentence was deliberately reworded, update the template in "
        "tests/test_docs_numbers.py in the same commit." % (doc, label or pattern.pattern))
    assert len(found) == 1, (
        "%s contains %d copies of this claim; a number stated twice can drift in one "
        "place only:\n  %s" % (doc, len(found), label or pattern.pattern))

    documented = [float(x) for x in
                  (found[0] if isinstance(found[0], tuple) else (found[0],))]
    live = [float(v) for v in values()]
    assert len(documented) == len(live), (
        "%s: the claim captures %d numbers but the construction produces %d"
        % (doc, len(documented), len(live)))

    for i, (d, v) in enumerate(zip(documented, live)):
        assert abs(d - v) <= tol, (
            "%s states %s where the code now computes %.6f (position %d of the claim, "
            "tolerance %s):\n  %s\n"
            "One of the two is stale. Recompute, correct the document in place with a "
            "dated note as docs/05 and docs/12 do, and never silently overwrite."
            % (doc, d, v, i + 1, tol, label or pattern.pattern))
