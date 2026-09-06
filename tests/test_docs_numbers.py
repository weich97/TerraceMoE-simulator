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
      "At 512 ranks they span {n} to {n},",
      lambda: list(_band(512)), 0.005)

claim("README.md",
      "the corrected reference thresholds are {n} for the measured PyTorch chain and {n} for the hypothetical fused target",
      lambda: _breakevens()[:2], 0.005)


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


# ---------------------------------------------------------------------------


@pytest.mark.parametrize("doc,pattern,values,tol,label",
                         CLAIMS,
                         ids=["%s::%s" % (c[0].split("-")[0],
                                          (c[4] or c[1].pattern)[:48])
                              for c in CLAIMS])
def test_documented_number_reproduces(doc, pattern, values, tol, label):
    """The sentence must still be there, and its numbers must still be the code's."""
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
