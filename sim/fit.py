# -*- coding: utf-8 -*-
"""Fit the a2a cost model to your own measurements, and audit what the fit supports.

`calibrate.py` ships distilled constants for one machine. This module is how those
constants were obtained and audited, and how you obtain your own: give it size
sweeps, it fits the model and reports which parts of the result the data actually
constrains.

## Input schema

A list of records, each one timed a2a call shape:

    {"frame": "<name of the machine/corpus>",   # fits are per frame; never pooled
     "world": 128,                              # ranks in the collective
     "bytes": 8388608,                          # total send buffer per rank
     "ms": 0.6466}                              # wall clock per call

`load_records(path)` reads a JSON list of those. Anything else -- our own raw
sweeps included -- stays out of this repository; only the distilled constants ship.

## Model

    t(world, S) = alpha(world) + wire / (beta_eff * 1e6)      [ms]
    wire        = S * (world - 1) / world                     [bytes]
    beta_eff    = beta_inf * x / (x + x_half),  x = S / world [GB/s]

`x_half` is the half-performance message size. `fit_frame` gives every world its
own alpha and shares (beta_inf, x_half), which is the honest split: alpha is a
machine property that does not transfer, the bandwidth pair does.

## What our own audit found (331 usable points, 4 datasets, 2 machines)

1. **beta_inf transfers.** Fitted independently per dataset, with alpha pinned to
   the direct measurements (the correct procedure -- see point 4):
   beta_inf = 117.8 / 111.0 / 113.4 / 130.3 GB/s and x_half = 100 / 79 / 54 /
   94 KiB, at 5.4 / 9.3 / 7.0 / 6.4% median relative error. The beta_inf spread is
   narrower than the machine's own run-to-run drift, which is why `calibrate.py`
   ships a single beta with confidence. beta_inf comes out identical whether alpha
   is pinned or free, which is what makes it the trustworthy half of the pair.

2. **alpha above world 128 is not supported.** Only one corpus reaches worlds 256
   and 512; it is also the one whose absolute bandwidth sits ~5x below the others
   and which this model fits worst (40% median). See `calibrate.py::ALPHA_PTS` and
   figure F10 for what that does to large-cluster claims.

3. **A saturating beta was tried and rejected.** Substituting `beta_eff` above for
   the flat beta in `calibrate.py` moves Tier-1 from 8.1% to 17.9% median and 12.5%
   to 101.6% worst -- the gate fails. The reason is that the two benchmark families
   are not interchangeable: the size-sweep corpus and the direct alpha corpus
   disagree by roughly 2x at comparable per-peer sizes, and the sweep corpus that
   drove the saturating fit is a **drift study** whose own run-to-run spread is
   2.1-3.1x. An instrument that scatters by 3x cannot adjudicate a 15% question,
   so the flat beta stands and this stays a documented negative result.

4. **alpha and x_half are not separately identifiable from size sweeps.** The
   round-trip self-check (`python -m sim.fit`) recovers a synthetic frame's
   predictions to 1e-11 while missing x_half by 18% and alpha by 4% -- the two
   parameters trade off against each other. Re-fitting the real corpora with alpha
   pinned to the direct measurements leaves beta_inf untouched (117.8 / 111.0 /
   113.4 / 130.3 GB/s, same as the free fit) but moves x_half from 282-368 KiB down
   to 54-100 KiB. Read that as: **beta_inf is identified, x_half is not** -- it is
   whatever absorbs the alpha degeneracy. So measure alpha directly, pin it, and
   fit only the bandwidth pair (`fit_beta_pinned`). Never quote an x_half that was
   fitted with alpha free.

Point 3 is the reason this module reports fit quality per frame instead of pooling
everything into one number: pooled fits hide exactly this kind of corpus conflict.

## The unresolved part, stated plainly

Two of our own benchmark families disagree about the same machine in the same week.
At comparable per-peer sizes the direct-alpha benchmark reports ~0.39 ms where the
size-sweep benchmark reports 0.52-0.65 ms. The shipped calibration is anchored to
the first and Tier-1 validates against the first, so the gate is internally
consistent but **not cross-validated against the second**. We are not able to
adjudicate this offline: the sweep corpus is a drift study whose own spread is
2.1-3.1x, wide enough to contain the disagreement without explaining it. Anyone
recalibrating on their own machine should run both styles and check they agree
before trusting either.
"""
from __future__ import annotations

import json
import math

# Fitted on the reference machine; see the audit notes above.
BETA_INF_GBPS = 118.0
X_HALF_BYTES = 320 * 1024


def load_records(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        recs = json.load(fh)
    out = []
    for r in recs:
        if r.get("ms", 0) > 0 and r.get("world", 0) >= 2 and r.get("bytes", 0) > 0:
            out.append((str(r.get("frame", "default")), int(r["world"]),
                        float(r["bytes"]), float(r["ms"])))
    return out


def beta_eff(per_peer_bytes: float, beta_inf: float = BETA_INF_GBPS,
             x_half: float = X_HALF_BYTES) -> float:
    """Effective bandwidth [GB/s] at a given per-peer message size."""
    x = max(float(per_peer_bytes), 0.0)
    return beta_inf * x / (x + x_half)


def predict_ms(world: int, total_bytes: float, alpha_ms: float,
               beta_inf: float = BETA_INF_GBPS,
               x_half: float = X_HALF_BYTES) -> float:
    wire = total_bytes * (world - 1) / world
    return alpha_ms + wire / (beta_eff(total_bytes / world, beta_inf, x_half) * 1e6)


def fit_frame(records: list, seeds=(2e4, 2e5, 2e6)) -> dict:
    """Fit one alpha per world plus a shared (beta_inf, x_half) for one frame.

    Returns {"alpha": {world: ms}, "beta_inf": GB/s, "x_half": bytes,
             "median_rel_err": float, "n": int}. Requires scipy.
    """
    from scipy.optimize import least_squares      # imported lazily: fitting is optional
    import numpy as np

    worlds = sorted({r[1] for r in records})
    idx = {w: i for i, w in enumerate(worlds)}
    w = np.array([r[1] for r in records], dtype=float)
    S = np.array([r[2] for r in records], dtype=float)
    t = np.array([r[3] for r in records], dtype=float)
    wi = np.array([idx[r[1]] for r in records])

    def model(p):
        alphas = p[:len(worlds)]
        binf, xh = p[len(worlds)], p[len(worlds) + 1]
        per_peer = S / w
        return alphas[wi] + (S * (w - 1) / w) / (binf * per_peer /
                                                 (per_peer + xh) * 1e6)

    lo = [0.0] * len(worlds) + [1.0, 1e2]
    hi = [20.0] * len(worlds) + [4000.0, 1e9]
    best = None
    for seed in seeds:
        p0 = [0.1] * len(worlds) + [110.0, seed]
        out = least_squares(lambda p: (model(p) - t) / t, p0,
                            bounds=(lo, hi), max_nfev=40000)
        err = float(np.median(np.abs((model(out.x) - t) / t)))
        if best is None or err < best[1]:
            best = (out.x, err)
    p, err = best
    return {"alpha": {wd: float(a) for wd, a in zip(worlds, p[:len(worlds)])},
            "beta_inf": float(p[len(worlds)]),
            "x_half": float(p[len(worlds) + 1]),
            "median_rel_err": err, "n": len(records)}


def fit_beta_pinned(records: list, alpha_of_world, seeds=(2e4, 2e5, 2e6)) -> dict:
    """Fit (beta_inf, x_half) with alpha supplied, not fitted -- the recommended path.

    `alpha_of_world` is a callable world -> ms, normally an interpolation over
    directly measured points. Pinning alpha removes the degeneracy documented
    above; without it x_half is meaningless.
    """
    from scipy.optimize import least_squares
    import numpy as np

    w = np.array([r[1] for r in records], dtype=float)
    S = np.array([r[2] for r in records], dtype=float)
    t = np.array([r[3] for r in records], dtype=float)
    a = np.array([alpha_of_world(x) for x in w], dtype=float)

    def model(p):
        per_peer = S / w
        return a + (S * (w - 1) / w) / (p[0] * per_peer / (per_peer + p[1]) * 1e6)

    best = None
    for seed in seeds:
        out = least_squares(lambda p: (model(p) - t) / t, [110.0, seed],
                            bounds=([1.0, 1e2], [4000.0, 1e9]), max_nfev=40000)
        err = float(np.median(np.abs((model(out.x) - t) / t)))
        if best is None or err < best[1]:
            best = (out.x, err)
    p, err = best
    return {"beta_inf": float(p[0]), "x_half": float(p[1]),
            "median_rel_err": err, "n": len(records)}


def audit(records: list, verbose: bool = True) -> dict:
    """Fit every frame separately and report. Frames are never pooled.

    Two guards worth reading the output for: a frame that fits far worse than the
    others is telling you its data is contaminated (a library artifact, a loaded
    machine, a different benchmark caliber) long before it tells you the model is
    wrong; and a world that only one frame covers can never be cross-checked, so
    no conclusion should rest on it.
    """
    frames = sorted({r[0] for r in records})
    results = {}
    for f in frames:
        rs = [r for r in records if r[0] == f]
        worlds = sorted({r[1] for r in rs})
        if len(rs) < len(worlds) + 4:
            continue
        results[f] = fit_frame(rs)
    if verbose and results:
        print("%-12s %6s %10s %12s %10s" %
              ("frame", "n", "beta_inf", "x_half (KiB)", "med err"))
        for f, r in results.items():
            print("%-12s %6d %9.1f %12.0f %9.1f%%"
                  % (f, r["n"], r["beta_inf"], r["x_half"] / 1024,
                     r["median_rel_err"] * 100))
        cover = {}
        for f, r in results.items():
            for wd in r["alpha"]:
                cover.setdefault(wd, []).append(f)
        lonely = [wd for wd, fs in sorted(cover.items()) if len(fs) == 1]
        if lonely:
            print("\nworlds covered by a single frame (uncheckable): %s" % lonely)
        worst = max(results.items(), key=lambda kv: kv[1]["median_rel_err"])
        if worst[1]["median_rel_err"] > 2 * min(
                r["median_rel_err"] for r in results.values()):
            print("frame %r fits %.1fx worse than the best frame -- suspect its data, "
                  "not the model, until shown otherwise"
                  % (worst[0], worst[1]["median_rel_err"] /
                     min(r["median_rel_err"] for r in results.values())))
    return results


def _demo() -> None:
    """Model self-check without any data: round-trip a synthetic frame."""
    truth_alpha, binf, xh = 0.30, 118.0, 320 * 1024
    recs = []
    for S in (2 ** 20, 4 * 2 ** 20, 16 * 2 ** 20, 64 * 2 ** 20, 256 * 2 ** 20):
        for wd in (8, 16, 128):
            recs.append(("synthetic", wd, float(S),
                         predict_ms(wd, S, truth_alpha, binf, xh)))
    got = fit_frame(recs)
    print("free alpha : beta_inf %.1f (truth %.1f), x_half %.0f KiB (truth %.0f), "
          "err %.2g%%   <- predictions exact, x_half off: the degeneracy"
          % (got["beta_inf"], binf, got["x_half"] / 1024, xh / 1024,
             got["median_rel_err"] * 100))
    pinned = fit_beta_pinned(recs, lambda _w: truth_alpha)
    print("pinned alpha: beta_inf %.1f, x_half %.0f KiB, err %.2g%%   <- both recovered"
          % (pinned["beta_inf"], pinned["x_half"] / 1024,
             pinned["median_rel_err"] * 100))
    assert math.isclose(got["beta_inf"], binf, rel_tol=0.02)
    assert math.isclose(pinned["x_half"], xh, rel_tol=0.05), (
        "pinning alpha must recover x_half; got %.0f KiB" % (pinned["x_half"] / 1024))


if __name__ == "__main__":
    _demo()
