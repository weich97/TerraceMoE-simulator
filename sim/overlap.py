# -*- coding: utf-8 -*-
"""Tier-2 assault: overlap-aware composition model families -- a systematic retrodiction evaluation.

## Background (why the step-level gate is red)

The sum of phase-level timings and the step-level delta disagree by ~5x: the two arms
overlap differently on their dual streams, so event-timed phase spans do not add up to
step time. Numerical reconnaissance sharpens the contradiction (data = the HOLDOUTS in
sim/validate.py, all measured):

  - The communication model's step delta (Δmodel) is almost entirely the fixed cost of
    "arrival chain + splits sync" (flag: 1000 ms of the 1018 ms); the pure communication
    delta is near zero on this bandwidth-flat machine.
  - The measured step delta (Δmeas) ranges from **-151 ms (flag, two-hop actually wins)**
    to +1833 ms (n4); the implied "exposure ratio" Δmeas/Δmodel runs from -0.15 to +0.62,
    **not even the sign is consistent**.

Conclusion first: **no single-parameter global overlap model can explain all seven points
at once** -- this module turns that sentence into a reproducible table: six structurally
different single-parameter families, all fitted on the calibration point (flag), all
evaluated on the six holdout points, all reported (including the failures -- especially
the failures).

## Discipline

  - Fitting uses flag only (matching the prespecified split of sim/validate.py);
    holdout points never participate.
  - Report the whole family set, pick no winner: even if some family passes retrodiction,
    that is only "model selection on seven points" -- **the Tier-2 gate stays red**, and
    unlocking requires freshly collected holdout points (see the experiment protocol in
    docs/07).
  - Every number in the table is produced directly by this module
    (`python -m sim.overlap`); the script is the provenance.

## Model families (each has exactly 1 free parameter; flag's one equation pins one parameter)

  M0  naive      Δ = Δmodel (no overlap; the current failing Tier-2 baseline, 0 parameters)
  M1  prop       Δ = φ·Δmodel (global proportional exposure)
  M2  hide/call  Δ = Δmodel - h·calls (a fixed number of ms hidden per call)
  M3  hide-fixed Δ = Δmodel - φ·fixed (a φ fraction of chain+splits hidden;
                  pure communication fully exposed)
  M4  hide∝comp  Δ = Δmodel - c·T_comp (hidden amount proportional to in-step compute
                  time; T_comp ≈ t_off - total one-hop communication, i.e. the off arm's
                  non-communication time)
  M5  expose-mb  Δ = Δmodel·(1 - λ/mbs) (more micro-batches, tighter pipeline, more
                  exposure -- a literal reading of the recon fact that the exposure
                  ratio rises monotonically with mbs)

Evaluation gate identical to Tier-2: holdout MAE ≤ 0.025, ≥4 points within ±0.035,
all sign points in the right direction.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core import step_delta
from .validate import HOLDOUTS

NEG_NAMES = {"tok2x", "tok4x", "k8m4", "n8", "n4"}   # the 4 prespecified sign points + the later n4;
# k8m2 (G=0.9935, n=1, within noise of 1) is not counted for direction -- matches
# the sign convention in validate.py


# ----------------------------------------------------------------------------
# Per-geometry decomposition (all derived from the Tier-1-validated communication
# model + measured step times)
# ----------------------------------------------------------------------------

@dataclass
class Decomp:
    name: str
    role: str
    g_meas: float
    t_off: float
    calls: int            # a2a calls per step (dispatch+combine, fwd+bwd)
    mbs: int
    d_model: float        # communication-model step delta [ms] (>0 = two-hop slower)
    d_meas: float         # measured step delta [ms]
    fixed: float          # of which: total fixed cost of chain+splits [ms]
    comm_off: float       # total communication of the off arm (one-hop) [ms]
    t_comp: float         # ≈ t_off - comm_off: the off arm's non-communication time [ms]


def decompose(cluster) -> list:
    out = []
    for h in HOLDOUTS:
        d = step_delta(cluster, h.geom)
        calls = d["calls_per_step"]
        chain = cluster.chain_us_per_row * h.geom.rows_hop_b() / 1000.0
        fixed = (chain + cluster.splits_sync_ms) * calls
        comm_off = d["per_call_off_ms"] * calls
        t_on = h.t_off_ms / h.g_measured
        out.append(Decomp(
            name=h.geom.name, role=h.role, g_meas=h.g_measured,
            t_off=h.t_off_ms, calls=calls, mbs=h.geom.mbs,
            d_model=d["step_delta_ms"], d_meas=t_on - h.t_off_ms,
            fixed=fixed, comm_off=comm_off,
            t_comp=h.t_off_ms - comm_off))
    return out


# ----------------------------------------------------------------------------
# Model families: predict(dec, param) -> predicted Δ; fit solves closed-form on flag
# ----------------------------------------------------------------------------

@dataclass
class Family:
    key: str
    desc: str
    n_params: int
    predict: object       # (Decomp, float) -> Δms
    fit: object           # (Decomp) -> float (solves the parameter on the calibration point)


def _families() -> list:
    return [
        Family("M0", "naive: Δ=Δmodel (no overlap)", 0,
               lambda d, p: d.d_model,
               lambda d: 0.0),
        Family("M1", "prop: Δ=φ·Δmodel", 1,
               lambda d, p: p * d.d_model,
               lambda d: d.d_meas / d.d_model),
        Family("M2", "hide/call: Δ=Δmodel-h·calls", 1,
               lambda d, p: d.d_model - p * d.calls,
               lambda d: (d.d_model - d.d_meas) / d.calls),
        Family("M3", "hide-fixed: Δ=Δmodel-φ·fixed", 1,
               lambda d, p: d.d_model - p * d.fixed,
               lambda d: (d.d_model - d.d_meas) / d.fixed),
        Family("M4", "hide∝comp: Δ=Δmodel-c·T_comp", 1,
               lambda d, p: d.d_model - p * d.t_comp,
               lambda d: (d.d_model - d.d_meas) / d.t_comp),
        Family("M5", "expose-mb: Δ=Δmodel·(1-λ/mbs)", 1,
               lambda d, p: d.d_model * (1.0 - p / d.mbs),
               lambda d: (1.0 - d.d_meas / d.d_model) * d.mbs),
    ]


def evaluate(cluster, verbose: bool = True) -> dict:
    """Fit every family + evaluate on the holdouts. Returns {family_key: metrics}."""
    decs = decompose(cluster)
    cal = next(d for d in decs if d.role == "calibration")
    holds = [d for d in decs if d.role == "holdout"]
    results = {}
    for fam in _families():
        p = fam.fit(cal)
        rows, errs, in_tol, signs_ok = [], [], 0, 0
        for d in holds:
            delta_pred = fam.predict(d, p)
            g_pred = d.t_off / (d.t_off + delta_pred)
            e = g_pred - d.g_meas
            errs.append(abs(e))
            if abs(e) <= 0.035:
                in_tol += 1
            if d.name in NEG_NAMES and g_pred < 1.0:
                signs_ok += 1
            rows.append((d.name, d.g_meas, g_pred, e))
        mae = sum(errs) / len(errs)
        n_neg = len([d for d in holds if d.name in NEG_NAMES])
        gate = (mae <= 0.025) and (in_tol >= 4) and (signs_ok == n_neg)
        results[fam.key] = {"param": p, "mae": mae, "in_tol": in_tol,
                            "signs_ok": signs_ok, "n_neg": n_neg,
                            "gate": gate, "rows": rows, "desc": fam.desc}
    if verbose:
        _report(decs, results)
    return results


def _report(decs, results) -> None:
    print("Recon table (the model step delta is almost entirely fixed cost; measured exposure ratios do not even agree in sign):")
    print("%-7s %6s %9s %+11s %+11s %9s %7s" %
          ("geom", "mbs", "G_meas", "d_meas_ms", "d_model_ms", "fixed", "expose"))
    for d in decs:
        print("%-7s %6d %9.4f %+11.1f %+11.1f %9.1f %7.3f" %
              (d.name, d.mbs, d.g_meas, d.d_meas, d.d_model, d.fixed,
               d.d_meas / d.d_model))
    print()
    print("Model-family retrodiction (fit uses flag only; gate = MAE≤0.025 and ≥4/6 within ±0.035 and all signs correct):")
    print("%-4s %-34s %10s %8s %7s %7s %6s" %
          ("fam", "structure", "param", "MAE", "±0.035", "signs", "gate"))
    for k, r in results.items():
        print("%-4s %-34s %10.4f %8.4f %5d/6 %5d/%d %6s" %
              (k, r["desc"], r["param"], r["mae"], r["in_tol"],
               r["signs_ok"], r["n_neg"], "PASS" if r["gate"] else "FAIL"))
    passed = [k for k, r in results.items() if r["gate"]]
    print()
    if passed:
        print("!! %s passed retrodiction -- but this is only model selection on seven points; the Tier-2 gate stays red;" % passed)
        print("   unlocking requires fresh holdout points collected per the docs/07 protocol, plus human review, before the gate moves.")
    else:
        print("**All families fail -- the single-parameter global overlap model road is exhausted; the negative result stands.**")
        print("The next step is not more parameters (1 calibration point pins exactly 1 parameter;")
        print("anything more means tuning on holdouts) but new measurements: each arm's own")
        print("dual-stream overlap timeline (docs/07 protocol).")


if __name__ == "__main__":
    from .calibrate import flat_supernode
    evaluate(flat_supernode())
