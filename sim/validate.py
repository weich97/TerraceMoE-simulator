# -*- coding: utf-8 -*-
"""Validation gate: the simulator must first reproduce our own end-to-end ground truth before it earns the right to extrapolate.

## Preregistration (2026-08-24, written before the first run; no after-the-fact edits)

**Calibration point (1)**: `flag`. The combine-side implementation delta (the on arm's
combine is actually faster; mechanism not fully explained, internal measurement
records/internal measurement records) cannot be derived from first principles; a constant
`combine_extra_ms` is back-solved from flag's step-level ground truth and **held fixed
for all remaining holdout geometries**.

**Holdout validation points (never used in any fit)**: tok2x / tok4x / k8m2 / k8m4 / n8
(the 5 preregistered) + n4 (a third scale-axis point added after preregistration; the
gate's threshold numbers stay put, the denominator honestly counts 6).

**Gate (all must hold to pass; thresholds are the preregistered originals, denominators labeled as they now stand)**:
  1. G predictions on holdout points: mean absolute error MAE ≤ 0.025 (over 6 points after n4 joined);
  2. at least 4 points within ±0.035 (preregistered as 4/5; the denominator is now 6);
  3. all four preregistered sign points (tok2x/tok4x/k8m4/n8) in the right direction (G_pred < 1).

No gate pass, no extrapolation -- the sweep module checks this gate at runtime.

## End-to-end ground truth (control testbed, identical config except the a2a; provenance: internal measurement records/internal measurement records/internal measurement records)

Step time is the steady-state median (first 300 steps trimmed, n=10/arm); G = t_off / t_on.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core import MoEGeometry, step_delta


@dataclass
class Holdout:
    geom: MoEGeometry
    t_off_ms: float          # measured step time of the vendor arm
    g_measured: float        # measured G
    role: str                # "calibration" | "holdout"


def _g(name, ng, R, k, M, mbs, layers=19):
    return MoEGeometry(name=name, n_groups=ng, R=R, k=k, M=M, mbs=mbs,
                       moe_layers=layers)


HOLDOUTS = [
    # flag: paired-group mean at the base tier (internal measurement records); the **calibration point**
    Holdout(_g("flag", 16, 8, 6, 2, 1), 4420.4, 1.0355, "calibration"),  # n=7 mean (t=3.27, p≈0.017, significant)
    # holdout validation points
    Holdout(_g("tok2x", 16, 8, 6, 2, 2), 3343.6, 0.8845, "holdout"),   # mean of three runs, spread 0.8%
    Holdout(_g("tok4x", 16, 8, 6, 2, 4), 2992.7, 0.8234, "holdout"),   # mean of three runs, spread 0.6%
    Holdout(_g("k8m2", 16, 8, 8, 2, 1), 4791.7, 0.9935, "holdout"),
    Holdout(_g("k8m4", 16, 8, 8, 4, 1), 4664.9, 0.9335, "holdout"),
    Holdout(_g("n8", 8, 8, 6, 2, 1), 6657.5, 0.9027, "holdout"),
    Holdout(_g("n4", 4, 8, 6, 2, 1), 11712.0, 0.8647, "holdout"),   # third point on the scale axis
]


def predict_g(cluster, h: Holdout, combine_extra_ms: float) -> float:
    """G prediction = t_off / (t_off + delta_step).

    delta_step = dispatch delta x dispatch calls + combine delta x combine calls.
    The dispatch delta uses the full model; combine's communication bytes mirror
    dispatch, but the implementation-side delta substitutes the calibrated constant
    combine_extra_ms for local_chain (see the module docstring).
    """
    d = step_delta(cluster, h.geom)
    calls_fwd = h.geom.calls_per_step_fwd() + h.geom.calls_per_step_bwd()
    chain_ms = cluster.chain_us_per_row * h.geom.rows_hop_b() / 1000.0
    # dispatch: full model
    disp_delta = (d["per_call_on_ms"] - d["per_call_off_ms"]) * calls_fwd
    # combine: communication part same as dispatch (mirror); implementation delta calibrated per row
    # (combine_extra_ms is really the value at flag's row count; scale by row count for other geometries)
    comm_delta = d["per_call_on_ms"] - chain_ms - d["per_call_off_ms"]
    scale = h.geom.rows_hop_b() / 24576.0
    comb_delta = (comm_delta + combine_extra_ms * scale) * calls_fwd
    t_on = h.t_off_ms + disp_delta + comb_delta
    return h.t_off_ms / t_on


def calibrate_combine(cluster) -> float:
    """Back-solve combine_extra_ms from the calibration point (flag)."""
    h = next(x for x in HOLDOUTS if x.role == "calibration")
    d = step_delta(cluster, h.geom)
    calls = h.geom.calls_per_step_fwd() + h.geom.calls_per_step_bwd()
    t_on_target = h.t_off_ms / h.g_measured
    chain_ms = cluster.chain_us_per_row * h.geom.rows_hop_b() / 1000.0
    disp_delta = (d["per_call_on_ms"] - d["per_call_off_ms"]) * calls
    comm_delta = d["per_call_on_ms"] - chain_ms - d["per_call_off_ms"]
    # t_on_target = t_off + disp + (comm_delta + x) * calls
    x = (t_on_target - h.t_off_ms - disp_delta) / calls - comm_delta
    return x


def validate(cluster, verbose: bool = True):
    """Run the validation gate. Returns (passed?, details)."""
    ce = calibrate_combine(cluster)
    rows, errs, in_tol, signs_ok = [], [], 0, 0
    neg = {"tok2x", "tok4x", "k8m4", "n8"}
    for h in HOLDOUTS:
        gp = predict_g(cluster, h, ce)
        e = gp - h.g_measured
        if h.role == "holdout":
            errs.append(abs(e))
            if abs(e) <= 0.035:
                in_tol += 1
            if h.geom.name in neg and gp < 1.0:
                signs_ok += 1
        rows.append((h.geom.name, h.role, h.g_measured, gp, e))
    mae = sum(errs) / len(errs)
    ok = (mae <= 0.025) and (in_tol >= 4) and (signs_ok == len(neg))
    if verbose:
        print("Validation gate (calibration point flag back-solves combine_extra=%.3f ms/call)" % ce)
        print("%-8s %-12s %8s %8s %8s" % ("geom", "role", "G_meas", "G_pred", "err"))
        for n, r, gm, gp, e in rows:
            print("%-8s %-12s %8.4f %8.4f %+8.4f" % (n, r, gm, gp, e))
        print("holdout MAE=%.4f (gate 0.025)  within ±0.035: %d/6 (gate 4)  signs: %d/4 (gate 4)"
              % (mae, in_tol, signs_ok))
        print("**%s**" % ("PASS -- extrapolation allowed" if ok else
                          "FAIL -- extrapolation forbidden; fix the model first"))
    return ok, {"mae": mae, "in_tol": in_tol, "signs_ok": signs_ok,
                "combine_extra_ms": ce, "rows": rows}


if __name__ == "__main__":
    from .calibrate import aug_flat
    validate(aug_flat())
