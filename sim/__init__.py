# -*- coding: utf-8 -*-
"""terrace-sim: an MoE-EP communication simulator calibrated from real measurements -- one machine's data, many machines' answers.

## Why this exists (the 2026-08-24 pivot)

T-A2A got a negative verdict on our **bandwidth-flat** supernode (internal measurement records):
the breakeven criterion r_be = (1-1/R)q/(q-1) demands a fast/slow bandwidth ratio > 1.31 (q=3);
the machine gives only ~1.0. But the criterion itself -- and the proposition that hierarchical
communication pays off on hierarchical clusters -- needs an answer across **many kinds of
clusters**, and we have exactly one. The pivot: turn every measurement banked on this machine
into a simulator with **swappable cluster parameters**.

## Methodology (this is the core of the deliverable, not the code)

    measure -> calibrate -> validate -> extrapolate

1. **Measure**: microbenchmarks measure only "primitives" -- the alpha(world) curve, the
   beta(size) curve (split by domain: intra-node / cross-node / cross-supernode), and the
   fixed per-call costs (splits host sync, local index chain).
2. **Calibrate**: every simulator parameter points at a measurement file; no hand-picked
   numbers (the manifest discipline of terrace/commodel.py carries over here).
3. **Validate (the step most simulation work skips)**: the simulator must first reproduce
   our own end-to-end ground truth -- the G values of the control testbed's 6 geometries
   (sign and magnitude, from +2.8% to -21.8%). Microbenchmark calibration and end-to-end
   validation are two mutually independent datasets; preregistered tolerances are in
   sim/validate.py, **frozen before the run**.
4. **Extrapolate**: only after validation passes may cluster parameters (hierarchy ratio,
   alpha curve, R, incast) be swapped to draw conclusions. Extrapolated results are always
   labeled "simulation" and listed strictly apart from measurements.

## Honesty boundary

- The simulator predicts **communication delta / G**, not absolute step time: the compute
  side (GEMM/router) is calibrated to a per-geometry constant from off-arm measurements --
  so it answers "how does the speed ratio move if you swap the communication scheme or the
  cluster", not "how fast does this model train".
- Cross-cluster extrapolation of alpha and beta rests on a "shape invariant, scale
  recalibrated" assumption; every extrapolation reports sensitivity.
- Not modeled: overlap/pipelining (neither control-testbed arm overlaps; baselines match),
  fault tolerance, and the tails of network jitter.
"""
from .core import ClusterSpec, MoEGeometry, one_hop_call, two_hop_call, step_delta
from .validate import HOLDOUTS, validate

__all__ = ["ClusterSpec", "MoEGeometry", "one_hop_call", "two_hop_call",
           "step_delta", "HOLDOUTS", "validate"]
