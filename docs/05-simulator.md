# The simulator: measured on one machine, answers for many clusters

`sim/` is a **measurement-calibrated** MoE-EP communication simulator: given a cluster spec
(α, β, and group size for the fast/slow tiers) and a MoE geometry (E/N_g/k/M/H/sequence/
micro-batch), it predicts one-hop and two-hop all-to-all communication time, and — only where
the validation gates pass — extrapolates the comparison to clusters you do not have.

## Methodology (this is the main deliverable; the code is just its carrier)

```
measure  ->  calibrate  ->  validate  ->  extrapolate
```

1. **Measure**: microbenchmarks measure primitives only — the α(world) curve, β (aligned-payload
   convention), and per-call fixed costs (splits host sync, arrival chain).
2. **Calibrate**: every simulator parameter points to one measurement, with its convention noted
   (`sim/calibrate.py`); this repo publishes the distilled constants, not the raw sweeps.
3. **Validate (the step most simulation work skips)**: the simulator must first reproduce
   **independent measurements on the same machine** before it earns the right to extrapolate.
   Two gate tiers, thresholds fixed ahead of the targets:
   - **Tier-1 (communication micro level)**: against directly measured one-hop/two-hop times on
     the same machine — median relative error ≤20%, max ≤35%, crossover position reproduced.
     **Current: pass** (8.1% / 12.5% / 2048–4096).
   - **Tier-2 (end-to-end step level)**: against measured G on 7 training geometries
     (1 calibration, 6 holdout; n4 is a scale-axis point added after preregistration).
     **Current: fail — step-level extrapolation stays locked.** The cause, named: the sum of
     phase-level timings differs from the step-level delta by ~5× (the two arms overlap
     differently across the dual streams; event-timed phase spans do not add up to step time).
     This is a known open problem, not something parameter tuning can fix; the test suite pins
     `test_tier2_step_gate_currently_fails_documented` — **the day this gate suddenly turns
     green, a human must check whether the problem was actually solved or the gate was loosened**.
4. **Extrapolate**: only tiers that passed their gate may extrapolate, and every output is
   labeled "simulated".

## Calibration highlights (details in the `sim/calibrate.py` comments)

- **β_fast has physical backing**: intra-node 8-die a2a measures 122.6 GB/s, within 0.2% of the
  aggregate-egress physical value (6 × intra-node link 112.1 + 1 × in-package direct 185)/7
  = 122.4, and run-to-run spread at this tier is <0.3% — the most stable number in the entire
  dataset.
- **α is a machine property; never assume it across machines**: on this machine, directly
  measured α(16)+α(8) ≈ α(128) (two-hop saves almost nothing on α); another machine's curve
  implies a 2.85× saving. Across machines you may borrow only the shape — re-calibrate locally,
  and report sensitivity.
- **Machine run-to-run drift is ~20%**: validation targets use the pooled median across runs;
  no single-run number is good enough to serve as a gate.

## The first extrapolation figure (communication level, Tier-1 unlocked; all numbers are **simulated**, not measured)

One-hop time / two-hop time (>1 = two-hop faster), 16 groups × R=8, k=6, T=4096 tokens/rank:

| Implementation tier \ hierarchy ratio | 1.03 (flat) | 3.2 | 8 | 15.7 |
|---|---|---|---|---|
| PyTorch arrival chain (measured 0.0875 µs/row) | 0.39 | 0.83 | 1.47 | 2.02 |
| Fused kernel (est. 0.012) | 0.79 | 1.47 | 2.18 | 2.63 |
| Zero implementation overhead (upper bound) | 0.94 | 1.68 | 2.36 | 2.77 |

(q=3; for q=2/q=6 and all six hierarchy-ratio tiers, run `python -m sim.sweep` to compute them
live — the table tracks the calibration; the doc stores only a snapshot.)

Three consistency anchors:

- The flat column is ≤1 — reproducing our measured negative verdict on the flat supernode
  (internal consistency);
- Columns at hierarchy ratio ≥3.2 are >1 — directionally consistent with public work
  (DeepSeek-V3 node-limited, TeleChat3-MoE +15%, Pangu Ultra MoE) (external consistency);
- The implementation tier's impact is on the same order as the hierarchy ratio's —
  **"fix the implementation first, then talk topology" holds on hierarchical machines too**.

## Three new figures: is two-hop worth it, at a glance (all **simulated**, Tier-1 scope)

### The breakeven map: which region does your cluster land in?

![breakeven map](assets/f8-breakeven-map.svg)

Green region = two-hop wins. **Breakeven hierarchy ratio** (the smallest hierarchy ratio at
which two-hop starts to win, q=3, T=4096):

| Implementation tier | breakeven hierarchy ratio |
|---|---|
| PyTorch arrival chain (measured 0.0875 µs/row) | **4.20** |
| Fused kernel (est. 0.012) | **1.57** |
| Zero implementation overhead (upper bound) | 1.16 |

One-sentence takeaway: **the implementation tier pulls breakeven from 4.2 down to 1.6** — on
common hierarchical machines like NVLink/IB (≈3.2), whether hierarchical communication is worth
doing depends on how well your arrival chain is written, not on the topology.

### Uncertainty bands: is the conclusion stable against calibration error?

![uncertainty bands](assets/f9-uncertainty.svg)

Calibration constants are not ground truth; they are measurements with spread (measured machine
run-to-run drift ±9~20%). 400 Monte Carlo draws propagate that spread into the extrapolation
(provenance noted item by item, see `sim/uncertainty.py`; fixed seed, reruns match bit for bit).
Two robustness anchors:

- Flat column, most favorable case (zero implementation overhead): p95 = **0.99 ≤ 1** — our
  negative verdict on the flat machine is robust to calibration error; but the margin is only
  0.01, so the claim stops at this precision and gets no further embellishment;
- 8× column, least favorable case (PyTorch chain): p5 = **1.37 > 1** — on hierarchical machines,
  the direction "two-hop wins" is robust to calibration error.

### Scale effects: where does the large-cluster advantage come from?

![scale effects](assets/f10-scale-alpha.svg)

Fix hierarchy ratio 3.2 and the fused tier, scale the cluster from 32 dies to 512 dies: the
two-hop ratio goes from 1.47 (128 dies) to 1.94 (512 dies) — while the "flat-α counterfactual
machine" control goes 1.44 → 1.38. **Nearly all of the extra advantage comes from α(world)
growing with scale** (one-hop pays α(512)=1.86 ms, two-hop pays α(64)+α(8)≈0.36 ms), and this
machine's α curve beyond 128 dies is a borrowed shape (shaded in the figure). The correct
reading of the conclusion: **α shape is a machine property — whether two-hop gains extra on a
large cluster requires re-calibrating α on any new machine.**

Geometry sensitivity (a 54-point (group count, R, k, M) grid, `sim.uncertainty.geometry_grid`)
points the same way: the larger the world, the lower the breakeven; at 512 dies several
geometries hit breakeven=1.0 (a pure α effect). Ranking the three axes by how far they actually
move breakeven (fused tier): **implementation tier largest** (4.20 → 1.16, Δ≈3.0) > scale axis
(32 → 512 dies, Δ≤1.9) > geometry axis ((k,M) moves ±0.2~0.3 at fixed world; under the PyTorch
tier the geometry axis widens to ±1.2, but the ordering stands).

## The Tier-2 step-level synthesis campaign record

All six families of single-parameter global overlap models are dead; the negative result plus
the breakthrough measurement protocol are in
**[docs/07-tier2-overlap.md](07-tier2-overlap.md)** — step-level extrapolation stays locked,
but "which measurement is missing, how to take it, and how to verify self-consistency" is now
written up as a protocol anyone with a cluster can run as-is.

## Bring your own machine

1. Measure three things with the same conventions as `docs/03 §4`: α (by world size), β
   (aligned-payload convention), and per-call fixed cost;
2. Fill in a `ClusterSpec` (model it on `sim/calibrate.py::flat_supernode`);
3. Replace `sim/validate_micro.py::MICRO_TARGETS` with the same-type microbenchmarks from your
   machine, and re-pass the gates;
4. Only after the gates are green, look at the `python -m sim.sweep` extrapolation.

## Run

```
python -m sim.validate_micro     # Tier-1 gate
python -m sim.validate           # Tier-2 gate (currently reports the failure, truthfully)
python -m sim.sweep              # extrapolation (checks the gates at entry)
python -m sim.overlap            # Tier-2 campaign: overlap model family battle report (docs/07)
python -m sim.uncertainty        # Monte Carlo uncertainty bands + breakeven (fixed seed)
python tools/gen_sim_figures.py  # regenerate F7-F12 (numbers computed live; only F12 embeds measurements)
python -m pytest tests/test_sim.py -q
```
