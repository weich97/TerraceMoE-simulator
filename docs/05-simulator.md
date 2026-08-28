# The simulator: measured on one machine, ratio sensitivities for new machines

`sim/` is a **measurement-calibrated** MoE-EP communication simulator: given a cluster spec
(α, β, and group size for the fast/slow tiers) and a MoE geometry (E/N_g/k/M/H/sequence/
micro-batch), it predicts one-hop and two-hop all-to-all communication time, and — only where
the validation gates pass — evaluates sensitivity to explicitly supplied machine parameters.
It does not turn nominal link rates into a target-platform verdict: a new target must measure
the five inputs used by the model and pass the communication-level gate locally.

## Methodology (this is the main deliverable; the code is just its carrier)

```
measure  ->  calibrate  ->  validate  ->  sensitivity analysis
```

1. **Measure**: microbenchmarks measure primitives only — the α(world) curve, β (aligned-payload
   convention), and per-call fixed costs (splits host sync, arrival chain).
2. **Calibrate**: every simulator parameter points to one measurement, with its convention noted
   (`sim/calibrate.py`); this repo publishes the distilled constants, not the raw sweeps.
3. **Validate (the step most simulation work skips)**: the simulator must first reproduce
   **independent measurements on the same machine** before it earns the right to extrapolate.
   Three gate tiers, thresholds fixed ahead of the targets:
   - **Tier-1 (communication micro level)**: against directly measured one-hop/two-hop times on
     the same machine — median relative error ≤20%, max ≤35%, crossover position reproduced.
   **Current: pass** (4.1% / 24.5% / 4096–8192).
   - **Tier-1b (cross-corpus, `sim/validate_sweep.py`)**: the same model against 64 distilled
     targets from a *different* benchmark family on **two machines** — the calibrated one, and
     a second one whose α, flat β, and x_half are fitted from C3 while the level split
     and arrival-chain assumptions remain inherited. Gate:
     median relative error ≤12%, at most 1 in 5 targets over 35%, median signed error within
     ±8%. **Current: pass** on all three corpora (machine A 1.9% median over 6 targets;
     machine B 9.3% over 44, bias −1.1%; machine A at world 16, collected 2026-08-26 after
     the constants were frozen, 8.0% over 14, bias −2.0%). The first two are same-corpus
     fit/consistency guards written after calibration, not out-of-sample transfer tests; the
     third is the post-freeze holdout. A fourth
     corpus, machine A at world 8, **fails** at 15.1% and −9.1%; it is reported as a drift
     probe on α(8) and held out of the conjunction, with the reasoning below.
   - **Tier-2 (end-to-end step level)**: against measured G on 7 training geometries
     (1 calibration, 6 holdout; n4 is a scale-axis point added after preregistration).
     **Current: fail — step-level extrapolation stays locked.** The cause, named: the sum of
     phase-level timings differs from the step-level delta by ~5× (the two arms overlap
     differently across the dual streams; event-timed phase spans do not add up to step time).
     This is a known open problem, not something parameter tuning can fix; the test suite pins
     `test_tier2_step_gate_currently_fails_documented` — **the day this gate suddenly turns
     green, a human must check whether the problem was actually solved or the gate was loosened**.
4. **Sensitivity analysis**: only tiers that passed their gate may be varied, and every output
   is labeled "simulated". A target-platform conclusion additionally requires target-local
   measurement and validation.

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
| PyTorch arrival chain (measured 0.0875 µs/row) | 0.40 | 0.87 | 1.51 | 2.04 |
| Hypothetical fused target (0.012) | 0.81 | 1.51 | 2.19 | 2.62 |
| Zero implementation overhead (upper bound) | 0.97 | 1.70 | 2.36 | 2.74 |

(q=3; for q=2/q=6 and all six hierarchy-ratio tiers, run `python -m sim.sweep` to compute them
live — the table tracks the calibration; the doc stores only a snapshot.)

Three consistency anchors:

- The flat column is ≤1 — consistent with our measured negative result on platform A;
- the higher-ratio columns are synthetic scenarios, not measurements or predictions for named
  platforms;
- The implementation tier's impact is on the same order as the hierarchy ratio's —
  **"fix the implementation first, then talk topology" holds on hierarchical machines too**.

## Three new figures: is two-hop worth it, at a glance (all **simulated**, Tier-1 scope)

### The breakeven map: which region does your cluster land in?

![breakeven map](assets/f8-breakeven-map.svg)

Green region = two-hop wins. **Breakeven hierarchy ratio** (the smallest hierarchy ratio at
which two-hop starts to win, q=3, T=4096):

| Implementation tier | breakeven hierarchy ratio |
|---|---|
| PyTorch arrival chain (measured 0.0875 µs/row) | **3.98** |
| Hypothetical fused target (0.012) | **1.49** |
| Zero implementation overhead (upper bound) | 1.10 |

One-sentence takeaway: in this calibrated sensitivity study, the implementation tier moves the
breakeven from 3.98 down to 1.49 (or 1.10 at the zero-overhead bound). Whether a target machine
lands on either side is unresolved until its effective ratio and call costs are measured.

### Uncertainty bands: is the conclusion stable against calibration error?

![uncertainty bands](assets/f9-uncertainty.svg)

Calibration constants are not ground truth; they are measurements with spread (measured machine
run-to-run drift ±9~20%). 400 Monte Carlo draws propagate that spread into the extrapolation
(provenance noted item by item, see `sim/uncertainty.py`; fixed seed, reruns match bit for bit).
Two robustness anchors:

- Flat column, most favorable case (zero implementation overhead): p95 = **1.03** — just above
  1. Read it precisely: once bandwidth saturation is modelled, the *byte account alone* on a
  flat fabric is close to neutral, so what actually makes two-hop lose there is the
  implementation overhead that this tier deliberately sets to zero. The measured verdict on the
  flat machine is unchanged (docs/03), and it is an implementation verdict, not a bytes verdict.
  Under the fused tier — the best any real implementation has reached — the flat column stays
  below 1;
- 8× synthetic scenario, least favorable case (PyTorch chain): p5 = **1.41 > 1**. This makes the
  sign robust inside the model's uncertainty calculation, not on an unmeasured target machine.

### Scale effects: where does the large-cluster advantage come from?

![scale effects](assets/f10-scale-alpha.svg)

Fix hierarchy ratio 3.2 and the fused tier, and scale the cluster up. Through **128 dies** the
answer is solid: 1.68 → 1.52 → 1.47 at 32 / 64 / 128, and those three numbers are *identical to
the digit* under every defensible treatment of α, because they use only the worlds we measured
directly (8, 16, 128).

**Past 128 dies this repository makes no claim.** An audit of every size sweep we own (331
usable a2a points, four datasets, two machines) found that the single corpus covering worlds
256 and 512 is also the corpus whose absolute bandwidth sits ~5× below all the others and which
the cost model fits worst — 40% median relative error, against 5–11% everywhere else. Push four
defensible treatments of those two α entries through the extrapolation and the 512-die ratio
lands anywhere in **1.37 – 2.94**, with the *direction of the trend flipping* between them
(under "no growth past 128" the ratio falls from 1.47 to 1.37). The figure plots that band
rather than a line, and `tests/test_sim.py` pins both halves: insensitivity below 128, and a
≥2× spread at 512 so the claim cannot be quietly re-hardened.

What survives is the mechanism, not the magnitude: two-hop's large-cluster upside is bought
with α(world) — one-hop pays α at the full world while two-hop pays α at two much smaller
worlds — and **α's shape is a machine property**, so on any new machine this question reopens
and must be re-measured, not inherited.

Geometry sensitivity (a 54-point (group count, R, k, M) grid, `sim.uncertainty.geometry_grid`)
is consistent with that mechanism, and carries the same caveat wherever it reaches past 128.
Ranking the three axes by how far they actually move breakeven (fused tier): **implementation
tier largest** (3.98 → 1.10, Δ≈2.9) > scale axis (Δ≤1.9, and only the ≤128 part of it is
trustworthy) > geometry axis ((k,M) moves up to 0.53 at fixed world; under the PyTorch tier the
geometry axis widens to 1.62, but the ordering stands).

## Calibration audit: what 331 measured points say about the constants

The shipped constants were distilled from one benchmark family. Re-fitting the
parametric cost model against **every size sweep we own** — 331 usable a2a points
across four datasets and two machines (`sim/fit.py`) — puts each constant on a
different footing:

| Constant | Verdict | Evidence |
|---|---|---|
| β (asymptotic bandwidth) | **corroborated** | fitted independently per dataset: 117.8 GB/s on the calibrated machine, 111.0–130.3 on the second — spread narrower than either machine's own drift, and unchanged whether α is pinned or free |
| α at worlds 8 / 16 / 128 | **corroborated** | direct measurements; every scale conclusion below 128 ranks is identical under all α treatments |
| α at worlds 256 / 512 | **not supported** | only one corpus reaches them, and it is the corpus with 5× lower absolute bandwidth and 40% fit error vs 5–11% elsewhere — see the scale section above |
| half-performance size `x_half` | **not identifiable** | trades off against α; moves 3–5× depending on whether α is pinned. Quote it only from a pinned-α fit |

### What changed, and the wrong turn on the way

The flat β is gone: the full-fabric and cross-node levels now use
β(x) = β∞·x/(x + x_half) with β∞ = 117.8 GB/s (fitted on this machine's own sweeps) and
x_half = 54 KiB (a **borrowed shape**: this machine's corpus has only 6 sizes and cannot
resolve the parameter, so it comes from the second machine's 19-size sweep — the same
borrow-the-shape-never-the-level discipline the α curve already follows). The intra-node
level keeps a flat β, because its value is physics-endorsed (link aggregation,
0.2% from measurement) and the sweep corpora never isolate it.

After correcting Hop A's local-group fraction from M/N_g to 1/N_g, the current result is
**Tier-1 median error 4.1%** (worst 24.5%, still
inside the 35% gate), and the entire bootstrap interval of x_half passes it (30 KiB → 2.5%,
87 KiB → 5.0%). Tier-1b then confirms the model on 64 targets from a different benchmark
family across two machines. The
extrapolated ratios move up 4–5% — two-hop's messages are larger per peer, so it
suffers less from saturation than one-hop does.

**The wrong turn is worth recording.** The first attempt at exactly this change
*failed*: with x_half fitted at 320 KiB, Tier-1 went to 17.9% median and 101.6%
worst. That x_half was fitted with α free, and α and x_half trade off against each
other — so the parameter was meaningless, and the model form got blamed for it.
Re-estimating with α pinned to its direct measurements, from the one corpus with
enough distinct sizes to resolve the parameter (19 sizes over 4.5 decades, giving
54 KiB with a bootstrap 90% interval of [30, 87]), the same model form halves the
error instead. **A model form cannot be judged while one of its parameters is
unidentified.**

One term still **rejected**: a world-scaling bandwidth degradation past ~128 ranks
is visible in one dataset at matched per-peer size (roughly 0.55× at 256 and 0.36×
at 512), but that is the low-confidence dataset, so it is recorded as unconfirmed
and stays out of the model.

One caveat that survives: the crossover position moved to the upper edge of the
prespecified window, so an x_half much above the fitted interval would fail the
gate. `tests/test_sim.py` holds the breakeven snapshot and the anchors so any
further drift shows up immediately.

**How much room x_half actually has, measured against the gates rather than the
fit.** Sweeping x_half and rerunning every gate gives a **one-sided** answer:
machine B rules out anything above **77 KiB** (`X_HALF_GATE_UPPER`, pinned by a
test), and nothing rules out the low end — 1 KiB passes every gate, because below
the knee β_eff is just β∞ across the payload range the corpora cover. So the gates
corroborate the upper half of the bootstrap interval [30, 87] KiB and are silent
about the lower half. x_half stays weakly determined, and that is consistent with
`sim/fit.py`'s identifiability caveat rather than a surprise.

An earlier revision of this page claimed a two-sided interval of [46, 76] KiB. That
lower bound came from a single-run version of the world-16 corpus; repeating the
sweep six times and taking medians dissolved it. Worth stating plainly, because the
error ran the flattering way: **an admissibility interval computed from a noisy
corpus comes out too tight, and reads as a stronger result than the data supports.**

The same sweeps disposed of an attempt that did not work. Two dense 13-size sweeps
were taken on the calibrated machine specifically to re-fit x_half from its own data
and retire the borrow. They could not, even at 12-sweep medians: the pinned-α fit
lands at x_half above 500 KiB with β∞ at 153 GB/s, against an asymptote of 103 GB/s
directly visible at the largest size in the same sweep. That is the α/x_half
degeneracy on a corpus whose small sizes are almost pure α, not a measurement, and
nothing from it is shipped. **x_half remains a borrowed shape.**

**And one gate now fails.** Those same world-8 medians, scored as a fourth Tier-1b
corpus, miss at 15.1% median with a −9.1% bias. The miss is localised: below 8 MB,
where α is 76–97% of the prediction, the model runs 18% fast; above it the wire term
dominates and it runs 10% slow. That is one number, α(8) = 0.111, against 0.129
measured the same day by the independent call-count scan. Setting α(8) to the
measured value clears the corpus (9.3%, −0.8%) and keeps Tier-1 green at 4.9%
against its 20% gate. **The constant is not changed**: both readings are direct
measurements of the same machine 15% apart, which is inside the documented 20%
drift, and the choice moves the breakeven ratio by under 2.1%. Retuning a constant to green a newly added
corpus, at the cost of the one gate that was passed blind, is the failure mode the
tiering exists to prevent. The corpus ships red, with the cause, held out of the
Tier-1b conjunction because it probes one constant's drift rather than the model
form.

## Payload: what the model varies, and what it deliberately does not

The cost of a call is driven by payload, so it is worth being explicit about which
payload dimensions the model reads and which it ignores.

**Read by the model.** Total send bytes per rank, per-peer bytes through the
saturating bandwidth term, hidden width, top-k, group count M, tokens per rank, and
element width. Element width is a parameter (`bytes_per_elem`, bf16 by default);
halving it for fp8 both halves the traffic and pushes the per-peer size toward the
unsaturated regime, and the model handles both effects, though no measurement in
this repository exercises a width other than bf16.

**Modelled separately, on purpose: routing skew** (`sim/imbalance.py`). A
variable-length all-to-all finishes when its busiest peer finishes, not its average
one. With the measured expert-load coefficient of variation of 0.136, the busiest
peer carries this much more than the mean:

| Collective | Peers | Inflation |
|---|---|---|
| one-hop | 128 | 1.42 |
| Hop B | 8 | 1.28 |
| Hop A | 16, aggregating q=3 experts per message | 1.19 |

**Skew favours two-hop**, and by a first-order amount: it shifts the ratio by +0.09
at hierarchy ratio 1.03 and by +0.40 at ratio 8. The mechanism is that the maximum
is taken over fewer peers on the two-hop side, and Hop A additionally aggregates q
expert loads into one message before the maximum is taken. This is the only modelled
effect that points the opposite way from the arrival chain.

It is kept out of `core.py` on purpose, and a test enforces that. The Tier-1 targets
come from a microbenchmark that divides its buffer equally across peers, so it is
balanced by construction; applying an inflation factor there would model an effect
the measurement does not contain, and the gate would reject it for the right reason.
Skew belongs to end-to-end prediction, which is Tier-2 territory and locked.

**Read as a per-call cost, not a payload cost: the launch path.** A call-count scan
([docs/09](09-phase-model.md), figure F16) measured one collective in two regimes
and found the fixed cost is 129 microseconds when the host runs ahead of the device
and 255 when it has to observe each call. The gap holds its share, 46 to 59% of the
total, across every payload from 64 KiB to 16 MB, so it is charged per call and
never as a fraction of the traffic. Two consequences for the numbers on this page:
the tabulated `alpha` is corroborated at the level by an independent benchmark and
belongs to the deep-queue regime, and the scan confirms that collectives do not
pipeline, which is the assumption `core.py` makes when it prices the two-hop chain
serially.

**Not modelled.** Metadata bytes: gate and slot values are packed alongside the
payload in the reference implementation, and the model counts only payload rows.
Capacity factors and token dropping are absent entirely. Both would need
measurements we do not have.

## The other half of the step: expert compute (`sim/compute.py`)

Communication is only one side of an MoE step. A measured bf16 GEMM roofline (256
samples per size, two independent campaigns) lets the repo say something about the
other side — and, more usefully, say precisely where the data stops.

| square GEMM | achieved TFLOPS | of peak |
|---|---|---|
| 1024 | 141.6 | **43%** |
| 2048 | 271.1 | 83% |
| 4096 | 327.4 | 100% |
| 8192 | 296.7 | 91% |
| 12288 | 291.4 | 89% |
| 16384 | 319.2 | 97% |

The curve is **not monotone** — it peaks at 4096 and dips through 8192–12288. That
is measured and reproduced, so it is kept rather than smoothed into a tidy fit.

**A reading of this curve that we withdrew.** This page used to say that at 1024 the
machine delivers 43% of its own peak, *therefore* narrow experts do not merely do
less work but do it less efficiently. Eight expert-FFN shapes were then measured
directly on two nodes, and they refute it: a GEMM whose smallest dimension is 1536
runs at **310 to 328 TFLOPS**, not at the 206 the square curve implies for a
dimension that size. The square-1024 dip is a **total-work** effect, not a shape
effect — a square 1024 matmul is 2.1 GFLOP and never fills the machine, while an
expert FFN of 1536 × 2048 × 5504 is 34.6 GFLOP and does. What survives is narrower
and still worth saying: efficiency falls when the total work *per call* is small,
which happens at high expert counts and small micro-batches, not merely because
d_expert is small.

**The bracket is closed.** The three indexing rules were scored against those eight
measurements:

| Rule | median abs error | bias | worst |
|---|---|---|---|
| smallest dimension | 19.0% | −19.0% | 37.1% |
| row count | 11.2% | −11.2% | 37.1% |
| **geometric mean** | **6.4%** | −6.4% | **10.9%** |

All three under-predict, because a square curve is the wrong prior for a large
non-square GEMM, but they are not equally wrong and the data now chooses. `sim/compute.py`
uses the geometric mean and reports a band from its measured worst case rather than
from an assumption spread. The residual −6.4% bias is left uncorrected: eight points
on one machine is not enough to fit a correction onto, and the direction is the safe
one, since predicting compute slightly slow understates communication's share, which
is exactly what `comm_share_upper_bound` is meant to bound from above.

Same-session square references came back at 328.7 (4096) and 296.6 (8192) against the
327.4 and 296.7 in the table, so the curve itself reproduced and the non-square shapes
can be read against it directly. The two nodes agree to 0.0–3.0% on every shape.

**Compute is never added to communication.** The two overlap by an amount nothing
on hand resolves — the same gap that keeps Tier-2 locked. `comm_share_upper_bound`
gives communication's share *if nothing overlapped*, which is a bound useful for
ranking geometries and for spotting regimes where communication cannot possibly
dominate. It is not a predicted share, and the module refuses to present it as one.

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
python -m sim.validate_sweep     # Tier-1b gate (cross-corpus, two machines, three corpora)
python -m sim.compute            # expert-FFN roofline, measured non-square shapes
python -m sim.imbalance         # routing skew: how much it favours two-hop
python -m sim.platforms          # calibrated platforms + where the methods pay off
python -m sim.profile            # is a given machine worth it, and which condition decides
python -m sim.phase              # phase spans; refuses step time until calibrated (docs/09)
python -m sim.validate           # Tier-2 gate (currently reports the failure, truthfully)
python -m sim.sweep              # extrapolation (checks the gates at entry)
python -m sim.overlap            # Tier-2 campaign: overlap model family battle report (docs/07)
python -m sim.uncertainty        # Monte Carlo uncertainty bands + breakeven (fixed seed)
python tools/gen_sim_figures.py  # regenerate F7-F12 (numbers computed live; only F12 embeds measurements)
python -m pytest tests/test_sim.py -q
```
