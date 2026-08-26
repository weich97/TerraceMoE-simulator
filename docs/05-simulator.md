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
   Three gate tiers, thresholds fixed ahead of the targets:
   - **Tier-1 (communication micro level)**: against directly measured one-hop/two-hop times on
     the same machine — median relative error ≤20%, max ≤35%, crossover position reproduced.
     **Current: pass** (2.8% / 24.5% / 4096–8192).
   - **Tier-1b (cross-corpus, `sim/validate_sweep.py`)**: the same model against 62 distilled
     targets from a *different* benchmark family on **two machines** — the calibrated one, and
     a second one whose constants are all re-fitted so only the model form is shared. Gate:
     median relative error ≤12%, at most 1 in 5 targets over 35%, median signed error within
     ±8%. **Current: pass** on all three corpora (machine A 1.9% median over 6 targets;
     machine B 9.3% over 44, bias −1.1%; machine A at world 16, collected 2026-08-26 after
     the constants were frozen, 11.3% over 12, bias −0.9%). The first two are a regression
     guard written after the calibration; the third is the one that had nothing fitted to it
     and clears the gate by 0.7 points, the thinnest margin of the three.
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
| PyTorch arrival chain (measured 0.0875 µs/row) | 0.40 | 0.88 | 1.55 | 2.12 |
| Fused kernel (est. 0.012) | 0.82 | 1.55 | 2.29 | 2.76 |
| Zero implementation overhead (upper bound) | 0.98 | 1.76 | 2.47 | 2.89 |

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
| PyTorch arrival chain (measured 0.0875 µs/row) | **3.87** |
| Fused kernel (est. 0.012) | **1.45** |
| Zero implementation overhead (upper bound) | 1.07 |

One-sentence takeaway: **the implementation tier pulls breakeven from 3.9 down to 1.5** — on
common hierarchical machines like NVLink/IB (≈3.2), whether hierarchical communication is worth
doing depends on how well your arrival chain is written, not on the topology.

### Uncertainty bands: is the conclusion stable against calibration error?

![uncertainty bands](assets/f9-uncertainty.svg)

Calibration constants are not ground truth; they are measurements with spread (measured machine
run-to-run drift ±9~20%). 400 Monte Carlo draws propagate that spread into the extrapolation
(provenance noted item by item, see `sim/uncertainty.py`; fixed seed, reruns match bit for bit).
Two robustness anchors:

- Flat column, most favorable case (zero implementation overhead): p95 = **1.04** — just above
  1. Read it precisely: once bandwidth saturation is modelled, the *byte account alone* on a
  flat fabric is close to neutral, so what actually makes two-hop lose there is the
  implementation overhead that this tier deliberately sets to zero. The measured verdict on the
  flat machine is unchanged (docs/03), and it is an implementation verdict, not a bytes verdict.
  Under the fused tier — the best any real implementation has reached — the flat column stays
  below 1;
- 8× column, least favorable case (PyTorch chain): p5 = **1.45 > 1** — on hierarchical machines,
  the direction "two-hop wins" is robust to calibration error.

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
tier largest** (3.87 → 1.07, Δ≈2.8) > scale axis (Δ≤1.9, and only the ≤128 part of it is
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

The result: **Tier-1 median error drops from 8.1% to 2.8%** (worst 12.5% → 24.5%, still
inside the 35% gate), and the entire bootstrap interval of x_half passes it (30 KiB → 2.5%,
87 KiB → 5.0%). Tier-1b then confirms the model on 62 targets from a different benchmark
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
preregistered window, so an x_half much above the fitted interval would fail the
gate. `tests/test_sim.py` holds the breakeven snapshot and the anchors so any
further drift shows up immediately.

**How much room x_half actually has, measured against the gates rather than the
fit.** The bootstrap interval [30, 87] KiB is the spread of a fit to one corpus.
Sweeping x_half and rerunning every gate gives a different and tighter number:
**all four gates pass only for x_half in [46, 76] KiB** (`X_HALF_ADMISSIBLE`, with
the endpoints pinned by a test). Both ends are set by corpora nothing was fitted to
— the machine A world-16 sweep collected after the freeze rules out the low end, and
machine B rules out the high end. The shipped 54 KiB was chosen before either check
and is left where it is, off the centre of that interval on purpose: recentring it
would be fitting a constant to the gates that are supposed to judge it.

That check also disposes of an attempt that did not work. Two dense 13-size sweeps
were taken on the calibrated machine specifically to re-fit x_half from its own data
and retire the borrow. They could not: single-run points in the mid range scatter
badly enough (one 8 MB point came in 2.7× slower than the 16 MB point beside it)
that the pinned-α fit lands at x_half above 1 MiB with β∞ at 152–168 GB/s, against
an asymptote of 103 GB/s directly visible at the largest size in the same sweep.
That is the α/x_half degeneracy reappearing on a noisy corpus, not a measurement,
and nothing from it is shipped. **x_half remains a borrowed shape.**

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
and found the fixed cost is 128 microseconds when the host runs ahead of the device
and 256 when it has to observe each call. The gap holds its share, 46 to 59% of the
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
Its headline for MoE: at 1024 the machine delivers 43% of its own peak, so narrow
experts do not merely do less work, they do it *less efficiently*, and a FLOP count
alone cannot see that.

**Compute time comes out as a bracket, and the bracket is the result.** The
roofline is square; an expert matmul is (rows × hidden × d_expert) with rows far
larger than d_expert, and we never measured a non-square GEMM. Three defensible
ways to index a square curve by a non-square shape — smallest dimension, geometric
mean, row count — agree to **1.19×** when d_expert = hidden, and diverge to
**2.25×** once d_expert ≤ hidden/2. So: for wide experts the compute number is
usable; for the fine-grained MoE regime it is assumption-dominated and must be
quoted as a range, or someone has to go measure tall-skinny GEMMs.

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
python -m sim.compute            # expert-FFN roofline and the bracket it implies
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
