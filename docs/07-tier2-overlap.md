# The Tier-2 campaign: overlap-aware synthesis models — a systematic negative result and a breakthrough protocol

**One sentence: six structurally distinct single-parameter overlap model families, all fitted
at the calibration point and evaluated at the holdout points, and none passes the gate —
"the phase ledger does not add up to the step ledger" is not one missing coefficient, it is one
missing kind of measurement. The protocol for that missing measurement is at the end of this
doc, ready for anyone with a cluster to run as-is.**

All figure numbers are produced live by `python -m sim.overlap`; the script is the source.

---

## 1. Full-family battle report: none lands inside the band

![overlap model families](assets/f7-overlap-families.svg)

| Family | Structure (Δ = predicted two-hop step delta) | Parameter (solved from flag) | Holdout MAE | within ±0.035 | negative-sign direction | Gate |
|---|---|---|---|---|---|---|
| M0 | naive: Δ = Δmodel (no overlap, current baseline) | — | 0.135 | 0/6 | 5/5 | fail |
| M1 | proportional exposure: Δ = φ·Δmodel | φ = −0.149 | 0.150 | 0/6 | 0/5 | fail |
| M2 | per-call hiding: Δ = Δmodel − h·calls | h = 2.47 ms | 0.062 | 4/6 | 3/5 | fail |
| M3 | hidden fixed cost: Δ = Δmodel − φ·fixed | φ = 1.12 | 0.136 | 0/6 | 0/5 | fail |
| M4 | hiding ∝ compute: Δ = Δmodel − c·T_comp | c = 0.302 | **0.048** | 2/6 | **5/5** | fail |
| M5 | exposure ∝ pipeline pressure: Δ = Δmodel·(1−λ/mbs) | λ = 1.149 | 0.087 | 2/6 | 2/5 | fail |

Gate = holdout MAE ≤ 0.025 AND ≥4/6 within ±0.035 AND every negative-sign point directionally correct — threshold numbers carried over from the Tier-2 preregistration; the denominator honestly counts the 6 holdout points that exist (n4 is a scale-axis point added after preregistration). Negative-sign points = the 4 preregistered ones (tok2x/tok4x/k8m4/n8) + n4; k8m2 (G=0.9935, within n=1 noise of 1) is excluded from the direction count.
Fitting used only the calibration point flag; the six holdout points never participated — the failures in the table are real failures, not overfitting failures.

Two reading notes:

- **The closest in magnitude (M2, MAE 0.062) flips sign on the scale axis**: it predicts
  two-hop winning at n8/n4, while measurement shows two-hop losing badly (G=0.90/0.86). The
  picture of hiding a fixed number of milliseconds per call over-hides once the call count doubles.
- **The one with perfect direction (M4, 5/5) misses the magnitude gate by 2×**: the picture of
  hidden time proportional to compute time captures the structure, but a single global
  coefficient cannot balance the tension between "flag: two-hop actually wins back 151 ms" and
  "n4: two-hop loses 1833 ms".

## 2. Why it stops here: the degrees-of-freedom ledger

The reconnaissance table (the first output block of `python -m sim.overlap`) puts the
contradiction at its sharpest:

| Geometry | mbs | measured G | Δ measured (ms) | Δ model (ms) | of which fixed cost | implied exposure ratio |
|---|---|---|---|---|---|---|
| flag | 1 | 1.0355 | **−152** | +1018 | 1000 | **−0.15** |
| tok2x | 2 | 0.8845 | +437 | +1033 | 990 | +0.42 |
| tok4x | 4 | 0.8234 | +642 | +1041 | 985 | +0.62 |
| k8m2 | 1 | 0.9935 | +31 | +1330 | 1327 | +0.02 |
| k8m4 | 1 | 0.9335 | +332 | +1411 | 1327 | +0.24 |
| n8 | 1 | 0.9027 | +718 | +2083 | 2001 | +0.34 |
| n4 | 1 | 0.8647 | +1833 | +4151 | 4002 | +0.44 |

- The communication model's step delta is **almost entirely the fixed cost of "arrival chain +
  splits sync"** (flag: 1000 of the 1018); on a bandwidth-flat machine the two arms' pure
  communication difference is near zero — so step-level prediction stakes everything on
  "how much of the fixed cost gets hidden by overlap".
- The implied exposure ratio runs from −0.15 to +0.62 — **not even the sign is consistent**:
  exposure is not one global constant, it is a function of geometry — and which geometric
  quantity it follows, and how, is something seven points cannot resolve (the mbs axis says it
  rises with pipeline pressure, the scale axis says it rises with call count, and the flag point
  says there is also a negative component).
- **One calibration point can pin down exactly one parameter.** Two-parameter families are
  underdetermined under this split; letting holdout points into the fit would cancel the
  validation. So serial enumeration of single-parameter families is all this road offers —
  enumeration complete, all dead, the negative result stands.

Honesty clause: evaluating six families serially against the same holdout set is itself model
selection — even if some family had passed the gate in retrospect, that would not constitute an
unlock (`tests/test_sim.py` pins this). **The Tier-2 gate stays red; step-level extrapolation
stays locked.**

## 3. The breakthrough protocol: measure "exposed time", not "phase span"

The contradiction is rooted in the measurement convention: event timing gives each
communication phase's **span**, but says nothing about "what runs in parallel with what". When
the two arms' overlap structure differs across the dual streams, the sum of phase spans and the
step-time difference can differ by several fold (we measured ~5×). The cure is to measure
**exposed time** directly. The protocol, for anyone with a cluster:

1. **Step ledger**: run both arms in identical configs for ≥10 steady-state steps (drop the
   first 300 steps), record the median step time.
2. **Phase ledger (the key difference)**: within the same run, use a profiler with **stream
   attribution** to capture ≥3 full steps: each communication op's [start, end] interval + the
   stream it runs on + the compute stream's busy/idle timeline.
3. **Compute each communication op's exposed time** = op interval − (op interval ∩
   compute-stream busy intervals). Accumulate separately by op type (dispatch / combine /
   splits sync / local chain).
4. **Self-consistency check (a prerequisite for any modeling)**: the two arms' Σexposed
   difference and their step-time difference should agree within ±10%. Disagreement means a
   third component remains (host blocking, stream-scheduling gaps) — go back to the timeline
   and find it.
5. **Model and validate**: rebuild the synthesis model with "per-op-type exposure ratio" as the
   first-class quantity; calibrate on one geometry, validate on **freshly collected** holdout
   geometries (≥4 of them, covering the token/load/scale axes) against the same Tier-2 gate.
   The old seven points serve as retrospective reference only, never as grounds for an unlock.

Estimated cost (for scheduling): two arms × (10 timed steps + 3 profiled steps) × ~5
geometries; calibration and the self-consistency check can be completed on a single machine,
while the holdout points need a machine allocation at target scale.

## 4. Relation to the other docs

- Communication-level extrapolation (Tier-1, unlocked) is unaffected by this doc — the
  bandwidth ledger is validated separately at the micro level (docs/05, median error 8.1%).
- This doc's negative result is confined to "step-level synthesis": do **not** treat the
  communication-level ratios of docs/05 as end-to-end speedups — that is exactly M0's mistake
  (MAE 0.135).
