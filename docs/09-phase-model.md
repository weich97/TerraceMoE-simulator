# The phase model, and the one measurement that would calibrate it

`sim/core.py` prices a single collective call and is validated by two gates.
`sim/phase.py` prices a whole training step by composing phases — dispatch, expert
compute, combine — and ships **deliberately uncalibrated**. Its structure is fixed,
its free parameters are named, and it raises rather than returning a plausible
number. This page is the measurement that would turn it green.

## Why it is not just "add the phases up"

Per MoE layer, per microbatch, the forward chain is a strict dependency:

```
router → [splits sync] → Hop A a2a → arrival chain → Hop B a2a
       → expert FFN
       → combine a2a → chain
```

Nothing inside that chain overlaps anything else inside it. Overlap comes from
*other* work in flight — other microbatches, other layers, the attention and dense
parts. So the exposed cost of a communication phase is

```
exposed = span − min(span, concurrent_compute_available × overlap_efficiency)
```

The measured data says this is the right shape. Implied exposure ratios track
**microbatch count**, not geometry size:

| micro-batches | 1 | 2 | 4 |
|---|---|---|---|
| implied exposure ratio | −0.15 | +0.42 | +0.62 |

With one microbatch there is little else in flight to hide behind; with four there
is a queue of independent work. That is why the six single-parameter overlap
families in [docs/07](07-tier2-overlap.md) all failed: they modelled exposure as a
property of the model when the data says it is a property of the **schedule**.

## What is missing

| Parameter | What it is | Why we do not have it |
|---|---|---|
| `overlap_efficiency` | how much theoretically-concurrent compute actually hides communication | a property of stream assignment and launch order in the framework build; never measured |
| `non_moe_compute_ms` | attention + dense + optimizer per step | derivable from step time minus everything else, but only once phases are separable |
| `chain_on_compute_stream` | whether the arrival chain competes with expert GEMMs or interleaves | unknown; it changes the sign of some predictions |

All three fall out of one measurement.

## The protocol

**Measure exposure time, not phase spans.** Event-timed spans are what we already
have, and they are exactly what does not add up — the sum of phase spans differs
from the step-time delta by ~5×, because spans overlap and a span says nothing
about what ran beside it.

1. **Step account.** Both arms, same config, ≥10 steady-state steps each (discard
   the first 300). Record the median step time. This is the number the model must
   reproduce, and it must be collected in the same run as step 2 so no machine
   drift sits between them.

2. **Timeline, with stream attribution.** In the same run, profile ≥3 complete
   steps per arm. For every op record: `[start, end]`, the **stream it ran on**,
   and the op type. A profiler that reports only durations is not sufficient —
   stream attribution is the entire point.

3. **Exposure per op.**
   ```
   exposed(op) = |[start, end]| − |[start, end] ∩ compute-stream-busy|
   ```
   Accumulate by op class: dispatch a2a, combine a2a, splits sync, arrival chain,
   expert GEMM.

4. **Self-consistency check — run this before fitting anything.** The two arms'
   difference in Σexposed must reconcile with their measured step-time difference
   to within ±10%. If it does not, a third component is unaccounted for (host
   blocking, scheduler gaps, launch queues) and the timeline has to be re-read.
   **A fit performed before this check passes is decoration**; the previous attempt
   failed precisely because phase accounting was trusted without it.

5. **Fit, then validate on points that were not fitted.** Calibrate the three
   parameters on one geometry. Validate on **newly collected** geometries — at
   least 4, spanning the token, load and scale axes — against the same gate as
   Tier-2 (MAE ≤ 0.025, ≥4/6 within ±0.035, sign correct on the negative points).
   The seven existing points are retrospective reference only; they do not unlock
   anything, and `tests/test_sim.py` pins that.

## Minimum viable run

One geometry, two arms, **4 nodes**:

| Item | Cost |
|---|---|
| 2 arms × (10 timed steps + 3 profiled steps) | minutes of compute |
| profiler overhead | inflates absolute times; only the *ratio* of exposure to step delta is used, and both arms carry the same overhead |
| node-hours | a few, dominated by startup |

Choose that geometry because **its answer is already known**: two-hop measures
1833 ms slower per step there. If the exposure account reproduces that delta, the
method works, and no amount of retuning can manufacture the agreement after the
fact.

Everything beyond the minimum buys holdouts, not mechanism: 4–6 more geometries
across the three axes, same protocol, to satisfy step 5.

## A second, much cheaper measurement: separating launch from fabric

The shipped `alpha(world)` lumps the per-call launch cost together with the
collective's own fixed cost, because nothing here separates them. Splitting them
matters in principle, since two-hop issues one more collective than one-hop and
launch is per call rather than per world, and because a device-initiated stack has a
very different launch cost: measured here at 36 to 57 microseconds for a kernel-side
put against 257 to 268 for a host-side point-to-point.

**The scan.** One node, eight ranks, minutes. Issue N back-to-back tiny collectives
at fixed world without synchronising between them, for N over a decade, and fit time
against N. The slope is the per-call cost that cannot be pipelined away, the
intercept is what a single call costs beyond it. Repeat at world 8 and 16 to check
that the per-call term really is world-independent, which is the assumption the split
rests on.

**Do it only when a node is free anyway.** The sensitivity is already bounded without
it: launch cannot exceed `alpha(8) = 111` microseconds, since the smallest measured
alpha contains both terms, and sweeping the whole plausible range moves the breakeven
hierarchy ratio from 1.45 to 1.78, reaching only 1.60 at that bound. Against the
arrival chain, which moves the same quantity from 3.87 to 1.45, this is second order.
`sim/profile.py::launch_sensitivity` reproduces the sweep.

**What the scan would not settle.** Whether to adopt a device-initiated dispatch
stack. That removes the host from the critical path and therefore changes overlap,
which is exactly the gap this page exists to close. The launch scan tightens a small
known term; the overlap protocol above is what answers the interesting question.

## What it would unlock

Tier-2 currently blocks all step-level extrapolation, which is why the platform
map in `sim/platforms.py` is stated in communication-level terms. With the phase
model calibrated, the same map could be given in end-to-end terms — "this platform
class gains X% per step" rather than "communication is X× cheaper" — which is the
form anyone deciding whether to adopt actually needs.
