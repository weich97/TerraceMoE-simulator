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

## The launch measurement: run, and what it found

This section used to describe a measurement worth taking. It was taken on
2026-08-26, on two nodes of platform A, and this is the result.

**What was run.** N back-to-back tiny collectives at fixed world with no
synchronisation between them, one sync at the end, N from 1 to 256, payloads 256 B
to 16 KiB, all-to-all and all-reduce, seven repetitions each, at world 8 and world
16. Then the whole scan again in a second mode where the host waits for each call to
land before issuing the next. World 8 was run on each of the two nodes separately,
which gives the constant an independent replicate rather than a single reading, and
once more with N pushed to 1024 to check that the plateau is a plateau. Every number
below is the median over repetitions of the slowest rank.

**Per-call cost, all-to-all, microseconds:**

| | world 8 | world 16 |
|---|---|---|
| deep queue, host runs ahead | 129  (124 and 134 over two nodes) | 134 |
| host observes each call | 255  (247 and 263) | 283 |
| gap | 126 | 149 |
| of which host CPU spent enqueuing | 58 | 76 |

Four things fall out.

**Collectives do not pipeline.** Per-call cost stops falling at N around 16 and is
flat from there to N = 1024, a sixty-fourfold range. Two back-to-back collectives cost
twice one. `sim/core.py` prices the two-hop chain serially on the strength of an
implementation note; that is now measured rather than asserted, which matters
because it is the assumption the whole two-hop comparison rests on.

**The cost is a fixed cost, not a payload cost.** 256 B and 16 KiB give the same
per-call number to within the run-to-run spread, as they have to: the wire term at
16 KiB is 0.12 microseconds.

**Its shape across worlds is not the tabulated shape.** The scan was repeated at
worlds 2 and 4 so the instrument could be checked where nothing is in dispute:

| world | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| measured, microseconds | 87 | 98 | 129 | 134 |
| tabulated `alpha` | 90 | 97 | 111 | 157 |

It reproduces worlds 2 and 4 to within 3% and then departs at 8 and 16, in opposite
directions. The measured curve climbs while a node fills, 87 to 129 across worlds 2
to 8, and then barely moves crossing to a second node, 129 to 134. That is what a
bandwidth-flat supernode ought to do, and it is corroborated independently: the
largest-payload asymptote today is 103 GB/s at world 8 and 100.7 at world 16, a
ratio of 0.978 against the 0.974 the calibration ships for exactly that flatness.
The tabulated 111 to 157 says something different.

**Nothing is changed on the strength of it**, and the reason is worth stating.
Setting `alpha(8)` to the measured 129 makes every validation corpus pass, including
the world-8 one that currently fails. Setting `alpha(16)` to the measured 134 — the
same scan, the same day, the same convention — breaks the world-16 corpus, from 8.0%
to 15.5%. Taking the half of a measurement that greens a gate while discarding the
half that reddens another is choosing by outcome, so neither half is taken. What
this leaves is an honest disagreement between two benchmarks of the same machine,
recorded in `sim/validate_sweep.py` as a corpus that ships red.

**Half the per-call cost is the host, and it is hidden only while the queue is
deep.** Submission costs 44 to 74 microseconds of host CPU. When the host runs
ahead that disappears under device execution. When something forces the host to
observe a call before issuing the next, per-call cost roughly doubles, and the extra
130 microseconds splits about evenly between submission and completion detection.
The two nodes disagree on the submission half by 44 against 68 microseconds, so
treat that split as indicative and the total gap as the measured quantity.

**And you cannot outgrow it by sending more.** The scan was repeated at world 16
over payloads from 64 KiB to 16 MB, which spans everything an MoE dispatch actually
sends. The gap between the two regimes does not close:

| payload per rank | deep queue | host exposed | gap | gap as share |
|---|---|---|---|---|
| 64 KiB | 160 | 297 | 137 | 46% |
| 256 KiB | 150 | 302 | 153 | 51% |
| 1 MB | 134 | 283 | 149 | 53% |
| 4 MB | 123 | 301 | 178 | 59% |
| 16 MB | 201 | 429 | 228 | 53% |

Microseconds per call. Repeated at world 8 on the other node the gap runs 90 to 228
microseconds and 39 to 59% of the total, so this is not a property of one world or
one node. Host exposure is a per-call cost, so it grows with the number of
collectives and not with what they carry; at every payload measured it is about half
the total. That is why it enters the comparison as one extra call and not as a
fraction of the traffic.

**The one host observation an MoE dispatch cannot avoid is the cheap one.** A
variable-length all-to-all needs its per-peer counts on the host before it can be
issued, and `core.py` carries 44 microseconds for that readback. Running the same
two-regime scan on a readback puts it at 78 microseconds with the queue deep and 123
when the host is already serialized, against 132 and 238 for an all-to-all. Two
readings, and only the second is a result. The absolute 78 is not comparable to the
shipped 44, because this op bundles a small device reduce with the readback so that
there is something to wait for, and the shipped number prices the copy alone. What
is comparable is the *penalty for being exposed*: 45 microseconds for the readback
against 106 for a collective. The splits sync is the least of the host observations
in the chain, which is the assumption behind treating it as a small additive
constant.

### What it changes

Not much, and that is the useful part. Charging two-hop one extra host exposure of
130 microseconds moves the breakeven hierarchy ratio on the reference geometry from
3.98 to 4.15. Remove the arrival chain and the same 130 microseconds moves it from
1.10 to 1.27. So the ordering is confirmed rather than disturbed: **the arrival
chain is worth more than the launch path, and fusing it comes first.**
`sim/profile.py::launch_sensitivity` carries the measured point in its sweep.

### What it does not settle

Which regime a real training step is in. Two collectives chained on one stream stay
in the deep-queue regime, because the device resolves the dependency without the
host. But an MoE dispatch is not two collectives on one stream: the variable-length
exchange requires a splits readback, which is a host observation by construction,
and `core.py` already carries 44 microseconds for it. Whether that readback is the
only host observation per layer, or whether framework-side control flow adds more,
is a timeline question, and the timeline protocol above is what answers it.

It also does not settle whether to adopt a device-initiated stack. The measured
device-side put floor of 36 to 57 microseconds against the 130 of host exposure says
the ceiling on that change is real. But removing the host from the critical path
also changes overlap, and overlap is exactly the gap this page exists to close.

## What it would unlock

Tier-2 currently blocks all step-level extrapolation, which is why the ratio-sensitivity
map in `sim/platforms.py` is stated only in communication-call terms. Even with a calibrated
phase model, a target-platform statement would still require target-local primitive
measurements and validation rather than a nominal link-rate label.
