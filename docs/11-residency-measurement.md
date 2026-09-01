# 11. Residency: what has to be measured before the memory verdict means anything

`sim/codesign.py` decides whether an architecture fits on an accelerator, and that
verdict sets the floor on expert parallelism, which in turn decides whether the
hierarchical dispatch question applies at all (a fast domain large enough to hold the
whole expert-parallel group withdraws it). So the memory model is load bearing, and
until this protocol is run, part of it is invented.

## What is arithmetic and what is not

An earlier version took a single `reserve_frac = 0.35`, meaning everything that is not
an expert weight. That number was made up, and it also conflated two kinds of quantity.
`codesign.residency` now splits them:

| Term | Status | Where it comes from |
|---|---|---|
| expert weights + optimizer state | **computed** | parameter count times bytes per parameter; exact |
| non-expert parameters + optimizer | **computed** when supplied | attention, embeddings, norms; currently omitted unless the caller passes them |
| activation residency | **assumed** | needs `activation_bytes_per_token_layer`, below |
| allocator, fragmentation, framework and communication buffers | **assumed** | needs `overhead_frac`, below |

Only the last two are inputs. A `MemoryProfile` whose `measured` flag is false marks
every result computed from it, so the assumption cannot be mistaken downstream for a
reading.

## The two numbers to measure

### A. `activation_bytes_per_token_layer`

Bytes resident per token per MoE layer at the peak of a step, under a stated recompute
policy. It is a per-token constant because the quantity scales with tokens per rank and
with layer count, and the constant is what does not.

Procedure, one node is enough:

1. Fix a geometry and record it: hidden width, expert width, top-k, `M`, sequence
   length, expert parallelism, and the recompute policy verbatim as configured. The
   constant is only valid under the policy it was measured with, so the policy is part
   of the reading, not context for it.
2. Run at micro-batch sizes 1, 2, 4 and 8, everything else held. At least three steps of
   steady state after warmup; take the peak allocator reading per step, not the average,
   and not the reserved figure.
3. Read peak allocated bytes at the same point in every step. Subtract the computed
   weight and optimizer residency, which `codesign.residency` reports separately, and
   subtract the overhead of B if it has already been measured.
4. Regress the remainder on tokens per rank. The slope divided by the MoE layer count is
   the constant. **Report the intercept too**: a large one means the split into
   per-token and fixed residency does not hold at this geometry, and the constant should
   not be used.
5. Repeat on a second node. Two nodes disagreeing by more than a few percent means the
   reading is not a machine property.

### B. `overhead_frac`

Allocator fragmentation plus framework and communication buffers, as a fraction of
capacity. This one cannot be derived at all.

1. On the same job, record reserved bytes against allocated bytes at the same point in
   each step. The gap is fragmentation and pool slack.
2. Record the resident bytes present before the first step, after the communication
   backend and the framework have initialised but before any activation exists. That is
   the fixed buffer cost.
3. Report the sum as a fraction of total capacity, and report it at more than one
   expert-parallelism degree, because the communication buffers scale with world size
   and the fixed part therefore is not fixed across the axis that matters most here.

## What the measurement changes

`min_ep_for_memory` returns the smallest expert parallelism whose residency fits.
That number decides:

- whether an architecture is runnable on a cluster at all,
- the expert-parallelism floor, which sets whether the group spans more than one fast
  domain and therefore whether the dispatch criterion returns a verdict or declines,
- the upper edge of the model-size band in `sim/envelope.py`.

None of those is trustworthy while A and B are guesses. The direction of the error is
not known either, which is why nothing in the repository currently reports a residency
verdict without the `measured: false` flag attached.

## Status

Not measured. Machines were not available when the model was written. Until then every
consumer receives `codesign.UNMEASURED`, whose values exist so the code runs and not so
anyone can quote them.
