# 12. The group cap: a known system gain against an unmeasured quality cost

## The gap

`M`, the cap on how many groups a token's experts may occupy, is the one architecture
knob that moves the dispatch verdict without touching parameter count, active parameters
per token, or anything else the model designer is usually optimising. It enters through
the deduplication quota `q = k/M`: tightening `M` at fixed `k` raises `q`, and the
slow-side saving is `(q-1)/q`.

The system side is computed. On the synthetic DGX H100 with NDR400 that
`sim.codesign.synthetic_dgx_h100` constructs (8 cards per node, 16 nodes, 450 against
50 GB/s, the measured alpha curve, `x_half` 54 KiB and the arrival chain priced at this
hidden width by `sim/machine.py`, all borrowed constants labeled in the constructor) at
hidden 7168, expert width 2048, 256 experts, `k = 8`, one MoE layer, 4096 tokens per
rank:

| M | q = k/M | G, PyTorch chain | G, fused kernel |
|---|---|---|---|
| 4 | 2 | 1.362 | 1.649 |
| 2 | 4 | 1.960 | 2.614 |
| 1 | 8 | 2.510 | 3.694 |

Reproduce with `python -m sim.codesign`; a test pins every cell. These are
dispatch-call ratios, not step times: the step-level gate fails. Every cap wins on
both chains, and tightening `M` is worth 1.36 to 2.51 on the PyTorch operator chain
and 1.65 to 3.69 fused. The chain still moves the verdict more than the cap does, but
by less than it appeared to: correcting its cost narrowed the gap between the two
columns from 1.6-2.1x to 1.2-1.5x.

> **Correction (2026-09-08).** The table read 0.948 / 1.204 / 1.391 and 1.490 / 2.234 /
> 2.979 until the arrival-chain constant was found to be over-charged by 2.8x and the
> row-gather bandwidth under-stated by 3x, both from the same denominator error
> ([sim/chain_remeasured.py](../sim/chain_remeasured.py)). The PyTorch-chain column
> rises 44 to 80 percent and the fused column 11 to 24 percent. The direction is again
> unchanged, but `M = 4` is now a **win** at 1.36 rather than the loss the 2026-09-01
> correction had just introduced -- so the sentence "M = 4 is a loss even at ratio 9"
> stood for exactly one week and was never right.

> **Correction (2026-09-01).** An earlier version of this table read 1.268 / 1.565 /
> 1.845 and 1.835 / 2.529 / 3.353, with `x_half` stated as 46 KiB. Those numbers were
> produced by a construction that was not recorded and cannot be reproduced from the
> shipped code; back-solving them yields a different arrival-chain cost per row at each
> `M`, which no single construction produces. The table above is computed by the
> constructor named beside it, and the numbers are pinned by `tests/test_codesign.py`
> so they cannot drift again. The stale table overstated the PyTorch-chain column by
> 30 to 34 percent and the fused column by 13 to 23 percent; the direction is
> unchanged, but the corrected PyTorch-chain column now puts `M = 4` below 1, which
> the stale table did not.

The quality side is measured at exactly one point. Corpus Q1 ran `E = 128` in 8 groups
of 16 with `k = 8` and `M = 4`, and found the conjunction costs `+0.0034` nats, 3.4% of
the preregistered no-loss margin. **`M = 2` and `M = 1` have never been measured.**

So the repository can say what tightening `M` buys and cannot say what it costs. Every
statement it makes about the tighter settings is a system-side statement only, and
`docs/06` and the paper's Section on the routing constraint both say so. This document
is the experiment that would close it.

## Design

Preregistered before any arm is run. Nothing below is to be changed after data is seen;
if it is, the change is logged as an amendment with its date and reason, in the manner
of the reporting policy in the paper.

**Arms.** `M in {1, 2, 4}` at `k = 8` held, each with the equal-quota constraint inside
the chosen groups, that is the full conjunction at each cap. Plus the unconstrained
top-k control. The `M = 4` arm and the control are Q1's, reused rather than rerun, which
is only legitimate if the geometry, seeds, token count, tokenizer and data order match
Q1 exactly; if any of them cannot be matched, `M = 4` is rerun too and Q1's figure is
reported beside it rather than merged with it.

**Geometry.** Q1's, unchanged: 13.14B parameters with 1.33B active, `E = 128` in 8
groups of 16. Changing the geometry and the cap together would confound them.

**Seeds and tokens.** Four seeds per arm and 62.9B tokens per arm, matching Q1, because
the deltas are paired by seed against the same control and pairing requires the same
seeds. Four is above this project's usual two-to-three because the pairing constraint
dictates it, not because more resolution is wanted.

**Endpoint.** Holdout validation loss on the same shard and probe as Q1, paired by seed,
read at 20k and 30k steps. The instrument is fixed to Q1's holdout probe in advance
precisely because Q1's own change of instrument happened after data had been seen.

**Margin.** The preregistered no-loss margin of 0.1 nats, unchanged.

**Secondary endpoints.** Downstream accuracy under two one-sided tests at plus or minus
1.0 percentage points, expert load entropy no lower than the control, step time
unchanged. Same as Q1, and same caveat: the load and step endpoints carry no margin, so
clearing them establishes less than the quality endpoint does.

## Decision rule, stated in advance

The point of measuring is to choose a cap, so the choice is written down before the
data exists.

Adopt the tightest `M` that satisfies both:

1. its quality delta against the unconstrained control is below 0.02 nats, a fifth of
   the preregistered margin and the line the quality figure already draws; and
2. its 90% interval excludes the interval of the next looser cap, so the ordering
   between caps is resolved rather than assumed.

If no cap tighter than 4 satisfies both, `M = 4` stands and the system gains in the
table above are recorded as unavailable at this geometry rather than as forgone.

If a tighter cap satisfies the first condition but not the second, report both and adopt
nothing: an unresolved ordering between two caps is not a reason to take the tighter one
just because it is cheaper on the transport.

## What would falsify the premise

The premise is that `M` is nearly free on quality, as it was at `M = 4`. It is refuted
if the quality delta grows faster than linearly in `1/M`, which would mean the cost is
in the group restriction itself rather than in how tight it is. Q1 already carries a
hint in that direction: at `M = 4` the group limit alone costs `+0.00276` while the
conjunction costs `+0.00339`, so most of the cost is already present at the loosest cap
that constrains anything.

## Status

Not run. No machines allocated. The system-side numbers above are computable today and
are reported as such; the quality side is blank and is not to be filled by analogy with
`M = 4`.
