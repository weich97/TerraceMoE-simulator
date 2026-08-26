# T-Route: the routing constraint, and what four modes cost

T-A2A cannot be expressed without a bound on how many groups a token reaches. This
page is that bound: what it is, how it differs from the two constraints it is built
from, and what each of the four resulting modes costs, measured.

The short version: **T-Route is the conjunction of two constraints, and it costs
38% of what one of them costs alone.** The combination is cheaper than one of its
parts, which is the result worth understanding.

## The four modes

![What each routing mode constrains](assets/f14-routing-modes.svg)

Standard top-k routing picks the `k` highest-scoring experts anywhere in the pool.
Two independent constraints can be layered on top:

**Group limit.** Confine a token's experts to `M` of the `N_g` groups, where a group
is a set of ranks sharing a high-bandwidth domain. This bounds cross-group fan-out
at `M`, which is what makes a two-hop dispatch expressible: without it a token can
reach every group and there is nothing to deduplicate.

**Equal quota.** Select exactly `k/M` experts inside each chosen group. This makes
every cross-group message the same size, which removes the variable-length exchange
that would otherwise have to precede the collective.

| Mode | Constraint | Cross-group fan-out | Message size | Known as |
|---|---|---|---|---|
| unconstrained top-k | none | up to min(k, N_g) | data dependent | the baseline |
| group-limited | group limit only | at most M | data dependent | DeepSeek-V2 device-limited, V3 node-limited |
| equal quota only | quota only, M = N_g | all N_g groups | constant | MoGE (Pangu) |
| **T-Route** | both, with M < N_g | **exactly M** | **constant** | this work |

The last row is the point. Neither existing constraint alone gives a compile-time
traffic envelope: the group limit bounds the fan-out but leaves sizes data
dependent, and the quota fixes sizes but sends to every group. Together, and only
together, both the count and the size of cross-group messages are known before the
first token is routed.

## What the modes cost, measured

![Measured cost of each mode](assets/f15-routing-costs.svg)

Testbed: 13.14B total parameters, 1.33B active, E = 128 experts in 8 groups of 16,
k = 8, M = 4. Four routing modes, four seeds each, 62.9B tokens per arm. All four
modes switch inside a single function, so one code path serves every arm, which is a
precondition for the comparison to mean anything. Thresholds were preregistered
before the runs.

| Mode | delta val loss | 90% CI | delta load entropy | delta load CV |
|---|---|---|---|---|
| group-limited | +0.00276 | [+0.00223, +0.00329] | +0.00018 | -0.0061 |
| equal quota only | +0.00895 | [+0.00726, +0.01064] | +0.00035 | -0.0137 |
| **T-Route** | **+0.00339** | [+0.00224, +0.00455] | +0.00026 | -0.0094 |

Deltas are paired by seed against unconstrained top-k on a holdout probe. Three
things to read off the table.

**The cost is small and it is resolved, not merely undetectable.** T-Route costs
+0.0034 nats, 3.4% of the preregistered no-loss threshold of 0.1. All three modes
have confidence intervals excluding zero, and all 24 paired seed deltas share the
sign (3 modes, 4 seeds, 2 readpoints). The within-arm standard deviation of the
holdout probe is 0.0008 to 0.0023 nats, an order of magnitude below the seed noise
of the training-log metric an earlier version of this analysis used. The effect was
measured first, then shown to be small.

**T-Route costs less than the quota alone, at 38% of it.** That is not an averaging
artifact. Requiring an equal quota in *every* group is a hard constraint: the router
must place experts in all `N_g` groups whether or not the token's affinities point
there. The group limit gives the quota a pressure-relief valve, because the choice
of *which* M groups remains free. Constraining more, in the right direction, costs
less.

**No hidden capacity loss.** A constraint that improved loss by degrading load
balance would be trading one cost for another. Expert-level load entropy under every
constrained mode is at or above the unconstrained control, and load CV is at or
below it, so the quality figure is the whole cost and not half of it.

## The other two axes

**Downstream, equivalent under preregistered TOST at +-1.0 percentage points.**
HellaSwag acc_norm delta +0.158 pp with 90% CI [-0.095, +0.411]; LAMBADA accuracy
delta -0.084 pp with CI [-0.406, +0.238], pooled over three readpoints.

That axis also carries a methodological caution worth repeating. LAMBADA read at a
single checkpoint gave delta -0.577 pp with a confidence interval touching the
equivalence boundary, which would have been reported as a measurable loss. Adding
two more readpoints showed each mode's estimate swinging 1 to 2 pp and changing sign
between readpoints, a swing larger than any single estimate. We therefore make only
an equivalence claim on that axis and do not quote a point cost.

**Step time, unchanged.** Both arms use the identical baseline one-hop all-to-all
and differ only in routing, so this isolates the constraint from the dispatch it
enables. G = t(unconstrained) / t(T-Route) measured 0.9882 and 1.0070 over two runs,
mean 0.9976, inside the +-1 to 3% run-to-run spread of same-shape jobs. The routing
constraint has no measurable step-time effect of its own.

Note what this also says: the load entropy advantage visible in the forerunner
testbed does **not** convert into step time. A proxy metric and a delivery metric
have to be measured separately, and here they disagree.

## Why adopt it separately from T-A2A

T-A2A pays off only on machines above a hierarchy ratio threshold (see
[docs/03](03-applicability.md) and the platform map in `sim/platforms.py`). T-Route
is not conditional in the same way. It buys a compile-time traffic envelope, each
token's cross-group fan-out bounded by M and every cross-group message a constant
size, at no measurable cost on any of the four axes above. That is useful for
capacity planning and for removing a variable-length exchange, on any machine.

Since it costs nothing measurable, adopting it is not a trade. It is also what makes
the two-hop dispatch available later, if and when the machine justifies it.

## Boundaries, unsoftened

- Cross-group balance is a **statistical property, not an architectural guarantee**.
  With M < N_g the choice of which M groups is data dependent, and under adversarial
  input the group-level CV can reach 1.0. `tests/test_routing.py` carries a
  reproducible counterexample.
- Validated at the scale and geometry above. The 3.4% figure is not a ratio to be
  extrapolated linearly to larger widths or more extreme M/N_g.
- The MoGE comparison mode requires k >= N_g and N_g dividing k, which holds only in
  this geometry. That cell cannot be tabulated against other geometries.
- The reference implementation is in `terrace/routing.py`, where all four modes are
  selected by a `mode` argument inside one function.

## Full ablation record

[docs/06](06-troute-results.md) carries the complete results: both readpoints with
confidence intervals, the per-seed scatter, the downstream TOST detail, the load
axis, the step-time runs, the small-scale forerunner 2x2, and the retraction record
for three results that did not survive scrutiny.
