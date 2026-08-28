<img src="docs/assets/logo.svg" alt="TerraceMoE Simulator logo" width="84" align="right">

# TerraceMoE Simulator: a measurement-calibrated cost model for hierarchical MoE communication

**What this is:** a communication-call cost model for screening hierarchical MoE
all-to-all after calibrating it on the target machine. It is not a training-throughput
predictor: the step-level gate fails. The repository also contains **T-Route**
(hierarchy-aligned routing constraints) and a reference **T-A2A** two-hop path.

![Where hierarchical dispatch pays off](docs/assets/f13-platform-map.svg)

The figure is a ratio-only sensitivity map. Run `python -m sim.platforms` to reproduce
the scenario table; do not read the synthetic rows as platform predictions.

| Scenario | fast/slow ratio | measured-chain G | hypothetical-fused G |
|---|---:|---:|---:|
| Platform A, measured constants | 1.03 | 0.37 | 0.80 |
| Synthetic sensitivity | 2.0 | 0.63 | 1.18 |
| Synthetic sensitivity | 4.0 | 1.00 | 1.67 |
| Synthetic sensitivity | 9.0 | 1.60 | 2.28 |
| Synthetic sensitivity | 18.0 | 2.14 | 2.69 |

(Only the first row uses a measured ratio. The 0.012 microsecond/row fused tier is a
design target, not a measured fused kernel.)

`python -m sim.profile` runs the target-machine checklist. A ratio alone does not
decide it; five decision conditions and one arrival-chain advisory identify what must
be measured or what precondition fails:

| Condition | Why it can sink an otherwise good machine |
|---|---|
| q = k/M ≥ 2 | at q=1 each token already sends one row per group, so there is nothing to deduplicate |
| ratio ≥ **effective** breakeven | the byte account needs 1.31 at R=8, q=3; the corrected reference thresholds are 3.98 for the measured PyTorch chain and 1.49 for the hypothetical fused target |
| EP spans >1 fast domain | if every expert fits inside one NVLink/HCCS domain there is no slow hop to save, so keep EP in the domain instead. Rack-scale domains make this common |
| messages bandwidth-bound | below the half-performance size, byte savings do not convert into time |
| α(N_g)+α(R) < α(EP) | two hops pay two fixed costs, so on a machine where α barely grows with world size the swap loses before any byte moves. **A machine property, must be measured** |
| arrival chain cost | advisory: already priced above, but it is the one item fully under your control, and fusing it moves the threshold further than most topology changes |

The checklist and the cost model can never disagree: a machine it clears is one the model scores above 1, and a test pins that.

**T-Route can be evaluated separately from T-A2A.** It bounds each token's
cross-group fan-out and fixes the per-selected-group quota; aggregate per-peer sizes
remain data-dependent. The measured validation-loss effect is small but nonzero
(+0.0034 nats). The recorded downstream analysis reports equivalence, but the released
artifact does not contain all per-seed/checkpoint inputs and estimator details needed
to reconstruct it independently. Load and step-time measurements are thinner; see
[docs/10](docs/10-troute.md) and [docs/06](docs/06-troute-results.md).

> **Calibration coverage:** only platform A has a measured hierarchy ratio, 1.03.
> Platform B's corpus does not separate fast and slow levels; it is a same-corpus
> fit/consistency check after machine-specific refitting, not an out-of-sample transfer
> test or a second ratio sample. C4 on platform A is the post-freeze holdout. All
> ratios above 1.03 in the map are synthetic sensitivities.

---

## Methods

**T-Route** adds two constraints on top of standard top-k routing:

1. **Group-limited**: each token's experts land in only M groups (group boundaries = communication hierarchy boundaries, M < N_g);
2. **Equal quota within groups**: exactly k/M experts in each selected group.

The first caps each token's cross-group fan-out at M; the second fixes how many expert
rows that token contributes to each selected group. These are per-token architectural
properties. Aggregate per-peer messages are data-dependent sums, so counts exchange,
padding, or capacity bounds may still be required. Relation to existing work:
DeepSeek-V3's node-limited routing has only the group limit; MoGE's equal quota is
equivalent to M = N_g. T-Route is the conjunction, with M < N_g.

**T-A2A** replaces the one-hop all-to-all in dispatch with two hops: across groups, send only 1 copy to a representative card of the target group (Hop A, deduplicated); within the group, scatter to the actual expert cards (Hop B). Per-token payload over the slow links drops from q = k/M rows to 1 row, at the cost of moving an extra q(1−1/R) rows on the fast side. **This trade pays only when the slow links are significantly more expensive**, which is exactly the applicability criterion.

## Applicability criterion (run this first)

```
python tools/breakeven.py --ratio <your measured fast-side/slow-side bandwidth ratio>
```

Necessary and sufficient condition for two-hop to come out ahead on bytes (R cards per group, q experts per group; **card = die = one EP rank**, since the design docs say "card" and the measurement docs say "die" where the physical unit matters):

$$r_{be} = \frac{(1-1/R)\,q}{q-1} \quad<\quad \frac{\beta_{fast}}{\beta_{slow}}$$

At R=8: q=2 → 1.75, q=3 → 1.31, q=6 → 1.05, q=8 → 1.00.

This byte-only inequality is a necessary sensitivity check, not a deployment verdict.
The full communication-call model additionally prices fixed call costs, message-size
saturation, and the arrival chain. A target must measure those quantities at its own
operating point and pass the communication-level gate; the failed step-level gate
precludes any training-throughput claim.

Two things you must watch:

- **Use measured ratios, not nominal ones.** Effective collective-communication bandwidth is often off from nominal by more than 2x, and you must measure at **your real message sizes**, because bandwidth's dependence on message size and alignment can manufacture an entirely false conclusion. We stepped on this.
- **Bytes are only half the ledger.** Two-hop pays the fixed overhead of one extra collective; passing the byte criterion only advances the target to a calibrated communication-call check. Full checklist in [docs/03-applicability.md](docs/03-applicability.md).

## T-Route's quality cost (independent of communication; stands on its own)

![What each routing mode constrains](docs/assets/f14-routing-modes.svg)

T-Route is the conjunction of two constraints that exist separately in the literature: a **group limit** (a token's experts confined to M of the N_g groups, as in DeepSeek-V2 device-limited and V3 node-limited routing) and an **equal quota** inside each chosen group (as in MoGE, where M = N_g). The group limit bounds per-token fan-out; the quota fixes per-token multiplicity within a selected group. Aggregate peer counts remain data dependent, so the conjunction supplies a bounded traffic envelope rather than a static traffic matrix.

![Measured cost of each mode](docs/assets/f15-routing-costs.svg)

![Quality axis](docs/assets/f1-loss-axis.svg)

The **complete ablation results, per-seed figures, and retraction records** for all four axes (quality / downstream / load / step time) are in [docs/06-troute-results.md](docs/06-troute-results.md). Headline table (13.14B total params / 1.33B active, 4 modes × 4 seeds, 62.9B tokens per arm, holdout val loss):

| Mode | Constraint | Δ vs unconstrained top-k | 90% CI |
|---|---|---|---|
| `group_limited` | group limit only (4 of 8) | +0.00276 | [+0.00223, +0.00329] |
| `quota_only` | equal quota only (= MoGE) | +0.00895 | [+0.00726, +0.01064] |
| **`full` (T-Route)** | group limit + equal quota | **+0.00339** | [+0.00224, +0.00455] |

- The cost is +0.0034 nats, **3.4%** of the prespecified no-loss threshold of 0.1 nats, and all 24 paired seed deltas (3 modes x 4 seeds x 2 readpoints) share the sign. This is not "no detectable difference": all three modes have confidence intervals excluding 0, so the effect is resolved first and only then shown to be small.

![Per-seed paired deltas](docs/assets/f2-per-seed.svg)

- **`full` costs about 38% of `quota_only` (roughly 62% lower)**: the group limit gives the quota a pressure-relief valve ("which M groups" remains free).
- Downstream (HellaSwag / LAMBADA, ±1.0 pp prespecified TOST): reported equivalent,
  but the released artifact lacks the complete estimator inputs needed for independent
  reconstruction.
- Load: expert-level load entropy in the constrained modes is no lower than the control, which rules out "hidden capacity loss traded for efficiency".

**Boundary**: cross-group balance is a statistical property, not an architectural guarantee (group-level CV can reach 1.0 under adversarial input); validated up to the scale above, so do not extrapolate linearly to larger widths or more extreme M/N_g.

## Results at a glance

### The cost model, and what validates it

![Uncertainty bands](docs/assets/f9-uncertainty.svg)

A collective call costs `alpha(world) + wire_bytes / beta_eff(per-peer bytes)`, with
`beta_eff = beta_inf * x / (x + x_half)`. Every constant is measured, and the model
answers to three gates rather than to a plot that looks right.

| Gate | What it checks | Result |
|---|---|---|
| Tier-1 | one-hop and two-hop times measured directly on the calibrated machine | **pass**, 4.1% median error, 24.5% worst |
| Tier-1b | 64 targets from another benchmark family; machine B is a same-corpus fit/consistency check, while 14 C4 targets on A are post-freeze holdouts | **pass** on all three corpora, 1.9% / 9.3% / 8.0% median |
| Tier-1b drift probe | a fourth corpus, 13 world-8 targets on the calibrated machine | **fail**, 15.1% median with a −9.1% bias, traced to one constant and left unretuned |
| Tier-2 | end-to-end step time on 7 training geometries | **fail, and step-level extrapolation stays locked** |

The band above is 400 Monte Carlo draws over the calibration uncertainty, including
the bootstrap interval of `x_half`. Two anchors read off it: at a hierarchy ratio of
8 the least favourable measured-chain case is still 1.41, so that simulated direction
is robust to propagated calibration error. At ratio 1.03 the zero-overhead p95 is
1.03 and crosses unity, so no robust sign claim is made for that corner.

### What one collective call actually costs

![Launch cost](docs/assets/f16-launch-cost.svg)

`alpha(world)` is the fixed cost of a call, and a call-count scan measured it
directly rather than reading it off a fit: N collectives issued back to back at
fixed world, N from 1 to 1024, payloads from 256 B to 16 MB, two worlds, two nodes.
Three results, all of them load-bearing elsewhere in this repository.

- **Collectives do not pipeline.** Per-call cost stops falling at N around 16 and
  stays flat to N = 1024. Two back-to-back calls cost twice one, which is the
  assumption `sim/core.py` makes when it prices the two-hop chain serially.
- **The shipped `alpha` is corroborated at the level**: 129 microseconds measured
  against a tabulated 111 at world 8, 134 against 157 at world 16, both inside the
  20% run-to-run drift the calibration documents, from a different benchmark months
  later. The 41% step between the two table entries is *not* confirmed; the scan
  sees 12%, inside its own spread.
- **The same call costs twice as much when the host has to watch it.** 128
  microseconds with the queue deep, 256 when the host observes each call before
  issuing the next. The gap is a per-call cost that holds its share, 46 to 59%,
  across every payload from 64 KiB to 16 MB.

Charging two-hop one extra host exposure moves the breakeven ratio from 3.98 to
4.15. Removing the arrival chain instead moves it from 3.98 to 1.10. The ordering is
the point: fuse the chain first, and the launch path is the next thing worth paying
for, not the first.

### Where the wins and losses come from

![Breakeven map](docs/assets/f8-breakeven-map.svg)

Green is where two-hop wins. The two panels differ only in the arrival chain, and
the measured-chain and hypothetical-fused targets move the corrected breakeven from
3.98 to 1.49. At ratio 3.2 this is a sensitivity comparison, not a prediction for a
named machine.

### End-to-end measurements on the calibrated machine

![Verdict testbed](docs/assets/f11-verdict-bed.svg)

Seven geometries, same configuration, only the all-to-all implementation swapped.
This machine has a hierarchy ratio of 1.03, so the communication-call model scores
two-hop as a loss. Five of six step-level directions agree, one is a measured miss,
and the step-level gate remains failed. The bed is diagnostic: more tokens per
micro-batch makes it worse because
the workload becomes bandwidth-bound, fewer nodes makes it worse because Hop A has
fewer peers to deduplicate across, and wider cross-group fan-out makes it worse
because two-hop converges to one-hop's traffic while keeping its extra costs.

### Scale, and where the data runs out

![Scale effect](docs/assets/f10-scale-alpha.svg)

Through 128 ranks every defensible treatment of `alpha` agrees to the digit. Past
that, the only corpus covering worlds 256 and 512 sits five times below every other
in absolute bandwidth and fits worst by a factor of four, so the figure plots the
band across four treatments instead of a line. At 512 ranks they span 1.37 to 2.94
and the direction of the trend flips, which is why this repository makes no claim
about clusters past 128 ranks.

## Repository layout

| Path | Contents | Status |
|---|---|---|
| `terrace/routing.py` | T-Route reference implementation; all four ablation modes switch inside one function | **Validated** (quality ablation + property tests) |
| `terrace/ta2a*.py` | T-A2A two-hop chain: planning, dispatch, packing, differentiable seam | **Validated bit-exact** (guarded by the repo's 291 CPU tests; end-to-end measured only on a flat supernode, see the criterion) |
| `terrace/ops/` | Arrival-chain fused kernels (AscendC): passthrough / K1 / K2 + executable CPU spec | passthrough passes bit-exact validation on device; **K1 algorithm proven correct, but the device-side translation has one unfixed multi-core scalar-write visibility bug (the earlier out-of-bounds is fixed); K2 not validated on hardware** (see [docs/04-kernel-status.md](docs/04-kernel-status.md)) |
| `sim/` | Measurement-calibrated cost model: cluster spec → one-hop/two-hop times, breakeven maps, Monte Carlo uncertainty bands, expert-FFN roofline, platform registry and the target checklist (see [docs/05-simulator.md](docs/05-simulator.md)) | **Tier-1 and Tier-1b gates passed** (4.1% median on C1; 1.9%/9.3%/8.0% on C2-C4, with C3 a same-corpus B fit and C4 the post-freeze A holdout); a world-8 drift probe fails and is not retuned. Tier-2 fails, so step-level extrapolation is banned. |
| `tools/breakeven.py` | Applicability criterion (closed form; together with sim/, two independent implementations of the same ledger, cross-checked by tests) | — |
| `tests/` | 291 tests, pure CPU (no NPU needed) | All green |
| `tools/onesided/` | One-sided transfer instrument: preregistered benchmark of aclshmem put vs collective a2a, plus hyper-parallel patches (free serialization + UAF, hard-coded block_dim) and EQ usage traps | Ruled a loss on the bandwidth-flat machine (best case 0.68× a2a, see [docs/08-onesided.md](docs/08-onesided.md)); the patches apply to every Ascend+shmem user |
| `docs/` | Design docs ×5, the routing constraint explained (docs/10), full ablation results (docs/06), the Tier-2 campaign (docs/07), the one-sided verdict (docs/08), and the phase model with the measurement that would calibrate it (docs/09); figures-first | — |
| `tools/gen_figures.py` | Result-figure generation (numbers embedded; figures reproducible) | — |

**Not in this repository**: training-framework integration (upstream training stack / Megatron shims), cluster and measurement scripts, raw experiment data, internal records. The seam contract is in [docs/02-ta2a-design.md](docs/02-ta2a-design.md); integrators write their own shims against the contract.

## Running the tests

```
python -m pytest tests/ -q        # torch only, runs on CPU, about 70 seconds
python -m sim.validate_micro      # simulator Tier-1 validation gate
python -m sim.sweep               # cross-cluster extrapolation (gate checked at entry; all output labeled "simulated")
```

## License

Apache-2.0
