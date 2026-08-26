# T-Route ablation results: the complete set

For what the constraint is and how the four routing modes differ, start at
[docs/10](10-troute.md). This page is the full record behind it.

**One sentence: T-Route (group-limited + equal-quota within groups) is lossless/neutral on all four axes — quality, downstream, load, step time — and every axis carries a preregistered criterion + paired seeds + a holdout set/independent measurement.** The numbers embedded in the figures match those embedded in `tools/gen_figures.py` — the script is the source, and the figures are reproducible.

Testbed: 13.14B total params / 1.33B active, E=128 = 8 groups × 16, k=8, M=4; 4 routing modes × 4 seeds = 16 arms, 62.9B tokens per arm; all four modes switch via `mode` inside **the same function** (a single shared code path is the precondition for a comparable ablation; implementation in `terrace/routing.py`).

---

## 1. Quality axis: +0.0034 nats, 1/5 of the tolerance

![quality axis](assets/f1-loss-axis.svg)

| Tier | Δ @20k (90% CI) | Δ @30k (90% CI) | drift between read points |
|---|---|---|---|
| group_limited | +0.00276 [+0.00223, +0.00329] | +0.00192 [+0.00114, +0.00270] | −0.00084 |
| quota_only (=MoGE) | +0.00895 [+0.00726, +0.01064] | +0.00895 [+0.00751, +0.01039] | **±0.00000** |
| **full (T-Route)** | **+0.00339 [+0.00224, +0.00455]** | +0.00372 [+0.00300, +0.00444] | +0.00033 |

Three reading notes:

- **This is not "we cannot detect a difference" — it is "the difference is small and we can detect it"**: all three tiers' CIs exclude 0. The holdout-set probe's within-arm sd is 0.0008–0.0023 nats, an order of magnitude below seed noise in the training-log convention (0.0056–0.0058); the earlier "indistinguishable" report on this axis was an insufficient ruler, not a zero effect.
- **full is ~62% cheaper than quota_only**: the group limit gives the equal quota a pressure-relief valve — quota_only must place experts in every group, while full places q per group only within the M selected groups, and "which M groups" stays free.
- Measurement protocol: all 16 arms' checkpoints re-measured uniformly (the `--skip-train` path, holdout shard, 4096 sequences per arm, same seed and same GBS, read points verified cell by cell across the 32-cell grid).

## 2. Per-seed: 24/24 paired differences share the same sign

![per-seed](assets/f2-per-seed.svg)

3 constraint tiers × 4 seeds × 2 read points = 24 paired differences, **all positive** — the effect is real, the direction consistent, and every one of them is far below the tolerance. One more witness for the point estimates' credibility: quota_only gives the identical five-decimal +0.00895 at two read points 10k steps apart.

## 3. Downstream axis: equivalent on both axes, plus a lesson in "single read points lie"

![downstream axis](assets/f3-downstream.svg)

| Axis | T-Route Δ | 90% CI | TOST @ ±1.0 pp (preregistered) |
|---|---|---|---|
| HellaSwag acc_norm (3000 questions, loglikelihood) | +0.158 pp | [−0.095, +0.411] | **equivalent** |
| LAMBADA accuracy (5153 questions, three read points pooled) | −0.084 pp | [−0.406, +0.238] | **equivalent** |

The right-hand panel is the most methodologically valuable piece: looking at LAMBADA @30k alone, T-Route Δ=−0.577, with the CI just breaching ±1.0 — **a single measurement would have falsely reported a loss**. After adding @10k/@20k, every mode's point estimate swings 1–2 pp between read points and changes sign, with swings larger than any single point estimate itself — that is the signature of noise. At n=4, LAMBADA cannot resolve sub-1 pp effects; on this axis the repo makes an equivalence call only and claims no specific cost value.

## 4. Load axis: the constraint did not buy efficiency with hidden capacity loss

![load axis](assets/f4-load-axis.svg)

| Paired diff (vs unconstrained, n=4) | group_limited | quota_only | full (T-Route) |
|---|---|---|---|
| load entropy Δ (≥0 good) | +0.00018 ±0.00033 | +0.00035 ±0.00026 | +0.00026 ±0.00040 |
| load CV Δ (≤0 good) | −0.0061 ±0.0125 | −0.0137 ±0.0094 | −0.0094 ±0.0147 |

The criterion asks a **relative** question: did the constrained tiers get worse relative to unconstrained (paired by seed, n=4, mean over each arm's final 50 windows)? All three tiers show entropy diff ≥0 and CV diff ≤0 — not only not worse, the direction is slightly better. The routing flip-rate half closes independently (constrained/unconstrained flip ratio 0.990).

## 5. Step-time axis: the constraint levies no step-time tax

![step time](assets/f5-step-neutral.svg)

Both arms **use the same baseline one-hop a2a and differ only in routing** (on = full, off = unconstrained top-k; both arms' mode/quota switches passed three independent cross-checks), 16 nodes × 8 dies end to end, median of 10 steady-state step times per arm:

| Run | G = t(unconstrained)/t(T-Route) |
|---|---|
| 1 | 0.9882 |
| 2 | 1.0070 |
| mean | **0.9976** (straddles 1.0; same-type jobs show ±1–3% run-to-run spread) |

**T-Route's step-time effect is zero (within noise).** Together with the three axes above: it is a quality-neutral + step-time-neutral structural constraint; what it buys is a **compile-time communication envelope** (per-token cross-group fan-out ≤M, fixed-length cross-group messages) — exactly the enabler for hierarchical communication on hierarchical clusters (docs/03). Note that the load-entropy advantage (forerunner figure F6) did **not** convert into step time — proxy metrics and deliverable metrics must be measured separately; that too is part of the methodology.

## 6. Forerunner: the small-scale 2×2 decomposition

![forerunner](assets/f6-forerunner.svg)

Before the main ablation, a small-scale synthetic testbed (0.5B/A0.1B, single node, 4 modes × 2 seeds) first validated the two constraints in a 2×2 decomposition:

| Mode | converged val loss (seed mean) | seed range | load entropy |
|---|---|---|---|
| global_topk | 6.0302 | 0.0003 | 0.924 |
| quota_only | 6.0313 | 0.0002 | 0.907 |
| group_limited | 6.0305 | 0.0011 | 0.930 |
| **full (T-Route)** | 6.0306 | 0.0002 | **0.946** |

The four tiers differ from one another by ≤0.0011 nats (below the within-seed range), and full has the highest load entropy. The forerunner carries less verdict weight than the main ablation (the task discriminates weakly); its value was falsifying method defects and delivering the ablation tooling.

## 7. Retractions and negative records (kept in full — they are why the numbers above are credible)

1. **Forerunner round one voided**: it originally compared training-set loss, which superficially showed full winning big (6.22 vs 6.62). Trajectory inspection ruled it invalid: not converged (it compared "who entered the steep-descent segment first"), overfit (31 epochs — memorization, not generalization), and descent-timing variance (two seeds of the same mode differed by 0.79). After the fix (holdout set + training to plateau), the clean result of Section 6 emerged.
2. **Forerunner run 4b retracted in full**: in that run the expert backward pass never propagated due to a backend defect — routed experts stayed frozen at initialization, so all four tiers being equal was inevitable and constitutes no losslessness evidence; the mechanism attribution built on it ("structured routing memorizes more") is retracted with it. Replacement evidence comes from the rerun after fixing the backend (hard task, 6 seeds paired, full Δ=+0.0011).
3. **Two LAMBADA single-read-point conclusions retracted** (see Section 3): at n=4, single-read-point estimates swing 1–2 pp and change sign.
4. Verdict discipline: equivalence bounds fixed before running; in overfit scenarios the comparison point is val-minimum (early stopping), not final-val.

## 8. Boundaries (reproduced verbatim, not softened)

- Inter-group balance is a **statistical property, not an architectural guarantee**: with M < N_g, "which M groups" is data-driven; under adversarial input the group-level CV can reach 1.0 (`tests/test_routing.py` ships a reproducible counterexample).
- Verified up to the scale and geometry above; do not linearly extrapolate the 3.4% ratio to larger widths or more extreme M/N_g.
- The MoGE control (quota_only) requires k ≥ N_g and N_g | k, satisfied only in this geometry; that cell must not be mixed into tables with other geometries.
- Step-time neutrality was measured on a **bandwidth-flat** machine; on hierarchical machines the constraint's communication gains go through the docs/03 criterion, which does not conflict with the neutrality result here (what it earns there is the envelope, not the balance measured here).
