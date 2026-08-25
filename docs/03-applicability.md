# Applicability Criterion: Which Clusters Should Use This, and Which Should Not

This doc is why the repo exists. **Conclusion first: this method targets clusters with pronounced hierarchy; on bandwidth-flat interconnects, two-hop does not pay — we measured that ourselves.**

## 1. Deriving the breakeven

Write β_f for the fast-side (intra-group) bandwidth, β_s for the slow-side (cross-group) bandwidth, R cards per group, q experts per token per selected group, and B payload bytes per row (H × dtype width). Transfer time per token per target group:

```
one-hop:  t1 = qB / β_s
two-hop:  t2 = 1·B / β_s  +  q(1 − 1/R)·B / β_f
```

Two-hop nets out ahead ⇔ t2 < t1 ⇔

**r_be = (1 − 1/R) · q / (q − 1)  <  β_f / β_s**

Key points:

- r_be **decreases monotonically in q**, with limit (1 − 1/R). R=8: q=2 → 1.75, q=3 → 1.31, q=6 → 1.05, q=8 → 1.00. **Small quotas (strong group-limiting) actually demand a deeper hierarchy** — there are fewer slow-side rows to save.
- The formula assumes both sides send one copy per (token, expert). If the two-hop side does **card-level dedup**, the q in the numerator becomes the expected hit-card count D(q) < q and the criterion relaxes (`tools/breakeven.py --dedup`).
- The derivation counts bytes only. **The α side (per-collective fixed overhead) is a second gate to clear**, see §4.

## 2. Where the interconnect classes land

| Interconnect | β_f/β_s (order of magnitude; trust your own measurements) | q=3 criterion (needs >1.31) |
|---|---|---|
| NVLink domain vs cross-node IB | ~3× and up | passes |
| in-server high-speed bus vs cross-server RoCE | ~8× | passes |
| intra-supernode vs cross-supernode | several × (and cross-supernode collectives often degrade under incast) | passes (if you actually stretch EP beyond the supernode — see §5) |
| **inside a flat supernode** (unified switch silicon, cross-node p2p ≈ intra-node) | **~1.0** | **fails** |

Public positive evidence for this method family (on hierarchical interconnects): DeepSeek-V3 training (node-limited routing + send once across nodes, then forward inside the node; NVLink vs IB); TeleChat3-MoE (hierarchical EP A2A, reports +15% training throughput at EP=16); Pangu Ultra MoE (same-family hierarchical A2A). **The shared precondition that makes these work is exactly hierarchy ratio ≫ r_be.**

## 3. Our negative result (flat supernode, full disclosure)

On a bandwidth-flat supernode (16 nodes × 8 cards, cross-node p2p to intra-node bandwidth ratio ≈ 1.0, flat to within 2.6%) we built T-A2A out in full and ran **same-config end-to-end comparisons across 7 geometries** (only the a2a implementation is swapped, everything else identical to the letter; step time is the steady-state median; G = t(one-hop)/t(two-hop), >1 = two-hop faster):

![Verdict testbed panorama](assets/f11-verdict-bed.svg)

| Axis | Geometry | G (measured) | Runs |
|---|---|---|---|
| — | baseline point (16 nodes, T=4096, k6M2, MBS=1) | **1.0355** | n=7, t=3.27, **p≈0.017 significant** |
| token ×2 | same, MBS=2 | 0.8845 | n=3, spread 0.8% |
| token ×4 | same, MBS=4 | 0.8234 | n=3, spread 0.6% |
| load | k8M2 | 0.9935 | n=1 |
| load | k8M4 | 0.9335 | n=1 |
| scale | 8 nodes | 0.9027 | n=1 |
| scale | 4 nodes | 0.8647 | n=1 |

All three axes trend in the direction the mechanism predicts, which is more convincing than any single number:

- **the token axis falls monotonically**: more tokens means more bandwidth-dominated, and two-hop moves a net ~8% more bytes through the only resource that matters here (per-card egress);
- **the scale axis falls monotonically**: fewer nodes means fewer peers Hop A can save, so two-hop's relative overhead grows;
- **the load axis points the same way**: the wider the cross-group fan-out (M), the closer two-hop gets to one-hop's communication shape, leaving only the extra cost;
- **the single positive value (+3.6%, significant) sits exactly where the micro-level books say it should**: in the same-machine communication microbenchmark, the 4096 token/rank tier lands precisely inside the window where two-hop wins on pure communication (one-hop 1.366 ms vs two-hop 1.217 ms, see `sim/validate_micro.py`); subtract implementation costs such as the arrival chain and this 3.6% is what remains. As soon as tokens grow (from the 8192 tier up), the micro window closes and end-to-end turns negative in step. Note this is **not** α savings — direct measurement on this machine gives α(16)+α(8) ≈ α(128); two-hop gains almost nothing on fixed overhead (docs/05, calibration notes).

**This negative result is scoped to "flat interconnect + this implementation"**: what got the negative verdict is "doing hierarchy on a machine that has none", not hierarchy itself. The same criterion returns the opposite verdict on machines with hierarchy ratio ≥ 1.5 (simulator extrapolation and uncertainty bands for hierarchical machines: [docs/05](05-simulator.md)). These 7 points also serve as the ground-truth set for the simulator's Tier-2 validation gate (`sim/validate.py::HOLDOUTS`, same source numbers).

## 4. Three things to check beyond the criterion (pitfall list)

1. **The α side**: two-hop adds one collective per dispatch. Measure how your platform's a2a fixed overhead grows with peer count — if α barely varies with peer count, two-hop gains nothing at the small-message end either.
2. **Message-size effects**: collective bandwidth can depend strongly on the alignment of per-peer byte counts (we observed implementation behavior stepping in power-of-2 stairs). **Benchmarks must use integer multiples of your real row width (H × dtype)** — otherwise you are measuring alignment effects and the entire conclusion is fake.
3. **Instrument resolution**: measure the instrument's own noise first (repeat spread, launch overhead floor); your threshold must exceed the noise. We crashed once at each of these three spots; the lessons are all written into the tests.

## 5. A boundary observation

As of our survey (2026-08), wherever a rack-level/supernode-level high-bandwidth domain exists, public training-side deployments keep EP inside the domain and go DP/PP across domains (the generation trained on 8-card NVLink domains routinely ran EP across nodes — exactly the classic scenario for this method family); no public work yet stretches EP beyond a **supernode-level** domain. If your scenario must cross it (the experts don't fit inside the domain, or fault tolerance left the domain degraded), the two-hop criterion passes easily at that level — but you would be first: there is no public data on α behavior and incast there, and you must measure it yourself.
