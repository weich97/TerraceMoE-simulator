# T-Route Design

## Definition

Standard top-k routing: a token's hidden vector scores affinity against E experts, take the top k. T-Route adds two constraints on top, with group boundaries = communication hierarchy boundaries (node / rack / supernode, decided by deployment) and the E experts split evenly into N_g groups:

1. **Group-limited**: first pick M groups (M < N_g) by "highest affinity score inside the group"; the token's experts come only from these M groups.
2. **Equal quota inside groups**: each selected group contributes exactly q = k/M experts (requires M | k).

## Five-step pseudocode

```
scores = sigmoid(h @ E_experts + bias)          # affinity scores; bias is the aux-loss-free balancing bias
group_score = max_pool(scores, per_group)       # representative score per group
groups = topM(group_score)                      # pick M groups
experts = for each selected group: topq(scores[that group])   # exactly q per group
gates = normalize(scores[experts])              # gate normalization
```

The four ablation modes (`global_topk` / `group_limited` / `quota_only` / `full`) switch via `mode` inside one function — **a single shared code path is the precondition for comparable ablations**. See `terrace/routing.py`.

## What holds unconditionally, and what holds only statistically

**Unconditional (architectural guarantees, any input)**:

- exactly k experts per token, all distinct;
- the experts span ≤ M groups ⇒ **per-token cross-group fan-out is bounded**;
- exactly q experts per selected group ⇒ **every cross-group message has a fixed row count** (q × H·dtype bytes per row).

**Statistical only (breakable by adversarial input)**:

- **Load balance across groups**. With M < N_g, "which M groups" is decided by the data; balance is driven by the bias, a statistical property. On real corpora the group-level CV sits in the same range as the unconstrained control, but an adversarial construction can push group-level CV to 1.0 (`tests/test_routing.py` carries a reproducible counterexample). A truly data-independent traffic matrix needs batch-level capacity constraints or global assignment — that is a different routing algorithm, and the quality ablations would have to be redone.

## Point-by-point differences vs prior work

| | Group-limited | Equal quota | M < N_g |
|---|---|---|---|
| DeepSeek-V3 node-limited | ✔ | ✘ | ✔ |
| MoGE (equal quota) | ✘ | ✔ | ✘ (M = N_g) |
| **T-Route** | ✔ | ✔ | ✔ |

Why the conjunction matters: group-limiting gives equal quota a pressure-release valve — `quota_only` must place experts in **every** group, while `full` places q in each of the M selected groups only, and "which M groups" stays free. The quality ablation confirms `full` costs about 38% of `quota_only`, roughly 62% lower (README table: +0.00339 vs +0.00895).

## Why these two constraints help communication

- Fan-out bound M ⇒ the **count** of cross-group messages is known at compile time;
- Equal quota q ⇒ the **size** of cross-group messages is known at compile time (fixed length; no need to exchange counts first to compute splits);
- Together, they are what lets Hop A of the hierarchical all-to-all (T-A2A) become a **fixed-shape** communication — the precondition for turning dispatch from a "data-dependent variable-length a2a" into "static orchestration".

Note: fixed shape covers Hop A (the cross-group leg) only. Inside a group (Hop B), "which q experts" still varies with the data; intra-group scatter remains variable-length — see doc 02.
