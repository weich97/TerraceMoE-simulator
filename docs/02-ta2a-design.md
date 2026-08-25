# T-A2A Design: Two-Hop Hierarchical All-to-All

## Structure

One-hop baseline: dispatch is a single `all_to_all_single` over the whole EP group (world = N_g × R); each of a token's k payload rows travels straight to its destination card — every row crosses the slow link.

T-A2A splits it into two hops:

```
Hop A (cross-group, slow link): the payload a token sends to group g goes out once, to the representative card in group g
                                (representative = the card with the same intra-group index as the sender; a pure function, no negotiation)
Hop B (intra-group, fast link): the representative card scatters the received rows to the actual expert cards inside the group
```

Per token per target group: slow-link payload drops from q rows to 1 row; the fast side picks up q(1−1/R) extra rows (the representative card's own share needs no move). **The applicability criterion lives in tools/breakeven.py and doc 03 — this trade only pays off when the slow link is significantly more expensive.**

## Data flow (matched to code)

1. **Routing output format**: T-Route emits per-token (group, intra-group slot, gate) — `terrace/routing.py`.
2. **Plan**: `terrace/ta2a.py::plan_ta2a` buckets the batch's tokens by target group and emits Hop A's send layout. Under equal quota each (token, group) has exactly q slots, so the layout is **fixed-shape** (the quota fast path); without quota it degrades to the variable-length path.
3. **Hop A**: counts exchange (small) + payload `all_to_all_single` (cross-group communicator).
4. **Arrival chain**: the representative card expands the received (row, slot) pairs into the intra-group scatter plan — pair expansion, stable bucket sort by owner card, intra-group counts histogram, send-buffer gather. This section is pure local tensor ops, and it is **the target of fused kernel K1** (see doc 04).
5. **Hop B**: intra-group `all_to_all_single` (fast-side communicator) delivers the rows to the expert cards; gate and slot metadata can be packed together with the payload (`terrace/ta2a_pack.py` — merging the small messages saves one collective's fixed overhead).
6. **combine**: mirror image, reversed.

## Backward (autograd seam contract)

Many training stacks wrap the MoE layer forward in a custom `autograd.Function` (gradient tracking off, backward hand-written), so T-A2A **does not depend on the outer autograd graph**:

- the payload path goes through the differentiable `_A2A` (`terrace/ep_dist.py`); its backward is automatically the transposed a2a;
- the metadata path (int, no gradient) goes through raw a2a;
- gates hang on their own differentiable edge — **do not weld gates and payload into one fused node**: the outer stack may call `.backward()` separately on the two outputs; welding them triggers "backward a second time", and the first call writes zero gradients into the other path (we hit this).

An integrator needs to provide only two things: the dispatch seam (replace the original token_permutation with T-A2A's permute) and the combine seam (mirror image). **Integration shims do not ship with the library** — they couple tightly to the specific training stack; this repo's tests guard bit-level correctness against a pure-PyTorch reference chain (280 tests repo-wide, runnable on CPU).

## Known structural costs (honest list)

From building the full thing and measuring it to the bottom: two hops cost more than bytes.

1. **One extra collective's fixed overhead.** Each dispatch goes from 1 a2a to 2 (+counts); the fixed overhead is on the order of 0.05–0.5 ms per call depending on implementation/platform. Small-message (α-dominated) scenarios need their own accounting.
2. **Variable-length splits need a host-side sync.** `all_to_all_single` wants its splits as a Python list — one device-to-host readback (~0.04 ms order). This also kills the naive "chunked pipelined two-hop overlap": every chunk pays one sync. Equal quota makes Hop A fixed-shape and drops that sync, but Hop B stays variable-length.
3. **The arrival chain's local tensor ops are not cheap.** The expand/bucket-sort/gather PyTorch composite chain is, in our measurements, the **largest single item** of two-hop extra cost — larger than the byte difference — which is exactly why fused kernel K1 exists (doc 04).
4. **Hop B does no card-level dedup by default**: a token routed to multiple experts on the same card still sends one copy per expert. Dedup can save more (the criterion tool's `--dedup` mode) but needs the arrival chain to cooperate on expansion.

**On hierarchical clusters these four are costs to manage; on flat clusters they are what kills the trade. Run the criterion first.**
