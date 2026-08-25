"""CPU/pure-torch reference for K2 -- the executable spec of the kernel semantics, bit-for-bit identical to the live chain.

Standalone file, depends only on torch (**does not import terrace.ta2a /
ta2a_fwd**): it is a direct transcription of the AscendC kernel's
(ascendc/op_kernel/terrace_k2_pack.cpp) two-pass counting sort, not a rewrite
of the live chain -- only then is the bit-for-bit reconciliation in
tests/test_terrace_k2_ref.py ("reference == live chain (plan_ta2a fast path +
deduplicated gather + _pack_quota_wire)") an informative proof, not the same
code checking itself. The K1 reference (k1_arrival_ref, in
terrace/ops/__init__.py) reuses live-chain primitives; the K2 reference is a
standalone file per the task requirement, and grafting it in takes only
`from .k2_ref import k2_pack_ref` in __init__.py (internal grafting notes (not
published with the repo)).

The live-chain stretch being replicated (quota fast path, equal quota
groups_m=M; the three call sites share the same isomorphic stretch:
ta2a_fwd.ta2a_moe_forward / ta2a_dispatch.ta2a_permute[_overlap]):

    u_src, u_node, node_counts, inverse = plan_ta2a(expert_idx, world,
                                                    n_experts, rpn, groups_m=M)
    payload = hidden[u_src]
    mask, gate_rows = _pack_quota_wire(expert_idx, gates, inverse, payload,
                                       n_rows, slots, quota, n_experts)

The bit-for-bit argument (full text in the kernel file header; here in
executable form):

  - plan's row order = the ascending set-bit positions of the occupancy table
    flattened node-major, i.e. (node ascending, token ascending); under equal
    quota, a token's k experts, once sorted ascending, split into exactly M
    segments (runs) of quota each, each run entirely on one node, nodes
    strictly ascending across runs -- enumerating runs by (t, j) and
    stable-sorting by destination node yields a dst permutation elementwise
    equal to plan's sel enumeration (at most one run per token per node;
    within-bucket order == ascending token order). The argsort(dest,
    stable=True) here is the same mathematical object as the kernel's two-pass
    scheme (histogram -> prefix cursors -> in-order placement).
  - within-row sort: expert ids are distinct within a row => the ascending
    permutation is unique; torch.sort, the kernel's insertion sort, and the
    live chain's _pack_quota_wire(sorted_rows=False) float32-key argsort all
    yield the same permutation; when the row is already ascending (the seam
    entry, routing_map_to_topk by construction) the sort is identity, so it is
    also bit-for-bit identical to the sorted_rows=True no-sort branch.
  - gates are pure movement throughout (gather/reshape, no arithmetic),
    bit-for-bit identical to the live chain's gather.

Contract (matches the torch-side csrc/kernel TORCH_CHECKs; fail loud):
  - gates.dtype == hidden.dtype: the C1 rounding-point contract -- the gate
    plane derives from the payload, so a mismatch must die loudly (the same
    failure shape as _pack_quota_wire at its index_put; the entry point of a
    drift defect in one internal commit); the caller owns the rounding point
    (the overlap seam casts before packing).
  - out-of-range expert ids raise directly: the live chain dies loudly at
    plan's scatter/gather; the reference is equally non-silent. The kernel side
    cannot raise and substitutes "skip the write + zeros containment" (no
    bit-level promise for corrupted inputs; see "containment of corrupted
    inputs" in the kernel file header).
  - the equal-quota invariants (each token exactly M nodes, each node exactly
    quota experts) are the caller's responsibility (plan_ta2a validates on the
    first call + every 256 calls); under the invariants the reference is
    bit-for-bit identical to the live chain, and on drifted inputs it promises
    nothing, same as the live chain (the live chain's searchsorted dies loudly
    out of range).
"""
from __future__ import annotations

import torch


def k2_pack_ref(hidden: torch.Tensor, expert_idx: torch.Tensor, gates: torch.Tensor,
                world: int, n_experts: int, rpn: int, groups_m: int):
    """Reference for the send-side fused packing chain: (payload, mask, gate_rows, u_src, node_counts).

    hidden [T, H]; expert_idx [T, k] int64 (distinct within a row, no ascending
    requirement); gates [T, k] with hidden's dtype. Returns payload [T*M, H],
    mask [T*M, quota] (ascending slot-id table, int64), gate_rows [T*M, quota],
    u_src [T*M] int64, node_counts [n_nodes] int64. Differentiability matches
    the live chain: payload (deduplicated gather of hidden) and gate_rows
    (permutation gather of gates) carry gradients; mask/u_src/node_counts are
    index/count planes.
    """
    if expert_idx.dim() != 2 or hidden.dim() != 2 or gates.dim() != 2:
        raise ValueError("k2_pack_ref: hidden/expert_idx/gates must be 2-D")
    T, k = expert_idx.shape
    if hidden.shape[0] != T or gates.shape != expert_idx.shape:
        raise ValueError(
            f"k2_pack_ref: geometry mismatch (hidden {tuple(hidden.shape)}, "
            f"expert_idx {tuple(expert_idx.shape)}, gates {tuple(gates.shape)})")
    if gates.dtype != hidden.dtype:
        # RuntimeError, same class and same noise as _pack_quota_wire's index_put failure shape.
        raise RuntimeError(
            f"k2_pack_ref: gates dtype {gates.dtype} != hidden dtype {hidden.dtype}"
            f" -- the C1 gate plane derives from the payload; cast at the caller"
            f" (the caller owns the rounding point)")
    if world <= 0 or rpn <= 0 or groups_m <= 0 or n_experts <= 0:
        raise ValueError("k2_pack_ref: bad geometry scalars")
    if world % rpn or n_experts % world or k % groups_m:
        raise ValueError(
            f"k2_pack_ref: world={world} rpn={rpn} n_experts={n_experts} k={k} "
            f"groups_m={groups_m} not divisible (same asserts as the live chain)")
    epr = n_experts // world
    slots = epr * rpn
    n_nodes = world // rpn
    quota = k // groups_m
    if slots > 63:
        raise ValueError(f"{slots} expert slots per node exceeds the chain bound")
    if T and (int(expert_idx.min()) < 0 or int(expert_idx.max()) >= n_experts):
        raise ValueError("k2_pack_ref: expert id out of range (the live chain dies "
                         "loudly in plan_ta2a's scatter; so does the reference)")

    dev = expert_idx.device
    P = T * groups_m                                 # send row count, static (what buys the sync-free path)

    # ---- within-row ascending sort (the kernel's pass-2 insertion sort; distinct
    #      within a row => unique permutation, identity when already ascending)
    e_sorted, order = torch.sort(expert_idx, dim=1, stable=True)
    g_sorted = torch.gather(gates, 1, order)

    # ---- run enumeration (t ascending, j ascending) and destination node;
    #      stable counting sort == plan's row order ----
    dest = torch.div(e_sorted[:, ::quota], slots,
                     rounding_mode="floor").reshape(-1)              # [T*M]
    sigma = torch.argsort(dest, stable=True)                         # row -> run
    node_counts = torch.bincount(dest, minlength=n_nodes)            # run histogram

    # ---- the five outputs: all gathers by sigma (in the kernel this is
    #      cursor-ordered placement, the same permutation)
    u_src = torch.div(sigma, groups_m, rounding_mode="floor")        # the run's token
    payload = hidden[u_src]
    mask = (e_sorted % slots).reshape(P, quota)[sigma] if P else \
        torch.zeros(0, quota, dtype=torch.int64, device=dev)
    gate_rows = g_sorted.reshape(P, quota)[sigma] if P else \
        hidden.new_zeros(0, quota)
    return payload, mask, gate_rows, u_src, node_counts
