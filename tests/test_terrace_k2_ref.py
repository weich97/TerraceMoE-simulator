"""K2 (send-side fused pack chain) CPU reference vs the live chain: bitwise reconciliation.

k2_pack_ref (terrace/ops/k2_ref.py) is a direct transcription of the AscendC
kernel's two-pass counting sort (an independent implementation that reuses no
live-chain primitives); the live-chain arm is the **verbatim composition** of the
quota fast path's send stage: plan_ta2a(groups_m=M) + hidden[u_src] deduplicated
gather + _pack_quota_wire. All five outputs of the two arms (payload / mask slot
table / gate_rows / u_src / node_counts) must be torch.equal (bitwise, no
tolerance) -- this is the CPU-side foundation for machine-level acceptance of the
K2 kernel: the kernel and k2_pack_ref are the same mathematical object (argument
in the kernel file header), and k2_pack_ref is nailed to the live chain here, so
by transitivity kernel == live chain.

Coverage axes (>=8 cases from the Cartesian product of geometry x row order x dtype):
  - geometry: slots 4/16/24/32 (spanning build_expansion's 24-bit float32 mask
    boundary -- irrelevant to the C1 slot table itself, but keeps the geometry
    spectrum consistent with the quota_wire regression testbed), quota 1 (k==M,
    slot table width 1), M=1 (single-node whole pack), T=1, odd T / quota=2;
  - row order: ascending (seam entry; routing_map_to_topk is ascending by
    construction) and shuffled (fused-forward entry; the pack side's in-row
    argsort branch). For ascending rows the live-chain arm takes the sort-free
    sorted_rows=True branch, for shuffled rows the argsort branch with =False --
    the single-path reference implementation must be bitwise equal to both at
    once;
  - dtype: fp32 / bf16 / fp16 (the gate plane is a pure bit move; not one dtype
    may drift).

Two more contracts nailed here: a gates/hidden dtype mismatch must die loudly in
both arms (the C1 rounding-point contract; the entry point of a drift defect in
one internal commit); ascending order within each slot-table row is a production
contract (the arrival side no longer sorts), asserted on the reference arm on its
own, not merely via coincidental agreement with the live chain.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terrace.ops.k2_ref import k2_pack_ref
from terrace.ta2a import plan_ta2a
from terrace.ta2a_fwd import _pack_quota_wire


def _chain(hidden, expert_idx, gates, world, n_experts, rpn, groups_m, sorted_rows):
    """Verbatim composition of the live chain (line-for-line isomorphic to the
    quota branch of ta2a_moe_forward / ta2a_permute)."""
    u_src, _, node_counts, inverse = plan_ta2a(expert_idx, world, n_experts, rpn,
                                               groups_m=groups_m)
    n_rows = u_src.numel()
    payload = hidden[u_src]
    slots = (n_experts // world) * rpn
    quota = expert_idx.shape[1] // groups_m
    mask, gate_rows = _pack_quota_wire(expert_idx, gates, inverse, payload,
                                       n_rows, slots, quota, n_experts,
                                       sorted_rows=sorted_rows)
    return payload, mask, gate_rows, u_src, node_counts


def _equal_quota(T, k, n_experts, n_nodes, m, seed, sort_rows):
    """T-Route equal-quota routing: exactly m nodes per token, exactly k//m
    experts per node."""
    g = torch.Generator().manual_seed(seed)
    per, quota = n_experts // n_nodes, k // m
    rows = []
    for _ in range(T):
        nodes = torch.randperm(n_nodes, generator=g)[:m]
        rows.append(torch.cat([torch.arange(a * per, (a + 1) * per)[
            torch.randperm(per, generator=g)[:quota]] for a in nodes]))
    idx = torch.stack(rows)
    return torch.sort(idx, dim=1).values if sort_rows else idx


# (world, rpn, n_experts, T, k, M) -- slots = (E//world)*rpn noted at end of line.
GEOMETRIES = [
    (4, 2, 8, 8, 4, 2),        # slots 4, same geometry as the distribution regression testbed
    (32, 8, 64, 48, 4, 2),     # slots 16, matches the testbed geometry
    (32, 8, 96, 32, 8, 4),     # slots 24, the last width for the f32 mask
    (32, 8, 128, 32, 8, 2),    # slots 32, geometry of the old mask's exact int64 path
    (16, 8, 32, 5, 2, 2),      # quota 1: k==M, slot table width 1
    (16, 4, 16, 12, 4, 1),     # M=1: single-node whole pack, 4 nodes
    (8, 4, 8, 1, 2, 2),        # T=1
    (64, 8, 128, 33, 6, 3),    # odd T, quota 2
]


@pytest.mark.parametrize("world,rpn,E,T,k,m", GEOMETRIES)
@pytest.mark.parametrize("sort_rows", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_k2_ref_bitwise_equals_chain(world, rpn, E, T, k, m, sort_rows, dtype):
    n_nodes = world // rpn
    quota = k // m
    for seed in range(2):
        idx = _equal_quota(T, k, E, n_nodes, m, 500 + seed, sort_rows)
        g = torch.Generator().manual_seed(seed)
        hidden = torch.randn(T, 16, generator=g).to(dtype)
        gates = torch.rand(T, k, generator=g).to(dtype)

        ref = k2_pack_ref(hidden, idx, gates, world, E, rpn, m)
        chain = _chain(hidden, idx, gates, world, E, rpn, m,
                       sorted_rows=sort_rows)

        names = ("payload", "mask", "gate_rows", "u_src", "node_counts")
        for name, a, b in zip(names, ref, chain):
            assert a.dtype == b.dtype, f"{name}: dtype {a.dtype} != {b.dtype}"
            assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
            assert torch.equal(a, b), (
                f"{name} reference implementation not bitwise equal to the live chain "
                f"(geom w{world}/rpn{rpn}/E{E}/T{T}/k{k}/M{m}, "
                f"sorted={sort_rows}, {dtype}, seed={seed})")
        # Shape/staticness: n_rows = T*M is what buys K2 its freedom from synchronization
        assert ref[0].shape == (T * m, 16)
        assert ref[1].shape == (T * m, quota) and ref[1].dtype == torch.int64
        # Ascending order within each slot-table row is the C1 production contract
        # (the arrival side no longer sorts); nail it separately
        assert torch.equal(ref[1], torch.sort(ref[1], dim=1).values)
        # Count conservation: per-node row counts sum to the number of rows sent
        assert int(ref[4].sum()) == T * m


def test_k2_ref_gradients_flow_like_the_chain():
    """The reference implementation is a pure gather composition, so autograd's
    adjoint (index_add / scatter) IS the live chain's backward -- the grafted
    TerraceK2PackFn.backward uses the same set of primitives (internal grafting
    records (not shipped with this repo)). Nailed here: payload/gate_rows carry
    gradient, and the gradients match the live-chain arm bitwise."""
    world, rpn, E, T, k, m = 4, 2, 8, 8, 4, 2
    idx = _equal_quota(T, k, E, world // rpn, m, 700, False)
    g = torch.Generator().manual_seed(3)
    hidden_r = torch.randn(T, 16, generator=g).requires_grad_(True)
    gates_r = torch.rand(T, k, generator=g).requires_grad_(True)
    hidden_c = hidden_r.detach().clone().requires_grad_(True)
    gates_c = gates_r.detach().clone().requires_grad_(True)

    pr, _, gr, _, _ = k2_pack_ref(hidden_r, idx, gates_r, world, E, rpn, m)
    pc, _, gc, _, _ = _chain(hidden_c, idx, gates_c, world, E, rpn, m,
                             sorted_rows=False)
    gp = torch.randn(pr.shape, generator=g)
    gg = torch.randn(gr.shape, generator=g)
    (pr * gp).sum().backward(retain_graph=True)
    (gr * gg).sum().backward()
    (pc * gp).sum().backward(retain_graph=True)
    (gc * gg).sum().backward()
    assert torch.equal(hidden_r.grad, hidden_c.grad)
    assert torch.equal(gates_r.grad, gates_c.grad)


def test_k2_ref_keeps_dtype_mismatch_loud():
    """A gates/hidden dtype mismatch must die loudly -- same failure shape in the
    reference arm and the live-chain arm (_pack_quota_wire hits RuntimeError at
    index_put; the gate plane derives from payload -- deriving it from gates would
    silently ship the wider gate plane into production, the entry point of one
    internal commit)."""
    world, rpn, E, T, k, m = 4, 2, 8, 8, 4, 2
    idx = _equal_quota(T, k, E, world // rpn, m, 11, True)
    hidden = torch.randn(T, 16, dtype=torch.bfloat16)
    gates = torch.rand(T, k)                                  # fp32: mismatched
    with pytest.raises(RuntimeError):
        k2_pack_ref(hidden, idx, gates, world, E, rpn, m)
    with pytest.raises(RuntimeError):
        _chain(hidden, idx, gates, world, E, rpn, m, sorted_rows=True)


def test_k2_ref_rejects_out_of_range_expert_ids():
    """Out-of-range expert ids: the live chain dies loudly at the plan's scatter,
    and the reference implementation fails just as loud (the kernel side cannot
    raise, and substitutes skip-the-write + zeros containment with no bitwise
    promise -- see the kernel file header; the reference implementation is the
    spec, and the spec must reject)."""
    world, rpn, E, T, k, m = 4, 2, 8, 4, 4, 2
    idx = _equal_quota(T, k, E, world // rpn, m, 13, True)
    idx[0, 0] = E                                             # out of range
    hidden = torch.randn(T, 16)
    gates = torch.rand(T, k)
    with pytest.raises(ValueError, match="out of range"):
        k2_pack_ref(hidden, idx, gates, world, E, rpn, m)
