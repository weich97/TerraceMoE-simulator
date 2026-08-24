"""plan_ta2a must refuse routings that satisfy fan-out == M but violate equal quota.

Why this file exists (2026-08-11 repo audit, finding 1.4): the structured fast path in
`build_expansion` reshapes a token's k destination slots into [M, k/M] runs, which is only
correct when every touched node holds EXACTLY k/M of the token's experts. The guard in
`plan_ta2a` used to verify only the weaker property "every token touches exactly M nodes" --
which is also precisely the precondition its docstring offered callers. The gap between the
two properties is a silent-corruption zone, reproduced concretely before fixing:

    k=4, M=2, experts split 3/1 across two nodes (fan-out exactly 2 -- old guard passes):
    reference mask 16 set bits, fast path 8; the second node's rows come out ZERO and the
    first node's mask carries a bit belonging to the OTHER node's expert
    (0b1101 expected, 0b1010 produced -- wrong experts activated, not merely dropped).

Reachable with the repo's own router: t_route(mode="group_limited") has exact fan-out but no
quota. Equal-quota modes ("full", "quota_only") are what guarantee the invariant.

The pre-existing tests/test_ta2a.py could not catch this: its fixture asserts k % m == 0 and
samples exactly k/m experts per group -- structurally unable to violate the quota. These
tests construct the violating inputs directly.
"""
from __future__ import annotations

import pytest
import torch

import terrace.ta2a as ta2a


def fresh_guard():
    """The guard verifies on the first call per (T, k, n_nodes, M) and every 256th after;
    clear its memory so each test exercises the verification path, not the cache."""
    ta2a._VERIFIED.clear()
    ta2a._VERIFY_COUNT.clear()


# world=4 ranks, rpn=2 -> 2 nodes; n_experts=8 -> 2 experts/rank, 4 slots/node.
# Node 0 owns e0..3, node 1 owns e4..7.
W, RPN, E = 4, 2, 8


def test_quota_violation_with_exact_fanout_is_refused():
    # THE regression: 3/1 split, fan-out exactly M=2. The old guard passed this and the
    # fast path corrupted the mask silently.
    fresh_guard()
    idx = torch.tensor([[0, 2, 3, 5]] * 4)          # node0 x3 + node1 x1
    with pytest.raises(ValueError, match="quota"):
        ta2a.plan_ta2a(idx, world=W, n_experts=E, rpn=RPN, groups_m=2)


def test_equal_quota_passes_and_row_count_is_exact():
    fresh_guard()
    idx = torch.tensor([[0, 2, 4, 6]] * 4)          # 2/2 split: the invariant holds
    u_src, u_node, counts, inverse = ta2a.plan_ta2a(idx, world=W, n_experts=E, rpn=RPN, groups_m=2)
    assert u_src.numel() == 4 * 2
    assert int(counts.sum()) == 4 * 2


def test_k_not_divisible_by_m_is_refused():
    # Floor quota would silently drop k % M experts per token on the arrival side.
    fresh_guard()
    idx = torch.tensor([[0, 2, 4]] * 4)             # k=3, M=2
    with pytest.raises(ValueError, match="divide"):
        ta2a.plan_ta2a(idx, world=W, n_experts=E, rpn=RPN, groups_m=2)


def test_fanout_violation_still_refused():
    # The original (weaker) check must keep working: fan-out != M.
    fresh_guard()
    idx = torch.tensor([[0, 1, 2, 3]] * 4)          # all four experts on node 0: fan-out 1
    with pytest.raises(ValueError, match="fan-out"):
        ta2a.plan_ta2a(idx, world=W, n_experts=E, rpn=RPN, groups_m=2)


def test_fast_path_equals_scatter_when_invariant_holds():
    # With the invariant satisfied, the structured path and the scatter path must agree
    # bit-for-bit -- this is the equivalence the performance claim rests on.
    from terrace.ta2a_fwd import build_expansion
    fresh_guard()
    torch.manual_seed(7)
    T, k, M = 16, 4, 2
    # Sample exactly k/M experts from each of M distinct nodes, random slots.
    rows = []
    for _ in range(T):
        picks = []
        for node in torch.randperm(2)[:M].tolist():
            base = node * 4                          # 4 expert slots per node
            picks += (base + torch.randperm(4)[: k // M]).tolist()
        rows.append(picks)
    idx = torch.tensor(rows)
    u_src, u_node, counts, inverse = ta2a.plan_ta2a(idx, world=W, n_experts=E, rpn=RPN, groups_m=M)
    n_rows = u_src.numel()
    m_fast = build_expansion(idx, inverse, n_rows, W, E, rpn=RPN, groups_m=M)
    m_ref = build_expansion(idx, inverse, n_rows, W, E, rpn=RPN, groups_m=None)
    assert torch.equal(m_fast, m_ref), "structured path diverged from scatter under a valid routing"


def test_group_limited_router_output_is_refused_not_corrupted():
    # End-to-end with the repo's own router: group_limited yields exact fan-out without
    # quota. Passing groups_m for it must be a loud refusal, never a wrong answer.
    from terrace.routing import TRouteConfig, t_route
    fresh_guard()
    torch.manual_seed(0)
    T, E_big, Ng, k, M = 256, 64, 8, 4, 2
    cfg = TRouteConfig(n_experts=E_big, n_groups=Ng, top_k=k, top_groups=M)
    affinity = torch.rand(T, E_big) * 0.998 + 1e-3          # sigmoid-range affinities
    bias = torch.zeros(E_big)
    idx, gates, group_idx = t_route(affinity, bias, cfg, mode="group_limited")
    # world such that one node == one group: 64 experts / 8 nodes, rpn=1, epr=8
    epr = E_big // 8
    dest = idx // (epr * 1)
    per = torch.zeros(T, 8).scatter_add_(1, dest, torch.ones(T, k))
    violates = bool((per[per > 0] != k / M).any())
    if not violates:
        pytest.skip("sampled batch happened to be equal-quota; nothing to refuse")
    with pytest.raises(ValueError):
        ta2a.plan_ta2a(idx, world=8, n_experts=E_big, rpn=1, groups_m=M)
