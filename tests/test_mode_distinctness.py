"""The four ablation modes must produce DISTINGUISHABLE routings, and unknown modes must crash.

Why this file exists (2026-08-11 repo audit, finding 2.8): with `mode` hard-wired to one
value inside `t_route` -- the shape of the 某上游分支的路由参数静默丢弃事故 incident, where the vendor
branch silently dropped the group-routing args and 48 arms of "four modes, no difference"
nearly shipped as a false-positive G2 (内部工程记录) -- the ENTIRE test suite stayed
green: G1 asserted only "k distinct experts per token", G2/G3 used full only, and the
backend tests compare two backends within one mode. Nothing asserted the modes differ.

These tests pin the discriminating signatures at real geometry (E=128, Ng=8, k=8, M=4):

  group span (distinct groups touched per token) is a MATHEMATICAL invariant per mode:
    full          == M exactly        (group-limit + equal quota)
    quota_only    == Ng exactly       (every group, k/Ng each -- MoGE)
    group_limited <= M                (free top-k within the M chosen groups)
    global_topk   unconstrained       (empirically ~5.4 at this geometry, and must be
                                       able to EXCEED M, else it is not unconstrained)

Chosen over "expert_idx differs pairwise" because spans cannot collide by luck: they are
structural, not sampled. A regression that collapses the mode dispatch (e.g. the `else`
branch swallowing every mode) breaks at least two of these immediately.
"""
from __future__ import annotations

import pytest
import torch

from terrace.routing import TRouteConfig, t_route

E, NG, K, M = 128, 8, 8, 4
T = 2048


@pytest.fixture(scope="module")
def routed():
    torch.manual_seed(0)
    cfg = TRouteConfig(n_experts=E, n_groups=NG, top_k=K, top_groups=M)
    aff = torch.rand(T, E) * 0.998 + 1e-3
    bias = torch.zeros(E)
    out = {}
    for mode in ("full", "group_limited", "quota_only", "global_topk"):
        idx, gates, _ = t_route(aff, bias, cfg, mode=mode)
        out[mode] = idx
    return out


def spans(idx: torch.Tensor) -> torch.Tensor:
    """Distinct groups touched per token."""
    g = idx // (E // NG)
    return torch.tensor([int(torch.unique(row).numel()) for row in g])


def test_full_span_is_exactly_m(routed):
    s = spans(routed["full"])
    assert bool((s == M).all()), f"full must touch exactly M={M} groups; got spans {s.unique().tolist()}"


def test_quota_only_span_is_exactly_ng(routed):
    s = spans(routed["quota_only"])
    assert bool((s == NG).all()), f"quota_only (MoGE) must touch all {NG} groups; got {s.unique().tolist()}"


def test_group_limited_span_at_most_m(routed):
    s = spans(routed["group_limited"])
    assert bool((s <= M).all()), f"group_limited must stay within M={M} groups; got max {int(s.max())}"


def test_global_topk_is_actually_unconstrained(routed):
    # If the unconstrained baseline never exceeds M groups, it is not unconstrained --
    # which is exactly what the vendor-branch collapse looked like.
    s = spans(routed["global_topk"])
    assert int(s.max()) > M, (
        f"global_topk never exceeded M={M} groups over {T} tokens -- the four modes have "
        f"collapsed (某上游分支 incident signature, 内部工程记录:)")


def test_modes_differ_pairwise(routed):
    modes = list(routed)
    diff_frac = {}
    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            a = torch.sort(routed[modes[i]], dim=1).values
            b = torch.sort(routed[modes[j]], dim=1).values
            frac = float((a != b).any(dim=1).float().mean())
            diff_frac[(modes[i], modes[j])] = frac
            assert frac > 0.5, (
                f"{modes[i]} and {modes[j]} pick identical experts on "
                f"{100 * (1 - frac):.1f}% of tokens -- modes are not distinguishable")


def test_unknown_mode_crashes():
    cfg = TRouteConfig(n_experts=E, n_groups=NG, top_k=K, top_groups=M)
    aff = torch.rand(4, E)
    bias = torch.zeros(E)
    for bad in ("Full", "quota", "", "toplevel"):
        with pytest.raises(ValueError, match="unknown routing mode"):
            t_route(aff, bias, cfg, mode=bad)
