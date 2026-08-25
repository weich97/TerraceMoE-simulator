import pytest
import torch

from terrace import Router, TRouteConfig, expert_load, t_route, update_bias_
from bench import dispatch_traffic, uniformity_cv

CFG = TRouteConfig(n_experts=64, n_groups=8, top_k=8, top_groups=4)


def _route(n_tok=512, seed=0):
    torch.manual_seed(seed)
    affinity = torch.sigmoid(torch.randn(n_tok, CFG.n_experts))
    return t_route(affinity, torch.zeros(CFG.n_experts), CFG)


def test_structural_guarantees():
    expert_idx, gates, group_idx = _route()
    g = CFG.group_size
    # exactly k distinct experts per token
    assert expert_idx.shape[1] == CFG.top_k
    assert all(len(set(row.tolist())) == CFG.top_k for row in expert_idx)
    # fan-out bound: exactly M distinct groups, consistent with expert ids
    groups_of_experts = expert_idx // g
    for r in range(expert_idx.shape[0]):
        assert set(groups_of_experts[r].tolist()) == set(group_idx[r].tolist())
        assert len(set(group_idx[r].tolist())) == CFG.top_groups
    # quota: exactly k/M experts in each selected group
    counts = torch.zeros(expert_idx.shape[0], CFG.n_groups, dtype=torch.long)
    counts.scatter_add_(1, groups_of_experts, torch.ones_like(groups_of_experts))
    picked = counts.gather(1, group_idx)
    assert (picked == CFG.per_group_k).all()


def test_gates_normalized():
    _, gates, _ = _route()
    assert torch.allclose(gates.sum(-1), torch.ones(gates.shape[0]), atol=1e-5)
    assert (gates > 0).all()


def test_bias_balances_load():
    torch.manual_seed(0)
    router = Router(hidden_size=64, cfg=CFG)
    # skew the router so raw routing is imbalanced
    with torch.no_grad():
        router.weight[: CFG.group_size] += 0.5

    def run(steps, use_bias):
        loads = torch.zeros(CFG.n_experts)
        for _ in range(steps):
            x = torch.randn(2048, 64)
            expert_idx, _, _ = router(x)
            load = expert_load(expert_idx, CFG.n_experts)
            if use_bias:
                update_bias_(router.bias, load, gamma=5e-3)
            loads += load
        return loads

    router.bias.zero_()
    base = run(50, use_bias=False)
    base_cv = (base.std() / base.mean()).item()

    router.bias.zero_()
    run(400, use_bias=True)               # let the bias converge
    settled = run(50, use_bias=True)      # measure after convergence
    cv = (settled.std() / settled.mean()).item()

    assert cv < 0.25, f"expert load CV {cv:.3f} too high"
    assert cv < base_cv, "bias update failed to improve balance"


def test_traffic_matrix_properties():
    torch.manual_seed(1)
    ep, cards_per_group = 64, 8
    n_tok = ep * 128
    affinity = torch.sigmoid(torch.randn(n_tok, CFG.n_experts))
    _, _, group_idx = t_route(affinity, torch.zeros(CFG.n_experts), CFG)
    src_card = torch.arange(n_tok) % ep

    mat = dispatch_traffic(group_idx, src_card, CFG, cards_per_group)
    # dedup invariant: every token contributes exactly M node-level messages
    tokens_per_node = n_tok // CFG.n_groups
    assert torch.allclose(mat.sum(dim=1), torch.full((CFG.n_groups,), float(tokens_per_node * CFG.top_groups)))
    # with unbiased random affinities the matrix should be near-uniform
    assert uniformity_cv(mat) < 0.2


# --- what is architectural vs what is merely statistical --------------------------
# An external reviewer showed that docs/01 claimed equal quota gives STRICT
# inter-group balance, while the algorithm only forces k/M experts inside the M
# groups a token CHOOSES BY AFFINITY -- nothing makes different groups get chosen
# equally often. Every balance test above uses uniform random affinity, under which
# both the strict and the statistical reading look identical, so the overclaim
# survived. These pin the real boundary.

def _group_load(expert_idx, n_groups, group_size):
    grp = (expert_idx // group_size).reshape(-1)
    return torch.bincount(grp, minlength=n_groups).float()


def _cv(x):
    return (x.std(unbiased=False) / x.mean()).item()


def test_group_balance_is_statistical_not_strict():
    """T-Route (M < N_g): adversarial affinity concentrates load on M groups.

    This is the reviewer's counterexample. It is not a bug -- it is the honest
    boundary of the mechanism, and terrace/routing.py's own docstring always said
    "group-level load balance is statistical". Only docs/01 said otherwise.
    """
    n_tok, e, n_g = 2048, 128, 8
    g = e // n_g
    cfg = TRouteConfig(n_experts=e, n_groups=n_g, top_k=8, top_groups=4)
    bias = torch.zeros(e)

    aff = torch.rand(n_tok, e) * 0.01
    aff[:, : 4 * g] += 1.0                      # every token prefers the first 4 groups
    expert_idx, _, _ = t_route(aff, bias, cfg, mode="full")
    load = _group_load(expert_idx, n_g, g)

    assert _cv(load) == pytest.approx(1.0, abs=1e-6), "adversarial load must be maximally skewed"
    assert (load[4:] == 0).all(), "the unfavoured groups receive nothing"
    assert (load[:4] > 0).all()


def test_strict_group_balance_holds_only_when_M_equals_Ng():
    """quota_only (M == N_g) IS strict: every token must touch every group.

    This is MoGE's actual guarantee and the reason quota_only measures group_cv
    exactly 0.0000 on the real reference-operating-point arms while full measures 0.064.
    """
    n_tok, e, n_g = 2048, 128, 8
    cfg = TRouteConfig(n_experts=e, n_groups=n_g, top_k=8, top_groups=n_g)
    bias = torch.zeros(e)

    for aff in (torch.rand(n_tok, e),                       # benign
                torch.rand(n_tok, e) * 0.01 + torch.cat(    # adversarial
                    [torch.ones(n_tok, 4 * (e // n_g)), torch.zeros(n_tok, e - 4 * (e // n_g))], 1)):
        expert_idx, _, _ = t_route(aff, bias, cfg, mode="quota_only")
        load = _group_load(expert_idx, n_g, e // n_g)
        assert _cv(load) == pytest.approx(0.0, abs=1e-9), "M == N_g must be strictly balanced"
        assert (load == n_tok * cfg.top_k / n_g).all()


def test_fan_out_and_message_size_are_unconditional():
    """What T-Route DOES guarantee architecturally, even adversarially: each token
    spans exactly M groups with exactly k/M experts each -- bounded fan-out and
    fixed-size node messages, which is what T-A2A compiles against."""
    n_tok, e, n_g, m = 2048, 128, 8, 4
    g = e // n_g
    cfg = TRouteConfig(n_experts=e, n_groups=n_g, top_k=8, top_groups=m)
    bias = torch.zeros(e)

    aff = torch.rand(n_tok, e) * 0.01
    aff[:, : 4 * g] += 1.0
    expert_idx, _, group_idx = t_route(aff, bias, cfg, mode="full")

    assert group_idx.shape[1] == m                                  # fan-out bound
    assert (torch.tensor([len(set(r.tolist())) for r in group_idx]) == m).all()
    per_group = (expert_idx // g)
    for row in per_group:
        counts = torch.bincount(row, minlength=n_g)
        nz = counts[counts > 0]
        assert (nz == cfg.top_k // m).all(), "fixed k/M per selected group"
