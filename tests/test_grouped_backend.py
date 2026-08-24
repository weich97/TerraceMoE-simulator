import torch

from terrace.layer import TerracedMoE, grouped_mm
from terrace.routing import TRouteConfig


def test_grouped_mm_matches_dense_blockdiag():
    torch.manual_seed(0)
    E, n_in, n_out = 4, 8, 6
    counts = torch.tensor([3, 0, 5, 2])          # includes an empty group
    N = int(counts.sum())
    x = torch.randn(N, n_in)
    w = torch.randn(E, n_in, n_out)
    out = grouped_mm(x, w, counts)
    # reference: each row multiplied by its group's weight
    ref = torch.empty(N, n_out)
    off = 0
    for g in range(E):
        n = int(counts[g])
        ref[off:off + n] = x[off:off + n] @ w[g]
        off += n
    assert torch.allclose(out, ref, atol=1e-5)


def test_backends_agree():
    torch.manual_seed(1)
    cfg = TRouteConfig(n_experts=16, n_groups=4, top_k=4, top_groups=2)
    layer = TerracedMoE(hidden=32, d_expert=16, d_shared=24, cfg=cfg)
    layer.eval()                                  # freeze bias update for determinism
    x = torch.randn(40, 32)
    for mode in ["full", "group_limited", "quota_only", "global_topk"]:
        layer.routing_mode = mode
        layer.expert_backend = "loop"
        y_loop, s_loop = layer(x)
        layer.expert_backend = "grouped"
        y_grp, s_grp = layer(x)
        assert torch.allclose(y_loop, y_grp, atol=1e-5), f"backend mismatch in {mode}"
        assert torch.equal(s_loop["expert_load"], s_grp["expert_load"])


def test_backends_agree_backward():
    torch.manual_seed(2)
    cfg = TRouteConfig(n_experts=8, n_groups=4, top_k=4, top_groups=2)
    x = torch.randn(24, 32)

    def run(backend):
        torch.manual_seed(9)
        layer = TerracedMoE(hidden=32, d_expert=16, d_shared=24, cfg=cfg)
        layer.expert_backend = backend
        y, _ = layer(x)
        y.sum().backward()
        return y.detach(), layer.w13.grad.clone(), layer.w2.grad.clone()

    y1, g13_1, g2_1 = run("loop")
    y2, g13_2, g2_2 = run("grouped")
    assert torch.allclose(y1, y2, atol=1e-5)
    assert torch.allclose(g13_1, g13_2, atol=1e-4)
    assert torch.allclose(g2_1, g2_2, atol=1e-4)
