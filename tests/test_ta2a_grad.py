"""T-A2A must be equivalent in the BACKWARD pass too, not just the forward.

Every T-A2A test so far compares outputs. That is exactly the check that cannot see the
failure this file exists to catch: a dispatch schedule can reproduce the forward bit for bit
and still deliver the wrong gradients -- or none -- and the symptom is not a crash or a bad
number, it is a router that quietly stops learning. This codebase has already met that shape
of bug twice (`npu_grouped_matmul` not propagating gradients on this stack; the vendor
routing branch dropping group kwargs), and both times the forward looked perfect.

The specific hazard here is the id plane. T-A2A ships the per-(row, slot) GATE alongside the
payload so the gate can be applied at the expert rather than at the origin, and it ships it
with `_a2a_raw` -- a bare `all_to_all_single` into a `new_empty` buffer, which has no
autograd node at all. Masks and slot indices legitimately need no gradient. The gate is not
like them: it is the router's output, and in an MoE the router learns through it.

Run with gloo on CPU so this is a unit test rather than a fleet job: world 4, rpn 2, so
n_nodes = 2 and the fabric hop is real (n_nodes < 2 would short-circuit to the baseline).
"""
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORLD, RPN, T, K, M, E, H, D = 4, 2, 8, 4, 2, 8, 6, 4


def _run(rank, world, q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from terrace.ep_dist import ep_moe_forward
        from terrace.ta2a_fwd import ta2a_moe_forward, init_ta2a_groups

        init_ta2a_groups(world, RPN)
        torch.manual_seed(100 + rank)
        epr = E // world

        # Same inputs for both paths; gradients compared, not just outputs.
        def fresh():
            g = torch.Generator().manual_seed(7 + rank)
            x = torch.randn(T, H, generator=g, dtype=torch.float32, requires_grad=True)
            # T-Route routing: exactly M nodes per token, exactly K/M experts on each.
            n_nodes, per, quota = world // RPN, E // (world // RPN), K // M
            rows = []
            for _ in range(T):
                gs = torch.randperm(n_nodes, generator=g)[:M]
                rows.append(torch.cat([torch.arange(a * per, (a + 1) * per)[
                    torch.randperm(per, generator=g)[:quota]] for a in gs]))
            idx = torch.stack(rows)
            gates = torch.rand(T, K, generator=g, dtype=torch.float32, requires_grad=True)
            w13 = torch.randn(epr, H, 2 * D, generator=g, dtype=torch.float32,
                              requires_grad=True) / (H ** 0.5)
            w2 = torch.randn(epr, D, H, generator=g, dtype=torch.float32,
                             requires_grad=True) / (D ** 0.5)
            w13.retain_grad(); w2.retain_grad()
            return x, idx, gates, w13, w2

        xb, idxb, gb, w13b, w2b = fresh()
        yb = ep_moe_forward(xb, idxb, gb, w13b, w2b, world, E)
        yb.sum().backward()

        xt, idxt, gt, w13t, w2t = fresh()
        yt = ta2a_moe_forward(xt, idxt, gt, w13t, w2t, world, E, RPN, groups_m=M)
        yt.sum().backward()

        def cmp(a, b):
            if a is None or b is None:
                return dict(present=(a is not None, b is not None), maxdiff=None)
            return dict(present=(True, True),
                        maxdiff=float((a - b).abs().max()),
                        scale=float(a.abs().max()))

        q.put({
            "rank": rank,
            "fwd_maxdiff": float((yb - yt).abs().max()),
            "x": cmp(xb.grad, xt.grad),
            "gates": cmp(gb.grad, gt.grad),
            "w13": cmp(w13b.grad, w13t.grad),
            "w2": cmp(w2b.grad, w2t.grad),
        })
    except Exception as e:                                   # noqa: BLE001
        q.put({"rank": rank, "error": f"{type(e).__name__}: {e}"})
    finally:
        dist.destroy_process_group()


def _collect():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_run, args=(r, WORLD, q)) for r in range(WORLD)]
    for p in procs:
        p.start()
    out = [q.get(timeout=180) for _ in range(WORLD)]
    for p in procs:
        p.join(timeout=30)
    return out


@pytest.mark.timeout(300)
def test_ta2a_backward_matches_baseline():
    """Forward equality is not enough: every gradient must match the baseline's.

    `gates` is the one to watch. If its gradient is absent while the baseline has one, the
    router is being trained on nothing -- silently, with a perfect forward.
    """
    res = _collect()
    errs = [r for r in res if "error" in r]
    if errs:
        pytest.skip(f"distributed CPU run unavailable: {errs[0]['error']}")

    for r in res:
        assert r["fwd_maxdiff"] < 1e-4, f"rank {r['rank']} forward differs: {r['fwd_maxdiff']}"

    for name in ("x", "gates", "w13", "w2"):
        for r in res:
            c = r[name]
            base_has, ta2a_has = c["present"]
            assert base_has, f"baseline produced no {name} gradient -- test fixture is wrong"
            assert ta2a_has, (
                f"rank {r['rank']}: T-A2A produced NO gradient for `{name}` while the "
                f"baseline did. The forward is identical, so nothing else would reveal "
                f"this; with `gates` it means the router receives no signal from the "
                f"expert path and silently stops learning.")
            tol = 1e-4 * max(c["scale"], 1.0)
            assert c["maxdiff"] <= tol, (
                f"rank {r['rank']}: {name} gradient differs by {c['maxdiff']:.3e} "
                f"(tolerance {tol:.3e}, scale {c['scale']:.3e})")
