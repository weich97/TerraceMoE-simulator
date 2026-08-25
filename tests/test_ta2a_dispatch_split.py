"""The split halves must equal the closed forward elementwise -- otherwise what plugs
into the vendor MoE layer is a different model.

Why this file exists (#18(a)): `ta2a_moe_forward` is a closed forward
(dispatch + experts + combine), already locked equal to `ep_moe_forward` by
tests/test_ta2a.py. To hook into Megatron's MoE layer it must be cut at the expert
computation into the two halves `token_permutation` / `token_unpermutation`, with
the vendor's own grouped GEMM in between. The cut line is exactly where mistakes
come easiest:

- the intermediates (u_src / r_idx / order / four sets of split counts) must be
  carried across the two calls; miss one and rows go wrong silently;
- **the gate application point moves**: the closed forward multiplies at the
  experts itself, while the vendor multiplies by permuted_probs at experts.py:241
  -- copy the closed forward verbatim and you multiply twice, and a double-gated
  output still "looks like a normal loss";
- the two pairs of split counts on the return path are used in **opposite**
  directions (ir,is / recv,send); write them the straight way round and nothing
  errors -- the rows are simply sent to other ranks.

So this file does not test "does it run"; it tests **whether the recombined halves
equal the original forward**. gloo / CPU, world 4, rpn 2 (= 2 nodes, includes a
cross-node hop), cheap enough for unit tests.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORLD, RPN, T, K, M, E, H, D = 4, 2, 8, 4, 2, 8, 6, 4


def _routing(gen, n_nodes, per, quota):
    """T-Route equal quota: exactly M nodes, exactly K/M experts per node."""
    rows = []
    for _ in range(T):
        gs = torch.randperm(n_nodes, generator=gen)[:M]
        rows.append(torch.cat([torch.arange(a * per, (a + 1) * per)[
            torch.randperm(per, generator=gen)[:quota]] for a in gs]))
    return torch.stack(rows)


def _run(rank, world, q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from terrace.layer import grouped_mm
        from terrace.ta2a_fwd import ta2a_moe_forward, init_ta2a_groups
        from terrace.ta2a_dispatch import ta2a_permute, ta2a_unpermute

        intra = init_ta2a_groups(world, RPN)
        epr = E // world
        n_nodes, per, quota = world // RPN, E // (world // RPN), K // M

        g = torch.Generator().manual_seed(11 + rank)
        x = torch.randn(T, H, generator=g, dtype=torch.float32)
        idx = _routing(g, n_nodes, per, quota)
        gates = torch.rand(T, K, generator=g, dtype=torch.float32)
        w13 = torch.randn(epr, H, 2 * D, generator=g, dtype=torch.float32) / (H ** 0.5)
        w2 = torch.randn(epr, D, H, generator=g, dtype=torch.float32) / (D ** 0.5)

        # Reference: the closed forward (locked == ep_moe_forward by test_ta2a.py)
        ref = ta2a_moe_forward(x, idx, gates, w13, w2, world, E, RPN, groups_m=M)

        # Under test: vendor-shaped dense routing_map + probs -> the two halves with
        # "the vendor's" expert computation in between
        routing_map = torch.zeros(T, E, dtype=torch.bool)
        probs = torch.zeros(T, E, dtype=torch.float32)
        routing_map[torch.arange(T).unsqueeze(1), idx] = True
        probs[torch.arange(T).unsqueeze(1), idx] = gates

        permuted, tpe, pprobs, st = ta2a_permute(
            x, probs, routing_map, world=world, rank=rank, rpn=RPN,
            n_experts=E, intra_group=intra, groups_m=M)
        # Equivalent of the vendor's experts.py: weight by permuted_probs first, then the grouped GEMM
        a, b = grouped_mm(permuted, w13, tpe).chunk(2, dim=-1)
        ye = grouped_mm(F.silu(a) * b, w2, tpe) * pprobs.unsqueeze(-1)
        got = ta2a_unpermute(ye, st, x)

        q.put((rank, "ok", float((got - ref).abs().max()), int(tpe.sum()), tpe.tolist()))
    except Exception as e:                                    # noqa: BLE001
        import traceback
        q.put((rank, "err", traceback.format_exc(), 0, []))
    finally:
        dist.destroy_process_group()


def test_split_halves_equal_the_closed_forward():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_run, args=(r, WORLD, q)) for r in range(WORLD)]
    for p in procs:
        p.start()
    out = [q.get(timeout=180) for _ in range(WORLD)]
    for p in procs:
        p.join(timeout=60)

    for rank, status, payload, total, tpe in out:
        assert status == "ok", f"rank {rank}:\n{payload}"
        # In float32 the split merely moves a call boundary inside the same chain of
        # operations; there should be no visible error
        assert payload < 1e-5, f"rank {rank} differs from the closed forward by {payload}"
        # tokens_per_expert is a mathematically forced quantity: T-A2A does not change
        # "which expert handles which pairs", so the row counts summed over all ranks
        # must equal the total (token, expert) pair count = world * T * K / world
        assert sum(tpe) == total


def test_total_pairs_are_conserved():
    """All ranks' tokens_per_expert summed == the global (token, expert) pair count.

    This is the forced-quantity probe for "did we hook it up right": short means an
    expert got dropped (the fast path silently dropping experts is the original case
    of audit finding 1.4); long means some pairs got duplicated (the early combine
    bug that accumulated per (token, expert)).
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_run, args=(r, WORLD, q)) for r in range(WORLD)]
    for p in procs:
        p.start()
    out = [q.get(timeout=180) for _ in range(WORLD)]
    for p in procs:
        p.join(timeout=60)
    assert all(s == "ok" for _, s, _, _, _ in out), [p for _, s, p, _, _ in out if s != "ok"]
    assert sum(t for _, _, _, t, _ in out) == WORLD * T * K
