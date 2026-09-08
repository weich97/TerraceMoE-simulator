# -*- coding: utf-8 -*-
"""One hop against two, measured, across a supernode boundary. The falsification test.

## What this is for

This repository has never measured two-hop dispatch beating one-hop. Its seven
end-to-end geometries all ran inside a single supernode where the hierarchy ratio is
1.03 and there is nothing to buy, and six of the seven lost. Every statement that
two-hop pays has been a model output.

Both supernode boundaries are now measured, and at the shallower of them the model
makes a specific, falsifiable prediction at this exact geometry with the arrival chain
that actually exists -- no fused kernel, no hypothetical tier. This measures it.

## What is measured, and against what

The comparison mirrors sim/core.py term for term, so the numbers are directly
commensurable with the model:

    one hop     one all-to-all over every rank, T*k rows
    two hop     Hop A across the boundary carrying T*M rows, of which half stay local;
                the arrival chain; Hop B inside the supernode carrying T*k rows;
                and the splits readback the variable-length exchange cannot avoid

The arrival chain is the repository's own `terrace.ops.k1_arrival_ref`, which its
docstring calls the executable spec of the fused kernel and which its tests hold
bit-identical to the live chain. It is not a per-row estimate: the real gather, the real
stable bucket sort, the real histogram, run on the device at the real shapes.

Groups are supernodes here, so n_groups is 2 and R is the 64 cards of one supernode.
Hop A therefore has exactly one cross-boundary peer per card, which is the narrowest
case there is and the one a separate sweep has already shown costs nothing.

Every phase is timed on its own with the ranks aligned first and the host waiting, which
is the regime an MoE dispatch runs in and the regime the additive model belongs to. The
two-hop total is their sum, which is what sim/core.py assumes and what the call-count
scan independently confirmed: collectives on this machine do not pipeline.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist

TOKENS = 4096
K = 6
EPR = 4                      # experts per card, for the chain's owner arithmetic


def max_over(seconds):
    t = torch.tensor([seconds * 1e6], dtype=torch.float32, device="npu")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item()) * 1e-6


def timed(fn, n):
    ts = []
    for _ in range(n):
        dist.barrier()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append(max_over(time.perf_counter() - t0))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", required=True, help="path holding the terrace package")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--calls", type=int, default=7)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    from terrace.ops import k1_arrival_ref

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    dist.init_process_group(backend="hccl", device_id=torch.device("npu", local))

    half = world // 2                       # cards in one supernode
    side = 0 if rank < half else 1
    counterpart = rank + half if side == 0 else rank - half
    sn_group = dist.new_group(ranks=list(range(half)))
    sn_other = dist.new_group(ranks=list(range(half, world)))
    mine = sn_group if side == 0 else sn_other

    rows = []
    for H in (2048, 4096):
        for M in (1, 2):
            q = K // M
            one_rows = TOKENS * K
            a_rows = TOKENS * M

            # --- one hop: everything crosses in a single collective
            oi = torch.ones(one_rows * H, dtype=torch.bfloat16, device="npu")
            oo = torch.empty_like(oi)
            f_one = lambda: dist.all_to_all_single(oo, oi)

            # --- hop A: T*M rows, half local, half to the one cross-boundary peer
            per = a_rows // 2 * H
            sa = [0] * world; ra = [0] * world
            sa[rank] = per; sa[counterpart] = per
            ra[rank] = per; ra[counterpart] = per
            ai = torch.ones(2 * per, dtype=torch.bfloat16, device="npu")
            ao = torch.empty_like(ai)
            f_a = lambda: dist.all_to_all_single(ao, ai, ra, sa)

            # --- the real arrival chain, at the real shapes
            rx = torch.randn(a_rows, H, dtype=torch.bfloat16, device="npu")
            rslot = torch.randint(0, half * EPR, (a_rows, q), device="npu")
            rgate = torch.randn(a_rows, q, dtype=torch.bfloat16, device="npu")
            f_chain = lambda: k1_arrival_ref(rx, rslot, rgate, quota=q,
                                             epr=EPR, rpn=half)

            # --- hop B: T*k rows inside the supernode
            bi = torch.ones(one_rows * H // half * half, dtype=torch.bfloat16,
                            device="npu")
            bo = torch.empty_like(bi)
            f_b = lambda: dist.all_to_all_single(bo, bi, group=mine)

            # --- the splits readback the variable-length exchange cannot avoid
            cnt = torch.zeros(half, dtype=torch.int32, device="npu")
            def f_splits():
                cnt.cpu()

            for f in (f_one, f_a, f_chain, f_b):
                for _ in range(3):
                    f()
            torch.npu.synchronize()

            t = {}
            for name, f in (("one_hop", f_one), ("hop_a", f_a), ("chain", f_chain),
                            ("hop_b", f_b), ("splits", f_splits)):
                t[name] = statistics.median(
                    [timed(f, args.calls) for _ in range(args.repeats)])
            two = t["hop_a"] + t["chain"] + t["hop_b"] + t["splits"]
            g = t["one_hop"] / two
            row = {"H": H, "M": M, "q": q, "tokens": TOKENS, "k": K,
                   "one_hop_ms": t["one_hop"] * 1e3, "hop_a_ms": t["hop_a"] * 1e3,
                   "chain_ms": t["chain"] * 1e3, "hop_b_ms": t["hop_b"] * 1e3,
                   "splits_ms": t["splits"] * 1e3, "two_hop_ms": two * 1e3, "G": g}
            rows.append(row)
            if rank == 0:
                print("H=%5d M=%d q=%d | one-hop %7.3f | hopA %6.3f chain %6.3f "
                      "hopB %6.3f splits %5.3f = two-hop %7.3f | G = %.3f  %s"
                      % (H, M, q, row["one_hop_ms"], row["hop_a_ms"], row["chain_ms"],
                         row["hop_b_ms"], row["splits_ms"], row["two_hop_ms"], g,
                         "TWO-HOP WINS" if g > 1 else "two-hop loses"), flush=True)
            del oi, oo, ai, ao, rx, rslot, rgate, bi, bo
            torch.npu.empty_cache()

    if rank == 0:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"tag": args.tag, "world": world, "rows": rows}, fh, indent=1)
        print("wrote %s" % args.out, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
