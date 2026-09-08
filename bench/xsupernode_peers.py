# -*- coding: utf-8 -*-
"""Cross-supernode bandwidth against how many peers each card spreads it over.

## Why this is the measurement that matters

`sim/tiers.py` found that a tier delivers more per card when a collective spreads its
bytes over more peers: cross-supernode 13.3 GB/s at 8 peers, 20.0 at 64. Both points come
from all-to-alls that also carry intra-supernode traffic, so both rest on a deconvolution,
and neither reaches the regime that matters most. Hop A of a two-hop dispatch runs at
world = group count, so it has the *fewest* cross-supernode peers of anything in the scheme --
one peer on a two-supernode machine. That is below everything measured, it is unpriced, and
it charges two-hop while crediting one-hop. It is the open risk against the supernode-boundary
verdict.

## The design

No subgroups and no deconvolution. One process group over all ranks, and
`all_to_all_single` with explicit split sizes so that **every card sends only across the
boundary**, to exactly `n` peers, and receives from exactly `n`. The measured time is
therefore purely cross-supernode: bandwidth is (bytes sent) / time with nothing to subtract.

Every card sends the same total, B/2, at every `n`; only the number of peers it is
divided among changes. So the aggregate load on the boundary is identical across the
sweep and the only variable is spread. Per-peer message size necessarily falls as `n`
rises, and that is not a confound to remove -- it is the same trade Hop A faces.

The pairing is a rotation, which keeps it a permutation at every offset: card `a` of one
supernode sends to card `(a + k) mod 64` of the other for k = 0..n-1, and receives from
`(a - k) mod 64`. Every card has exactly n senders and n receivers, all distinct.

A final phase times a plain all-to-all over all 128 ranks, which is the like-for-like
partner of the number already measured on the other supernode pair, so this run also says
whether the boundary ratio replicates on a second pair of supernodes.
"""
import argparse
import json
import os
import statistics
import time

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist

PEERS = [1, 2, 4, 8, 16, 32, 64]
SIZES_MIB = [64, 128, 256]


def max_over_ranks(seconds):
    t = torch.tensor([seconds * 1e6], dtype=torch.float32, device="npu")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item()) * 1e-6


def timed(fn, n_calls):
    ts = []
    for _ in range(n_calls):
        dist.barrier()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append(max_over_ranks(time.perf_counter() - t0))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--calls", type=int, default=7)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    dist.init_process_group(backend="hccl", device_id=torch.device("npu", local))

    half = world // 2                       # cards per supernode
    my_local = rank % half
    other_base = half if rank < half else 0

    rows = []
    for mib in SIZES_MIB:
        total_bytes = mib * 1024 * 1024
        cross_bytes = total_bytes // 2       # what each card sends across, fixed
        for n in PEERS:
            if n > half or cross_bytes % (n * 2) != 0:
                continue
            chunk_elems = cross_bytes // (n * 2)          # bf16
            send_splits = [0] * world
            recv_splits = [0] * world
            for k in range(n):
                send_splits[other_base + ((my_local + k) % half)] = chunk_elems
                recv_splits[other_base + ((my_local - k) % half)] = chunk_elems
            inp = torch.ones(n * chunk_elems, dtype=torch.bfloat16, device="npu")
            out = torch.empty(n * chunk_elems, dtype=torch.bfloat16, device="npu")

            def call():
                dist.all_to_all_single(out, inp, recv_splits, send_splits)

            for _ in range(3):
                call()
            torch.npu.synchronize()
            reps = [timed(call, args.calls) for _ in range(args.repeats)]
            t = statistics.median(reps)
            spread = (max(reps) - min(reps)) / max(min(reps), 1e-12)
            gbps = cross_bytes / t / 1e9
            rows.append({"phase": "peers", "peers": n, "bytes": total_bytes,
                         "cross_bytes": cross_bytes, "ms": t * 1e3,
                         "spread": spread, "gbps": gbps})
            if rank == 0:
                print("  %4d MiB  %3d cross-supernode peer(s)  %8.4f ms  %6.2f GB/s per card"
                      " (spread %4.1f%%)" % (mib, n, t * 1e3, gbps, 100 * spread),
                      flush=True)
            del inp, out
            torch.npu.empty_cache()

        # like-for-like with the other supernode pair: a plain a2a over everything
        numel = total_bytes // 2
        if numel % world == 0:
            gi = torch.ones(numel, dtype=torch.bfloat16, device="npu")
            go = torch.empty_like(gi)

            def gcall():
                dist.all_to_all_single(go, gi)

            for _ in range(3):
                gcall()
            torch.npu.synchronize()
            t = statistics.median([timed(gcall, args.calls) for _ in range(args.repeats)])
            rows.append({"phase": "global", "peers": world - 1, "bytes": total_bytes,
                         "cross_bytes": total_bytes // 2, "ms": t * 1e3})
            if rank == 0:
                print("  %4d MiB  plain a2a over all %d ranks   %8.4f ms"
                      % (mib, world, t * 1e3), flush=True)
            del gi, go
            torch.npu.empty_cache()

    if rank == 0:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"tag": args.tag, "world": world, "rows": rows}, fh, indent=1)
        print("wrote %s" % args.out, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
