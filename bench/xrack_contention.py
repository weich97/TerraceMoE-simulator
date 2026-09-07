# -*- coding: utf-8 -*-
"""How does cross-rack bandwidth hold up when many node pairs cross at once?

The rack boundary was measured with exactly one node pair crossing, which is the least
contended case there is. Real expert parallelism crosses with every node in the rack at
once, sharing whatever uplinks the rack has. Nothing in the cost model prices that, and
it is the term that could invert the verdict the single-pair measurement suggests.

The design holds structure fixed and varies only pressure. Sixteen nodes, eight in each
rack, paired node i with node i+8. Each pair forms a 16-rank subgroup whose all-to-all
is exactly the configuration already measured: seven intra-node peers and eight across
the boundary. Then the same subgroup a2a is timed with 1, 2, 4 and 8 pairs running
concurrently. Every subgroup does identical work at every step, so a slowdown is
contention on the shared boundary and nothing else.

One more phase costs nothing while the allocation is held: a single all-to-all over all
128 ranks, which is the full cross-rack fabric at the scale expert parallelism would
actually use.

Timed in the percall convention -- ranks aligned, then one call timed with the host
waiting -- because that is the regime an MoE dispatch is in and the regime the shipped
additive model belongs to.
"""
import argparse
import json
import os
import statistics
import time

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist

SIZES_MIB = [32, 64, 128]
PAIRS = [1, 2, 4, 8]


def max_over(group, seconds):
    t = torch.tensor([seconds * 1e6], dtype=torch.float32, device="npu")
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return float(t.item()) * 1e-6


def timed(group, inp, out, n):
    """Median of n calls, ranks aligned before each, host waiting for each."""
    ts = []
    for _ in range(n):
        dist.barrier()                      # global: line every rank up
        torch.npu.synchronize()
        t0 = time.perf_counter()
        dist.all_to_all_single(out, inp, group=group)
        torch.npu.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--calls", type=int, default=9)
    ap.add_argument("--nodes-per-rack", type=int, default=8)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    dist.init_process_group(backend="hccl", device_id=torch.device("npu", local))

    npr = args.nodes_per_rack
    per_node = world // (2 * npr)
    my_node = rank // per_node
    my_pair = my_node % npr                        # node i pairs with node i+npr

    # every rank must construct every group, in the same order
    groups = []
    for i in range(npr):
        members = (list(range(i * per_node, (i + 1) * per_node))
                   + list(range((i + npr) * per_node, (i + npr + 1) * per_node)))
        groups.append(dist.new_group(ranks=members))
    world_group = dist.new_group(ranks=list(range(world)))
    mine = groups[my_pair]

    rows = []
    for mib in SIZES_MIB:
        nbytes = mib * 1024 * 1024
        pair_numel = nbytes // 2
        inp = torch.ones(pair_numel, dtype=torch.bfloat16, device="npu")
        out = torch.empty_like(inp)
        for _ in range(3):
            dist.all_to_all_single(out, inp, group=mine)
        torch.npu.synchronize()
        dist.barrier()

        for active in PAIRS:
            reps = []
            for _ in range(args.repeats):
                if my_pair < active:
                    t = timed(mine, inp, out, args.calls)
                else:
                    # inactive ranks must still hit the same global barriers
                    for _ in range(args.calls):
                        dist.barrier()
                    t = 0.0
                reps.append(t)
            worst = max_over(None, max(reps))       # slowest active pair, globally
            med = statistics.median([r for r in reps if r > 0]) if my_pair < active else 0.0
            med = max_over(None, med)
            rows.append({"phase": "pairs", "active_pairs": active, "bytes": nbytes,
                         "ms": med * 1e3, "worst_ms": worst * 1e3})
            if rank == 0:
                print("  %4d MiB  %d pair(s) crossing   %8.4f ms" %
                      (mib, active, med * 1e3), flush=True)
        del inp, out
        torch.npu.empty_cache()

        # the whole fabric across both racks, same size
        g_numel = nbytes // 2
        if g_numel % world == 0:
            gi = torch.ones(g_numel, dtype=torch.bfloat16, device="npu")
            go = torch.empty_like(gi)
            for _ in range(3):
                dist.all_to_all_single(go, gi, group=world_group)
            torch.npu.synchronize()
            t = timed(world_group, gi, go, args.calls)
            t = max_over(None, t)
            rows.append({"phase": "global", "active_pairs": npr, "bytes": nbytes,
                         "ms": t * 1e3, "worst_ms": t * 1e3})
            if rank == 0:
                print("  %4d MiB  all %d ranks           %8.4f ms" %
                      (mib, world, t * 1e3), flush=True)
            del gi, go
            torch.npu.empty_cache()

    if rank == 0:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"tag": args.tag, "world": world, "nodes_per_rack": npr,
                       "per_node": per_node, "rows": rows}, fh, indent=1)
        print("wrote %s" % args.out, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
