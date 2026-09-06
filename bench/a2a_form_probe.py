# -*- coding: utf-8 -*-
"""Does a collective's fixed cost ADD to its transfer, or OVERLAP it?

The TerraceMoE cost model prices one all-to-all as t = alpha(world) + wire/beta, every
fixed cost paid in series with the bytes. Recalibrating the model under the alternative
rule -- the two overlapping, combining in quadrature -- fits this machine's size-sweep
corpora three times better at identical parameter count, and fits the Tier-1 corpus
twice as badly. The two corpora come from two different benchmark styles, so the
repository could not tell whether the machine or the instrument was speaking.

This settles it, by running BOTH styles at the SAME world over the SAME sizes:

    burst    N collectives back to back, one synchronisation at the end, cost = total/N.
             The host runs ahead and never observes an individual call. This is the
             style the alpha table came from, and it was only ever run at tiny payloads.
    percall  ranks aligned on a barrier, then one collective timed on its own.
             This is the style the size sweeps came from, and it was only ever run at
             large payloads.

The two forms are indistinguishable at both ends -- at tiny sizes both give alpha, at
huge sizes both give wire/beta -- and differ most where the two terms are comparable.
On this machine alpha is about 0.13 ms and beta about 105 GB/s, so they cross near a
13 MB buffer, and at that point the additive rule predicts 2*alpha while quadrature
predicts 1.41*alpha. A 29% gap is far outside the run-to-run drift. The grid below is
dense through that decade for exactly this reason.

Every reported time is the slowest rank's, reduced with MAX, because a collective ends
when its last participant ends. Sizes are powers of two and small multiples, so
per-peer bytes stay aligned at every world -- unaligned sizes measure the allocator's
size classes rather than the link, which this project has been burned by before.
"""
import argparse
import json
import os
import statistics
import time

import torch
import torch_npu  # noqa: F401  (registers the npu backend)
import torch.distributed as dist

# MiB. Dense from 1 to 32 where the fixed cost and the transfer are comparable.
SIZES_MIB = [0.0625, 0.125, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24,
             32, 48, 64, 96, 128, 192, 256]


def max_over_ranks(seconds):
    """Slowest rank's time. Reduced in microseconds as float32, because HCCL will not
    all-reduce a double; at these magnitudes float32 keeps seven digits, far more than
    the measurement carries."""
    t = torch.tensor([seconds * 1e6], dtype=torch.float32, device="npu")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item()) * 1e-6


def burst(inp, out, n):
    """N calls back to back, one sync at the end: the host never sees a single call."""
    dist.barrier()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        dist.all_to_all_single(out, inp)
    torch.npu.synchronize()
    return max_over_ranks((time.perf_counter() - t0) / n)


def percall(inp, out, n):
    """One call at a time, ranks aligned first and the host waiting for each."""
    out_ts = []
    for _ in range(n):
        dist.barrier()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        dist.all_to_all_single(out, inp)
        torch.npu.synchronize()
        out_ts.append(max_over_ranks(time.perf_counter() - t0))
    return statistics.median(out_ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--burst-n", type=int, default=32)
    ap.add_argument("--percall-n", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    dist.init_process_group(backend="hccl",
                            device_id=torch.device("npu", local))

    rows = []
    for mib in SIZES_MIB:
        nbytes = int(mib * 1024 * 1024)
        if nbytes % (world * 2) != 0:
            continue
        numel = nbytes // 2                       # bf16
        inp = torch.ones(numel, dtype=torch.bfloat16, device="npu")
        out = torch.empty_like(inp)

        for _ in range(args.warmup):
            dist.all_to_all_single(out, inp)
        torch.npu.synchronize()

        b = [burst(inp, out, args.burst_n) for _ in range(args.repeats)]
        p = [percall(inp, out, args.percall_n) for _ in range(args.repeats)]
        row = {"world": world, "bytes": nbytes, "per_peer_bytes": nbytes / world,
               "burst_ms": statistics.median(b) * 1e3,
               "burst_spread": (max(b) - min(b)) / max(min(b), 1e-12),
               "percall_ms": statistics.median(p) * 1e3,
               "percall_spread": (max(p) - min(p)) / max(min(p), 1e-12)}
        rows.append(row)
        if rank == 0:
            print("%10d B  per-peer %9.0f B   burst %8.4f ms (spread %4.1f%%)   "
                  "percall %8.4f ms (spread %4.1f%%)   ratio %.2f"
                  % (nbytes, nbytes / world, row["burst_ms"],
                     100 * row["burst_spread"], row["percall_ms"],
                     100 * row["percall_spread"],
                     row["percall_ms"] / row["burst_ms"]), flush=True)
        del inp, out
        torch.npu.empty_cache()

    if rank == 0:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"tag": args.tag, "world": world,
                       "burst_n": args.burst_n, "percall_n": args.percall_n,
                       "repeats": args.repeats, "rows": rows}, fh, indent=1)
        print("wrote %s" % args.out, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
