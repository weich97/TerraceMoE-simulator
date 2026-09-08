# -*- coding: utf-8 -*-
"""Re-measure the arrival chain: the constant everything hinges on.

`calibrate.CHAIN_US_PER_ROW` is 0.0875 microseconds per Hop-B row, and that file calls
it "the constant everything hinges on -- it alone moves the breakeven ratio from 1.10 to
3.98". Two readings stand behind it, and they agree: 2.15 ms at 24576 rows from a single
measurement, and a nine-row-count sweep on two nodes months later giving 0.0987.

Measuring two-hop against one-hop on 2026-09-08 produced a third reading, incidentally,
because that run timed the real chain rather than charging a per-row estimate: **0.0424
microseconds per row at the same shape**. Half. If that is right the reference threshold
falls from 3.98 to 2.49, which changes which machines the method pays on, so it cannot
be left as a by-product of another experiment.

This reproduces both of the calibration's own sweeps so the readings are directly
comparable:

    row sweep       output pairs from 1024 to 65536 at the reference hidden width,
                    which is where the linear form and its floor were established
    width sweep     hidden width 1024 to 8192 at 24576 pairs, which is where the
                    index-versus-gather split came from

The chain is `terrace.ops.k1_arrival_ref`, the repository's own reference chain, whose
docstring calls it bit-for-bit identical to the live chain and whose five stages are the
five `sim/machine.py` names: pair expansion, owner stable bucket sort, the i_send
histogram, the [pairs, H] send gather, and the gate flat gather.

Purely local work, so this needs one card and no collectives. The denominator is output
pairs, matching `core.py`, which charges the chain at `rows_hop_b`.
"""
import argparse
import json
import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401

PAIRS = [1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536]
WIDTHS = [1024, 2048, 4096, 8192]
QUOTA = 6
EPR = 4
RPN = 64


def time_chain(fn, calls, repeats):
    out = []
    for _ in range(repeats):
        ts = []
        for _ in range(calls):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.npu.synchronize()
            ts.append(time.perf_counter() - t0)
        out.append(statistics.median(ts))
    return statistics.median(out), (max(out) - min(out)) / max(min(out), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--calls", type=int, default=15)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    from terrace.ops import k1_arrival_ref

    torch.npu.set_device(args.device)
    rows = []

    def one(pairs, H):
        n_in = pairs // QUOTA
        rx = torch.randn(n_in, H, dtype=torch.bfloat16, device="npu")
        rslot = torch.randint(0, RPN * EPR, (n_in, QUOTA), device="npu")
        rgate = torch.randn(n_in, QUOTA, dtype=torch.bfloat16, device="npu")
        f = lambda: k1_arrival_ref(rx, rslot, rgate, quota=QUOTA, epr=EPR, rpn=RPN)
        for _ in range(5):
            f()
        torch.npu.synchronize()
        ms, spread = time_chain(f, args.calls, args.repeats)
        del rx, rslot, rgate
        torch.npu.empty_cache()
        return ms * 1e3, spread

    print("row sweep at H = 2048 (the reference width)")
    for pairs in PAIRS:
        ms, sp = one(pairs, 2048)
        rows.append({"sweep": "rows", "pairs": pairs, "H": 2048, "ms": ms,
                     "us_per_row": ms * 1000.0 / pairs, "spread": sp})
        print("  %6d pairs   %8.4f ms   %7.4f us/row   (spread %4.1f%%)"
              % (pairs, ms, ms * 1000.0 / pairs, 100 * sp), flush=True)

    print()
    print("width sweep at 24576 pairs (the shape the calibration quotes)")
    for H in WIDTHS:
        ms, sp = one(24576, H)
        rows.append({"sweep": "width", "pairs": 24576, "H": H, "ms": ms,
                     "us_per_row": ms * 1000.0 / 24576, "spread": sp})
        print("  H = %5d       %8.4f ms   %7.4f us/row   (spread %4.1f%%)"
              % (H, ms, ms * 1000.0 / 24576, 100 * sp), flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"quota": QUOTA, "epr": EPR, "rpn": RPN, "rows": rows}, fh, indent=1)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
