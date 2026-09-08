# -*- coding: utf-8 -*-
"""Is the reference arrival chain representative of the live one, or just cheaper?

The re-measurement puts the chain 3.3x below the shipped constant. Two explanations:
the software stack improved, or terrace.ops.k1_arrival_ref is a cheaper way to compute
what the live chain computes. Its docstring names the only difference -- r_idx by
arithmetic rather than by table lookup -- and its sort primitive turns out to BE the
live chain's `_stable_argsort_small`. This times both sequences at the same shapes so
the explanation is not a guess.
"""
import argparse, json, statistics, sys, time
import torch, torch_npu  # noqa: F401

QUOTA, EPR, RPN = 6, 4, 64


def timeit(fn, calls, repeats):
    out = []
    for _ in range(repeats):
        ts = []
        for _ in range(calls):
            torch.npu.synchronize(); t0 = time.perf_counter()
            fn(); torch.npu.synchronize()
            ts.append(time.perf_counter() - t0)
        out.append(statistics.median(ts))
    return statistics.median(out) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--repo", required=True)
    ap.add_argument("--calls", type=int, default=15); ap.add_argument("--repeats", type=int, default=7)
    a = ap.parse_args(); sys.path.insert(0, a.repo)
    from terrace.ops import k1_arrival_ref
    from terrace.ta2a_fwd import _expand_arrival_quota, _stable_argsort_small, fixed_hist

    torch.npu.set_device(0)
    rows = []
    for pairs in (8192, 24576, 65536):
        for H in (2048, 8192):
            n_in = pairs // QUOTA
            rx = torch.randn(n_in, H, dtype=torch.bfloat16, device="npu")
            rslot = torch.randint(0, RPN * EPR, (n_in, QUOTA), device="npu")
            rgate = torch.randn(n_in, QUOTA, dtype=torch.bfloat16, device="npu")

            def ref():
                k1_arrival_ref(rx, rslot, rgate, quota=QUOTA, epr=EPR, rpn=RPN)

            def live():
                r_idx, slot_idx = _expand_arrival_quota(rslot)
                owner = slot_idx // EPR
                ordo = _stable_argsort_small(owner, RPN)
                ri, si = r_idx[ordo], slot_idx[ordo]
                i_send = fixed_hist(owner, RPN)
                send_buf = rx[ri]
                gate_pairs = rgate.reshape(-1)[ordo]
                return send_buf, gate_pairs, ri, si, i_send

            for f in (ref, live):
                for _ in range(5):
                    f()
            torch.npu.synchronize()
            tr, tl = timeit(ref, a.calls, a.repeats), timeit(live, a.calls, a.repeats)
            rows.append({"pairs": pairs, "H": H, "ref_ms": tr, "live_ms": tl,
                         "live_over_ref": tl / tr})
            print("  %6d pairs  H=%5d   reference %7.4f ms   live %7.4f ms   live/ref %.3f"
                  % (pairs, H, tr, tl, tl / tr), flush=True)
            del rx, rslot, rgate
            torch.npu.empty_cache()
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
