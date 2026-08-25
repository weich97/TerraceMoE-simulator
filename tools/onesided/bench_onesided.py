#!/usr/bin/env python3
"""**True one-sided**: in-kernel one-sided writes via CANN aclshmem, against collective a2a.

Why ordinary means cannot measure one-sided (instrument first, then what we measure):

  · PyTorch's cross-device `tensor.copy_()` carries a fixed ~257-268 µs overhead per
    call, which swamps everything: growing the payload from 4 MB to 40 MB does not move
    the time at all, and the reported "bandwidth" is just bytes ÷ a constant. Any
    conclusion about one-sided drawn with it is a dead number -- we have retracted a
    whole set of such tables before.
  · hyper-parallel's `put_mem` takes a different path: it launches an AscendC kernel
    directly on the current stream, **no aclnn, no host tiling; offset/size/target_pe
    are all passed as device pointers and the host never reads them back**; inside the
    kernel, `aclshmem_ptr(target, pe)` yields a directly addressable remote GM pointer,
    and MTE moves GM→UB→remote GM. That is exactly what `copy_()`'s 257 µs lacks.
  · The control target has to stand first: at aligned sizes (multiples of 4096 B),
    intra-node collective a2a measures ~104 GB/s on this machine, against a single-die
    aggregate egress physical value of 122.4 (docs/05) -- a2a already takes ~85%.

------------------------------------------------------------------------------
Criterion (written down before the run; no after-the-fact edits)

  **The ceiling itself is only ~15%.** a2a is already at ~85% of the aggregate egress
  physical value, so the theoretical ceiling for any intra-node transport is those
  remaining ~15 percentage points. Therefore:

    one-sided ≥ 1.15 × a2a (on ≥2 aligned size tiers)  -> one-sided really reaches line rate; worth pursuing
    otherwise                                          -> the one-sided route is closed

  1.15 is not pulled from thin air: it is equivalent to "going from 85% to 100% of
  line rate". Below it, one-sided is merely trading away protocol overhead, and
  protocol overhead is not the bottleneck on this machine.

  **Instrument floor**: this run measures **bandwidth**, and the floor is launch
  overhead; the two are different quantities, which is what makes the floor gate
  valid. Any tier whose time is under 3x the floor -> that tier is voided.
  (Counterexample: were the thing measured the fixed overhead itself, the floor would
  equal the signal and this gate would stay red forever -- the same trick depends on
  what is being measured.)

------------------------------------------------------------------------------
Why we do not import hyper_parallel

The top-level `hyper_parallel/__init__.py` imports the whole DTensor / FSDP / PP / CP
stack and **unconditionally** runs `override_functions()`, monkey-patching
`BackwardHookFunction`. All the risk of coexisting in-process with a training
framework lives there.
`platform/torch/symmetric_memory/symmetric_memory.py:29-35` itself demonstrates the
bypass: plain `torch.ops.load_library` + `torch.classes.SymmetricMemory.Manager/Ops`.

**Aside**: the `tcp://127.0.0.1:8662` hard-coded in `attr_init` is not a problem for
this run -- Hop B is intra-node 8 dies, all on one host, so the loopback address is
exactly enough. Only cross-node needs a change, and the C++ side `attr_init` takes an
address parameter anyway (the README's "hard-coded" claim applies only to the Python
wrapper).

Usage (single node, 8 dies):
  torchrun --nnodes=1 --nproc_per_node=8 tools/onesided/bench_onesided.py --out onesided.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

try:
    import torch_npu  # noqa: F401
except Exception as exc:                       # pragma: no cover - only runs on the cluster
    raise SystemExit("torch_npu required: %s" % exc)

H = 2048                 # verdict-testbed hidden dim; each row is H*2 = 4096 B
_MGR = _OPS = None


def _load(lib: str):
    """Load libaclshmem_torch.so and grab the two TorchScript classes.

    Path priority: explicit --lib > TERRACE_SHMEM_LIB > the copy inside the installed wheel.
    If none is found, **exit loudly** -- no silent fallback, because a fallback would
    disguise "did not measure" as "measured, and it was slow".
    """
    global _MGR, _OPS
    cands = [lib] if lib else []
    env = os.environ.get("TERRACE_SHMEM_LIB")
    if env:
        cands.append(env)
    try:
        import hyper_parallel.platform.torch.symmetric_memory as _sm
        cands.append(os.path.join(os.path.dirname(os.path.abspath(_sm.__file__)),
                                  "libaclshmem_torch.so"))
    except Exception:                                       # noqa: BLE001
        pass
    for c in cands:
        if c and os.path.exists(c):
            torch.ops.load_library(c)
            _MGR = torch.classes.SymmetricMemory.Manager()
            _OPS = torch.classes.SymmetricMemory.Ops()
            return c
    raise SystemExit(
        "libaclshmem_torch.so not found. Tried: %s\n"
        "Build it first per tools/onesided/build_shmem.sh (offline clusters must pre-stage gitcode.com/cann/shmem v1.3.0)."
        % cands)


def timed(fn, iters=10) -> float:
    """Same convention as every other microbenchmark in this repo: 3 discarded warmups + mean over iters."""
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    # Sizes are integer multiples of H -> bytes per peer are integer multiples of 4096 B,
    # same shape as the real Hop B. Unaligned sizes fall into the fake sawtooth caused by
    # power-of-two steps (implementation behavior): that measures alignment effects, not
    # the transport.
    ap.add_argument("--rows", type=int, nargs="+",
                    default=[512, 1024, 1536, 2048, 3072, 4096, 6144],
                    help="rows per peer (1 row = H*2 = 4096 B); 512 rows = 2 MB")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--lib", default="")
    ap.add_argument("--heap-mb", type=int, default=4096)
    ap.add_argument("--out", default="onesided_shmem.json")
    args = ap.parse_args()

    os.environ.setdefault("SYMMETRIC_MEMORY_HEAP_SIZE", str(args.heap_mb * 1024 * 1024))
    dist.init_process_group(backend="hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.npu.set_device(rank % 8)
    lib = _load(args.lib)
    _MGR.attr_init(rank, world, args.heap_mb * 1024 * 1024, "tcp://127.0.0.1:8662")
    # the official hyper-parallel shmem_alltoall form uses one stream per peer
    streams = [torch.npu.Stream() for _ in range(world)]
    if rank == 0:
        print("[onesided] loaded %s; world=%d heap=%d MB" % (lib, world, args.heap_mb), flush=True)

    dt = torch.bfloat16

    def T(x, d=torch.int64):
        return torch.tensor([x], dtype=d, device="npu")

    # **offset/size tensors hoisted out of the loop.** An audit found the reference
    # implementation building 3-5 fresh `torch.tensor([x], device='npu')` per call --
    # each put issued costs a few extra small H2D allocations, exactly the class of
    # overhead we are trying to beat.
    zero, one_i32 = T(0), T(1, torch.int32)

    recs, floor = [], [None]
    for rows in args.rows:
        n_elem = rows * H
        per_peer_mb = n_elem * 2 / 1e6
        # Symmetric memory: send region (one slice per peer) and recv region (one slice per peer)
        send = _MGR.malloc([world * n_elem], dt)
        recv = _MGR.malloc([world * n_elem], dt)
        torch.fill_(send, 1.0)
        sig = _MGR.malloc([1], torch.int32)
        torch.zero_(sig)
        offs = [T(p * n_elem) for p in range(world)]      # also hoisted out of the loop
        size_t = T(n_elem)
        dist.barrier()

        def run_onesided():
            # XOR butterfly: in round r, exchange with r^rank. Each round, every die has
            # exactly one in and one out -- contention-free by construction, the classic
            # contention-free p2p schedule.
            for r in range(1, world):
                pe = rank ^ r
                _OPS.put_mem(recv, offs[rank], send, offs[pe], size_t, pe)
            torch.npu.synchronize()

        # Control arm: collective a2a with the same payload (measured ~76-104 GB/s on this
        # machine at these sizes)
        sv = torch.ones(world * n_elem, dtype=dt, device="npu")
        rv = torch.empty_like(sv)

        def run_a2a():
            dist.all_to_all_single(rv, sv)

        # Third arm: the official shmem_alltoall form (one stream per peer + put_mem_signal +
        # signal_wait_until). The butterfly arm occupies only one link per round and by
        # construction cannot exploit link parallelism; this arm is the parallel form the
        # library actually runs in production -- **no verdict is allowed before measuring the
        # strongest legitimate implementation**.
        # Completion detection uses a cumulative signal count (op=1 is add): no per-round
        # reset, no host sync; sig is freshly allocated and zeroed per size tier and n_calls
        # resets with it, so the bookkeeping stays self-consistent.
        n_calls = [0]
        expect_t = T(-1, torch.int32)   # after the first add_(world) it equals world-1, i.e. the GT threshold

        def run_official():
            for pe in range(world):
                with torch.npu.stream(streams[pe]):
                    _OPS.put_mem_signal(recv, offs[rank], send, offs[pe], size_t,
                                        sig, zero, one_i32, 1, pe)
            n_calls[0] += 1
            # compare_op offers only 0=EQ/1=GT/2=LT (kernel source); there is no GE.
            # EQ is fatal: a fast rank enters the next round early and lands its +1 in my
            # sig, the exact-equality window gets overshot -> permanent spin (observed: all
            # ranks hang here on device sync timeout).
            # GT + (8k-1) is equivalent to GE 8k, immune to overshoot under a monotonically
            # increasing counter.
            # The threshold tensor is incremented in place on device (expect_t prebuilt in
            # the outer scope); no tensors built on the hot path -- same discipline as
            # offs/size_t.
            expect_t.add_(world)
            _OPS.signal_wait_until(recv, sig, zero, expect_t, 1)
            torch.npu.synchronize()

        # **Instrument floor**: the same world-1 puts, 1 element each. This run measures
        # bandwidth while the floor is launch overhead; the two are different quantities --
        # which is why this gate is right for this run (were the measurement the fixed
        # overhead itself, the same gate would stay red forever).
        if floor[0] is None:
            s1, r1 = _MGR.malloc([world], dt), _MGR.malloc([world], dt)
            o1, z1 = [T(p) for p in range(world)], T(1)

            def run_floor():
                for r in range(1, world):
                    pe = rank ^ r
                    _OPS.put_mem(r1, o1[rank], s1, o1[pe], z1, pe)
                torch.npu.synchronize()

            floor[0] = timed(run_floor, args.iters) * 1e3
            _MGR.free(s1)
            _MGR.free(r1)

        to = sorted(timed(run_onesided, args.iters) for _ in range(args.reps))[args.reps // 2]
        ta = sorted(timed(run_a2a, args.iters) for _ in range(args.reps))[args.reps // 2]
        tf2 = sorted(timed(run_official, args.iters) for _ in range(args.reps))[args.reps // 2]
        nbytes = (world - 1) * n_elem * 2          # bytes this die actually moves out
        rec = {"rows": rows, "per_peer_mb": per_peer_mb,
               "onesided_ms": to * 1e3, "a2a_ms": ta * 1e3,
               "onesided_gbps": nbytes / to / 1e9,
               "a2a_gbps": (world - 1) / world * (world * n_elem * 2) / ta / 1e9,
               "ratio": ta / to if to else 0.0,
               "official_ms": tf2 * 1e3,
               "official_gbps": nbytes / tf2 / 1e9,
               "ratio_official": ta / tf2 if tf2 else 0.0,
               "over_floor": (to * 1e3) / floor[0] if floor[0] else 0.0,
               "official_over_floor": (tf2 * 1e3) / floor[0] if floor[0] else 0.0}
        recs.append(rec)
        if rank == 0:
            print("%5d rows (%6.2f MB/peer)  butterfly %6.1f  official %6.1f  a2a %6.1f GB/s"
                  "  -> butterfly **%.2f×** official **%.2f×**  (floor %.1f×/%.1f×)"
                  % (rows, per_peer_mb, rec["onesided_gbps"], rec["official_gbps"],
                     rec["a2a_gbps"], rec["ratio"], rec["ratio_official"],
                     rec["over_floor"], rec["official_over_floor"]), flush=True)
        for t in (send, recv, sig):
            _MGR.free(t)
        del sv, rv

    if rank != 0:
        dist.destroy_process_group()
        return

    out = {"records": recs, "floor_ms": floor[0], "world": world, "lib": lib}
    for r in recs:
        cands = []
        if r["over_floor"] >= 3.0:
            cands.append(r["ratio"])
        if r["official_over_floor"] >= 3.0:
            cands.append(r["ratio_official"])
        r["best_ratio"] = max(cands) if cands else None
    voided = [r["rows"] for r in recs if r["best_ratio"] is None]
    if voided:
        print(flush=True)
        print("!! The following tiers ran under 3x the instrument floor (%.3f ms); **readings do not count**: %s"
              % (floor[0], voided), flush=True)
    live = [r for r in recs if r["best_ratio"] is not None]

    print(flush=True)
    wins = [r for r in live if r["best_ratio"] >= 1.15]
    if len(live) < 3:
        out["verdict"] = "INVALID: fewer than 3 valid tiers"
        print("!! Only %d valid tiers -- **no verdict issued**." % len(live), flush=True)
    elif len(wins) >= 2:
        out["verdict"] = "one-sided reaches line rate; worth pursuing"
        print("**one-sided ≥1.15× a2a on %d tiers** -> %s" % (len(wins), out["verdict"]), flush=True)
        print("   best tier: %.2f× @ %.2f MB/peer"
              % (max(w["best_ratio"] for w in wins),
                 max(wins, key=lambda w: w["best_ratio"])["per_peer_mb"]), flush=True)
    else:
        out["verdict"] = "the one-sided route is closed (on this machine)"
        best = max(live, key=lambda r: r["best_ratio"])
        print("one-sided (best of both implementations) tops out at **%.2f×** (@ %.2f MB/peer), short of 1.15 -> **%s**"
              % (best["best_ratio"], best["per_peer_mb"], out["verdict"]), flush=True)
        print("   Reason: a2a is already at ~85%% of the aggregate egress physical value; the ceiling was only ~15 points to begin with;", flush=True)
        print("   one-sided trades away protocol overhead, and protocol overhead is not the bottleneck on this machine.", flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("wrote %s" % os.path.abspath(args.out), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
