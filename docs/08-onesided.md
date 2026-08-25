# One-sided transfers vs collective a2a: a preregistered negative verdict + two upstream library patches

**One sentence: within an 8-die node of a bandwidth-flat supernode, doing MoE-dispatch-shaped
exchange with CANN aclshmem in-kernel one-sided writes (via hyper-parallel's put path), across
4 configurations — two implementations (butterfly / official alltoall shape) × two core counts —
the best reaches only 0.68× of collective a2a; the preregistered criterion (≥1.15× on ≥2 valid
tiers) is met by zero tiers — the one-sided road is closed on this machine.** The two upstream
library defects fixed along the way (per-call malloc/free: the free implies a device sync that
serializes concurrent puts, and the kernel's SyncAll uses already-freed memory — a UAF;
block_dim hard-coded to 1 with no setter) plus one usage trap (completion counting with EQ
deadlocks under repeated measurement) ship with this repo as reusable patches/lessons
(`tools/onesided/`), applicable to any Ascend + shmem user.

![one-sided vs a2a](assets/f12-onesided.svg)

## 1. Result matrix

| Implementation × block_dim | valid tiers (≥3× instrument floor) | one-sided bandwidth | vs a2a (~104 GB/s) |
|---|---|---|---|
| Butterfly (single-stream serial) × 1 | 6/7 | 13–15 GB/s plateau | 0.13–0.16× |
| Butterfly × 8 | 4/7 | 63–70 | **0.68×** (best) |
| Official alltoall shape (one stream per peer) × 1 | 5/7 | 23–58, climbing with size | 0.25–0.56× |
| Official shape × 8 | 0 (all tiers below floor, and large sizes hit device timeout) | — | instrument limit, not a measurement |

(Provenance note: the butterfly ×8 row comes from the two-arm instrument version that predates
the official arm — the butterfly protocol is identical across both versions; official ×8 hit
device timeouts in the three-arm version, no valid readings. The verdict strings in the
archived JSON carry the wording of the version current at the time.)

Criterion (fixed before running): any legitimate one-sided implementation at ≥1.15× a2a on ≥2
valid tiers → worth pursuing further; otherwise closed. Where 1.15 comes from: measured a2a
already sits at ~85% of the single-die aggregate-egress physical value (122.4 GB/s, docs/05);
1.15× is equivalent to "going from 85% to 100% of line rate".

**Why the verdict is safe (physics footnote)**:

- Butterfly scheduling runs 7 serial rounds; of each die's 7 legs, 6 go over intra-node links
  (112.1 GB/s) and 1 over the in-package direct link (185); the perfect-utilization ceiling
  = 7/(6/112.1 + 1/185) ≈ 118.8 GB/s ≈ **1.14× a2a** — nominally <1.15, but only 0.8 percentage
  points short, within the measurement noise of the inputs; **what actually kills this shape is
  the measurement**: 8 cores already push a single put to 70 GB/s, still 40% short of its own
  ceiling, and the whole gap sits in the kernel's UB-relay implementation — the same bottleneck
  as the official arm;
- The official multi-link shape's abstract ceiling ≈ 122.4/104 ≈ 1.18×, not physically
  excluded, but the existing kernel is a GM→UB→remote-GM MTE relay (every byte crosses UB
  twice), and measurement tops out at 0.56×. Reaching 1.15 needs a new DMA-direct transport —
  that is rewriting the communication library, not using it.

## 2. Two upstream defects, one usage trap (the part useful to everyone)

`tools/onesided/build_shmem.sh` patches hyper-parallel's symmetric memory automatically at
build time (idempotent, with marker checks); every patch comes from a real crash:

1. **Upstream defect: one `aclrtMalloc + aclrtMemset + aclrtFree` set per put (32 B sync area)**.
   `aclrtFree` typically implies a device sync — this alone voids the whole set of bandwidth
   readings (all puts serialized; no two puts were ever actually in flight); and the kernel's
   `SyncAll` uses exactly that **already-freed** memory (a UAF, saved from crashing only by the
   serialization above). Patch: statically reuse the sync area + switch zeroing to
   `aclrtMemsetAsync` **enqueued on the same stream**. Honesty footnote: same-stream enqueue is
   not decoration — our first patch version did only the static reuse and kept the
   host-synchronous memset; that memset, not ordered against the stream, hit the flag of an
   in-flight kernel and caused a permanent spin (measured: 3/8 dies hung). The original was
   naturally immune to this race ("fresh buffer every time"); once the buffer is reused,
   stream-ordered zeroing is a correctness requirement, not an optimization. The
   `put_mem_signal` variant, called concurrently on multiple streams by the official alltoall,
   uses a 64-slot rotating pool.
2. **`DEFAULT_BLOCK_DIM = 1` hard-coded, no setter**. The kernel itself is written for multiple
   cores (`size_per_core = size_ / aiv_num_`), but upstream grants only 1 block — single-core
   MTE is a bandwidth ceiling (measured 13–15 GB/s plateau; opening up 8 cores gives 5×
   immediately). Patch: tunable via the environment variable `TERRACE_SHMEM_BLOCK_DIM`;
   **the default stays 1, behavior verbatim-unchanged** — auto-enabling a capability the moment
   it becomes available amounts to treating "it compiles" as evidence of "it behaves correctly"
   (the same lesson as docs/04).
3. **Usage trap (not an upstream bug): completion counting with exact equality (EQ) is certain
   death under repeated exchanges**. `signal_wait_until`'s comparison enum offers only
   EQ/GT/LT. Under the official alltoall's one-shot usage (fresh signal, single exchange,
   discarded after use) EQ is safe — the count rises monotonically to the target and stops. But
   in **repeated-exchange** settings like benchmarks or training, a fast rank enters the next
   round early and lands its +1 in your signal; the "exactly equals 8k" window gets overshot →
   permanent spin (measured: device sync timeout on all ranks). Fix: **GT + (target−1)**,
   immune to overshoot under a monotonically increasing count. Same-family lesson: rewrite any
   distributed predicate of the form "wait until count == N" as ≥.

## 3. Instrument discipline (why these numbers are credible)

- **Launch-floor gate**: for each size tier, first measure the floor with the same call count
  and a 1-element payload; any tier whose time is under 3× the floor has its reading **voided**
  (hollow points in the figure). A bandwidth reading's resolution must beat the effect under
  measurement.
- **Aligned payloads**: every payload is an integer multiple of the true row width
  (H×dtype = 4096 B) — unaligned payloads fall into implementation behavior that steps at
  powers of 2, and you end up measuring alignment effects, not the transport.
- **Control arm co-located**: a2a and one-sided are measured alternately in the same process,
  on the same buffers, under the same timer.
- **Criterion before data**: 1.15× / ≥2 tiers / 3× floor were all fixed before the first run.

## 4. Reproduce

```
# 1) Build (needs the hyper-parallel sources; offline clusters pre-stage 3rdparty/shmem @ v1.3.0)
bash tools/onesided/build_shmem.sh /path/to/hyper-parallel-master
# 2) Run (single node, 8 dies; one run per core count)
TERRACE_SHMEM_BLOCK_DIM=1 torchrun --nnodes=1 --nproc_per_node=8 \
    tools/onesided/bench_onesided.py --out onesided_bd1.json
TERRACE_SHMEM_BLOCK_DIM=8 torchrun --nnodes=1 --nproc_per_node=8 \
    tools/onesided/bench_onesided.py --out onesided_bd8.json
```

The criterion, the floor gate, and the verdict are all built into the bench; the output is the
conclusion. On a different machine, first re-calibrate under the docs/05 conventions (the a2a
physical fraction changes, so the meaning of the 1.15 threshold changes — re-derive the
threshold for your machine, do not copy it).
