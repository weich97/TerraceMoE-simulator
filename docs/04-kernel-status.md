# Fused Kernel (AscendC) Status

The arrival chain (doc 02, data-flow step 4) as a PyTorch composite chain is the largest single item of two-hop extra cost — larger than the byte difference. `terrace/ops/` is the effort to fuse it into a single kernel; honest status per component:

| Component | What it is | Status |
|---|---|---|
| `passthrough` | identity copy, for link verification only | **bit-level verified on device** (0 errors across four shapes) |
| `k1_arrival` | arrival-chain fusion: pair expansion + stable bucket sort by owner + counts histogram + send gather | **algorithm proven correct; one unfixed bug in the device-side translation** (below) |
| `k2_pack` | send-side pack chain | compiles; **no on-device bit-level verification yet** |
| CPU executable spec | `terrace/ops/__init__.py::k1_arrival_ref` etc. | bit-for-bit identical to the current composite chain, guarded by tests |

## Behavior without the .so

With `TERRACE_CUSTOM_OPS` unset or 0, everything runs the CPU/composite-chain reference implementation — **bit-level equivalent, zero behavior change** — the kernel is a pure accelerator, not a correctness dependency. **The default is off**: compiling ≠ computing correctly, so enabling the fused kernels requires an explicit `TERRACE_CUSTOM_OPS=1` (reason: we lived through an incident where a kernel that had never passed bit-level checks went live automatically because it compiled; default-on treats build success as evidence of behavioral correctness).

## K1's known bug (localized to surgical precision — fixes welcome)

The diagnosis completed in two steps:

1. **The MTE out-of-bounds exception is gone**: root cause was a missing VECOUT queue on the GM→UB→GM move (see lesson 1 below);
   after adding it, the device exception disappeared.
2. **The remaining error is pinned to "scalar GM writes from non-core-0 are not visible".** One run with dumps produced three
   mutually exclusive pieces of evidence:
   - `send_buf` (the MTE move path): **0 errors** — payload all correct;
   - `i_send` (scalar writes done by core 0 only): **0 errors** — first-pass counts all correct;
   - `slot_idx`/`gate_pairs`/`r_idx` (each core doing its own scalar writes): the wrong positions are **exactly the dsts
     owned by non-core-0** (with 1 input row the errors sit at [1,2]; with 2 input rows at [1,4,5];
     both dumps ran multi-core, and the only varying axis is the input row count).
   MTE path all correct + single-core scalar writes all correct + multi-core scalar writes selectively lost ⇒
   a cross-core visibility problem with `GlobalTensor::SetValue`.
- **Fix direction**: move the metadata writes off scalar `SetValue` onto a UB buffer + `DataCopy` (MTE3) —
  the same path the payload uses, already proven visible.
- The fastest entry for a fixer: work from `k1_arrival_ref` (the bit-exact spec, guarded by `tests/test_terrace_k1_arrival.py`)
  against `op_kernel/terrace_k1_arrival.cpp`.

## Three portability lessons (for whoever edits the kernels)

1. **A GM→UB→GM move must have both a VECIN and a VECOUT queue.** A queue's position decides which two pipelines it synchronizes (VECIN pairs MTE2→V, VECOUT pairs V→MTE3); with only VECIN, nobody inserts that MTE3 barrier and the write emits dirty data — **no compile error, no load error, no runtime error; only bit-level comparison exposes it**. `tests/test_terrace_k1_arrival.py` carries a source-level guard pinning this down.
2. **The bit-level criterion is attainable — do not settle for approximate.** The arrival chain has only gather/permutation, no reduction; the output must equal the reference bit for bit — anything less means an index bug, not a precision issue.
3. **Verify against a .so that actually loaded, and stop at the first device exception** — test cases after the card enters a fault state are echoes of the same failure, not independent evidence.
