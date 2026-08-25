"""terrace.ops: torch-side wrapper -- loader and fallback switch for the custom T-A2A AscendC kernels.

Engineering scaffold (built overnight on 2026-08-20; see the header comments of
the files in this directory and the internal design records (not published with
the repo)). Registered ops:
  - terrace_passthrough: copies the input out unchanged; smoke benchmark for the
    full chain (msopgen project -> opp package -> aclnn -> torch.library ->
    autograd.Function -> fallback switch);
  - terrace_k1_arrival (landed 2026-08-20): arrival-side fused chain (expansion +
    stable bucket sort by owner + i_send histogram + send-buffer gather + gate
    gather), C1 quota wire format, bit-for-bit identical to the live composed
    chain -- see the executable spec k1_arrival_ref below and the two-pass
    argument in ascendc/op_kernel/terrace_k1_arrival.cpp.
K2 (send-side packing chain) interface draft: see the comment at the bottom of
this file and the header of the build script ascendc/build.sh.

Switch semantics (TERRACE_CUSTOM_OPS, read once at process start):
  **unset / "0"** -> off. Do not try to load the .so; everything goes through
                     the live composed chain (the verified, bit-exact path).
  "1"            -> on. Try to load; on failure, **print one WARNING line** and
                    behave as "0". Fail-loud, not silent: the fallback must be
                    visible in the logs, but it must not kill training.
  "require"/"2"  -> hard. A load failure raises RuntimeError directly. For
                    bench/acceptance scenarios of the "tonight this MUST run on
                    the kernel" kind, so a fallback cannot silently swap kernel
                    readings for composed-chain readings.

**The default changed from "1" to "0" as the 2026-08-24 incident fix**: the
moment the `.so` first compiled successfully, the K1 kernel -- which had NOT
passed bit-for-bit validation -- entered the training path automatically. From
then on, every T-A2A-on arm run crashed on all 128 ranks at step 0 (K1 index
bug -> wrong i_send -> Hop B splits mismatch), wasting two verdict-testbed
runs. "Compiles" does not mean "computes correctly", and at the time the only
gate was "does the .so dlopen". Enabling a kernel now requires an explicit
sign-off.

Why read the environment once instead of on every call: dispatch passes through
here on every layer, every microbatch; one os.environ lookup + string compare
on the hot path is pure waste, and the training process never flips the switch
mid-run. Tests that need to flip it use reset() (below).
"""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass

import torch

_LOG = logging.getLogger("terrace.ops")

_ENV_SWITCH = "TERRACE_CUSTOM_OPS"
_ENV_LIB = "TERRACE_OPS_LIB"          # explicit .so path, bypasses the default search
_LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")

# Ops that must all exist after a successful load. Missing any one counts as a
# load failure (fail loud): better to fall back wholesale than to sit in the
# half-dead ".so loaded but schema not registered" state that only blows up on
# the first call.
# After k1_arrival landed (2026-08-20), an old .so containing only passthrough
# falls back wholesale -- deliberately: a half-old, half-new op set that blows
# up mid-training is worse than slow. Rebuild on the cluster per the header of
# the build script ascendc/build.sh.
_REQUIRED_OPS = ("passthrough", "k1_arrival")


class OpsLoadError(RuntimeError):
    """The custom op library failed to load (.so not found / dlopen failed / schema missing)."""


@dataclass(frozen=True)
class OpsState:
    """The loader's one-shot verdict. requested is the raw environment switch; loaded is the final fact."""
    requested: str        # "0" | "1" | "require" (normalized)
    loaded: bool          # torch.ops.terrace.* is available
    lib: str | None       # path of the .so actually loaded (None if not loaded)
    reason: str           # plain-language reason when loaded=False; "ok" when loaded=True


_STATE: OpsState | None = None


def _normalized_switch() -> str:
    """Normalize the switch. **The default is "0" (off), not "1".**

    The 2026-08-24 incident: the `.so` first compiled successfully at 07:25;
    because the default was "1", `custom_ops_enabled()` became true on the spot
    and **the K1 kernel, which had not passed bit-for-bit validation, went
    straight into the training path**. From then on, every T-A2A-on arm run
    crashed on all 128 ranks simultaneously at step 0 with
    `RuntimeError: Split sizes dosen't match total dim 0 size`
    (K1's slot_idx computed wrong -> wrong i_send -> Hop B splits mismatch);
    the r4 and isub verdict-testbed runs were wasted, while the runner still
    reported "done".

    bitcheck's verdict line had long said "K1 must not go on the testbed" --
    **but no mechanism enforced it**; the only gate was "does the .so dlopen".
    "Compiles" does not mean "computes correctly".

    Changed to fail-safe: **no op enters the path unless someone explicitly asks**.
        unset / "0"     -> off (reference implementation, zero behavior change)
        "1"             -> on (explicit request)
        "require" / "2" -> on, and blow up if it cannot load

    The cost is that every site that wants the ops has to say so explicitly --
    which is exactly the point: it turns "enable a kernel" into an action
    **someone signs off on**.
    """
    raw = os.environ.get(_ENV_SWITCH, "0").strip()
    if raw in ("2", "require"):
        return "require"
    if raw == "1":
        return "1"
    return "0"


def _find_library() -> str:
    """Locate the build artifact. An explicit env wins; the default searches terrace/ops/lib/*.so (where the cluster build lands)."""
    explicit = os.environ.get(_ENV_LIB)
    if explicit:
        if not os.path.isfile(explicit):
            raise OpsLoadError(f"{_ENV_LIB}={explicit} does not exist")
        return explicit
    hits = sorted(glob.glob(os.path.join(_LIB_DIR, "*.so")))
    if not hits:
        raise OpsLoadError(
            f"no .so under {_LIB_DIR} (expected on machines without CANN; build on the cluster per the header of the build script ascendc/build.sh)")
    return hits[0]


def _try_load() -> str:
    """Load the .so and verify the schema is complete. Returns the library path; any failure raises OpsLoadError."""
    path = _find_library()
    try:
        torch.ops.load_library(path)
    except Exception as e:                          # dlopen/registration can fail in many ways
        raise OpsLoadError(f"load_library({path}) failed: {e}") from e
    ns = getattr(torch.ops, "terrace", None)
    missing = [op for op in _REQUIRED_OPS
               if ns is None or not hasattr(ns, op)]
    if missing:
        raise OpsLoadError(
            f"{path} loaded but torch.ops.terrace.{{{','.join(missing)}}} missing -- "
            f"the TORCH_LIBRARY registration did not take effect; check csrc/terrace_ops.cpp")
    return path


def _initialize() -> OpsState:
    switch = _normalized_switch()
    if switch == "0":
        return OpsState(requested="0", loaded=False, lib=None,
                        reason=f"{_ENV_SWITCH}=0 (explicitly off; no load attempted)")
    try:
        path = _try_load()
        _LOG.info("terrace custom ops loaded: %s", path)
        return OpsState(requested=switch, loaded=True, lib=path, reason="ok")
    except OpsLoadError as e:
        if switch == "require":
            raise RuntimeError(
                f"{_ENV_SWITCH}={os.environ.get(_ENV_SWITCH)} requires the custom ops, "
                f"but loading failed: {e}") from e
        # Fail-loud fallback: exactly one WARNING line (with logging unconfigured,
        # lastResort still prints to stderr), then behave as TERRACE_CUSTOM_OPS=0.
        _LOG.warning("TERRACE_CUSTOM_OPS downgraded to 0 (%s) -- using the live composed chain", e)
        return OpsState(requested=switch, loaded=False, lib=None, reason=str(e))


def status() -> OpsState:
    """Lazy init with caching. The training process makes this verdict exactly once in its lifetime."""
    global _STATE
    if _STATE is None:
        _STATE = _initialize()
    return _STATE


def custom_ops_enabled() -> bool:
    """The dispatch side's future gate: only when True may torch.ops.terrace.* be called."""
    return status().loaded


def reset() -> None:
    """Forget the cached verdict; the next status() re-reads the environment. Test/debug hook; training code must not call it.

    Note that torch.ops.load_library is process-level irreversible: reset()
    only resets **this module's** verdict; registered schemas do not disappear.
    Tests use it to flip the switch semantics, not to unload the .so.
    """
    global _STATE
    _STATE = None


# --------------------------------------------------------------------------------------
# autograd.Function boilerplate + functional entry points
# --------------------------------------------------------------------------------------

class TerracePassthroughFn(torch.autograd.Function):
    """Boilerplate: forward calls the custom kernel, backward uses the composed chain.

    passthrough is an identity copy, so its "composed-chain backward" happens to
    be identity too -- but the boilerplate is written out in full, because K1/K2
    have this shape (internal design records (not published with the repo):
    backward stays on the live composed chain):

      K1 has landed in this shape (see TerraceK1ArrivalFn below): forward calls
      torch.ops.terrace.k1_arrival; backward is the live chain's
      index_add_/gather. When K2 is implemented for real, change here:
        forward:  payload, mask, gate_rows, ... = torch.ops.terrace.k2_pack(...)
        backward: payload's backward = index_add_(0, u_src, grad) (the
                  scatter-add adjoint of the deduplicated gather); gate_rows'
                  backward = the gather grad[inverse, slot_flat] -- all existing
                  composed-chain primitives, bit-level semantics identical to
                  today's.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # When K1 is implemented for real, change here: swap in the matching
        # torch.ops.terrace.* call and ctx.save
        return torch.ops.terrace.passthrough(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # The adjoint of an identity copy is identity. For K1/K2 backward see the
        # class docstring -- stay on the composed chain, write no backward kernel
        # (roadmap decision: both kernels are pure permutation/copy; the backward
        # semantics already exist).
        return grad_out


def passthrough(x: torch.Tensor) -> torch.Tensor:
    """Identity copy, for chain validation only. Takes the custom op when the kernel is available, otherwise the composed-chain equivalent.

    Contract (guarded by tests/test_terrace_ops_scaffold.py): both paths return
    output bit-for-bit equal to the input, as a new tensor (no storage shared
    with the input), with identity gradient flow back.
    """
    if custom_ops_enabled():
        return TerracePassthroughFn.apply(x)
    # Composed-chain equivalent: clone is the existing-primitive spelling of
    # "copy out unchanged" (differentiable, identity adjoint).
    return x.clone()


# --------------------------------------------------------------------------------------
# K1: arrival-side fused chain (C1 quota wire format, landed 2026-08-20)
# --------------------------------------------------------------------------------------

def _stable_ordo(owner: torch.Tensor, rpn: int) -> torch.Tensor:
    """The live chain's stable bucket-sort primitive (deferred import to avoid the module-level ops <-> ta2a_fwd cycle)."""
    from ..ta2a_fwd import _stable_argsort_small
    return _stable_argsort_small(owner, rpn)


def k1_arrival_ref(rx: torch.Tensor, rslot: torch.Tensor, rgate: torch.Tensor,
                   quota: int, epr: int, rpn: int, my_local: int = 0):
    """CPU/composed-chain reference for K1 -- the executable spec of the kernel semantics, bit-for-bit identical to the live chain.

    Replicates bit-for-bit the arrival segment of ta2a_fwd.ta2a_moe_forward /
    ta2a_dispatch.ta2a_permute (the quota fast-path branch, after C1, before
    Hop B):

        r_idx, slot_idx = _expand_arrival_quota(rslot)
        owner = slot_idx // epr
        ordo  = _stable_argsort_small(owner, rpn)
        r_idx, slot_idx = r_idx[ordo], slot_idx[ordo]
        from ..ta2a_fwd import fixed_hist   # deferred import, avoids the module-level cycle
        i_send = fixed_hist(owner, rpn)     # the same fixed-length histogram as the live chain
        send_buf, gate_pairs = rx[r_idx], rgate.reshape(-1)[ordo]

    The AscendC kernel's two-pass scheme (count -> prefix cursors -> expand and
    write rows) is the same mathematical object: stable counting sort == stable
    ascending argsort (buckets in ascending owner order, within-bucket in
    ascending flattened position p = r*quota + i). Full argument in the file
    header of ascendc/op_kernel/terrace_k1_arrival.cpp. r_idx uses ordo // quota
    instead of a table lookup: _expand_arrival_quota's pre-sorted r_idx is just
    arange(R) expanded by quota, so the row number of the p-th pair is always
    p // quota.

    Differentiability matches the live chain: send_buf (gather of rx) and
    gate_pairs (gather of flattened rgate) carry gradients; r_idx/slot_idx/
    i_send are index/count planes (integer, naturally gradient-free).

    my_local is unused by the math of this segment -- the K1 interface reserves
    it per spec for the expert-order rearrangement half after Hop B (the
    isomorphic bucket sort of exp_j = my_slot - my_local*epr); the kernel/tiling
    already carries it.
    """
    R, q = rslot.shape
    assert q == quota, f"rslot dim 1 {q} != quota {quota} (C1 wire-format contract)"
    slot_flat = rslot.reshape(-1)
    owner = slot_flat // epr
    ordo = _stable_ordo(owner, rpn)
    r_idx = torch.div(ordo, quota, rounding_mode="floor")
    slot_idx = slot_flat[ordo]
    from ..ta2a_fwd import fixed_hist   # deferred import, avoids the module-level cycle
    i_send = fixed_hist(owner, rpn)     # the same fixed-length histogram as the live chain
    send_buf = rx[r_idx]
    gate_pairs = rgate.reshape(-1)[ordo]
    return send_buf, gate_pairs, r_idx, slot_idx, i_send


def _k1_grad_rx(g_send, r_idx: torch.Tensor, rx_shape) -> torch.Tensor:
    """Adjoint of the live chain's `send_buf = rx[r_idx]`. r_idx has duplicate rows
    (quota pairs per row); the add-reduction order = the index enumeration order,
    bit-for-bit the same as the autograd adjoint of index
    (guarded by tests/test_terrace_k1_arrival.py::test_fn_backward_formula_*)."""
    grad_rx = g_send.new_zeros(rx_shape)
    grad_rx.index_add_(0, r_idx, g_send)
    return grad_rx


def _k1_grad_rgate(g_gate, rslot: torch.Tensor, epr: int, rpn: int,
                   rgate_shape) -> torch.Tensor:
    """Adjoint of the live chain's `gate_pairs = flattened rgate[ordo]`. ordo is
    not a kernel output (downstream does not need it); here it is recomputed
    from rslot with the live-chain primitive (_stable_argsort_small is the
    0.107ms-class float32 composite-key sort, not the 5.32ms int64 stable sort;
    see the measurement comment on ta2a_fwd._stable_argsort_small). ordo is a
    permutation, so the scatter has no duplicates and the add introduces no
    reduction-order divergence."""
    ordo = _stable_ordo(rslot.reshape(-1) // epr, rpn)
    flat = g_gate.new_zeros(rgate_shape[0] * rgate_shape[1])
    flat.index_add_(0, ordo, g_gate)
    return flat.view(rgate_shape)


class TerraceK1ArrivalFn(torch.autograd.Function):
    """K1: forward calls the AscendC kernel; backward uses the live composed chain (roadmap decision).

    Backward semantics = the adjoints of the live chain's two gathers, all
    existing primitives, bit-exact (formulas in the two _k1_grad_* helpers
    above; the segment-graph entry point k1_arrival_segment shares the same
    ones):
      - adjoint of send_buf = rx[r_idx]:      grad_rx = zeros.index_add_(0, r_idx, g);
      - adjoint of gate_pairs = flattened rgate[ordo]: flattened
        grad_rgate.index_add_(0, ordo, g).

    Scope: callers whose backward walks the whole segment graph **in one pass**
    (the fused forward ta2a_moe_forward and the legacy 3-arg seam ta2a_permute).
    The vendor overlap seam enters the permute2 segment with two separate
    .backward() calls and cannot use this fused node -- see the _K1SendEdge
    docstring.
    """

    @staticmethod
    def forward(ctx, rx, rslot, rgate, quota, epr, rpn, my_local):
        send_buf, gate_pairs, r_idx, slot_idx, i_send = torch.ops.terrace.k1_arrival(
            rx, rslot, rgate, quota, epr, rpn, my_local)
        ctx.save_for_backward(rslot, r_idx)
        ctx.k1_geom = (quota, epr, rpn, rx.shape, rgate.shape)
        ctx.mark_non_differentiable(r_idx, slot_idx, i_send)
        return send_buf, gate_pairs, r_idx, slot_idx, i_send

    @staticmethod
    def backward(ctx, g_send, g_gate, _g_r, _g_s, _g_i):
        rslot, r_idx = ctx.saved_tensors
        quota, epr, rpn, rx_shape, rgate_shape = ctx.k1_geom
        grad_rx = None
        if ctx.needs_input_grad[0] and g_send is not None:
            grad_rx = _k1_grad_rx(g_send, r_idx, rx_shape)
        grad_rgate = None
        if ctx.needs_input_grad[2] and g_gate is not None:
            grad_rgate = _k1_grad_rgate(g_gate, rslot, epr, rpn, rgate_shape)
        return grad_rx, None, grad_rgate, None, None, None, None


def k1_arrival(rx: torch.Tensor, rslot: torch.Tensor, rgate: torch.Tensor,
               quota: int, epr: int, rpn: int, my_local: int = 0):
    """Arrival-side fused chain: (send_buf, gate_pairs, r_idx, slot_idx, i_send).

    Takes the custom op when the kernel is available (NPU), otherwise the
    composed-chain reference -- the two paths are bit-for-bit identical
    (tests/test_terrace_k1_arrival.py guards the CPU side; NPU bit-exactness is
    guarded by the cluster device smoke test, command in the header of the
    build script ascendc/build.sh). Callers (the entry points in ta2a_fwd /
    ta2a_dispatch) carry their own custom_ops_enabled() gate and take the
    verbatim live chain on fallback, without passing through this function --
    the fallback here is for direct calls/tests.
    """
    if custom_ops_enabled():
        return TerraceK1ArrivalFn.apply(rx, rslot, rgate, quota, epr, rpn, my_local)
    return k1_arrival_ref(rx, rslot, rgate, quota, epr, rpn, my_local)


# --------------------------------------------------------------------------------------
# K1 segment-graph version: for the 6-arg overlap seam only (2026-08-20)
# --------------------------------------------------------------------------------------

class _K1SendEdge(torch.autograd.Function):
    """Hang the kernel-computed send_buf back onto the rx leaf: zero work forward; backward = the adjoint of rx[r_idx].

    Why the overlap seam cannot use TerraceK1ArrivalFn directly: that is a
    **fused node**, shared by the token path (permute2_graph) and the gate path
    (permute2_prob_graph). The vendor gmm's hand-written backward enters the
    permute2 segment with **two** separate .backward() calls for those two roots
    (prob first, then token; step-by-step replication in
    tests/test_ta2a_overlap_seam.py) --
      1. the second call hits "Trying to backward through the graph a second
         time" (the first already freed that node's saved tensors), and the
         vendor code has no place to pass retain_graph;
      2. even if it did not blow up, the first call would first write the
         materialized **zero gradients** into the other path's .grad: one extra
         [pairs, H]-sized scatter, and -0.0 + x is no longer bit-safe in the
         sign bit.
    The live chain has no such problem: the two gather subgraphs are disjoint,
    each hanging only under its own detach leaf. So the segment-graph version
    separates the kernel's **data** output from the **graph**: the kernel runs
    once (no_grad), and the two float outputs each hang on an independent edge
    rooted at rx_d / rgate_d respectively -- isomorphic to the live chain; the
    vendor orchestration runs unchanged, and the seat contract (7+3), the
    detach boundary, and the splits handoff all stay put.
    """

    @staticmethod
    def forward(ctx, rx, r_idx, send_buf):
        ctx.save_for_backward(r_idx)
        ctx.rx_shape = rx.shape
        return send_buf            # already produced by the kernel; autograd aliases it and attaches grad_fn

    @staticmethod
    def backward(ctx, g_send):
        (r_idx,) = ctx.saved_tensors
        grad_rx = None
        if ctx.needs_input_grad[0] and g_send is not None:
            grad_rx = _k1_grad_rx(g_send, r_idx, ctx.rx_shape)
        return grad_rx, None, None


class _K1GateEdge(torch.autograd.Function):
    """Hang the kernel-computed gate_pairs back onto the rgate leaf. See the _K1SendEdge docstring.

    The gradient lands back in rgate's [R, quota] shape (the same shape as the
    live chain's reshape backward); the vendor replays row-count splits along
    Hop A with no awareness of the layout.
    """

    @staticmethod
    def forward(ctx, rgate, rslot, gate_pairs, epr, rpn):
        ctx.save_for_backward(rslot)
        ctx.k1_gate_geom = (epr, rpn, rgate.shape)
        return gate_pairs

    @staticmethod
    def backward(ctx, g_gate):
        (rslot,) = ctx.saved_tensors
        epr, rpn, rgate_shape = ctx.k1_gate_geom
        grad_rgate = None
        if ctx.needs_input_grad[0] and g_gate is not None:
            grad_rgate = _k1_grad_rgate(g_gate, rslot, epr, rpn, rgate_shape)
        return grad_rgate, None, None, None, None


def k1_arrival_segment(rx: torch.Tensor, rslot: torch.Tensor, rgate: torch.Tensor,
                       quota: int, epr: int, rpn: int, my_local: int = 0):
    """The **segment-graph version** of K1: returns bit-for-bit the same values as k1_arrival; the graph has the same shape as the live chain's.

    The kernel (the reference implementation on fallback) runs only once, under
    no_grad, computing the **data** from detached inputs; the graph is rebuilt
    from two disjoint edges -- send_buf hangs back onto rx, gate_pairs onto
    rgate. r_idx / slot_idx / i_send are integer index/count planes, naturally
    outside the gradient, and are handed over as-is (they pass through no
    Function, so no fake gradients get materialized for them).

    The caller is the arrival segment of ta2a_dispatch.ta2a_permute_overlap
    (inside the permute2 segment graph, under the detach leaves rx_d /
    rgate_d); callers with a single-pass backward keep using k1_arrival.
    """
    with torch.no_grad():
        send_buf, gate_pairs, r_idx, slot_idx, i_send = k1_arrival(
            rx.detach(), rslot, rgate.detach(), quota, epr, rpn, my_local)
    send_buf = _K1SendEdge.apply(rx, r_idx, send_buf)
    gate_pairs = _K1GateEdge.apply(rgate, rslot, gate_pairs, epr, rpn)
    return send_buf, gate_pairs, r_idx, slot_idx, i_send


# --------------------------------------------------------------------------------------
# K2 interface draft (comment only; finalize against the chain after C1 lands --
# do not write callers against this)
#
#   terrace::k2_pack(Tensor hidden, Tensor expert_idx, Tensor gates,
#                    int world, int n_experts, int rpn, int groups_m)
#       -> (Tensor payload, Tensor mask, Tensor gate_rows,
#           Tensor u_src, Tensor node_counts)
#     Replaces the whole stretch from plan_ta2a(...) to _pack_quota_wire(...)
#     (ta2a_permute :106-:121; the overlap half shares the same stretch). The
#     equal-quota fast path is fully shape-static; u_src/node_counts must still
#     be handed back (the combine half and the Hop A count exchange need them).
#     The C1 packing side (_pack_quota_wire) is K2's territory; K1 does not
#     touch it.
# --------------------------------------------------------------------------------------

__all__ = [
    "OpsLoadError", "OpsState", "TerracePassthroughFn", "TerraceK1ArrivalFn",
    "custom_ops_enabled", "k1_arrival", "k1_arrival_ref", "k1_arrival_segment",
    "passthrough", "reset", "status",
]
