"""CPU-side bit-level contract and integration-switch semantics for K1
(terrace_k1_arrival, the arrival-side fused chain).

The kernel itself runs on the cluster NPUs (bit-level validation: the k1 smoke
command in the header comment of the build script ascendc/build.sh + an internal
benchmark script, not shipped with the repo); this file guards **everything that
can be proven locally**:

  1. Executable spec: terrace.ops.k1_arrival_ref (pure torch, a bitwise mirror
     of the kernel semantics) is reconciled bit-for-bit against the current
     composite chain **verbatim** (_expand_arrival_quota + owner stable bucket
     sort + bincount + two gathers, copied line by line) — multiple geometries x
     multiple dtypes, incl. degenerate inputs (C1 zeros-containment rows, R=0);
     the backward (adjoint of the composite chain) is bitwise as well.
  2. Fallback path: with no .so, the k1_arrival wrapper takes the reference
     implementation and the results match the current chain bit for bit; the
     formula of TerraceK1ArrivalFn.backward (the backward the kernel path will
     actually run) is reconciled bit-for-bit against the composite chain's
     autograd.
  3. Switch semantics: k1_arrival is listed in _REQUIRED_OPS (an old .so
     degrades as a whole, no partial registration); "0" is an explicit off,
     "require" must blow up when there is no .so — the same contract as
     passthrough.
  4. Wiring proof (gloo, world 4 / rpn 2): swap terrace.ops.k1_arrival for a
     counting reference implementation, force the gate open; the fused forward
     and the legacy 3-arg seam's forward + all gradients must be bitwise equal
     to the current chain without K1, and the fake kernel's call count must be
     > 0 (live-path evidence — equivalence assertions are blind to "the gate
     never opened"; the silent-failure discipline of the internal engineering
     records).

Why the reference implementation is enough to represent the kernel on CPU: the
two are the same mathematical object (a stable counting sort); the
bitwise-consistency argument between the two-pass method (count -> prefix
cursors -> expanded row writes) and a stable ascending argsort is in the header
of terrace/ops/ascendc/op_kernel/terrace_k1_arrival.cpp; CPU unit tests cannot
exercise device behavior (internal engineering records), so the NPU bit level is
guarded separately by cluster smoke runs — this file does not pretend to be
them.
"""
from __future__ import annotations

import os
import sys
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import terrace.ops as tops  # noqa: E402
from terrace.ta2a_fwd import (_expand_arrival_quota,  # noqa: E402
                              _stable_argsort_small)


@pytest.fixture()
def clean_ops(monkeypatch):
    """One independent verdict per case (same pattern as
    test_terrace_ops_scaffold): clear env, clear caches."""
    monkeypatch.delenv("TERRACE_CUSTOM_OPS", raising=False)
    monkeypatch.delenv("TERRACE_OPS_LIB", raising=False)
    tops.reset()
    yield monkeypatch
    tops.reset()


def _chain(rx, rslot, rgate, quota, epr, rpn):
    """The current composite chain, verbatim (the arrival-segment quota branch of
    ta2a_fwd.ta2a_moe_forward / ta2a_dispatch.ta2a_permute, i.e. a line-by-line
    copy of the integration point's else branch) — K1's functional spec."""
    r_idx, slot_idx = _expand_arrival_quota(rslot)
    owner = slot_idx // epr
    ordo = _stable_argsort_small(owner, rpn)
    r_idx, slot_idx = r_idx[ordo], slot_idx[ordo]
    i_send = torch.bincount(owner, minlength=rpn)
    return rx[r_idx], rgate.reshape(-1)[ordo], r_idx, slot_idx, i_send


def _mk(R, quota, epr, rpn, H, dtype, seed, degenerate=False):
    """Arrival plane of the C1 wire format: each row holds ascending,
    duplicate-free slot ids (the construction in _pack_quota_wire), or all-zero
    rows when degenerate=True (zeros containment: straggler rows = slot 0 / gate 0)."""
    g = torch.Generator().manual_seed(seed)
    slots = epr * rpn
    assert quota <= slots
    if degenerate:
        rslot = torch.zeros(R, quota, dtype=torch.int64)
    else:
        scores = torch.rand(R, slots, generator=g)
        rslot = torch.sort(torch.topk(scores, quota, dim=1).indices,
                           dim=1).values.to(torch.int64)
    rx = torch.randn(R, H, generator=g).to(dtype)
    rgate = torch.rand(R, quota, generator=g).to(dtype)
    return rx, rslot, rgate


NAMES = ("send_buf", "gate_pairs", "r_idx", "slot_idx", "i_send")

# Geometry coverage: quota 1/2/3/4/5, epr 1..8, rpn 1..16, slots spanning 4..63,
# H not a power of 2.
GEOMS = [
    # (R, quota, epr, rpn, H)
    (16, 2, 2, 8, 64),     # matches the testbed geometry (slots 16)
    (64, 1, 2, 8, 32),     # quota=1: the r_idx == ordo degenerate case
    (33, 3, 2, 4, 48),     # non-power-of-2 R/quota
    (128, 4, 4, 8, 16),    # slots 32
    (7, 2, 8, 4, 96),      # epr>rpn
    (1, 1, 1, 1, 24),      # minimal geometry
    (256, 2, 2, 16, 8),    # rpn 16
    (40, 5, 3, 4, 40),     # odd geometry where quota does not divide slots (slots 12)
    (48, 7, 9, 7, 8),      # slots 63: the current chain's int64 mask upper-bound geometry
]


# ======================================================================================
# 1. Executable spec: reference impl == current chain verbatim, bitwise
#    (multi-geometry x multi-dtype >= 8 cases)
# ======================================================================================

@pytest.mark.parametrize("geom", GEOMS, ids=lambda g: f"R{g[0]}q{g[1]}e{g[2]}r{g[3]}")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_ref_bitwise_equals_chain(geom, dtype):
    R, quota, epr, rpn, H = geom
    for seed in range(2):
        rx, rslot, rgate = _mk(R, quota, epr, rpn, H, dtype, 100 + seed)
        want = _chain(rx, rslot, rgate, quota, epr, rpn)
        got = tops.k1_arrival_ref(rx, rslot, rgate, quota, epr, rpn)
        for name, w, g in zip(NAMES, want, got):
            assert w.dtype == g.dtype and w.shape == g.shape, name
            assert torch.equal(w, g), f"{name} not bitwise equal to the current chain (seed {seed})"


def test_ref_bitwise_on_degenerate_rows():
    """C1 zeros containment: all-zero slot rows (slot 0 x quota) are legal wire
    input; still bitwise."""
    R, quota, epr, rpn, H = 24, 2, 2, 8, 32
    rx, rslot, rgate = _mk(R, quota, epr, rpn, H, torch.float32, 5, degenerate=True)
    for name, w, g in zip(NAMES, _chain(rx, rslot, rgate, quota, epr, rpn),
                          tops.k1_arrival_ref(rx, rslot, rgate, quota, epr, rpn)):
        assert torch.equal(w, g), name


def test_ref_bitwise_on_empty_arrival():
    """R=0 (a rank that received no rows): all five outputs empty/zero, same
    shape and value as the current chain."""
    quota, epr, rpn, H = 2, 2, 8, 16
    rx = torch.zeros(0, H)
    rslot = torch.zeros(0, quota, dtype=torch.int64)
    rgate = torch.zeros(0, quota)
    want = _chain(rx, rslot, rgate, quota, epr, rpn)
    got = tops.k1_arrival_ref(rx, rslot, rgate, quota, epr, rpn)
    for name, w, g in zip(NAMES, want, got):
        assert torch.equal(w, g), name
    assert got[4].shape == (rpn,) and int(got[4].sum()) == 0


def test_ref_backward_bitwise_equals_chain_autograd():
    """The reference implementation's backward (gather adjoint) is bitwise equal
    to the current chain's autograd."""
    R, quota, epr, rpn, H = 12, 2, 2, 4, 8
    rx0, rslot, rgate0 = _mk(R, quota, epr, rpn, H, torch.float32, 9)
    g = torch.Generator().manual_seed(1)
    gs = torch.randn(R * quota, H, generator=g)
    gg = torch.randn(R * quota, generator=g)

    rx1, rg1 = rx0.clone().requires_grad_(True), rgate0.clone().requires_grad_(True)
    s1, p1, *_ = _chain(rx1, rslot, rg1, quota, epr, rpn)
    ((s1 * gs).sum() + (p1 * gg).sum()).backward()
    rx2, rg2 = rx0.clone().requires_grad_(True), rgate0.clone().requires_grad_(True)
    s2, p2, *_ = tops.k1_arrival_ref(rx2, rslot, rg2, quota, epr, rpn)
    ((s2 * gs).sum() + (p2 * gg).sum()).backward()
    assert torch.equal(rx1.grad, rx2.grad)
    assert torch.equal(rg1.grad, rg2.grad)


# ======================================================================================
# 2. Fallback path: with no .so the wrapper == current chain; the kernel path's
#    backward formula bitwise
# ======================================================================================

def test_wrapper_falls_back_bitwise_without_so(clean_ops):
    assert tops.custom_ops_enabled() is False, "local no-CANN premise broken?"
    R, quota, epr, rpn, H = 16, 2, 2, 8, 64
    rx, rslot, rgate = _mk(R, quota, epr, rpn, H, torch.bfloat16, 3)
    want = _chain(rx, rslot, rgate, quota, epr, rpn)
    got = tops.k1_arrival(rx, rslot, rgate, quota, epr, rpn, my_local=1)
    for name, w, g in zip(NAMES, want, got):
        assert torch.equal(w, g), f"fallback path {name} not bitwise equal to the current chain"


def test_fn_backward_formula_bitwise_equals_chain_autograd():
    """The formula of TerraceK1ArrivalFn.backward (the backward the kernel path
    really runs on the cluster: ordo recompute + two index_add) reconciled
    bit-for-bit against the current chain's autograd — no .so needed: the
    formula is a pure composite chain, so call the static method directly with a
    hand-built ctx."""
    R, quota, epr, rpn, H = 20, 3, 2, 4, 16
    rx0, rslot, rgate0 = _mk(R, quota, epr, rpn, H, torch.float32, 21)
    g = torch.Generator().manual_seed(2)
    gs = torch.randn(R * quota, H, generator=g)
    gg = torch.randn(R * quota, generator=g)

    rx1, rg1 = rx0.clone().requires_grad_(True), rgate0.clone().requires_grad_(True)
    s1, p1, r_idx, _, _ = _chain(rx1, rslot, rg1, quota, epr, rpn)
    ((s1 * gs).sum() + (p1 * gg).sum()).backward()

    ctx = types.SimpleNamespace(
        saved_tensors=(rslot, r_idx),
        k1_geom=(quota, epr, rpn, rx0.shape, rgate0.shape),
        needs_input_grad=(True, False, True, False, False, False, False))
    grad_rx, _, grad_rgate, *_ = tops.TerraceK1ArrivalFn.backward(ctx, gs, gg,
                                                                  None, None, None)
    assert torch.equal(grad_rx, rx1.grad)
    assert torch.equal(grad_rgate, rg1.grad)


# ======================================================================================
# 3. Switch semantics (k1-specific parts; the generic contract belongs to
#    test_terrace_ops_scaffold)
# ======================================================================================

def test_k1_is_in_required_ops():
    """An old .so containing only passthrough must degrade as a whole — a
    half-new, half-old op set is worse than slow."""
    assert "k1_arrival" in tops._REQUIRED_OPS


def test_switch_off_uses_ref_without_load_attempt(clean_ops):
    clean_ops.setenv("TERRACE_CUSTOM_OPS", "0")
    rx, rslot, rgate = _mk(8, 2, 2, 4, 16, torch.float32, 11)
    got = tops.k1_arrival(rx, rslot, rgate, 2, 2, 4)
    want = _chain(rx, rslot, rgate, 2, 2, 4)
    for w, g in zip(want, got):
        assert torch.equal(w, g)
    assert tops.status().requested == "0"


def test_switch_require_fails_hard_on_k1_call(clean_ops):
    clean_ops.setenv("TERRACE_CUSTOM_OPS", "require")
    rx, rslot, rgate = _mk(8, 2, 2, 4, 16, torch.float32, 12)
    with pytest.raises(RuntimeError, match="TERRACE_CUSTOM_OPS"):
        tops.k1_arrival(rx, rslot, rgate, 2, 2, 4)


# ======================================================================================
# 4. Wiring proof (gloo distributed layer): forced kernel path == current chain,
#    and the fake kernel really gets called
# ======================================================================================

WORLD, RPN, T, K, M, E, H, D = 4, 2, 8, 4, 2, 8, 6, 4


def _routing(gen, n_nodes, per, quota):
    rows = []
    for _ in range(T):
        gs = torch.randperm(n_nodes, generator=gen)[:M]
        rows.append(torch.cat([torch.arange(a * per, (a + 1) * per)[
            torch.randperm(per, generator=gen)[:quota]] for a in gs]))
    return torch.stack(rows)


def _run(rank, world, q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # Ports in use: 29577/29591/29613/29623/29627/29641/29645/29661/29665.
    os.environ.setdefault("MASTER_PORT", "29677")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        import terrace.ops as ops_mod
        from terrace.layer import grouped_mm
        from terrace.ta2a_fwd import ta2a_moe_forward, init_ta2a_groups
        from terrace.ta2a_dispatch import ta2a_permute, ta2a_unpermute

        intra = init_ta2a_groups(world, RPN)
        epr = E // world
        n_nodes, per, quota = world // RPN, E // (world // RPN), K // M

        calls = [0]
        real_enabled, real_k1 = ops_mod.custom_ops_enabled, ops_mod.k1_arrival

        def fake_k1(rx, rslot, rgate, quota_, epr_, rpn_, my_local=0):
            # Wiring assertions: the geometry the integration point hands the
            # kernel must be self-consistent (a swapped argument can still look
            # right in the equivalence assertions — e.g. a geometry where epr
            # and rpn coincide — so pin it down right here).
            assert rx.dim() == 2 and rslot.shape == rgate.shape
            assert rslot.shape[1] == quota_ == quota
            assert epr_ == epr and rpn_ == RPN
            assert my_local == rank % RPN
            assert rslot.dtype == torch.int64 and rgate.dtype == rx.dtype
            calls[0] += 1
            return ops_mod.k1_arrival_ref(rx, rslot, rgate, quota_, epr_, rpn_,
                                          my_local)

        def one_pass(forced):
            if forced:
                ops_mod.custom_ops_enabled = lambda: True
                ops_mod.k1_arrival = fake_k1
            try:
                g = torch.Generator().manual_seed(31 + rank)
                x0 = torch.randn(T, H, generator=g)
                idx = _routing(g, n_nodes, per, quota)
                gates0 = torch.rand(T, K, generator=g)
                w13_0 = torch.randn(epr, H, 2 * D, generator=g) / (H ** 0.5)
                w2_0 = torch.randn(epr, D, H, generator=g) / (D ** 0.5)
                G = torch.randn(T, H, generator=g)
                routing_map = torch.zeros(T, E, dtype=torch.bool)
                routing_map[torch.arange(T).unsqueeze(1), idx] = True
                probs_dense = torch.zeros(T, E)
                probs_dense[torch.arange(T).unsqueeze(1), idx] = gates0

                # Entry 1: fused forward (the integration point in ta2a_fwd)
                xF = x0.clone().requires_grad_(True)
                gF = gates0.clone().requires_grad_(True)
                w13F = w13_0.clone().requires_grad_(True)
                w2F = w2_0.clone().requires_grad_(True)
                yF = ta2a_moe_forward(xF, idx, gF, w13F, w2F, world, E, RPN,
                                      groups_m=M)
                yF.backward(G)

                # Entry 2: legacy 3-arg seam (the integration point in ta2a_dispatch)
                hidL = x0.clone().requires_grad_(True)
                probL = probs_dense.clone().requires_grad_(True)
                w13L = w13_0.clone().requires_grad_(True)
                w2L = w2_0.clone().requires_grad_(True)
                permL, tpeL, ppL, stL = ta2a_permute(
                    hidL, probL, routing_map, world=world, rank=rank, rpn=RPN,
                    n_experts=E, intra_group=intra, inter_group=None, groups_m=M)
                a, b = grouped_mm(permL, w13L, tpeL).chunk(2, dim=-1)
                eoL = grouped_mm(F.silu(a) * b, w2L, tpeL) * ppL.unsqueeze(-1)
                outL = ta2a_unpermute(eoL, stL, hidL)
                outL.backward(G)

                return {"yF": yF.detach(), "xF": xF.grad, "gF": gF.grad,
                        "w13F": w13F.grad, "w2F": w2F.grad,
                        "yL": outL.detach(), "xL": hidL.grad, "pL": probL.grad,
                        "w13L": w13L.grad, "w2L": w2L.grad,
                        "tpe": tpeL, "perm": permL.detach(), "pp": ppL.detach()}
            finally:
                ops_mod.custom_ops_enabled = real_enabled
                ops_mod.k1_arrival = real_k1

        base = one_pass(forced=False)
        calls_baseline = calls[0]           # must still be 0: fake kernel unreachable unless forced
        forced = one_pass(forced=True)
        diffs = {}
        for name in base:
            same = torch.equal(base[name], forced[name])
            diffs[name] = None if same else str(base[name].dtype)
        q.put({"rank": rank, "status": "ok", "diffs": diffs,
               "calls": calls[0], "calls_baseline": calls_baseline})
    except Exception:                                      # noqa: BLE001
        import traceback
        q.put({"rank": rank, "status": "err", "trace": traceback.format_exc()})
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def k1_wiring():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_run, args=(r, WORLD, q)) for r in range(WORLD)]
    for p in procs:
        p.start()
    out = [q.get(timeout=240) for _ in range(WORLD)]
    for p in procs:
        p.join(timeout=60)
    for r in out:
        assert r["status"] == "ok", f"rank {r['rank']}:\n{r.get('trace')}"
    return out


@pytest.mark.timeout(300)
def test_k1_forced_path_bitwise_equals_chain(k1_wiring):
    """With the kernel path forced (fake kernel = reference implementation), both
    integration points' forward and all gradients must be bitwise equal to the
    current chain — argument wiring, tensor hand-off, and autograd stitching at
    the integration points are all correct."""
    for r in k1_wiring:
        for name, d in r["diffs"].items():
            assert d is None, (
                f"rank {r['rank']}: {name} forced K1 path not bitwise equal to "
                f"the current chain ({d})")


@pytest.mark.timeout(300)
def test_k1_forced_path_is_live(k1_wiring):
    """Live-path evidence: the fused forward + the legacy seam each cross the
    integration point once, so the fake kernel is called exactly 2 times; the
    unforced baseline pass must be 0 (the gate defaults to hard-off).
    Equivalence assertions are blind to "the gate never opened"; if this test
    fails, the one above proved nothing (internal engineering records)."""
    for r in k1_wiring:
        assert r["calls_baseline"] == 0, f"rank {r['rank']}: gate open on the baseline pass"
        assert r["calls"] == 2, (
            f"rank {r['rank']}: fake kernel called {r['calls']} times, expected 2 "
            f"(fused forward 1 + legacy seam 1) — the integration points did not "
            f"take the kernel branch")


# ======================================================================================
# Engineering discipline: this file itself LF + py_compile (files under ops/ are
# locked by test_terrace_ops_scaffold)
# ======================================================================================

def test_this_file_compiles_and_is_lf():
    import py_compile
    py_compile.compile(__file__, doraise=True)
    with open(__file__, "rb") as f:
        assert b"\r\n" not in f.read()


def test_gm_ub_gm_kernels_declare_both_vecin_and_vecout():
    """Kernels doing GM→UB→GM **must declare both a VECIN and a VECOUT queue**.

    A pitfall measured on 2026-08-24 (first on passthrough, then found in K1 in
    the exact same shape): in AscendC, a `TQue`'s **position** decides which two
    pipelines it synchronizes —
        VECIN 's EnQue/DeQue pairs MTE2 -> V
        VECOUT's EnQue/DeQue pairs V    -> MTE3
    Declare only VECIN and then issue MTE3 (DataCopy to GM) straight from the
    VECIN LocalTensor, and **nobody inserts** that MTE2→MTE3 barrier: what gets
    copied out may not have finished arriving, or may already have been
    overwritten by the next round's AllocTensor reuse.

    The symptom is a **silent data error**, not a compile error and not a
    crash — on passthrough it showed up as "output mostly zeros" (8x256 with
    2047/2048 wrong), while op build/load/execute stayed all green. K1's CopyRow
    at the time copied passthrough's broken template word for word, with a
    comment that even said "same as passthrough".
    **The guardrail sits at the source level because the compiler does not
    report this class of error, loading does not report it, and only bitwise
    comparison exposes it.**
    """
    import glob
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kdir = os.path.join(root, "terrace/ops/ascendc/op_kernel")
    if not os.path.isdir(kdir):
        pytest.skip("kernel directory absent")
    checked = 0
    for path in sorted(glob.glob(os.path.join(kdir, "*.cpp"))):
        src = open(path, encoding="utf-8").read()
        # Only kernels that DataCopy from UB back to GM are in scope; pure
        # readers and scalar-only writers are not
        writes_gm = re.search(r"DataCopy\(\s*\w*[Gg]m", src) is not None
        if not writes_gm:
            continue
        checked += 1
        name = os.path.basename(path)
        assert "QuePosition::VECIN" in src, "%s writes GM but has no VECIN queue" % name
        assert "QuePosition::VECOUT" in src, (
            "%s does a GM→UB→GM transfer but has **no VECOUT queue** — the "
            "MTE2/MTE3 barrier is inserted by the queue framework pairing on "
            "position; without VECOUT nobody inserts it and dirty data goes out. "
            "This is the silent error measured on passthrough on 2026-08-24." % name)
    assert checked >= 2, (
        "only %d GM-writing kernels scanned; the guardrail may not be scanning "
        "anything" % checked)


def test_custom_ops_default_is_off_not_on():
    """**With TERRACE_CUSTOM_OPS unset, the custom ops must be off.**

    The 2026-08-24 incident: the default was "1", so the moment the `.so` first
    compiled successfully (07:25, k1-rebuild), the K1 kernel — which had not
    passed bitwise validation — automatically entered the training dispatch
    path. Every T-A2A on-arm run after that blew up at step 0 on all 128 ranks
    at once:
        RuntimeError: Split sizes dosen't match total dim 0 size
    (K1's slot_idx indexing wrong -> i_send wrong -> Hop B splits mismatch)
    Two verdict-testbed runs (r4 and isub) burned for nothing, while the runner
    still reported "done" and wrote the DONE flag.

    bitcheck's verdict line had said "K1 must not go on the testbed" all along,
    but **no mechanism enforced it** — the only gate was "does the .so dlopen".
    A successful compile ≠ correct arithmetic.

    What this guardrail pins is that default: **no sign-off, no kernel on the
    path.**
    """
    import importlib
    saved = os.environ.pop("TERRACE_CUSTOM_OPS", None)
    try:
        importlib.reload(tops)
        assert tops._normalized_switch() == "0", (
            "TERRACE_CUSTOM_OPS unset normalized to %r — must be '0'. "
            "Default-on means any single successful compile ships an unvalidated "
            "kernel into the training path."
            % tops._normalized_switch())
        # Reverse control: an explicit request must really turn it on, otherwise
        # this guardrail locks the feature out for good
        os.environ["TERRACE_CUSTOM_OPS"] = "1"
        assert tops._normalized_switch() == "1"
        os.environ["TERRACE_CUSTOM_OPS"] = "require"
        assert tops._normalized_switch() == "require"
    finally:
        os.environ.pop("TERRACE_CUSTOM_OPS", None)
        if saved is not None:
            os.environ["TERRACE_CUSTOM_OPS"] = saved
        importlib.reload(tops)
