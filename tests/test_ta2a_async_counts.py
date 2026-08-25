"""A3-lite (2026-08-21): issue Hop A's counts exchange early, wait asynchronously --
zero bit-level change + live path.

Why this file exists: A3 (constant-izing the plan to kill both counts exchanges) was
ruled a **net loss of −7.03 ms, not implemented** (internal design record, not
shipped with this repo: equal quota only fixes the per-token fanout M, not the
per-destination load; the capacity-padding upper bound can only take the geometric
worst case n_nodes/M = 8x and rpn = 8x, and the redundant bytes far exceed the α
saved; the tokens_per_expert hard probe and the fail-loud overflow are two more
structural blockers). **The only salvageable fragment** is this file: the inter
counts exchange depends only on the node_counts handed over by plan_ta2a and does
not have to wait for the local gather + packing to finish -- issue it right after
plan with `async_op=True`, move the wait to where the splits are actually needed,
and cover its α₁₂₈ with local work.

**It must change nothing**: the same set of collectives, the same order (it is still
the first collective on the inter group), the same splits, the same numerics. So the
criteria are:
  1. **Bit-level**: on/off arms A/B in the same process; both seams' forward, four
     leaf gradients, and the four detach-leaf `.grad` values read by the vendor's
     manual replay are all `torch.equal`;
  2. **Same collective count**: async only moves the issue/wait positions, not the
     count -- both arms' fwd/bwd counts are equal item by item (absolute values are
     guarded by test_ta2a_gate_at_arrival; this file locks "the two arms are
     equal");
  3. **Same splits**: `st.send_l / st.recv_l` (= vendor disp.input_splits/
     output_splits) equal item by item across arms -- issuing early must not alter
     the two lists handed to the vendor;
  4. **Live path**: the on arm must actually go through `async_op=True` (recording
     the async flag of each counts exchange), and the off arm must never do so.
     Equivalence assertions are blind to "the switch never opened" (internal
     engineering record).

Separate file, separate commit: on the testbed the A3-lite readings must be
attributable separately from the bundled A1'/A2 readings (switch
TERRACE_TA2A_ASYNC_COUNTS=0 returns to synchronous issue at the original position).

gloo / CPU, world 4, rpn 2 (2 nodes, with a real cross-node hop).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import terrace.ta2a_dispatch as disp_mod  # noqa: E402

WORLD, RPN, T, K, M, E, H, D = 4, 2, 8, 4, 2, 8, 6, 4


def test_switch_defaults_on_and_can_be_turned_off(monkeypatch):
    """Switch: unset / non-"0" = on (factory default); "0" = off, back to
    synchronous issue at the original position."""
    monkeypatch.delenv("TERRACE_TA2A_ASYNC_COUNTS", raising=False)
    disp_mod.reset_async_counts()
    assert disp_mod.async_counts_enabled() is True
    monkeypatch.setenv("TERRACE_TA2A_ASYNC_COUNTS", "0")
    disp_mod.reset_async_counts()
    assert disp_mod.async_counts_enabled() is False
    monkeypatch.setenv("TERRACE_TA2A_ASYNC_COUNTS", "1")
    disp_mod.reset_async_counts()
    assert disp_mod.async_counts_enabled() is True
    monkeypatch.delenv("TERRACE_TA2A_ASYNC_COUNTS", raising=False)
    disp_mod.reset_async_counts()


def test_counts_buffers_are_geometry_only():
    """_hopa_counts depends only on geometry + node_counts: the entire justification
    for issuing early lives in this one fact.

    It reads no payload, no mask, no gate -- so moving it to after plan_ta2a and
    before packing cannot possibly read anything that has not been computed yet.
    """
    dev = torch.device("cpu")
    n_nodes = WORLD // RPN
    node_counts = torch.tensor([3, 5], dtype=torch.long)
    send, recv = disp_mod._hopa_counts(WORLD, n_nodes, RPN, 1, dev, node_counts)
    assert send.shape == (WORLD,) and recv.shape == (WORLD,)
    assert send.dtype == torch.long and recv.dtype == torch.long
    # node n's designated peer = n*rpn + my_local; every other position is always 0
    assert send.tolist() == [0, 3, 0, 5]
    assert int(send.sum()) == int(node_counts.sum())


def _routing(gen, n_nodes, per, quota):
    rows = []
    for _ in range(T):
        gs = torch.randperm(n_nodes, generator=gen)[:M]
        rows.append(torch.cat([torch.arange(a * per, (a + 1) * per)[
            torch.randperm(per, generator=gen)[:quota]] for a in gs]))
    return torch.stack(rows)


def _bitdiff(a, b):
    if a is None or b is None:
        return ("missing", None)
    if a.dtype != b.dtype:
        return (f"{a.dtype} vs {b.dtype}", float((a.float() - b.float()).abs().max()))
    if torch.equal(a, b):
        return None
    return (str(a.dtype), float((a.float() - b.float()).abs().max()))


def _run(rank, world, q, dtype_name):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # Already taken: 29577/29591/29613/29623/29627/29641/29645/29661/29665/29677/
    #                29681/29685/29697/29709/29713.
    os.environ.setdefault(
        "MASTER_PORT", "29725" if dtype_name == "float32" else "29729")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        import terrace.ta2a_dispatch as dm
        from terrace.layer import grouped_mm
        from terrace.ep_dist import _a2a_raw
        from terrace.ta2a_fwd import init_ta2a_groups
        from terrace.ta2a_dispatch import (ta2a_permute, ta2a_unpermute,
                                           ta2a_permute_overlap,
                                           ta2a_unpermute_overlap)

        pdt = getattr(torch, dtype_name)
        intra = init_ta2a_groups(world, RPN)
        epr = E // world
        n_nodes, per, quota = world // RPN, E // (world // RPN), K // M

        real_a2a = dist.all_to_all_single
        counter, seen = [0], {"async": 0, "sync": 0}
        real_switch = dm.async_counts_enabled

        def counting_a2a(*args, **kwargs):
            counter[0] += 1
            # The counts exchange's signature: 1-D long, length = group size.
            # Record whether it was issued async.
            x = args[1]
            if x.dim() == 1 and x.dtype == torch.long and x.shape[0] in (world, RPN):
                seen["async" if kwargs.get("async_op") else "sync"] += 1
            return real_a2a(*args, **kwargs)

        real_early = dm.early_hopb_counts_enabled

        def one_pass(async_on, gm):
            # **Flip both switches together**: A3-lite (early inter counts) and A6
            # (early intra counts) are two applications of the same mechanism. If
            # the control arm turned off only A3, A6's two async issues would still
            # be there, "sync arm async==0" would fail on the spot, and that failure
            # would look like A6 being broken when the real cause is a control arm
            # that was not switched off cleanly.
            dm.async_counts_enabled = (lambda: True) if async_on else (lambda: False)
            dm.early_hopb_counts_enabled = (lambda: True) if async_on else (lambda: False)
            g = torch.Generator().manual_seed(29 + rank)
            x0 = torch.randn(T, H, generator=g).to(pdt)
            idx = _routing(g, n_nodes, per, quota)
            gates0 = torch.rand(T, K, generator=g)          # fp32: router output precision
            w13_0 = (torch.randn(epr, H, 2 * D, generator=g) / (H ** 0.5)).to(pdt)
            w2_0 = (torch.randn(epr, D, H, generator=g) / (D ** 0.5)).to(pdt)
            G = torch.randn(T, H, generator=g).to(pdt)
            routing_map = torch.zeros(T, E, dtype=torch.bool)
            routing_map[torch.arange(T).unsqueeze(1), idx] = True
            probs_dense = torch.zeros(T, E)
            probs_dense[torch.arange(T).unsqueeze(1), idx] = gates0

            dist.all_to_all_single = counting_a2a
            try:
                hidL = x0.clone().requires_grad_(True)
                probL = probs_dense.clone().to(pdt).requires_grad_(True)
                w13L = w13_0.clone().requires_grad_(True)
                w2L = w2_0.clone().requires_grad_(True)
                counter[0] = 0
                permL, tpeL, ppL, stL = ta2a_permute(
                    hidL, probL, routing_map, world=world, rank=rank, rpn=RPN,
                    n_experts=E, intra_group=intra, inter_group=None, groups_m=gm)
                a, b = grouped_mm(permL, w13L, tpeL).chunk(2, dim=-1)
                eoL = grouped_mm(F.silu(a) * b, w2L, tpeL) * ppL.unsqueeze(-1)
                outL = ta2a_unpermute(eoL, stL, hidL)
                fwd = counter[0]
                counter[0] = 0
                outL.backward(G)
                bwd = counter[0]

                hidO = x0.clone().requires_grad_(True)
                probO = probs_dense.clone().requires_grad_(True)
                save = []
                permO, tpeO, ppO, _share, stO = ta2a_permute_overlap(
                    hidO, probO, routing_map, world=world, rank=rank, rpn=RPN,
                    n_experts=E, intra_group=intra, inter_group=None, groups_m=gm,
                    save_tensors=save, run_shared_experts=None)
                w13O = w13_0.clone().requires_grad_(True)
                w2O = w2_0.clone().requires_grad_(True)
                dI = permO.detach().requires_grad_(True)
                pI = ppO.detach().requires_grad_(True)
                a, b = grouped_mm(dI, w13O, tpeO).chunk(2, dim=-1)
                eoO = grouped_mm(F.silu(a) * b, w2O, tpeO) * pI.unsqueeze(-1)
                outO = ta2a_unpermute_overlap(eoO, stO, save)
                (p1g, p1p, _n, p2ind, p2g, p2pd, p2pg, u1ind, u1g, u2ind) = save

                outO.backward(G)
                grad_red = _a2a_raw(u2ind.grad, stO.send_l, stO.recv_l)
                u1g.backward(grad_red)
                eoO.backward(u1ind.grad)
                p2pg.backward(pI.grad)
                ggr = _a2a_raw(p2pd.grad, stO.recv_l, stO.send_l)
                p2g.backward(dI.grad)
                gpl = _a2a_raw(p2ind.grad, stO.recv_l, stO.send_l)
                torch.autograd.backward([p1g, p1p], grad_tensors=[gpl, ggr])
            finally:
                dist.all_to_all_single = real_a2a

            return {
                "L.out": outL.detach(), "L.gx": hidL.grad, "L.gp": probL.grad,
                "L.gw13": w13L.grad, "L.gw2": w2L.grad, "L.perm": permL.detach(),
                "O.out": outO.detach(), "O.gx": hidO.grad, "O.gp": probO.grad,
                "O.gw13": w13O.grad, "O.gw2": w2O.grad, "O.perm": permO.detach(),
                "O.p2in": p2ind.grad, "O.p2prob": p2pd.grad,
                "O.u1in": u1ind.grad, "O.u2in": u2ind.grad,
                "_counts": (fwd, bwd),
                "_splits": (list(stO.send_l), list(stO.recv_l),
                            list(stL.send_l), list(stL.recv_l)),
            }

        report = {}
        try:
            for gm in (M, None):
                seen["async"] = seen["sync"] = 0
                sync = one_pass(async_on=False, gm=gm)
                seen_sync = dict(seen)
                seen["async"] = seen["sync"] = 0
                asyn = one_pass(async_on=True, gm=gm)
                seen_async = dict(seen)
                diffs = {n: _bitdiff(asyn[n], sync[n])
                         for n in asyn if not n.startswith("_")}
                report[f"gm={gm}"] = {
                    "diffs": diffs,
                    "counts_same": asyn["_counts"] == sync["_counts"],
                    "counts": (asyn["_counts"], sync["_counts"]),
                    "splits_same": asyn["_splits"] == sync["_splits"],
                    "seen_async": seen_async, "seen_sync": seen_sync,
                }
        finally:
            dm.async_counts_enabled = real_switch
        dm.early_hopb_counts_enabled = real_early

        q.put({"rank": rank, "status": "ok", "report": report})
    except Exception:                                      # noqa: BLE001
        import traceback
        q.put({"rank": rank, "status": "err", "trace": traceback.format_exc()})
    finally:
        dist.destroy_process_group()


def _spawn(dtype_name):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_run, args=(r, WORLD, q, dtype_name))
             for r in range(WORLD)]
    for p in procs:
        p.start()
    out = [q.get(timeout=240) for _ in range(WORLD)]
    for p in procs:
        p.join(timeout=60)
    for r in out:
        assert r["status"] == "ok", f"rank {r['rank']}:\n{r.get('trace')}"
    return out


@pytest.fixture(scope="module")
def ac_fp32():
    return _spawn("float32")


@pytest.fixture(scope="module")
def ac_bf16():
    return _spawn("bfloat16")


def _assert_bitwise(results, label):
    for r in results:
        for tag, rep in r["report"].items():
            for name, d in rep["diffs"].items():
                assert d is None, (
                    f"rank {r['rank']} {label} {tag}: {name} async arm not bitwise"
                    f" equal to sync arm {d} -- A3-lite only moves the issue/wait"
                    f" positions and must not change any value")


def _assert_contract(results, label):
    for r in results:
        for tag, rep in r["report"].items():
            assert rep["counts_same"], (
                f"rank {r['rank']} {label} {tag}: collective counts differ between "
                f"arms {rep['counts']} -- going async moves the issue/wait "
                f"positions, not the count")
            assert rep["splits_same"], (
                f"rank {r['rank']} {label} {tag}: st.send_l/recv_l differ between "
                f"arms -- those two lists are exactly what gets handed to vendor "
                f"disp.input_splits/output_splits")


def _assert_live(results, label):
    for r in results:
        for tag, rep in r["report"].items():
            sa, ss = rep["seen_async"], rep["seen_sync"]
            # Two counts exchanges per seam (inter + intra), 4 total across both
            # seams. **All async as of 2026-08-22**: A3-lite covers the two inter
            # exchanges, A6 the two intra ones. A6's justification is in the
            # internal measurement records -- the five-tier attribution proved
            # "sending one fewer" is worth zero (A1'' exactly 0); only "moving it
            # to where real compute can cover it" pays (A3-lite −0.34 ms per
            # exchange, consistent across the full and small tiers). A6 moves the
            # intra counts ahead of the sort and the big [pairs,H] gather and uses
            # them as cover.
            assert sa["async"] == 4 and sa["sync"] == 0, (
                f"rank {r['rank']} {label} {tag}: async arm counts exchanges {sa}, "
                f"expected async=4 (two seams x inter+intra) / sync=0 -- "
                f"a switch is off or the early issue is not wired in (equivalence "
                f"assertions are blind to this, internal engineering record)")
            assert ss["async"] == 0 and ss["sync"] == 4, (
                f"rank {r['rank']} {label} {tag}: sync arm counts exchanges {ss}, "
                f"expected all-sync -- the A/B control arm is compromised")


@pytest.mark.timeout(300)
def test_fp32_async_counts_is_bitwise(ac_fp32):
    """fp32: both seams' forward, four leaf gradients, and four detach-leaf .grad
    values, async == sync."""
    _assert_bitwise(ac_fp32, "fp32")


@pytest.mark.timeout(300)
def test_fp32_async_counts_keeps_contract(ac_fp32):
    _assert_contract(ac_fp32, "fp32")


@pytest.mark.timeout(300)
def test_fp32_async_counts_is_live(ac_fp32):
    _assert_live(ac_fp32, "fp32")


@pytest.mark.timeout(300)
def test_bed_dtype_async_counts_is_bitwise(ac_bf16):
    """Bed dtype convention (bf16 payload/weights + fp32 router probs): same as
    above, bitwise."""
    _assert_bitwise(ac_bf16, "bf16")


@pytest.mark.timeout(300)
def test_bed_dtype_async_counts_keeps_contract(ac_bf16):
    _assert_contract(ac_bf16, "bf16")


@pytest.mark.timeout(300)
def test_bed_dtype_async_counts_is_live(ac_bf16):
    _assert_live(ac_bf16, "bf16")


def test_this_file_compiles_and_is_lf():
    import py_compile
    py_compile.compile(__file__, doraise=True)
    with open(__file__, "rb") as f:
        assert b"\r\n" not in f.read()


def test_early_hopb_switch_defaults_on_and_can_be_turned_off(monkeypatch):
    """Three-state self-check of the A6 switch (TERRACE_TA2A_EARLY_HOPB).

    Default on -- it is a **pure scheduling reorder**: the value of i_send, the
    alltoall semantics, the contents of i_recv, and every downstream bit stay
    identical; only the issue point moves earlier. Unlike A1'/A5 it touches no
    layout or reduction order, so it needs no eq gate. The switch exists only for
    A/B runs on the testbed.
    """
    monkeypatch.delenv("TERRACE_TA2A_EARLY_HOPB", raising=False)
    disp_mod.reset_early_hopb()
    assert disp_mod.early_hopb_counts_enabled() is True
    monkeypatch.setenv("TERRACE_TA2A_EARLY_HOPB", "0")
    disp_mod.reset_early_hopb()
    assert disp_mod.early_hopb_counts_enabled() is False
    monkeypatch.setenv("TERRACE_TA2A_EARLY_HOPB", "1")
    disp_mod.reset_early_hopb()
    assert disp_mod.early_hopb_counts_enabled() is True
    monkeypatch.delenv("TERRACE_TA2A_EARLY_HOPB", raising=False)
    disp_mod.reset_early_hopb()


def test_bincount_is_permutation_blind():
    """A6's premise: i_send = bincount(owner) does not depend on ordering, so it can
    be computed before the sort.

    This premise used to live only in a comment ("histogram is permutation-blind:
    skip owner[ordo]"). A6 stakes the whole communication's issue point on it -- a
    premise like that has to be executable, not just a sentence.
    """
    import torch
    g = torch.Generator().manual_seed(5)
    owner = torch.randint(0, 8, (1024,), generator=g)
    perm = torch.randperm(1024, generator=g)
    a = torch.bincount(owner, minlength=8)
    b = torch.bincount(owner[perm], minlength=8)
    assert torch.equal(a, b), ("bincount is not permutation-blind -- A6's premise "
                               "fails, must revert")


def test_sync_probe_is_off_by_default_and_present_in_both_seams():
    """The discriminating probe defaults to off, and **both seams carry it** (the
    verdict testbed runs the overlap one).

    Internal records falsified my original mechanism story for fixed_hist: measured
    host sync is worth only 0.042-0.046 ms, while the bincount it replaced is
    0.797 -- sync explains at most 6%. Yet fixed_hist really did get -1.166 on the
    machine (reproduced in two rounds). A gain with an unexplained mechanism cannot
    be used to justify the next cut, hence the discrimination. Instrumenting only
    the legacy seam would make the whole experiment run hollow -- the verdict
    testbed never touches it.
    """
    import re
    monkeypatch = None
    disp_mod.reset_sync_probe()
    assert disp_mod.sync_probe_enabled() is False, \
        "the discriminating probe must default to off"
    src = open(disp_mod.__file__, encoding="utf-8").read()
    hits = [m.start() for m in re.finditer(r"if sync_probe_enabled\(\):", src)]
    assert len(hits) == 2, ("probe should be inserted once in each of the legacy "
                            "and overlap seams; found %d" % len(hits))
    i_overlap = src.index("def ta2a_permute_overlap")
    assert any(h > i_overlap for h in hits), (
        "no probe in the overlap seam -- that is the one the verdict testbed runs; "
        "instrumenting only legacy makes the experiment run hollow")
