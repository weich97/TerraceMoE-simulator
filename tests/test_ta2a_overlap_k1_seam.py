"""Bit-level contract and wiring proof for K1 hooked into the overlap 6-arg seam
(inside the permute2 segment graph).

Why this file exists (the number-one cut from the 2026-08-20 byte audit): the K1
kernel has existed for a while, and both the legacy 3-arg seam and the fused forward
already use it, but **the verdict testbed runs the overlap 6-arg seam**, which was
deliberately left unwired until now -- the bulk of the on-arm dispatch residual of
6.2–8.8ms/call is exactly this arrival expansion (256 MiB nominal memory traffic +
10-plus operators). The hard part of wiring it in is not the math (same kernel, same
math as the legacy branch) but the **segment graph**:

  - The vendor gmm's hand-written backward calls .backward() **twice** into the same
    segment, once for permute2_graph and once for permute2_prob_graph. K1 is a fused
    node that produces both outputs at once; using it directly would
    (a) hit "backward through the graph a second time" on the second call (there is
    no retain_graph to pass),
    (b) have the first call write materialized zero gradients into the other path's
    .grad.
    So the integration point goes through terrace.ops.k1_arrival_segment: the kernel
    runs once outside the graph to produce the data, and each of the two float
    outputs hangs its own **independent** edge back to its detach leaf -- isomorphic
    to the two disjoint gather subgraphs of the current chain.
  - The seat contract (7+3), detach boundaries, splits handoff, and everything the
    vendor's hand-written backward can see must not change.

This file locks four layers, all torch.equal (bitwise), no tolerances:
  1. The data plane of the segment-graph wrapper k1_arrival_segment == the current
     chain verbatim (multiple geometries x multiple dtypes);
  2. The two edges are truly **independent**: two .backward() calls in the vendor's
     order do not blow up, and both gradient paths equal the current chain's
     autograd (also run as two backwards) bit for bit; the integer planes
     (r_idx/slot_idx/i_send) carry no grad;
  3. Seam level: with the overlap seam forced onto K1, the forward + four leaf
     gradients + the four detach-leaf .grad values the vendor reads are all bitwise
     equal to an **unforced** same-process baseline pass (= before-vs-after
     regression; the baseline pass is the pre-K1 current chain verbatim); within the
     same pass, the overlap arm and the legacy arm are bitwise equal (= cross-seam
     bitparity still holds with K1 on);
  4. Live-path evidence: in the forced pass, the segment-graph variant at the
     overlap entry is called exactly 1 time and the kernel exactly 2 times
     (1 legacy seam + 1 overlap segment graph); in the baseline pass both counters
     must be 0 -- equivalence assertions are blind to "the switch never opened"
     (internal engineering record).

gloo / CPU, world 4, rpn 2 (2 nodes, with a real cross-node hop); two beds: uniform
fp32 precision and the bed dtype convention (bf16 payload/weights + fp32 router
probs).
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

import terrace.ops as tops  # noqa: E402
from terrace.ta2a_fwd import (_expand_arrival_quota,  # noqa: E402
                              _stable_argsort_small)

WORLD, RPN, T, K, M, E, H, D = 4, 2, 8, 4, 2, 8, 6, 4

NAMES = ("send_buf", "gate_pairs", "r_idx", "slot_idx", "i_send")


def _chain(rx, rslot, rgate, quota, epr, rpn):
    """The current composition chain verbatim (line-by-line copy of the integration
    point's else branch) -- K1's functional spec."""
    r_idx, slot_idx = _expand_arrival_quota(rslot)
    owner = slot_idx // epr
    ordo = _stable_argsort_small(owner, rpn)
    r_idx, slot_idx = r_idx[ordo], slot_idx[ordo]
    i_send = torch.bincount(owner, minlength=rpn)
    return rx[r_idx], rgate.reshape(-1)[ordo], r_idx, slot_idx, i_send


def _mk(R, quota, epr, rpn, Hh, dtype, seed):
    """Arrival plane in the C1 wire format: each row holds ascending, non-repeating
    slot ids (the construction used by _pack_quota_wire)."""
    g = torch.Generator().manual_seed(seed)
    slots = epr * rpn
    assert quota <= slots
    scores = torch.rand(R, slots, generator=g)
    rslot = torch.sort(torch.topk(scores, quota, dim=1).indices,
                       dim=1).values.to(torch.int64)
    rx = torch.randn(R, Hh, generator=g).to(dtype)
    rgate = torch.rand(R, quota, generator=g).to(dtype)
    return rx, rslot, rgate


# Geometry coverage matches test_terrace_k1_arrival (including the alignment-bed
# slots=16 and the degenerate quota=1).
GEOMS = [
    (16, 2, 2, 8, 64),     # alignment-bed geometry (slots 16)
    (64, 1, 2, 8, 32),     # quota=1: degenerate case where r_idx == ordo
    (33, 3, 2, 4, 48),     # non-power-of-2 R/quota
    (7, 2, 8, 4, 96),      # epr>rpn
    (40, 5, 3, 4, 40),     # quota does not divide slots (slots 12)
]


# ======================================================================================
# 1. Segment-graph wrapper data plane == current chain verbatim
# ======================================================================================

@pytest.mark.parametrize("geom", GEOMS, ids=lambda g: f"R{g[0]}q{g[1]}e{g[2]}r{g[3]}")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_segment_forward_bitwise_equals_chain(geom, dtype):
    """k1_arrival_segment only changes how k1_arrival attaches to the graph; the
    data must be bitwise unchanged."""
    R, quota, epr, rpn, Hh = geom
    rx, rslot, rgate = _mk(R, quota, epr, rpn, Hh, dtype, 300)
    want = _chain(rx, rslot, rgate, quota, epr, rpn)
    got = tops.k1_arrival_segment(rx, rslot, rgate, quota, epr, rpn, my_local=1)
    for name, w, g in zip(NAMES, want, got):
        assert w.dtype == g.dtype and w.shape == g.shape, name
        assert torch.equal(w, g), f"segment-graph {name} not bitwise equal to the current chain"


def test_segment_integer_planes_carry_no_grad():
    """r_idx/slot_idx/i_send are integer index/count planes: never attached to the
    graph, never materialized into fake gradients."""
    R, quota, epr, rpn, Hh = 16, 2, 2, 8, 32
    rx, rslot, rgate = _mk(R, quota, epr, rpn, Hh, torch.float32, 301)
    rx.requires_grad_(True)
    rgate.requires_grad_(True)
    send_buf, gate_pairs, r_idx, slot_idx, i_send = tops.k1_arrival_segment(
        rx, rslot, rgate, quota, epr, rpn)
    assert send_buf.grad_fn is not None and gate_pairs.grad_fn is not None, \
        "the two edges are not attached to the graph -- the vendor backward_func " \
        "returns immediately when it sees grad_fn is None"
    assert send_buf.grad_fn is not gate_pairs.grad_fn, \
        "the two paths share one node = the vendor's two .backward() calls will collide"
    for name, t in zip(NAMES[2:], (r_idx, slot_idx, i_send)):
        assert not t.requires_grad and t.grad_fn is None, name
        assert t.dtype == torch.int64, name


# ======================================================================================
# 2. The two edges are independent: two .backward() calls in the vendor's order do
#    not blow up, and match the current chain bitwise
# ======================================================================================

@pytest.mark.parametrize("geom", GEOMS[:3], ids=lambda g: f"R{g[0]}q{g[1]}")
def test_segment_split_adjoints_survive_two_backwards(geom):
    """The vendor gmm runs the prob path first, then the token path, one .backward()
    each: the segment-graph variant must survive both, and both gradient paths must
    equal the current chain's autograd (also run as two backwards) bit for bit.

    This is the file's core failure-mode anchor: using the fused TerraceK1ArrivalFn
    directly throws "Trying to backward through the graph a second time" on the
    second .backward().
    """
    R, quota, epr, rpn, Hh = geom
    rx0, rslot, rgate0 = _mk(R, quota, epr, rpn, Hh, torch.float32, 302)
    g = torch.Generator().manual_seed(7)
    gs = torch.randn(R * quota, Hh, generator=g)
    gg = torch.randn(R * quota, generator=g)

    # Current-chain reference: the two gather subgraphs are disjoint, so two
    # separate backwards already work there
    rxC, rgC = rx0.clone().requires_grad_(True), rgate0.clone().requires_grad_(True)
    sC, pC, *_ = _chain(rxC, rslot, rgC, quota, epr, rpn)
    (pC * gg).sum().backward()
    (sC * gs).sum().backward()

    rxS, rgS = rx0.clone().requires_grad_(True), rgate0.clone().requires_grad_(True)
    sS, pS, *_ = tops.k1_arrival_segment(rxS, rslot, rgS, quota, epr, rpn)
    (pS * gg).sum().backward()          # vendor order: prob path first
    assert rxS.grad is None, ("the gate path's backward must not write into the "
                              "token path's leaf (not even zero gradients)")
    (sS * gs).sum().backward()          # then the token path -- a fused node dies here

    assert torch.equal(rxS.grad, rxC.grad), \
        "token-path gradient not bitwise equal to the current chain"
    assert torch.equal(rgS.grad, rgC.grad), \
        "gate-path gradient not bitwise equal to the current chain"


def test_segment_backward_bitwise_in_bed_dtype():
    """Under the bed dtype (bf16 payload) both gradient paths are bitwise equal as
    well -- index_add_ and the index adjoint must use the same reduction in bf16
    too."""
    R, quota, epr, rpn, Hh = 16, 2, 2, 8, 6
    rx0, rslot, rgate0 = _mk(R, quota, epr, rpn, Hh, torch.bfloat16, 303)
    g = torch.Generator().manual_seed(8)
    gs = torch.randn(R * quota, Hh, generator=g).to(torch.bfloat16)
    gg = torch.randn(R * quota, generator=g).to(torch.bfloat16)

    rxC, rgC = rx0.clone().requires_grad_(True), rgate0.clone().requires_grad_(True)
    sC, pC, *_ = _chain(rxC, rslot, rgC, quota, epr, rpn)
    (pC.float() * gg.float()).sum().backward()
    (sC.float() * gs.float()).sum().backward()

    rxS, rgS = rx0.clone().requires_grad_(True), rgate0.clone().requires_grad_(True)
    sS, pS, *_ = tops.k1_arrival_segment(rxS, rslot, rgS, quota, epr, rpn)
    (pS.float() * gg.float()).sum().backward()
    (sS.float() * gs.float()).sum().backward()

    assert torch.equal(rxS.grad, rxC.grad)
    assert torch.equal(rgS.grad, rgC.grad)


# ======================================================================================
# 3+4. Seam level: overlap seam forced onto K1 == baseline pass == same-pass legacy
#      arm; live-path counters
# ======================================================================================

def _routing(gen, n_nodes, per, quota):
    rows = []
    for _ in range(T):
        gs = torch.randperm(n_nodes, generator=gen)[:M]
        rows.append(torch.cat([torch.arange(a * per, (a + 1) * per)[
            torch.randperm(per, generator=gen)[:quota]] for a in gs]))
    return torch.stack(rows)


def _bitdiff(a, b):
    """None = bitwise equal; otherwise returns (dtype note, max|delta|) to localize
    the failure."""
    if a is None or b is None:
        return ("missing", None)
    if a.dtype != b.dtype:
        return (f"{a.dtype} vs {b.dtype}", float((a.float() - b.float()).abs().max()))
    if torch.equal(a, b):
        return None
    return (str(a.dtype), float((a.float() - b.float()).abs().max()))


def _run(rank, world, q, dtype_name):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # Already taken: 29577/29591/29613/29623/29627/29641/29645/29661/29665/29677.
    os.environ.setdefault(
        "MASTER_PORT", "29681" if dtype_name == "float32" else "29685")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        import terrace.ops as ops_mod
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

        counts = {"k1": 0, "seg": 0}
        in_seg = [False]
        real_enabled = ops_mod.custom_ops_enabled
        real_k1 = ops_mod.k1_arrival
        real_seg = ops_mod.k1_arrival_segment

        def fake_k1(rx, rslot, rgate, quota_, epr_, rpn_, my_local=0):
            # Wiring assertions: the geometry the integration point hands the kernel
            # must be self-consistent (a swapped argument can still look correct
            # under the equivalence assertions -- e.g. geometries where epr and rpn
            # are equal -- so pin it down right here).
            assert rx.dim() == 2 and rslot.shape == rgate.shape
            assert rslot.shape[1] == quota_ == quota
            assert epr_ == epr and rpn_ == RPN
            assert my_local == rank % RPN
            assert rslot.dtype == torch.int64 and rgate.dtype == rx.dtype
            if in_seg[0]:
                # The segment-graph variant must fetch its data **outside** the
                # graph: a kernel output carrying its own grad_fn would weld the two
                # paths back into one node, and the two independent edges would hang
                # there for nothing.
                assert not torch.is_grad_enabled()
                assert not rx.requires_grad and not rgate.requires_grad
                assert rx.grad_fn is None and rgate.grad_fn is None
            counts["k1"] += 1
            return ops_mod.k1_arrival_ref(rx, rslot, rgate, quota_, epr_, rpn_,
                                          my_local)

        def counting_seg(rx, rslot, rgate, quota_, epr_, rpn_, my_local=0):
            # The segment-graph inputs must be exactly permute2's two detach leaves
            # (seat 4 and seat 6).
            assert rx.requires_grad and rgate.requires_grad
            assert rx.grad_fn is None and rgate.grad_fn is None
            counts["seg"] += 1
            in_seg[0] = True
            try:
                return real_seg(rx, rslot, rgate, quota_, epr_, rpn_, my_local)
            finally:
                in_seg[0] = False

        def one_pass(forced):
            if forced:
                ops_mod.custom_ops_enabled = lambda: True
                ops_mod.k1_arrival = fake_k1
                ops_mod.k1_arrival_segment = counting_seg
            try:
                g = torch.Generator().manual_seed(41 + rank)
                x0 = torch.randn(T, H, generator=g).to(pdt)
                idx = _routing(g, n_nodes, per, quota)
                gates0 = torch.rand(T, K, generator=g)       # fp32: router output precision
                w13_0 = (torch.randn(epr, H, 2 * D, generator=g) / (H ** 0.5)).to(pdt)
                w2_0 = (torch.randn(epr, D, H, generator=g) / (D ** 0.5)).to(pdt)
                G = torch.randn(T, H, generator=g).to(pdt)   # upstream gradient
                routing_map = torch.zeros(T, E, dtype=torch.bool)
                routing_map[torch.arange(T).unsqueeze(1), idx] = True
                probs_dense = torch.zeros(T, E)
                probs_dense[torch.arange(T).unsqueeze(1), idx] = gates0

                # ---- legacy 3-arg seam (plain autograd); probs cast to payload
                # dtype per the vendor legacy layer's convention before entering
                # the seam ----
                hidL = x0.clone().requires_grad_(True)
                probL = probs_dense.clone().to(pdt).requires_grad_(True)
                w13L = w13_0.clone().requires_grad_(True)
                w2L = w2_0.clone().requires_grad_(True)
                permL, tpeL, ppL, stL = ta2a_permute(
                    hidL, probL, routing_map, world=world, rank=rank, rpn=RPN,
                    n_experts=E, intra_group=intra, inter_group=None, groups_m=M)
                a, b = grouped_mm(permL, w13L, tpeL).chunk(2, dim=-1)
                eoL = grouped_mm(F.silu(a) * b, w2L, tpeL) * ppL.unsqueeze(-1)
                outL = ta2a_unpermute(eoL, stL, hidL)
                outL.backward(G)

                # ---- overlap 6-arg seam + step-by-step replay of the vendor's
                # hand-written backward ----
                hidO = x0.clone().requires_grad_(True)
                probO = probs_dense.clone().requires_grad_(True)
                save = []
                permO, tpeO, ppO, _share, stO = ta2a_permute_overlap(
                    hidO, probO, routing_map, world=world, rank=rank, rpn=RPN,
                    n_experts=E, intra_group=intra, inter_group=None, groups_m=M,
                    save_tensors=save, run_shared_experts=None)
                w13O = w13_0.clone().requires_grad_(True)
                w2O = w2_0.clone().requires_grad_(True)
                dI = permO.detach().requires_grad_(True)
                pI = ppO.detach().requires_grad_(True)
                a, b = grouped_mm(dI, w13O, tpeO).chunk(2, dim=-1)
                eoO = grouped_mm(F.silu(a) * b, w2O, tpeO) * pI.unsqueeze(-1)
                outO = ta2a_unpermute_overlap(eoO, stO, save)
                (p1g, p1p, ncpu, p2ind, p2g, p2pd, p2pg, u1ind, u1g, u2ind) = save
                seats = {
                    "n": len(save),
                    "num_cpu_is_none": ncpu is None,
                    "leaves": all(t.requires_grad and t.grad_fn is None
                                  for t in (p2ind, p2pd, u1ind, u2ind)),
                    "graphs": all(t.grad_fn is not None
                                  for t in (p1g, p1p, p2g, p2pg, u1g)),
                }

                outO.backward(G)                            # unpermute2 segment
                grad_red = _a2a_raw(u2ind.grad, stO.send_l, stO.recv_l)
                u1g.backward(grad_red)                      # unpermute1 segment
                eoO.backward(u1ind.grad)                    # experts equivalent
                p2pg.backward(pI.grad)                      # inside gmm: prob path first
                _ggr = _a2a_raw(p2pd.grad, stO.recv_l, stO.send_l)
                p2g.backward(dI.grad)                       # inside gmm: then token path
                _gpl = _a2a_raw(p2ind.grad, stO.recv_l, stO.send_l)
                torch.autograd.backward([p1g, p1p], grad_tensors=[_gpl, _ggr])

                return {
                    "L.perm": permL.detach(), "L.pp": ppL.detach(),
                    "L.tpe": tpeL, "L.out": outL.detach(),
                    "L.gx": hidL.grad, "L.gp": probL.grad,
                    "L.gw13": w13L.grad, "L.gw2": w2L.grad,
                    "O.perm": permO.detach(), "O.pp": ppO.detach(),
                    "O.tpe": tpeO, "O.out": outO.detach(),
                    "O.gx": hidO.grad, "O.gp": probO.grad,
                    "O.gw13": w13O.grad, "O.gw2": w2O.grad,
                    # the four detach leaves the vendor's manual replay actually
                    # reads -- direct products of K1's two edges
                    "O.p2in": p2ind.grad, "O.p2prob": p2pd.grad,
                    "O.u1in": u1ind.grad, "O.u2in": u2ind.grad,
                }, seats
            finally:
                ops_mod.custom_ops_enabled = real_enabled
                ops_mod.k1_arrival = real_k1
                ops_mod.k1_arrival_segment = real_seg

        base, _base_seats = one_pass(forced=False)
        counts_baseline = dict(counts)
        forced, seats = one_pass(forced=True)

        regress = {name: _bitdiff(base[name], forced[name]) for name in base}
        cross = {}
        for a_name, b_name in (("O.perm", "L.perm"), ("O.pp", "L.pp"),
                               ("O.tpe", "L.tpe"), ("O.out", "L.out"),
                               ("O.gx", "L.gx"), ("O.gw13", "L.gw13"),
                               ("O.gw2", "L.gw2")):
            cross[a_name] = _bitdiff(forced[a_name], forced[b_name])
        # probs gradients: legacy lands on a pdt leaf, overlap on an fp32 leaf; the
        # only legitimate difference is the exact upcast in .to's backward
        # (lossless), so after upcasting we still require bitwise equality -- zero
        # tolerance.
        cross["O.gp"] = _bitdiff(forced["O.gp"].float(), forced["L.gp"].float())

        q.put({"rank": rank, "status": "ok", "regress": regress, "cross": cross,
               "seats": seats, "counts": counts, "counts_baseline": counts_baseline})
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
def seam_fp32():
    return _spawn("float32")


@pytest.fixture(scope="module")
def seam_bf16():
    return _spawn("bfloat16")


@pytest.mark.timeout(300)
def test_overlap_k1_matches_pre_k1_chain_fp32(seam_fp32):
    """Before vs after: the overlap seam forced onto K1 matches the same-process
    baseline pass (= the pre-K1 current chain verbatim) on the forward, the four
    leaf gradients, and the four detach-leaf .grad values, all bitwise."""
    for r in seam_fp32:
        for name, d in r["regress"].items():
            assert d is None, (f"rank {r['rank']}: {name} forced-K1 not bitwise "
                               f"equal to the baseline pass {d}")


@pytest.mark.timeout(300)
def test_overlap_k1_matches_legacy_seam_fp32(seam_fp32):
    """Cross-seam bitparity still holds with K1 on (both seams in the same pass,
    same inputs, same kernel)."""
    for r in seam_fp32:
        for name, d in r["cross"].items():
            assert d is None, (f"rank {r['rank']}: {name} overlap not bitwise "
                               f"equal to legacy {d}")


@pytest.mark.timeout(300)
def test_overlap_k1_is_live(seam_fp32):
    """Live-path evidence: in the forced pass the segment-graph variant at the
    overlap entry is called exactly 1 time, the kernel exactly 2 times
    (1 legacy seam + 1 overlap segment graph); in the baseline pass both counters
    must be 0. Equivalence assertions are blind to "the switch never opened" --
    without this test the two above prove nothing (internal engineering record)."""
    for r in seam_fp32:
        assert r["counts_baseline"] == {"k1": 0, "seg": 0}, \
            f"rank {r['rank']}: switch is open in the baseline pass {r['counts_baseline']}"
        assert r["counts"]["seg"] == 1, (
            f"rank {r['rank']}: overlap segment-graph variant called "
            f"{r['counts']['seg']} times, expected 1"
            f" -- the integration point did not take the K1 branch")
        assert r["counts"]["k1"] == 2, (
            f"rank {r['rank']}: kernel called {r['counts']['k1']} times, expected 2"
            f" (1 legacy seam + 1 overlap segment graph)")


@pytest.mark.timeout(300)
def test_overlap_k1_keeps_save_tensors_contract(seam_fp32):
    """Zero change to the seat contract: with K1 on it is still 10 seats, seat 3 is
    None, 4 detach leaves, 5 graph-bearing."""
    for r in seam_fp32:
        s = r["seats"]
        assert s["n"] == 10, f"rank {r['rank']}: save_tensors seat count {s['n']} != 10"
        assert s["num_cpu_is_none"], f"rank {r['rank']}: seat 3 should be None"
        assert s["leaves"], f"rank {r['rank']}: detach-leaf seats are not leaves"
        assert s["graphs"], (f"rank {r['rank']}: graph seats missing grad_fn "
                             f"(gradients would vanish silently)")


@pytest.mark.timeout(300)
def test_overlap_k1_matches_pre_k1_chain_bed_dtype(seam_bf16):
    """Bed dtype convention (bf16 payload/weights + fp32 router probs): before vs
    after is bitwise here too."""
    for r in seam_bf16:
        for name, d in r["regress"].items():
            assert d is None, (f"rank {r['rank']}: {name} forced-K1 not bitwise "
                               f"equal to the baseline pass {d}")


@pytest.mark.timeout(300)
def test_overlap_k1_matches_legacy_seam_bed_dtype(seam_bf16):
    """Bed dtype: cross-seam bitparity still holds with K1 on (K1 does not move the
    gate plane's rounding point)."""
    for r in seam_bf16:
        for name, d in r["cross"].items():
            assert d is None, (f"rank {r['rank']}: {name} overlap not bitwise "
                               f"equal to legacy {d}")


@pytest.mark.timeout(300)
def test_overlap_k1_is_live_bed_dtype(seam_bf16):
    for r in seam_bf16:
        assert r["counts_baseline"] == {"k1": 0, "seg": 0}
        assert r["counts"] == {"k1": 2, "seg": 1}, \
            f"rank {r['rank']}: counts {r['counts']}"


# ======================================================================================
# Engineering discipline: this file itself stays LF and passes py_compile
# ======================================================================================

def test_this_file_compiles_and_is_lf():
    import py_compile
    py_compile.compile(__file__, doraise=True)
    with open(__file__, "rb") as f:
        assert b"\r\n" not in f.read()
