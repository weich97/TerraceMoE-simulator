"""Bit-level contract, call-count ledger, and live-path evidence for the A1/A2
collective packing (2026-08-21).

Why this file exists (A1/A2 of the internal design records, not shipped with the
repo): after splitting the w128/w8 a2a curves into `α + β`, α₁₂₈ = 0.45 ms,
α₈ = 0.058 ms, while β₈ ≈ β₁₂₈ ≈ 110 GB/s —
**bytes are nearly free, calls are expensive**. One dispatch originally took 8
collectives (inter 4 + intra 4); the vendor's alltoall_seq side has only about 3.
Packing merges lanes that share splits into a single call: Hop A 4 -> 2
([id‖payload‖gate]), Hop B 4 -> 3 ([slot‖gate]), split-half seam forward 10 -> 7
(combine untouched). **This is pure byte rearrangement** — it moves no reduction
order, pairing order, sort key, dtype, or rounding point — so the only admissible
criterion is `torch.equal`, with no tolerance.

Hop B's payload plane exp_rx is **deliberately not packed**: packing saves α, but
every large plane packed in pays two extra HBM copies of itself; on the verdict
testbed that is +0.317 ms to save 0.116 ms = a net loss of 0.20 ms (arithmetic in
the Hop B section of terrace/ta2a_pack.py) — the same physics as the 2026-08-01
"torch.cat gate packing" failure.

Four locks:

1. **Unit layer** (no process group): pack/unpack round-trips bit-for-bit against
   the **original tensors** — one set for Hop A (int64 container, id plane at the
   row head; two branches, the id_w=1 bitmask and the id_w=quota slot-index
   table) and one for Hop B (int64 container), multiple geometries x fp32/bf16;
   pad region zeroed, row-width formula, int64 full width never truncated,
   **splits are row counts and do not scale**, dtype mismatch fails loud.
2. **Seam layer** (gloo, world 4 / rpn 2, 2 nodes with a real cross-node hop):
   **same-process A/B** — the packed arm and the unpacked arm (monkeypatch
   `pack_enabled`) run the same batch of inputs in the same pass; the forwards of
   the legacy 3-arg seam and the overlap 6-arg seam, the four leaf gradients, and
   the `.grad` of the four detached leaves the vendor's manual replay actually
   reads are all bitwise equal. Two dtype tiers — fp32 and the testbed config
   (bf16 payload/weights + fp32 router probs) — and two branches, the quota fast
   path (gm=M) and the generic branch (gm=None).
3. **Vendor contract layer**: `st.send_l / st.recv_l` (exactly what gets handed
   to `disp.input_splits/output_splits`) are **item-for-item equal across the two
   arms and still row counts** — what packing scales by `F/gw` is an **internal**
   copy of the wire tensor, so the vendor's two manual replays (rx_d.grad /
   rgate_d.grad) run unchanged, each still **one** a2a. The replay in this test
   reproduces the vendor's order step by step (same order as
   tests/test_ta2a_overlap_seam.py); both `.backward()` calls into the permute2
   segment must stay live — if Hop B packing were welded into a single fused
   node, the second call would blow up on the spot (internal engineering
   records, 2026-08-20).
4. **Call counts and live path**: packed arm (default small form) legacy seam
   fwd/bwd = 8/6, unpacked arm = 10/6; overlap seam dispatch half 6 vs 8. On top
   of that, the call counts of `hopa_pack / hopa_pack_small / hopb_pack_meta` are
   recorded one by one — equivalence assertions are blind to "the gate never
   opened" (internal engineering records); without live-path evidence the first
   three layers are vacuous. (Whether the C1 wire format itself silently falls
   back is guarded by tests/test_ta2a_quota_wire_bitparity.py; since A1′ its
   observation point has also moved to the input arguments of `hopa_pack`.)

gloo / CPU, world 4, rpn 2.
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

import terrace.ta2a_pack as pk  # noqa: E402

WORLD, RPN, T, K, M, E, H, D = 4, 2, 8, 4, 2, 8, 6, 4

# Split-half seam call counts before / after packing (golden values pinned in
# sync with test_ta2a_gate_at_arrival.py).
# **2026-08-22: the default form changed from full to small, and these three
# numbers moved from 7/6/5 to 8/6/6 with it.**
# Not a regression — the full form measured a net loss of 2.05 ms/pass on the
# verdict testbed: the one collective it saved was bought with two HBM copies of
# the payload, a losing trade (internal measurement records).
PACKED_FWD, PACKED_BWD = 8, 6
PLAIN_FWD, PLAIN_BWD = 10, 6
# Overlap-seam dispatch half (permute_overlap) call counts, small form:
#   inter 3 = counts + payload + [id‖gate]
#   intra 3 = counts + exp_rx + [slot‖gate]
# Unpacked is 4 + 4 = 8; the full form is 2 + 3 = 5 (running it requires an
# explicit TERRACE_TA2A_PACK=full).
PACKED_DISP, PLAIN_DISP = 6, 8


# ======================================================================================
# 1. Unit layer: pack/unpack == the original three tensors, bitwise
# ======================================================================================

# (n, hidden, gate_w, id_w): id_w=1 is the generic arm's bitmask (1-D); >1 is the
# C1 slot-index table.
# Covers nonzero/zero pad, gate width 1, a wide gate plane (generic arm
# gw=slots), and verdict-testbed widths.
HOPA_GEOMS = [(5, 6, 2, 2), (5, 6, 4, 1), (1, 7, 3, 3), (17, 2048, 3, 3),
              (4, 5, 1, 1), (3, 24, 24, 1), (9, 2048, 24, 1)]


@pytest.mark.parametrize("geom", HOPA_GEOMS,
                         ids=lambda g: f"n{g[0]}h{g[1]}g{g[2]}i{g[3]}")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hopa_pack_roundtrip_is_bitwise(geom, dtype):
    """Hop A: id + payload + gate -> [n, W] int64 -> back; all three blocks
    bitwise, dtype/shape unchanged."""
    n, hidden, gate_w, id_w = geom
    g = torch.Generator().manual_seed(1000 + n * 31 + hidden + id_w)
    payload = torch.randn(n, hidden, generator=g).to(dtype)
    gate = torch.rand(n, gate_w, generator=g).to(dtype)
    ids = (torch.randint(0, 1 << 40, (n,), generator=g, dtype=torch.int64)
           if id_w == 1 else
           torch.randint(0, 24, (n, id_w), generator=g, dtype=torch.int64))

    buf, lay = pk.hopa_pack(payload, gate, ids)
    per_word = 8 // dtype.itemsize
    assert buf.shape == (n, lay.words) and buf.dtype == torch.int64
    assert buf.is_contiguous()
    assert lay.id_w == id_w and lay.id_1d is (id_w == 1 and ids.dim() == 1)
    # Row width: id plane + just enough words to hold (H + gw) floats; slack < 1 word
    assert (lay.words - id_w) * per_word >= hidden + gate_w
    assert (lay.words - id_w - 1) * per_word < hidden + gate_w
    # id plane at the row head => the float region always starts on an 8-byte boundary
    assert (id_w * 8) % 8 == 0
    # Pad region explicitly zeroed (the receive side never reads it)
    tail = id_w * per_word + hidden + gate_w
    if tail < lay.words * per_word:
        pad = buf.view(dtype)[:, tail:]
        assert torch.equal(pad, torch.zeros_like(pad))

    got_p, got_g, got_i = pk.hopa_unpack(buf, lay)
    assert torch.equal(got_p, payload) and got_p.dtype == payload.dtype
    assert torch.equal(got_g, gate) and got_g.dtype == gate.dtype
    assert torch.equal(got_i, ids) and got_i.shape == ids.shape
    assert got_p.is_contiguous() and got_g.is_contiguous() and got_i.is_contiguous(), (
        "unpack must return contiguous tensors: the 2026-08-01 torch.cat packing "
        "was dragged down precisely by downstream consuming non-contiguous slices")


def test_hopa_id_plane_survives_full_int64_range():
    """The bitmask is a **full-width** int64 value (slots can reach 63 bits); no
    narrowing path may truncate it."""
    ids = torch.tensor([0, 1, (1 << 62) + 12345, -1, (1 << 63) - 1],
                       dtype=torch.int64)
    payload = torch.randn(5, 6).to(torch.bfloat16)
    gate = torch.rand(5, 3).to(torch.bfloat16)
    buf, lay = pk.hopa_pack(payload, gate, ids)
    _p, _g, got = pk.hopa_unpack(buf, lay)
    assert torch.equal(got, ids)


@pytest.mark.parametrize("geom", HOPA_GEOMS,
                         ids=lambda g: f"n{g[0]}h{g[1]}g{g[2]}i{g[3]}")
def test_hopa_splits_are_plain_row_counts(geom):
    """dim 0 of the wire tensor is the row count — splits **do not scale**; each
    segment is exactly the original rows.

    This is the direct reason the vendor contract stays untouched:
    `st.send_l / st.recv_l` (= disp.input_splits / output_splits) are
    item-for-item identical to pre-packing, so the vendor's manual backward
    replay does not need to know packing exists.
    """
    n, hidden, gate_w, id_w = geom
    payload = torch.randn(n, hidden)
    gate = torch.rand(n, gate_w)
    ids = (torch.zeros(n, dtype=torch.int64) if id_w == 1
           else torch.zeros(n, id_w, dtype=torch.int64))
    buf, _lay = pk.hopa_pack(payload, gate, ids)
    assert buf.shape[0] == n, "dim 0 must be the row count"
    row_splits = [n // 3, n - n // 3 - n // 4, n // 4]
    assert sum(row_splits) == n
    off = 0
    for sz in row_splits:                       # segment ownership matches pre-packing, row for row
        assert torch.equal(buf[off:off + sz], buf.narrow(0, off, sz))
        off += sz


@pytest.mark.parametrize("P", [1, 5, 13, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hopb_meta_pack_roundtrip_is_bitwise(P, dtype):
    """Hop B: slot ids (int64) + gate -> int64 container -> back; both blocks bitwise."""
    g = torch.Generator().manual_seed(2000 + P * 17)
    slot = torch.randint(0, 24, (P,), generator=g, dtype=torch.int64)
    gate = torch.rand(P, generator=g).to(dtype)

    buf = pk.hopb_pack_meta(slot, gate)
    assert buf.shape == (P, 2) and buf.dtype == torch.int64 and buf.is_contiguous()
    # Slot plane at the row head + row width counted in int64 words => every row
    # start and the float-region start sit on an 8-byte boundary
    assert pk.hopb_meta_words() == 2

    got_s, got_g = pk.hopb_unpack_meta(buf, dtype)
    assert torch.equal(got_s, slot) and got_s.dtype == torch.int64
    assert torch.equal(got_g, gate) and got_g.dtype == dtype
    assert got_s.is_contiguous() and got_g.is_contiguous()


def test_hopb_meta_pack_pad_is_zeroed():
    """Pad region explicitly zeroed: the receive side never reads it, but
    uninitialized bits must not go on the wire either."""
    slot = torch.arange(7, dtype=torch.int64)
    gate = torch.rand(7).to(torch.bfloat16)
    buf = pk.hopb_pack_meta(slot, gate)
    fv = buf.view(torch.bfloat16)
    assert torch.equal(fv[:, 5:], torch.zeros_like(fv[:, 5:]))


def test_pack_dtype_mismatch_is_loud():
    """dtype / shape mismatch must die on the spot — the same failure mode as the
    old sparse plane at index_put (an internal commit's fp32 gate-plane drift got
    in through exactly this kind of silent spot)."""
    ids3 = torch.zeros(3, dtype=torch.int64)
    with pytest.raises(RuntimeError):           # Hop A: gate dtype differs from payload
        pk.hopa_pack(torch.randn(3, 4).to(torch.bfloat16), torch.rand(3, 2), ids3)
    with pytest.raises(RuntimeError):           # Hop A: id plane must be int64
        pk.hopa_pack(torch.randn(3, 4), torch.rand(3, 2),
                     torch.zeros(3, dtype=torch.int32))
    with pytest.raises(RuntimeError):           # Hop A: all three planes must agree on row count
        pk.hopa_pack(torch.randn(3, 4), torch.rand(3, 2),
                     torch.zeros(4, dtype=torch.int64))
    with pytest.raises(RuntimeError):           # Hop B: slot plane must be int64
        pk.hopb_pack_meta(torch.zeros(3, dtype=torch.int32), torch.rand(3))
    with pytest.raises(RuntimeError):           # Hop B: gate plane must be floating point
        pk.hopb_pack_meta(torch.zeros(3, dtype=torch.int64),
                          torch.zeros(3, dtype=torch.int64))
    with pytest.raises(RuntimeError):           # Hop B: both planes must share one shape
        pk.hopb_pack_meta(torch.zeros(3, dtype=torch.int64), torch.rand(4))


def test_pack_switch_defaults_on_and_can_be_turned_off(monkeypatch):
    """Gate: unset / non-"0" is on; "0" is off (the on-testbed A/B and the
    one-command rollback rely on it)."""
    monkeypatch.delenv("TERRACE_TA2A_PACK", raising=False)
    pk.reset()
    assert pk.pack_enabled() is True
    monkeypatch.setenv("TERRACE_TA2A_PACK", "0")
    pk.reset()
    assert pk.pack_enabled() is False
    monkeypatch.setenv("TERRACE_TA2A_PACK", "1")
    pk.reset()
    assert pk.pack_enabled() is True
    monkeypatch.delenv("TERRACE_TA2A_PACK", raising=False)
    pk.reset()


# ======================================================================================
# 2+3+4. Seam layer: same-process A/B, vendor contract, call counts and live path
# ======================================================================================

def _routing(gen, n_nodes, per, quota):
    """T-Route equal-quota: exactly M nodes, exactly K/M experts per node."""
    rows = []
    for _ in range(T):
        gs = torch.randperm(n_nodes, generator=gen)[:M]
        rows.append(torch.cat([torch.arange(a * per, (a + 1) * per)[
            torch.randperm(per, generator=gen)[:quota]] for a in gs]))
    return torch.stack(rows)


def _bitdiff(a, b):
    """None = bitwise equal; otherwise (description, max|Δ|) to localize the failure."""
    if a is None or b is None:
        return ("missing", None)
    if a.dtype != b.dtype:
        return (f"{a.dtype} vs {b.dtype}", float((a.float() - b.float()).abs().max()))
    if torch.equal(a, b):
        return None
    return (str(a.dtype), float((a.float() - b.float()).abs().max()))


def _run(rank, world, q, dtype_name):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # Ports in use: 29577/29591/29613/29623/29627/29641/29645/29661/29665/29677/
    #               29681/29685/29697. This bed takes one per dtype.
    os.environ.setdefault(
        "MASTER_PORT", "29709" if dtype_name == "float32" else "29713")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        import terrace.ta2a_pack as pack_mod
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
        counter = [0]

        def counting_a2a(*args, **kwargs):
            counter[0] += 1
            return real_a2a(*args, **kwargs)

        # Live-path counters: must be > 0 on the packed arm, == 0 on the unpacked arm.
        # **Count both Hop A packers**: the small form (default since 2026-08-22)
        # calls hopa_pack_small, the full form calls hopa_pack. Count only one of
        # them and the live-path evidence silently drops to 0 on a form switch —
        # and "the gate never opened" is exactly what this file exists to prevent
        # (internal engineering records).
        hits = {"hopa": 0, "hopb": 0}
        real_hopa = pack_mod.hopa_pack
        real_hopa_small = pack_mod.hopa_pack_small
        real_hopb = pack_mod.hopb_pack_meta
        real_switch = pack_mod.pack_enabled      # grab the original **before** it is overridden

        def counting_hopa(*a, **kw):
            hits["hopa"] += 1
            return real_hopa(*a, **kw)

        def counting_hopa_small(*a, **kw):
            hits["hopa"] += 1
            return real_hopa_small(*a, **kw)

        def counting_hopb(*a, **kw):
            hits["hopb"] += 1
            return real_hopb(*a, **kw)

        pack_mod.hopa_pack = counting_hopa
        pack_mod.hopa_pack_small = counting_hopa_small
        pack_mod.hopb_pack_meta = counting_hopb

        def one_pass(packed, gm):
            pack_mod.pack_enabled = (lambda: True) if packed else (lambda: False)
            g = torch.Generator().manual_seed(23 + rank)
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
                # ---- legacy 3-arg seam (plain autograd) ----
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
                seam_fwd = counter[0]
                counter[0] = 0
                outL.backward(G)
                seam_bwd = counter[0]

                # ---- overlap 6-arg seam + step-by-step replica of the vendor's
                # handwritten backward ----
                hidO = x0.clone().requires_grad_(True)
                probO = probs_dense.clone().requires_grad_(True)
                save = []
                counter[0] = 0
                permO, tpeO, ppO, _share, stO = ta2a_permute_overlap(
                    hidO, probO, routing_map, world=world, rank=rank, rpn=RPN,
                    n_experts=E, intra_group=intra, inter_group=None, groups_m=gm,
                    save_tensors=save, run_shared_experts=None)
                disp_fwd = counter[0]
                w13O = w13_0.clone().requires_grad_(True)
                w2O = w2_0.clone().requires_grad_(True)
                dI = permO.detach().requires_grad_(True)
                pI = ppO.detach().requires_grad_(True)
                a, b = grouped_mm(dI, w13O, tpeO).chunk(2, dim=-1)
                eoO = grouped_mm(F.silu(a) * b, w2O, tpeO) * pI.unsqueeze(-1)
                outO = ta2a_unpermute_overlap(eoO, stO, save)
                (p1g, p1p, ncpu, p2ind, p2g, p2pd, p2pg, u1ind, u1g, u2ind) = save

                outO.backward(G)                            # unpermute2 segment
                grad_red = _a2a_raw(u2ind.grad, stO.send_l, stO.recv_l)
                u1g.backward(grad_red)                      # unpermute1 segment
                eoO.backward(u1ind.grad)                    # experts equivalent
                # The vendor's gmm enters the permute2 segment through two
                # .backward() calls: the prob lane first, then the token lane.
                # If Hop B packing were welded into one fused node, the second
                # call would blow up right here.
                p2pg.backward(pI.grad)
                ggr = _a2a_raw(p2pd.grad, stO.recv_l, stO.send_l)
                p2g.backward(dI.grad)
                gpl = _a2a_raw(p2ind.grad, stO.recv_l, stO.send_l)
                torch.autograd.backward([p1g, p1p], grad_tensors=[gpl, ggr])
            finally:
                dist.all_to_all_single = real_a2a

            return {
                "L.perm": permL.detach(), "L.pp": ppL.detach(), "L.tpe": tpeL,
                "L.out": outL.detach(), "L.gx": hidL.grad, "L.gp": probL.grad,
                "L.gw13": w13L.grad, "L.gw2": w2L.grad,
                "O.perm": permO.detach(), "O.pp": ppO.detach(), "O.tpe": tpeO,
                "O.out": outO.detach(), "O.gx": hidO.grad, "O.gp": probO.grad,
                "O.gw13": w13O.grad, "O.gw2": w2O.grad,
                # the four detached leaves the vendor's manual replay actually reads
                "O.p2in": p2ind.grad, "O.p2prob": p2pd.grad,
                "O.u1in": u1ind.grad, "O.u2in": u2ind.grad,
                # contract and ledger (not part of the bitwise comparison)
                "_counts": (seam_fwd, seam_bwd, disp_fwd),
                "_splits": (list(stO.send_l), list(stO.recv_l),
                            list(stL.send_l), list(stL.recv_l)),
                "_seats": (len(save), ncpu is None),
            }

        report = {}
        try:
            for gm in (M, None):
                hits["hopa"] = hits["hopb"] = 0
                plain = one_pass(packed=False, gm=gm)
                hits_plain = dict(hits)
                hits["hopa"] = hits["hopb"] = 0
                packed = one_pass(packed=True, gm=gm)
                hits_packed = dict(hits)

                diffs = {}
                for name in packed:
                    if name.startswith("_"):
                        continue
                    if name.endswith(".tpe"):
                        diffs[name] = (None if torch.equal(packed[name], plain[name])
                                       else ("tpe", -1.0))
                    else:
                        diffs[name] = _bitdiff(packed[name], plain[name])
                report[f"gm={gm}"] = {
                    "diffs": diffs,
                    "counts_packed": packed["_counts"],
                    "counts_plain": plain["_counts"],
                    "splits_same": packed["_splits"] == plain["_splits"],
                    "splits": packed["_splits"],
                    "seats": packed["_seats"],
                    "hits_packed": hits_packed,
                    "hits_plain": hits_plain,
                }
        finally:
            pack_mod.hopa_pack, pack_mod.hopb_pack_meta = real_hopa, real_hopb
            pack_mod.hopa_pack_small = real_hopa_small
            pack_mod.pack_enabled = real_switch

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
def pack_fp32():
    return _spawn("float32")


@pytest.fixture(scope="module")
def pack_bf16():
    return _spawn("bfloat16")


def _assert_bitwise(results, label):
    for r in results:
        for tag, rep in r["report"].items():
            for name, d in rep["diffs"].items():
                assert d is None, (
                    f"rank {r['rank']} {label} {tag}: {name} packed arm vs "
                    f"unpacked arm not bitwise equal {d} — packing may only "
                    f"rearrange bytes")


def _assert_contract(results, label):
    for r in results:
        for tag, rep in r["report"].items():
            cp, cq = rep["counts_packed"], rep["counts_plain"]
            assert cp == (PACKED_FWD, PACKED_BWD, PACKED_DISP), (
                f"rank {r['rank']} {label} {tag}: packed-arm call counts {cp}, "
                f"expected {(PACKED_FWD, PACKED_BWD, PACKED_DISP)} (legacy "
                f"fwd/bwd, overlap dispatch) — more means packing was silently "
                f"reverted, fewer means an exchange was silently dropped")
            assert cq == (PLAIN_FWD, PLAIN_BWD, PLAIN_DISP), (
                f"rank {r['rank']} {label} {tag}: unpacked-arm call counts {cq}, "
                f"expected {(PLAIN_FWD, PLAIN_BWD, PLAIN_DISP)} — the control "
                f"arm changed form; this file's 'pre-change formula' premise no "
                f"longer holds, re-review required")
            assert rep["splits_same"], (
                f"rank {r['rank']} {label} {tag}: st.send_l/recv_l differ "
                f"between the arms {rep['splits']} — those two lists are exactly "
                f"the row counts handed to the vendor's disp.input_splits/"
                f"output_splits; packing must not change their semantics")
            n_send, n_recv, l_send, l_recv = rep["splits"]
            assert len(n_send) == WORLD and len(n_recv) == WORLD
            assert sum(n_send) == T * M, (
                f"rank {r['rank']} {label} {tag}: Hop A splits sum {sum(n_send)} "
                f"!= T*M={T * M} — what goes to the vendor must be **row "
                f"counts**, not scaled byte-segment counts")
            assert (n_send, n_recv) == (l_send, l_recv)
            assert rep["seats"] == (10, True), (
                f"rank {r['rank']} {label} {tag}: seat contract changed {rep['seats']}")


def _assert_live(results, label):
    for r in results:
        for tag, rep in r["report"].items():
            assert rep["hits_plain"] == {"hopa": 0, "hopb": 0}, (
                f"rank {r['rank']} {label} {tag}: the unpacked arm went through "
                f"packing {rep['hits_plain']} — the A/B control arm is "
                f"compromised; every equivalence assertion is vacuous")
            # legacy seam once + overlap seam once
            assert rep["hits_packed"] == {"hopa": 2, "hopb": 2}, (
                f"rank {r['rank']} {label} {tag}: packed-arm pack call counts "
                f"{rep['hits_packed']}, expected 1 per seam for the two seams — "
                f"the gate did not open or one seam is not wired up (equivalence "
                f"assertions are blind to this; internal engineering records)")


@pytest.mark.timeout(300)
def test_fp32_pack_is_bitwise(pack_fp32):
    """fp32: both seams' forwards, four leaf gradients, four detached-leaf
    .grad — packed == unpacked."""
    _assert_bitwise(pack_fp32, "fp32")


@pytest.mark.timeout(300)
def test_fp32_vendor_contract_and_counts(pack_fp32):
    """Call counts drop to 7/6 (dispatch half 8 -> 5); splits are still row
    counts, item-for-item equal across the arms."""
    _assert_contract(pack_fp32, "fp32")


@pytest.mark.timeout(300)
def test_fp32_pack_is_live(pack_fp32):
    """Live-path evidence: the packed arm really packed, the unpacked arm really did not."""
    _assert_live(pack_fp32, "fp32")


@pytest.mark.timeout(300)
def test_bed_dtype_pack_is_bitwise(pack_bf16):
    """Testbed config (bf16 payload/weights + fp32 router probs): same as above, bitwise."""
    _assert_bitwise(pack_bf16, "bf16")


@pytest.mark.timeout(300)
def test_bed_dtype_vendor_contract_and_counts(pack_bf16):
    _assert_contract(pack_bf16, "bf16")


@pytest.mark.timeout(300)
def test_bed_dtype_pack_is_live(pack_bf16):
    _assert_live(pack_bf16, "bf16")


# ======================================================================================
# Engineering discipline: this file itself is LF + py_compile
# ======================================================================================

def test_this_file_compiles_and_is_lf():
    import py_compile
    py_compile.compile(__file__, doraise=True)
    with open(__file__, "rb") as f:
        assert b"\r\n" not in f.read()
