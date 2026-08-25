"""The two split-half seams (legacy 3-arg vs overlap 6-arg) yield **bitwise-equal**
forward and all gradients on the same input.

Why this file exists (fix lock for the eqov alignment-bed FAIL, 2026-08-20):
the overlap seam's T-A2A drifts monotonically against the vendor overlap on the
alignment bed (1e-5@20 → 1.38e-4@100, calibration ratio 10.6×, bound 3×), while the
legacy seam on the same bed stays ≤2e-5. The drift hunt (bitwise comparison of the
two split-half paths) localized the single numeric difference: an internal commit
made the gate plane follow probs.dtype (fp32), while the legacy half runs
payload.dtype (bf16 on the bed). The fp32 gate, through the vendor gmm's probs
multiply, promotes expert_out to fp32, dragging the combine's two-hop return and
both levels of index_add reduction into fp32 -- the token plane at the dispatch exit
is still bitwise equal (so the per-layer ASSERTs never fired), but every reduction's
rounding departs from the legacy-validated path and compounds step by step. Fix:
round the gate to payload.dtype right where it enters gate_rows (same rounding point
as legacy; cast∘gather == gather∘cast, elementwise identical).

So this file locks two invariants, all torch.equal (bitwise), no tolerances:
  1. Uniform fp32 precision: both seams' forward output and all four gradients
     (hidden/probs/w13/w2) are bitwise equal (the same operation sequence behind a
     different call boundary should introduce no rounding difference at all);
  2. Bed dtype convention (bf16 payload/weights + fp32 router probs): the overlap
     seam receives fp32 probs (what the vendor overlap layer hands over is router
     precision), the legacy seam receives bf16 probs (the vendor legacy layer casts
     to hidden dtype upstream of dispatch -- hard fact: the legacy half's gate_rows
     is payload.dtype, fp32 probs would die on the spot in index_put's dtype-match
     check, and the legacy gate ran 100 steps clean). Both arms' forward and
     hidden/w13/w2 gradients are bitwise equal; the probs gradients land on a bf16
     leaf and an fp32 leaf respectively, the only legitimate difference being the
     exact upcast in .to's backward (bf16 -> fp32 lossless), so we assert overlap's
     fp32 gradient == legacy's bf16 gradient after upcast, bitwise -- still zero
     tolerance.

The overlap arm's backward replays the vendor's hand-written orchestration step by
step (same order as tests/test_ta2a_overlap_seam.py); the legacy arm uses plain
autograd. gloo / CPU, world 4, rpn 2 (2 nodes, with a real cross-node hop).
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

WORLD, RPN, T, K, M, E, H, D = 4, 2, 8, 4, 2, 8, 6, 4


def _routing(gen, n_nodes, per, quota):
    """T-Route equal-quota: exactly M nodes, exactly K/M experts per node."""
    rows = []
    for _ in range(T):
        gs = torch.randperm(n_nodes, generator=gen)[:M]
        rows.append(torch.cat([torch.arange(a * per, (a + 1) * per)[
            torch.randperm(per, generator=gen)[:quota]] for a in gs]))
    return torch.stack(rows)


def _bitdiff(a, b):
    """None = bitwise equal; otherwise returns (dtype pair, max|delta|) to localize
    the failure."""
    if a is None or b is None:
        return ("missing", None)
    if a.dtype != b.dtype:
        return (f"{a.dtype} vs {b.dtype}",
                float((a.float() - b.float()).abs().max()))
    if torch.equal(a, b):
        return None
    return (str(a.dtype), float((a.float() - b.float()).abs().max()))


def _run(rank, world, q, payload_dtype_name):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # One port per bed, to avoid colliding with the other dist test files
    # (29577/29591/29613 already taken).
    os.environ.setdefault(
        "MASTER_PORT", "29623" if payload_dtype_name == "float32" else "29627")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from terrace.layer import grouped_mm
        from terrace.ep_dist import _a2a_raw
        from terrace.ta2a_fwd import init_ta2a_groups
        from terrace.ta2a_dispatch import (ta2a_permute, ta2a_unpermute,
                                           ta2a_permute_overlap,
                                           ta2a_unpermute_overlap)

        pdt = getattr(torch, payload_dtype_name)
        intra = init_ta2a_groups(world, RPN)
        epr = E // world
        n_nodes, per, quota = world // RPN, E // (world // RPN), K // M

        g = torch.Generator().manual_seed(11 + rank)
        x0 = torch.randn(T, H, generator=g).to(pdt)
        idx = _routing(g, n_nodes, per, quota)
        gates0 = torch.rand(T, K, generator=g)          # fp32: router output precision
        w13_0 = (torch.randn(epr, H, 2 * D, generator=g) / (H ** 0.5)).to(pdt)
        w2_0 = (torch.randn(epr, D, H, generator=g) / (D ** 0.5)).to(pdt)
        G = torch.randn(T, H, generator=g).to(pdt)      # upstream gradient

        routing_map = torch.zeros(T, E, dtype=torch.bool)
        routing_map[torch.arange(T).unsqueeze(1), idx] = True
        probs_dense = torch.zeros(T, E)
        probs_dense[torch.arange(T).unsqueeze(1), idx] = gates0   # fp32 dense

        # ---- legacy arm: plain autograd. probs cast to payload dtype per the
        # vendor legacy layer's convention before entering the seam (.to is the
        # identity for fp32, same line of code on both beds). ----
        hidL = x0.clone().requires_grad_(True)
        # clone before .to: for fp32 .to is the identity, and setting requires_grad
        # on probs_dense directly would turn the overlap arm's clone below into a
        # non-leaf (.grad never lands); each arm must own its own leaf.
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

        # ---- overlap arm: the 6-arg seam's two halves + step-by-step orchestration
        # of the vendor's hand-written backward (same order as
        # test_ta2a_overlap_seam.py). probs enter the seam still fp32 (what the
        # vendor overlap layer hands over is router precision). ----
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
        (p1g, p1p, _ncpu, p2ind, p2g, p2pd, p2pg, u1ind, u1g, u2ind) = save

        fwd = {
            "permuted": _bitdiff(permL.detach(), permO.detach()),
            "pprobs": _bitdiff(ppL.detach(), ppO.detach()),
            "out": _bitdiff(outL.detach(), outO.detach()),
            "tpe": None if torch.equal(tpeL, tpeO) else ("tpe", -1.0),
        }

        outO.backward(G)                                   # unpermute2 segment
        grad_red = _a2a_raw(u2ind.grad, stO.send_l, stO.recv_l)
        u1g.backward(grad_red)                             # unpermute1 segment
        eoO.backward(u1ind.grad)                           # experts equivalent
        p2pg.backward(pI.grad)                             # inside gmm: prob path first
        ggr = _a2a_raw(p2pd.grad, stO.recv_l, stO.send_l)
        p2g.backward(dI.grad)                              # inside gmm: then token path
        gpl = _a2a_raw(p2ind.grad, stO.recv_l, stO.send_l)
        torch.autograd.backward([p1g, p1p], grad_tensors=[gpl, ggr])

        # probs gradients: legacy lands on a pdt leaf, overlap on an fp32 leaf. The
        # only legitimate difference is the exact upcast in .to's backward
        # (pdt -> fp32 lossless; same dtype when pdt is fp32), so after upcast we
        # still require bitwise equality -- zero tolerance, not an approximate
        # comparison.
        bwd = {
            "hidden": _bitdiff(hidL.grad, hidO.grad),
            "probs": _bitdiff(probL.grad.float(), probO.grad.float()
                              if probO.grad is not None else None),
            "w13": _bitdiff(w13L.grad, w13O.grad),
            "w2": _bitdiff(w2L.grad, w2O.grad),
        }
        q.put({"rank": rank, "status": "ok", "fwd": fwd, "bwd": bwd})
    except Exception:                                      # noqa: BLE001
        import traceback
        q.put({"rank": rank, "status": "err", "trace": traceback.format_exc()})
    finally:
        dist.destroy_process_group()


def _spawn(payload_dtype_name):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_run, args=(r, WORLD, q, payload_dtype_name))
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
def parity_fp32():
    return _spawn("float32")


@pytest.fixture(scope="module")
def parity_bf16():
    return _spawn("bfloat16")


@pytest.mark.timeout(300)
def test_fp32_forward_bitwise_equal(parity_fp32):
    """Uniform fp32 precision: the same operation sequence behind a different call
    boundary; the forward must be bitwise equal."""
    for r in parity_fp32:
        for name, d in r["fwd"].items():
            assert d is None, f"rank {r['rank']}: forward {name} not bitwise equal {d}"


@pytest.mark.timeout(300)
def test_fp32_all_grads_bitwise_equal(parity_fp32):
    """Uniform fp32 precision: the four gradients from the replayed vendor
    orchestration equal autograd bitwise."""
    for r in parity_fp32:
        for name, d in r["bwd"].items():
            assert d is None, f"rank {r['rank']}: gradient {name} not bitwise equal {d}"


@pytest.mark.timeout(300)
def test_bed_dtype_forward_bitwise_equal(parity_bf16):
    """Bed dtype convention (bf16 payload + fp32 probs): with the gate plane rounded
    at the same point, the forward is bitwise equal.

    Pre-fix failure mode (anchored here against regression): pprobs was fp32 (the
    dtype alone already differs), expert_out got promoted to fp32 by the multiply,
    the combine ran entirely in fp32, and out differed from legacy at the 1e-2
    level -- precisely the injection source of the eqov bed's 1.38e-4@100 drift.
    """
    for r in parity_bf16:
        for name, d in r["fwd"].items():
            assert d is None, f"rank {r['rank']}: forward {name} not bitwise equal {d}"


@pytest.mark.timeout(300)
def test_bed_dtype_all_grads_bitwise_equal(parity_bf16):
    """Bed dtype: hidden/w13/w2 bitwise equal; probs gradient bitwise equal after
    upcast (see the comment in _run)."""
    for r in parity_bf16:
        for name, d in r["bwd"].items():
            assert d is None, f"rank {r['rank']}: gradient {name} not bitwise equal {d}"
