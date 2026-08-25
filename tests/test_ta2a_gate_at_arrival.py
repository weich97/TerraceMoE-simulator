"""Step 2 (BACKWARD-PLAN): gating applied at the **arrival rank** must be
bitwise equal to gating applied at the expert rank.

Why this file exists (the regression lock from implementing Step 2 on
2026-08-20): since 2026-08-20, `ta2a_moe_forward` no longer ships each
(row, slot) pair's gate to the expert rank over the intra-node exchange (the old
`my_gate = _a2a(...)`); the gate stays on the arrival rank and is multiplied in
when the expert result lands on the return trip
(`red.index_add_(0, r_idx, ret * gate_pairs)`). The legality argument for the
transform is a single sentence: elementwise multiplication commutes with data
movement — `ye * gate` (expert side) and `ret * gate` (arrival side) are the
same multiplication over the same operands; the exchange only moves rows.
**That sentence must stay pinned by a test forever** — otherwise any future
change touching pairing order (ordo reshuffles, where r_idx/slot_idx come from)
could put the gate on the wrong row, and a wrong-row forward still has the
correct shape while the loss still looks like a normal loss.

The control arm is the "pre-change formula" itself: the legacy split-half seam
(ta2a_permute → vendor-style `* pprobs` → ta2a_unpermute) applies gating at the
expert rank — byte-for-byte isomorphic to the fused forward before 2026-08-20.
So this file also tightens "fused forward == split-half composition" from a 1e-5
tolerance (test_ta2a_dispatch_split) to zero tolerance, and covers all four
gradients plus bf16.

Three groups of assertions, all torch.equal (bitwise, no tolerance):
  1. Forward: y bitwise equal across the two arms (fp32 and bf16; both the
     groups_m=M topk branch and the =None nonzero branch — the two branches pair
     in different orders, and gate alignment may err in neither);
  2. Four gradients: x / gates / w13 / w2 bitwise equal (the gates control takes
     the dense probs leaf's gather at the routed positions; gather's backward
     only scatters, never merges, values unchanged);
  3. Collective counts: fused forward 9 in forward, 5 in backward; control arm
     forward **8**, backward 6 — Step 2's saving is the fused arm's one exchange
     each way, and the control arm's 10 -> 8 is the A1''/A2 packing (itemized
     ledger in the comment above SEAM_FWD_A2A). Exact values pinned in both
     directions: an increase = the optimization was silently reverted, a further
     decrease = an exchange was silently dropped
     (the Measure item of BACKWARD-PLAN Step 2 requires this assertion).

**After the packing, this file's standing as evidence got stronger**: the
fused-forward arm (ta2a_fwd) is untouched — its three planes each travel on
their own; the split-half seam arm takes the A1/A2 packing. Assertion groups 1
and 2 are therefore also a bitwise proof of "packed == unpacked" — forward and
all four gradients, fp32 and bf16, quota branch and generic branch, one pass
each.

The historical record, to prevent another mix-up: what the 2026-08-01
measurement rejected and rolled back was the **torch.cat gate packing** (an
internal commit → an internal commit; whole-payload copying is the physical
reason it was slow; item one of BACKWARD-PLAN "Do not re-propose"). Step 2 is
not isomorphic to it: zero copies, zero new tensors, it only deletes one pair of
exchanges. The 2026-08-13 doc saying "Step 2 was already rejected by
measurement" misattributed that result; Step 2 had never been implemented or
timed before 2026-08-20. The 2026-08-21 A1′/A2 packing is not isomorphic to the
08-01 attempt either: (1) it does not use torch.cat — the receive side unpacks
back to **contiguous** tensors, and the downstream gather sees shapes
byte-for-byte identical to pre-packing (the physical reason 08-01 was slow was
downstream consuming non-contiguous slices, not the packing itself); (2) it
saves α (call count), not β (bytes) — the α₁₂₈=0.45 ms decomposition was only
measured on 08-20/21; the 08-01 attempt had no such ledger.

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

WORLD, RPN, T, K, M, E, H, D = 4, 2, 8, 4, 2, 8, 6, 4

# Collective call counts of the fused forward after Step 2 (the fused forward is
# **unpacked** — it is this file's fixed reference arm).
FUSED_FWD_A2A, FUSED_BWD_A2A = 9, 5

# Call counts of the split-half seam (ta2a_dispatch). **Since 2026-08-22 the
# default form is A1'' (small), 10/6 -> 8/6**.
#
# History: the A1′ (full) of 2026-08-21 was 10/6 -> 7/6, estimated at
# −0.85 ms/pass. The verdict testbed overturned it: dispatch 14.222 vs unpacked
# 12.171, **a net loss of 2.05 ms/pass**
# (internal measurement records). Back-solving gives the true cost of the two
# payload HBM copies for pack/unpack ≈ 3.0 ms, against the original estimate of
# 0.106 ms — off by a factor of 28.
#
# Itemized ledger for the small form (forward, verdict-testbed geometry):
#   Hop A  4 -> 3: counts + payload (its own call) + [id‖gate] (saves 1 call,
#          α₁₂₈=0.450 ms; the container is 32 B/row instead of 4128 B/row, the
#          copy is negligible ⇒ net ≈ −0.43 ms)
#   Hop B  4 -> 3: counts + exp_rx + [slot‖gate] (saves 1 call, α₈=0.058 ms;
#          net +0.056 ms)
# The full form (TERRACE_TA2A_PACK=full) still runs; its counts are 7/6, kept
# only to reproduce that reading and for A/B.
#          The payload plane exp_rx is deliberately not packed: packing it pays
#          an extra 0.317 ms of copies to save 0.116 ms, a net loss of
#          0.20 ms — the same physics as the 2026-08-01 torch.cat packing
#          failure (arithmetic in ta2a_pack).
#   combine 2 -> 2: untouched (only 2 calls to begin with; measured phase ties
#          the vendor)
# Backward 6 -> 6 is **deliberately unchanged**: the two float lanes each hang
# on an independent edge (terrace/ta2a_pack.py::_PackedEdge), and the backward
# is still the same few _a2a_raw calls as before packing — fusing into a single
# node would collide with the vendor gmm's two .backward() calls into the
# permute2 segment (internal engineering records, 2026-08-20), and the overlap
# seam's Hop A backward is manually replayed by the vendor anyway; we cannot
# change it. So the backward count may neither drop nor rise.
# The assertion still pins **exact values** (not <=): a count increase = packing
# silently reverted, a further decrease = an exchange silently dropped; both
# directions must go red on the spot.
SEAM_FWD_A2A, SEAM_BWD_A2A = 8, 6


def _routing(gen, n_nodes, per, quota):
    """T-Route equal-quota: exactly M nodes, exactly K/M experts per node.

    Each row sorted ascending: routing_map_to_topk recovers indices via nonzero,
    which is naturally ascending — both arms must be fed the **same** [T, K]
    permutation, otherwise the inputs are not bitwise identical and the
    assertions test something else entirely.
    """
    rows = []
    for _ in range(T):
        gs = torch.randperm(n_nodes, generator=gen)[:M]
        rows.append(torch.cat([torch.arange(a * per, (a + 1) * per)[
            torch.randperm(per, generator=gen)[:quota]] for a in gs]))
    return torch.sort(torch.stack(rows), dim=1).values


def _bitdiff(a, b):
    """None = bitwise equal; otherwise (description, max|Δ|) to localize the failure."""
    if a is None or b is None:
        return ("missing", None)
    if a.dtype != b.dtype:
        return (f"{a.dtype} vs {b.dtype}", float((a.float() - b.float()).abs().max()))
    if torch.equal(a, b):
        return None
    return (str(a.dtype), float((a.float() - b.float()).abs().max()))


def _run(rank, world, q, payload_dtype_name):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # The golden counts measure the call counts of the **as-shipped
    # configuration**, so explicitly clear any env override of the packing gate:
    # when an operator A/Bs on the cluster with TERRACE_TA2A_PACK=0, this file
    # must not go red along with them (bitwise equivalence of the packed-on and
    # packed-off paths is guarded separately by tests/test_ta2a_pack_bitparity.py).
    os.environ.pop("TERRACE_TA2A_PACK", None)
    # Ports in use: 29577/29591/29613/29623/29627. This bed takes one per dtype.
    os.environ.setdefault(
        "MASTER_PORT", "29641" if payload_dtype_name == "float32" else "29645")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from terrace.layer import grouped_mm
        from terrace.ta2a_fwd import ta2a_moe_forward, init_ta2a_groups
        from terrace.ta2a_dispatch import ta2a_permute, ta2a_unpermute

        pdt = getattr(torch, payload_dtype_name)
        intra = init_ta2a_groups(world, RPN)
        epr = E // world
        n_nodes, per, quota = world // RPN, E // (world // RPN), K // M

        # Counters: each arm's all_to_all_single count (forward / backward
        # tracked separately). terrace.ep_dist / ta2a_fwd / ta2a_dispatch hold
        # the same module object, so patching the module attribute reaches every
        # call site.
        counter = [0]
        real_a2a = dist.all_to_all_single

        def counting_a2a(*args, **kwargs):
            counter[0] += 1
            return real_a2a(*args, **kwargs)

        report = {}
        for gm in (M, None):
            g = torch.Generator().manual_seed(11 + rank)
            x0 = torch.randn(T, H, generator=g).to(pdt)
            idx = _routing(g, n_nodes, per, quota)
            gates0 = torch.rand(T, K, generator=g).to(pdt)
            w13_0 = (torch.randn(epr, H, 2 * D, generator=g) / (H ** 0.5)).to(pdt)
            w2_0 = (torch.randn(epr, D, H, generator=g) / (D ** 0.5)).to(pdt)
            G = torch.randn(T, H, generator=g).to(pdt)

            # ---- new arm: fused forward, gate applied at the arrival rank
            # (post-Step 2 code) ----
            xN = x0.clone().requires_grad_(True)
            gN = gates0.clone().requires_grad_(True)
            w13N = w13_0.clone().requires_grad_(True)
            w2N = w2_0.clone().requires_grad_(True)
            dist.all_to_all_single = counting_a2a
            try:
                counter[0] = 0
                yN = ta2a_moe_forward(xN, idx, gN, w13N, w2N, world, E, RPN,
                                      groups_m=gm)
                fused_fwd = counter[0]
                counter[0] = 0
                yN.backward(G)
                fused_bwd = counter[0]
            finally:
                dist.all_to_all_single = real_a2a

            # ---- control arm = the pre-change formula: legacy split-half, gate
            # shipped to the expert rank via the my_gate exchange, multiplied on
            # the expert side by the "vendor" (its experts.py equivalent inlined
            # here) ----
            routing_map = torch.zeros(T, E, dtype=torch.bool)
            routing_map[torch.arange(T).unsqueeze(1), idx] = True
            probs_dense = torch.zeros(T, E, dtype=pdt)
            probs_dense[torch.arange(T).unsqueeze(1), idx] = gates0

            xR = x0.clone().requires_grad_(True)
            pR = probs_dense.clone().requires_grad_(True)
            w13R = w13_0.clone().requires_grad_(True)
            w2R = w2_0.clone().requires_grad_(True)
            dist.all_to_all_single = counting_a2a
            try:
                counter[0] = 0
                perm, tpe, pprobs, st = ta2a_permute(
                    xR, pR, routing_map, world=world, rank=rank, rpn=RPN,
                    n_experts=E, intra_group=intra, inter_group=None, groups_m=gm)
                a, b = grouped_mm(perm, w13R, tpe).chunk(2, dim=-1)
                ye = grouped_mm(F.silu(a) * b, w2R, tpe) * pprobs.unsqueeze(-1)
                yR = ta2a_unpermute(ye, st, xR)
                seam_fwd = counter[0]
                counter[0] = 0
                yR.backward(G)
                seam_bwd = counter[0]
            finally:
                dist.all_to_all_single = real_a2a

            tag = f"gm={gm}"
            report[tag] = {
                "y": _bitdiff(yN.detach(), yR.detach()),
                "x": _bitdiff(xN.grad, xR.grad),
                # The dense probs leaf's gather backward only scatters the [T,K]
                # gradient to the routed positions — values unchanged — so
                # reading it back at the routed positions reconciles bit-for-bit
                # with the fused arm's gates gradient.
                "gates": _bitdiff(gN.grad, None if pR.grad is None
                                  else pR.grad.gather(1, idx)),
                "w13": _bitdiff(w13N.grad, w13R.grad),
                "w2": _bitdiff(w2N.grad, w2R.grad),
                "counts": (fused_fwd, fused_bwd, seam_fwd, seam_bwd),
            }
        q.put({"rank": rank, "status": "ok", "report": report})
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
def arrival_fp32():
    return _spawn("float32")


@pytest.fixture(scope="module")
def arrival_bf16():
    return _spawn("bfloat16")


def _assert_bitwise(results, dtype_label):
    for r in results:
        for tag, rep in r["report"].items():
            for name in ("y", "x", "gates", "w13", "w2"):
                assert rep[name] is None, (
                    f"rank {r['rank']} {dtype_label} {tag}: {name} "
                    f"gate-at-arrival vs gate-at-expert not bitwise equal "
                    f"{rep[name]} — pairing alignment broken")


def _assert_counts(results, dtype_label):
    for r in results:
        for tag, rep in r["report"].items():
            ff, fb, sf, sb = rep["counts"]
            assert (ff, fb) == (FUSED_FWD_A2A, FUSED_BWD_A2A), (
                f"rank {r['rank']} {dtype_label} {tag}: fused-forward collective "
                f"counts fwd={ff}/bwd={fb}, expected "
                f"{FUSED_FWD_A2A}/{FUSED_BWD_A2A} — more means Step 2 was "
                f"silently reverted, fewer means an exchange was silently dropped")
            assert (sf, sb) == (SEAM_FWD_A2A, SEAM_BWD_A2A), (
                f"rank {r['rank']} {dtype_label} {tag}: control-arm collective "
                f"counts fwd={sf}/bwd={sb}, expected "
                f"{SEAM_FWD_A2A}/{SEAM_BWD_A2A} — more means the A1′/A2 packing "
                f"was silently reverted (or the C1/Step 2 form changed), fewer "
                f"means an exchange was silently dropped")


@pytest.mark.timeout(300)
def test_fp32_gate_position_bitwise_equal(arrival_fp32):
    """fp32: forward and all four gradients, gate at the arrival rank == gate at
    the expert rank, bitwise."""
    _assert_bitwise(arrival_fp32, "fp32")


@pytest.mark.timeout(300)
def test_bf16_gate_position_bitwise_equal(arrival_bf16):
    """bf16 (production payload precision): same as above, bitwise."""
    _assert_bitwise(arrival_bf16, "bf16")


@pytest.mark.timeout(300)
def test_fp32_collective_counts(arrival_fp32):
    """The saving is pinned by counts: fused forward 9/5 (unpacked), split-half
    seam 8/6 (after the A1''/A2 packing)."""
    _assert_counts(arrival_fp32, "fp32")


@pytest.mark.timeout(300)
def test_bf16_collective_counts(arrival_bf16):
    _assert_counts(arrival_bf16, "bf16")
