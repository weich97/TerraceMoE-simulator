"""T-A2A split into the two halves of a dispatcher, so it can hook into the vendor's real training step (#18(a)).

`ta2a_fwd.ta2a_moe_forward` is a **closed** forward: dispatch, expert compute, and combine
all live inside it. Megatron's MoE layer wants a different shape — a pair of methods with
the **vendor's own** expert compute in between:

    permuted, tokens_per_expert, permuted_probs = token_permutation(h, probs, routing_map)
    expert_out = <vendor's grouped GEMM / aux loss / expert bias / shared experts>
    output, _  = token_unpermutation(expert_out)

So this file cuts that closed forward open at the expert compute and passes the
intermediates on a state object. Why this route (rather than replacing moe_layer.forward
wholesale) is argued in internal design records (not published with the repo):
a whole-layer replacement would fork us away from `--moe-router-enable-expert-bias`,
shared experts, and seq-aux-loss — vendor features **already validated in the four-tier
ablation** — turning a validated path back into an unvalidated one.

Three points that must be stated plainly:

1. **The gate is NOT multiplied here.** The vendor applies it at experts.py:241 via
   `permuted_probs.unsqueeze(-1) *`; our only job is to hand over each (row, slot) gate in
   the same order. The original fused forward multiplied at the expert itself
   (`wgt = my_gate[order]`); copying that would **multiply twice**.
2. **`tokens_per_expert` is mathematically forced**: T-A2A only changes payload dedup and
   the routing transport, not "which expert processes which (token, expert) pairs", so the
   counts we compute must equal the vendor's `preprocess(routing_map)` element for element.
   This is the hardest probe for "is the seam wired right" (enable with TERRACE_TA2A_ASSERT=1).
3. **EP=8 is a no-op by construction**: 8 dies on one node hold exactly one EP group, no
   cross-node hop, so we hand straight back to the vendor implementation. In the EP-tier
   sweep the EP=8 delta must be ≈0; failing that means the seam is wired wrong.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

from .ep_dist import _a2a, _a2a_raw
from .ta2a import plan_ta2a
from .ta2a_fwd import (build_expansion, _stable_argsort_small, _expand_arrival,
                       _expand_arrival_quota, _pack_quota_wire, _send_index,
                       _splits_to_lists, fixed_hist)
# K1 kernel gate (terrace.ops lazy-imports ta2a_fwd inside function bodies only; no cycle
# at module level).
from . import ops as _tops
# Collective packing (A1/A2, 2026-08-21): Hop A's [payload‖gate] and Hop B's [slot‖gate]
# each merge into one collective; dispatch goes 8 -> 6 collectives. Pure byte reshuffling;
# splits semantics and the vendor contract are untouched — the itemized gain/cost ledger
# and "which planes deliberately stay unpacked, and why" live in the terrace/ta2a_pack.py
# module header. Gate TERRACE_TA2A_PACK=0 runs the pre-packing chain verbatim (including
# collective ordering): zero behavior change.
from . import ta2a_pack as _pk
# Drift probe (active only with TERRACE_DRIFT_PROBE=1; otherwise each site adds one cached
# bool read). Both seams use the **same-named** probe points: with the on arm, run legacy
# and overlap twice on the same machine, and the CPU bit-for-bit contract of
# tests/test_ta2a_seam_bitparity.py can be diffed on the NPU
# (python -m terrace.drift_probe compare ov.log lg.log).
from . import drift_probe as _dp


class TA2AState:
    """Intermediates passed between the two halves. One per layer per microbatch.

    Why not a module-level global: a transformer has N MoE layers and the forward walks
    them in order (layer i's unpermutation happens before layer i+1's permutation), but
    recompute makes the same layer's permutation run twice — under that interleaving a
    global gets overwritten by whichever layer wrote last, showing up as "some layers'
    combine used another layer's plan", and only with recompute enabled. Hanging it on
    the dispatcher instance maps one-to-one to the layer by construction.
    """

    __slots__ = ("u_src", "r_idx", "order", "R", "T", "hidden",
                 "send_l", "recv_l", "is_l", "ir_l", "intra", "inter", "dtype")


# --------------------------------------------------------------------------------------
# A3-lite (2026-08-21): launch Hop A's counts exchange early, wait asynchronously
# --------------------------------------------------------------------------------------
#
# A3 (constant-folding the plan to kill both counts exchanges) was judged a net loss of
# −7.03 ms and is not implemented (internal design records, not published with the repo:
# equal-quota fixes only the per-token fan-out M, not the per-destination load, so the
# capacity bound must take the geometric worst case n_nodes/M = 8x and rpn = 8x). **The
# only salvageable fragment** is this: the inter counts exchange does not need to wait for
# local packing to finish — it depends only on the node_counts that plan_ta2a hands over.
# Move it to right after plan, send it async, and let the local gather + packing time
# cover its α₁₂₈.
#
#   Zero redundant bytes, zero numerical change, no capacity assumption needed; the
#   recovery = min(α₁₂₈, local packing time), and it must be measured directly on the
#   machine (verdict testbed α₁₂₈ = 0.45 ms; the packing segment is a 33 MB gather plus a
#   ~0.05 ms packing copy — the desk estimate says only a small part gets covered — so it
#   is one independent cut with an independent reading; do not co-mingle it with A1′/A2).
#
# The intra exchange (near :459) has no such opportunity: i_send comes from the arrival
# expansion, and there is no coverable local work between it and Hop B.
#
# Gate TERRACE_TA2A_ASYNC_COUNTS=0 falls back to the original in-place synchronous send,
# for standalone A/B attribution on the machine.

_ENV_ASYNC_COUNTS = "TERRACE_TA2A_ASYNC_COUNTS"
_ASYNC_COUNTS: bool | None = None


_SYNC_PROBE_ENV = "TERRACE_TA2A_SYNC_PROBE"
_SYNC_PROBE = None


def sync_probe_enabled() -> bool:
    """Discriminating probe: one **purely discarded** .tolist() right after i_send. Off by default.

    This is not an optimization; it is an **instrument**. Internal records falsified my
    original mechanistic explanation of fixed_hist: the host sync measures at only
    0.042-0.046 ms, while the bincount it replaced cost 0.797 — sync explains at most 6%.
    Yet fixed_hist genuinely got -1.166 on the machine (reproduced twice). A gain with an
    unexplained mechanism must not be used to justify the next cut, hence the
    discrimination:

      Run one tier with this probe on. If dispatch goes back to ~9.2 => that 1.166 is
      mostly the price of "doing one host sync at that spot", and the sync family (the
      guards, the two _splits_to_lists) gets a real anchor;
      if it stays at ~8.1 => the 1.166 is the cost of bincount's own implementation, the
      sync family has no anchor, and the effort moves elsewhere immediately.

    **Both outcomes are useful** — exactly what a discriminating experiment should be.
    """
    global _SYNC_PROBE
    if _SYNC_PROBE is None:
        _SYNC_PROBE = os.environ.get(_SYNC_PROBE_ENV, "0").strip() == "1"
    return _SYNC_PROBE


def reset_sync_probe() -> None:
    """Test/debug hook."""
    global _SYNC_PROBE
    _SYNC_PROBE = None


_EARLY_HOPB_ENV = "TERRACE_TA2A_EARLY_HOPB"
_EARLY_HOPB = None


def early_hopb_counts_enabled() -> bool:
    """A6 gate: launch Hop B's counts exchange early and async. On by default.

    Why default-on: this is a **pure scheduling reorder** — the value of i_send, the
    alltoall semantics, the contents of i_recv, and every downstream bit are unchanged;
    only the launch point moves earlier. Unlike A1'/A5 it touches neither layout nor
    reduction order, so no eq gate is needed. The gate exists only for A/B on the machine
    (TERRACE_TA2A_EARLY_HOPB=0 runs the original text).

    Read once per process lifetime: dispatch is the per-layer per-microbatch hot path.
    """
    global _EARLY_HOPB
    if _EARLY_HOPB is None:
        _EARLY_HOPB = os.environ.get(_EARLY_HOPB_ENV, "1").strip() != "0"
    return _EARLY_HOPB


def reset_early_hopb() -> None:
    """Forget the A6 gate's cached verdict. Test/debug hook; training code must not call it."""
    global _EARLY_HOPB
    _EARLY_HOPB = None


def async_counts_enabled() -> bool:
    """A3-lite gate. Reads the environment once per process lifetime (dispatch is a hot path)."""
    global _ASYNC_COUNTS
    if _ASYNC_COUNTS is None:
        _ASYNC_COUNTS = os.environ.get(_ENV_ASYNC_COUNTS, "1").strip() != "0"
    return _ASYNC_COUNTS


def reset_async_counts() -> None:
    """Forget the cached verdict. Test/debug hook; training code must not call it."""
    global _ASYNC_COUNTS
    _ASYNC_COUNTS = None


def _group_world(group) -> int:
    """World size of the communication group; 0 if unavailable (callers fall back to the old behavior).

    From 2026-08-24 on, Hop A's communication group may be a **cross-node subgroup**
    (n_nodes ranks) instead of the whole EP group (world ranks). The lengths of counts and
    splits must follow it — on a length mismatch HCCL blows up on the spot rather than
    failing silently, which is good; but both paths still have to get the length right.
    """
    if group is None:
        return 0
    try:
        return int(dist.get_world_size(group=group))
    except Exception:                                       # noqa: BLE001
        return 0


def _hopa_counts(world: int, n_nodes: int, rpn: int, my_local: int, dev,
                 node_counts: torch.Tensor, inter_world: int = 0):
    """The two buffers for Hop A's counts exchange. The scatter index is a pure function of the geometry (cached; see _send_index).

    When `inter_world == n_nodes` we are on the **cross-node subgroup**: in-group rank i is
    exactly node i, so send is just node_counts — those 112 zero paddings are precisely
    what this eliminates. Otherwise fall back to the whole EP group (length world, only
    n_nodes nonzeros), verbatim the pre-2026-08-24 behavior.
    """
    if inter_world == n_nodes:
        send = node_counts.to(torch.long)
        return send, torch.empty_like(send)
    send = torch.zeros(world, dtype=torch.long, device=dev)
    send[_send_index(n_nodes, rpn, my_local, dev)] = node_counts
    return send, torch.empty_like(send)


def _per(src: torch.Tensor, n_out: int):
    """Exactly how many contributions per reduction target (only when divisible, else None).
    Holds by construction under equal-quota: combine's first level takes exactly quota
    contributions per row, the second exactly M per token — with it, the drift probe can
    use a **fixed-shape** reshape-sum as the deterministic reference to gauge the
    accumulator width of the device index_add."""
    n = int(src.shape[0])
    return (n // n_out) if (n_out and n % n_out == 0) else None


def routing_map_to_topk(routing_map: torch.Tensor, probs: torch.Tensor):
    """[T, E] boolean map + [T, E] probabilities -> [T, k] expert ids and gates. Each row ascending.

    The vendor's routing_map is a dense boolean map, while plan_ta2a wants top-k indices.
    This **requires exactly k per row** and fails loud: under drop-less routing that is
    always true; once it isn't (say a capacity-drop router gets wired in later), the shape
    changes silently, and reshaping by k would mix different tokens' experts into one
    row — the kind of error invisible in the loss.

    Extraction uses topk + ascending sort rather than nonzero (2026-08-20, knife-forging B):
      - Bit-identical: exactly-k-per-row is verified first, so the **set** topk selects
        over {0,1} values is determined (all k ones taken, independent of tie order);
        after the ascending sort it matches nonzero's row-major order bit for bit.
        Index values < E << 2^24 sort exactly in float32 (integer sort is this device's
        most expensive primitive, see _stable_argsort_small).
      - Saves two host syncs: nonzero's output shape is data-dependent and must sync; the
        guard used to pay `.item()` and `.all()` on top. Now the guard's min/max merge
        into one tolist — 3 syncs -> 1 for the whole function. Counts are summed in
        float32 (k <= E < 2^24, exact; int64 summation is a 37x slower primitive,
        internal benchmark script (not published with the repo)).
    """
    rm_f = routing_map.to(torch.float32)
    counts = rm_f.sum(dim=1)                     # exact: at most E ones per row, E << 2^24
    mn_mx = torch.stack((counts.min(), counts.max())).tolist()   # the only sync in this function
    if mn_mx[0] != mn_mx[1]:
        raise RuntimeError(
            "T-A2A requires exactly k experts per token (drop-less); observed per-row expert counts differ")
    k = int(mn_mx[0])
    if k == 0:
        raise RuntimeError("T-A2A got an all-empty routing_map (k=0); impossible under drop-less routing")
    sel = torch.topk(rm_f, k, dim=1).indices                     # exactly each row's k set bits
    if routing_map.shape[1] < (1 << 24):
        expert_idx = torch.sort(sel.to(torch.float32), dim=1).values.to(torch.int64)
    else:                                        # unreachable guard: integer sort needed only when E >= 2^24
        expert_idx = torch.sort(sel, dim=1).values
    gates = probs.gather(1, expert_idx)
    return expert_idx, gates


def ta2a_permute(hidden_states, probs, routing_map, *, world, rank, rpn,
                 n_experts, intra_group, inter_group=None, groups_m=None):
    """The dispatch half: returns (rows sorted by local expert, row count per local expert, per-row gates, state).

    Line-for-line isomorphic to the first half of `ta2a_moe_forward` (the dispatch
    segment, up to the expert-order arrangement); the only differences: no gate multiply,
    the intermediates go into state, and every collective runs on the passed-in group
    (the vendor's EP group) instead of the default global group.
    """
    epr = n_experts // world
    slots = epr * rpn
    dev = hidden_states.device
    my_local = rank % rpn
    # Read the gate once: both hops of one dispatch must take the same path (half-packed
    # is a third form nobody has validated), and pack_enabled is a module attribute tests
    # can monkeypatch.
    pack_mode = _pk.pack_mode()
    packing = pack_mode != "off"

    expert_idx, gates = routing_map_to_topk(routing_map, probs)
    T, k = expert_idx.shape

    u_src, u_node, node_counts, inverse = plan_ta2a(
        expert_idx, world, n_experts, rpn, groups_m=groups_m)
    n_rows = u_src.numel()
    n_nodes = world // rpn
    # A3-lite: the counts exchange depends only on node_counts; launch it early and cover
    # its α₁₂₈ with the gather + packing below (see the A3-lite section at the top of this
    # file). With the gate off it stays at the original spot as a synchronous send: zero
    # behavior change.
    h_cnt = send = recv = None
    if async_counts_enabled():
        send, recv = _hopa_counts(world, n_nodes, rpn, my_local, dev, node_counts,
                                  inter_world=_group_world(inter_group))
        h_cnt = dist.all_to_all_single(recv, send, group=inter_group, async_op=True)
    payload = hidden_states[u_src]
    if groups_m:
        assert k % groups_m == 0, f"k={k} not divisible by groups_m={groups_m}"
    quota = (k // groups_m) if groups_m else None
    if quota is not None:
        # C1 quota wire format (2026-08-20): Hop A's id plane switches from int64 bitmask
        # to [n_rows, quota] ascending slot ids, the gate plane from [n_rows, slots]
        # sparse to [n_rows, quota] dense — ascending slot ids == the output order of the
        # old arrival-side topk; the pair sequence and every downstream sort stay
        # bit-for-bit unchanged (argument and exhaustive check in _pack_quota_wire).
        # sorted_rows: routing_map_to_topk is ascending per row by construction, so the
        # packing side needs zero sorts.
        mask, gate_rows = _pack_quota_wire(expert_idx, gates, inverse, payload,
                                           n_rows, slots, quota, n_experts,
                                           sorted_rows=True)
    else:
        # One modulo, shared by the mask plane and the gate plane; sorted_rows:
        # routing_map_to_topk is ascending per row by construction (mask bits unchanged
        # bit for bit, see build_expansion).
        slot_flat = expert_idx.reshape(-1) % slots
        mask = build_expansion(expert_idx, inverse, n_rows, world, n_experts, rpn,
                               groups_m=groups_m, sorted_rows=True,
                               slot_flat=slot_flat)
        gate_rows = payload.new_zeros(n_rows, slots)
        gate_rows[inverse, slot_flat] = gates.reshape(-1)
    _dp.note("seam.payload", payload)
    _dp.note("seam.gate", gate_rows)
    _dp.note_int("seam.mask", mask)

    if h_cnt is None:
        send, recv = _hopa_counts(world, n_nodes, rpn, my_local, dev, node_counts,
                                  inter_world=_group_world(inter_group))
        dist.all_to_all_single(recv, send, group=inter_group)
    else:
        h_cnt.wait()
    send_l, recv_l = _splits_to_lists(send, recv)      # one sync, not two

    # mask is the id plane under either wire format: [n_rows] int64 bitmask (general) or
    # the [n_rows, quota] ascending slot table (C1 fast path). The three planes share
    # splits; A1′ merges them into one collective.
    if pack_mode == "small":
        # A1'' (2026-08-22, default form changed per verdict-testbed measurement): **merge
        # only the two small planes** (id ‖ gate, 32 B/row); the payload [n, 2048] takes
        # its own collective — **verbatim the same line** `_a2a(payload, ...)` as the
        # unpacked arm, so the payload path's gradients and collectives change zero.
        # Hop A collectives 4 -> 3, saving 1 α₁₂₈ ≈ 0.45 ms; the copy volume is 129x less
        # than the full form. Why not full (2 collectives): full has to copy the payload
        # into the container and back out; the verdict testbed measured those two HBM
        # passes at ≈ 3.0 ms, eating the 0.96 ms saved and losing 2.05 ms on top
        # (internal measurement records).
        with torch.no_grad():
            _sbuf, _lay = _pk.hopa_pack_small(gate_rows.detach(), mask)
            _rbuf = _pk.hopa_exchange_raw(_sbuf, send_l, recv_l, group=inter_group)
            _rgate_raw, rmask = _pk.hopa_unpack_small(_rbuf, _lay)
        _pk.assert_not_aliased(_rbuf, _rgate_raw, rmask)
        rx = _a2a(payload, send_l, recv_l, group=inter_group)   # verbatim the same line as the off arm
        rgate = _pk.attach_edge(gate_rows, _rgate_raw, send_l, recv_l, inter_group)
    elif pack_mode == "full":
        # A1′: id + payload + gate merged into one collective (int64 container, id plane
        # leads each row to keep 8-byte alignment). The data lands outside the graph; the
        # graph is rebuilt via two independent edges — backward is still the same two
        # _a2a_raw lines as before packing, gradients bit-for-bit unchanged, backward
        # collective count unchanged.
        # **Verdict testbed measured a net loss of 2.05 ms/call**; kept only to reproduce
        # that reading and for A/B, no longer the default.
        with torch.no_grad():
            _sbuf, _lay = _pk.hopa_pack(payload.detach(), gate_rows.detach(), mask)
            _rbuf = _pk.hopa_exchange_raw(_sbuf, send_l, recv_l, group=inter_group)
            _rx_raw, _rgate_raw, rmask = _pk.hopa_unpack(_rbuf, _lay)
        _pk.assert_not_aliased(_rbuf, _rx_raw, _rgate_raw, rmask)
        rx = _pk.attach_edge(payload, _rx_raw, send_l, recv_l, inter_group)
        rgate = _pk.attach_edge(gate_rows, _rgate_raw, send_l, recv_l, inter_group)
    else:
        rx = _a2a(payload, send_l, recv_l, group=inter_group)
        rmask = _a2a_raw(mask, send_l, recv_l, group=inter_group)
        rgate = _a2a(gate_rows, send_l, recv_l, group=inter_group)
    _dp.note("seam.rx", rx)
    _dp.note("seam.rgate", rgate)
    _dp.note_int("seam.rmask", rmask)

    R = rx.shape[0]
    # K1 (AscendC kernel, 2026-08-20): under the quota fast path, the whole arrival chain
    # below (pair expansion, owner stable bucket sort, i_send histogram, [pairs, H] send
    # gather, gate flat gather) fuses into one kernel. The argument that the two-pass
    # scheme matches the current chain's stable order bit for bit is in the header of
    # terrace/ops/ascendc/op_kernel/terrace_k1_arrival.cpp; the CPU executable spec is
    # terrace.ops.k1_arrival_ref. The else branch is the pre-K1 chain **verbatim** and is
    # the only path when the kernel is absent (TERRACE_CUSTOM_OPS=0 / no .so): zero
    # behavior change. exp_rx/gate_pairs are hoisted above the collectives only so both
    # branches enter Hop B with the same five tensors — pure dataflow reorder; operators
    # and operands unchanged.
    _h_i = _i_recv_early = None          # A6's handles; the K1 branch does not send early, stays None
    if quota is not None and _tops.custom_ops_enabled():
        exp_rx, gate_pairs, r_idx, slot_idx, i_send = _tops.k1_arrival(
            rx, rmask, rgate, quota, epr, rpn, my_local)
    else:
        if quota is not None:
            r_idx, slot_idx = _expand_arrival_quota(rmask)  # the slot table IS the pair table; skips bit extraction + topk
        else:
            r_idx, slot_idx = _expand_arrival(rmask, slots, quota)
        owner = slot_idx // epr
        # A6: the histogram is permutation-blind — computable the moment owner exists, no
        # need to wait for the sort. Fire Hop B's counts exchange **async** right after,
        # and cover it with the sort + two index gathers + the big [pairs,H] gather below.
        # Pure scheduling reorder: i_send values, alltoall semantics, i_recv contents,
        # everything downstream bit-for-bit unchanged.
        # (Basis: internal measurements showed "one fewer collective" is worth zero;
        # "moved to where real compute can cover it" is where the gain is.)
        i_send = fixed_hist(owner, rpn)          # fixed-length histogram, avoids bincount's hidden host sync
        if sync_probe_enabled():
            _ = i_send.tolist()   # discriminating probe: purely discarded, only pays one host sync (see sync_probe_enabled)
        if early_hopb_counts_enabled():
            _i_recv_early = torch.empty_like(i_send)
            _h_i = dist.all_to_all_single(_i_recv_early, i_send, group=intra_group,
                                          async_op=True)
        ordo = _stable_argsort_small(owner, rpn)
        r_idx, slot_idx = r_idx[ordo], slot_idx[ordo]
        exp_rx = rx[r_idx]
        # C1 fast path: the dense gate table is isomorphic to the pair enumeration, so the
        # 2-D gather degenerates to a flat gather — rgate.reshape(-1)[ordo] equals
        # rgate[r_idx, slot_idx] element for element (flat position row*quota+i is exactly
        # that pair's pre-ordo position), bit-for-bit equal.
        gate_pairs = (rgate.reshape(-1)[ordo] if quota is not None
                      else rgate[r_idx, slot_idx])
    _dp.note("seam.exprx", exp_rx)
    _dp.note("seam.gpairs", gate_pairs)
    _dp.note_int("seam.slot", slot_idx)
    _dp.note_int("seam.isend", i_send)
    # With A6 on and on the non-K1 branch, the async send already happened above — here we
    # only wait for it to land. The K1 branch's i_send comes out of the kernel, too late
    # to send early; it stays synchronous in place.
    if _h_i is not None:
        _h_i.wait()
        i_recv = _i_recv_early
    else:
        i_recv = torch.empty_like(i_send)
        dist.all_to_all_single(i_recv, i_send, group=intra_group)
    is_l, ir_l = _splits_to_lists(i_send, i_recv)

    # Step 2 (the fused forward dropped the my_gate exchange on 2026-08-20 and applies the
    # gate at the arrival rank) **does not apply to the seam**: here the my_gate exchange
    # IS the delivery of the vendor-contract permuted_probs — the multiply point is in the
    # vendor's hands (experts.py:241), not ours to choose; dropping it = changing the seam
    # semantics, not saving work.
    node_rx = _a2a(exp_rx, is_l, ir_l, group=intra_group)
    if packing:
        # A2: Hop B's two **one-scalar-per-pair** planes (slot id int64 + gate) merge into
        # one collective. The payload plane exp_rx deliberately stays **unpacked** —
        # merging it costs two extra 100-MB-scale HBM copies (+0.317 ms) to save 1 α₈
        # (0.058 ms), a 0.20 ms net loss on the verdict testbed; the arithmetic and the
        # zero-copy follow-up live in the Hop B section of terrace/ta2a_pack.py.
        with torch.no_grad():
            _bbuf = _pk.hopb_pack_meta(slot_idx, gate_pairs.detach())
            _rbb = _a2a_raw(_bbuf, is_l, ir_l, group=intra_group)
            my_slot, _mg_raw = _pk.hopb_unpack_meta(_rbb, gate_pairs.dtype)
        my_gate = _pk.attach_edge(gate_pairs, _mg_raw, is_l, ir_l, intra_group)
    else:
        my_slot = _a2a_raw(slot_idx, is_l, ir_l, group=intra_group)
        my_gate = _a2a(gate_pairs, is_l, ir_l, group=intra_group)

    exp_j = my_slot - my_local * epr
    order = _stable_argsort_small(exp_j, epr)
    tokens_per_expert = fixed_hist(exp_j, epr)                 # permutation-blind: skips exp_j[order]

    st = TA2AState()
    st.u_src, st.r_idx, st.order = u_src, r_idx, order
    st.R, st.T, st.hidden = R, T, hidden_states.shape[1]
    st.send_l, st.recv_l, st.is_l, st.ir_l = send_l, recv_l, is_l, ir_l
    st.intra, st.inter, st.dtype = intra_group, inter_group, hidden_states.dtype
    permuted, pprobs = node_rx[order], my_gate[order]
    _dp.note("seam.permuted", permuted)
    _dp.note("seam.pprobs", pprobs)
    _dp.note_int("seam.tpe", tokens_per_expert)
    return permuted, tokens_per_expert, pprobs, st


def ta2a_unpermute(expert_out, st: TA2AState, out_like):
    """The combine half: send expert outputs back to their origin tokens along dispatch's reverse.

    **No gate multiply** — the vendor already multiplied with permuted_probs in
    experts.py. Each returned row carries "the weighted sum over all relevant experts on
    the destination node", so the origin adds once per (token, node) (u_src names exactly
    that token); adding per (token, expert) was an early bug — it duplicates rows M times
    and re-applies gates that were already consumed.
    """
    back_pairs = expert_out.new_empty(expert_out.shape)
    back_pairs[st.order] = expert_out
    ret = _a2a(back_pairs, st.ir_l, st.is_l, group=st.intra)   # reverse: ir, is
    red = ret.new_zeros(st.R, ret.shape[1])
    red.index_add_(0, st.r_idx, ret)
    _dp.note("seam.eout", expert_out)
    _dp.note("seam.ret", ret)
    _dp.note("seam.red", red)
    _dp.check_reduction("seam.red", red, ret, st.r_idx, st.R, _per(ret, st.R))
    back = _a2a(red, st.recv_l, st.send_l, group=st.inter)     # reverse: recv, send
    y = out_like.new_zeros(st.T, st.hidden)
    y = y.index_add(0, st.u_src, back)
    _dp.note("seam.back", back)
    _dp.note("seam.y", y)
    _dp.check_reduction("seam.y", y, back, st.u_src, st.T, _per(back, st.T))
    return y


# --------------------------------------------------------------------------------------
# The overlap family (--moe-alltoall-overlap-comm + alltoall_seq): the two halves of the
# 6-arg seam (#18a/18c, Phase B)
#
# Vendor seam (upstream training stack (version pinned) moe_feature/overlap, read-only
# study conclusions; line numbers per the 2607 unpacked copy):
#
#   MoELayerOverlapAllToAllSeq.forward (moe_layer_overlap_all2allseq.py:69) calls
#     (share_experts_output, dispatched_input, tokens_per_expert, global_probs) =
#         token_dispatcher.token_permutation(
#             hidden_states, scores, routing_map, shared_experts, save_tensors, moe_ctx)
#   — that is the "6 args": 3 tensors + the shared-expert module (may be None) +
#   the save_tensors list + the layer's autograd.Function ctx. The combine side (:83) is
#   3-arg and **returns a single tensor** (not legacy's 2-tuple):
#   output = token_dispatcher.token_unpermutation(expert_output, mlp_bias, save_tensors).
#
#   Key structure: the whole layer is one autograd.Function with a **hand-written
#   backward**. The forward cuts compute into independent subgraphs ("segments"; the
#   vendor builds them with forward_func, segment inputs are detached leaves); the
#   EP-group all_to_all between segments deliberately sits outside the segments, outside
#   autograd, and backward replays it by hand using input_splits/output_splits on the
#   dispatcher. backward unpacks save_tensors by **fixed position**
#   (moe_layer_overlap_all2allseq.py:157-168), so the dispatcher halves must append
#   exactly 7 + 3 entries in this order:
#     permutation, 7: permute1_graph, permuted_probs_graph,
#         num_global_tokens_per_local_expert_cpu, permute2_input_detach, permute2_graph,
#         permute2_prob_detach, permute2_prob_graph
#     unpermutation, 3: unpermute1_input_detach, unpermute1_graph,
#         unpermute2_input_detach
#   The expert-side gmm is also a hand-written backward
#   (grouped_mlp_with_comp_and_comm_overlap_all2allseq.py): it calls backward_func on
#   permute2_graph / permute2_prob_graph, then replays the two detached leaves' .grad
#   backward along the EP group with (input_splits, output_splits) and hands them back to
#   the moe layer; the moe-layer backward does the same manual replay for
#   unpermute2_input_detach.grad.
#
# How the T-A2A halves plug in (segment boundaries match the vendor's one for one — that
# is the whole point of this adaptation):
#   - The one EP a2a the vendor replays by hand == T-A2A's Hop A (the cross-node fabric
#     hop, same EP group). Hand over disp.input_splits=send_l, disp.output_splits=recv_l,
#     and the vendor backward's two manual replays are **exactly Hop A's reverse** — not
#     one line of vendor code changes.
#   - Hop B (the intra-node a2a) hides entirely inside the permute2 / unpermute1
#     subgraphs, using ep_dist._A2A (a differentiable all_to_all with its own backward).
#     The hand-written backward only calls .backward() on those segments; autograd runs
#     Hop B in reverse automatically, and vendor code never needs to know it exists.
#   - Segment boundary == detach point: landing tensors rx / rgate
#     (= permute2_input_detach / permute2_prob_detach), return tensor back
#     (= unpermute2_input_detach).
#   - Gate precision follows payload.dtype, the **same rounding point** as the legacy
#     half (on entry into gate_rows on the dispatch side). One internal commit went with
#     "stay faithful to the vendor overlap's probs plane (router output precision)", i.e.
#     probs.dtype; the eqov alignment testbed (2026-08-20, slots=16, noise floor 1.3e-5)
#     falsified that choice: an fp32 gate plane, through the vendor gmm's probs multiply,
#     promotes expert_out to fp32, which drags combine's two return hops and both
#     index_add reduction levels into fp32 — every reduction rounds differently from the
#     validated legacy path (bf16 plane). Below the dispatch assertion tolerance (the
#     exit token plane is still bit-for-bit equal) but growing step by step:
#     1e-5@20 → 1.38e-4@100, calibration ratio 10.6× (bound 3×). After switching back to
#     payload.dtype, the two seams are bit-for-bit equal in forward and in all gradients
#     under the testbed configuration (bf16 payload + fp32 router probs)
#     (tests/test_ta2a_seam_bitparity.py). The probs gradient still flows back to the
#     fp32 leaf: .to's backward is an exact upcast, the vendor's hand-written
#     orchestration never sees the cast node, the seat contract is unchanged.
#
# The 18c (shared experts x A2A overlap) checklist:
#   Comes free (rides the vendor's scheduling; we write nothing):
#     - Shared-expert forward overlapping Hop A: this function calls run_shared_experts
#       back after Hop A launches async and before the wait, the same segment position as
#       the vendor's token_permutation;
#     - the placement of the shared-expert backward (the moe-layer backward handles
#       share_experts_graph uniformly);
#     - overlap of expert dW with dispatch's backward a2a (inherent to the gmm
#       hand-written backward, transparent to the two halves);
#     - activation recompute (should_recompute_activation) touches only gmm internals,
#       never dispatch.
#   Explicitly not doing (left to follow-up pieces; the wrapper gate falls back, raises
#   under REQUIRE):
#     - phase timing (dispatch/combine per-phase timestamps) — its own work item;
#       (groups_m passthrough was finished by Phase B's second piece: the wrapper decides
#       the geometry on the first batch and passes M into the groups_m parameter of both
#       seams in this file, see usercustomize._ta2a_groups_m_for)
#     - moe_zero_memory level0/level1: those tiers' backward recomputes dispatch per the
#       *vendor's* permute, not isomorphic to T-A2A;
#     - TP>1 / moe_tp_extend_ep (vendor swaps in tp-ep hybrid communication groups) /
#       capacity-drop routing;
#     - alltoall (non-seq) 5-arg overlap, mc2moe, fb_overlap, the balanced_moe family —
#       not adapted.
#
# The 18c verification conclusion (2026-08-21, zero machine time, in-repo + 2607
# read-only):
#   The `moe_shared_expert_overlap` **switch has nothing to do with alltoall_seq**; it is
#   the four-stage overlap switch (pre_forward_comm / linear_fc1_forward_and_act /
#   linear_fc2_forward+post_forward_comm / get_output) of Megatron's native alltoall
#   dispatcher, validated at `megatron/core/transformer/transformer_config.py:646-651`
#   (0.12.1). The alltoall_seq overlap family **ships its own seq-flavored overlap** and
#   never reads that switch: the layer unconditionally hands shared_experts to
#   `token_permutation` (moe_layer_overlap_all2allseq.py:59-70), and the dispatcher runs
#   it after both async a2a launches and before the wait
#   (overlap/token_dispatcher.py:179-206); conversely, the layer **explicitly rejects**
#   the switch (overlap/moe_layer.py:118-121 raise ValueError).
#   ⇒ The inference "moe_shared_expert_overlap=False ⇒ shared experts run serially" does
#   not hold; our seam calls run_shared_experts back at the same segment position, so the
#   overlap is **already realized**. Under the verdict-testbed geometry (16 nodes /
#   EP=128 / h=2048 / d_shared=768 / T=4096 tok/rank) the shared-expert GEMM is only
#   6·T·h·d/F = 0.118 ms while Hop A is 0.726 ms (α₁₂₈ 0.45 + wire 0.276) — it covers
#   16%, and it is the **only** compute in the microbatch with no data dependence on
#   dispatch; the uncovered 0.61 ms can only come from cutting communication
#   (A1′/A2/A4/K1/K2).
#   Full argument and the item-by-item alternatives: internal design records (not
#   published with the repo).


_ENV_SHARED_OVERLAP = "TERRACE_SHARED_OVERLAP"
_SHARED_OVERLAP: bool | None = None


def shared_overlap_enabled() -> bool:
    """The 18c A/B gate (default on). =0 moves shared experts to run after ALL dispatch collectives.

    Why this gate exists: overlap is a **device-side** matter (HCCL on the communication
    stream, GEMM on the compute stream) — no desk analysis can read it, and "we believe
    it overlaps" is not a reading. Turning it off = same collectives, same order, same
    numerics; only the shared-expert kernels enqueue after all dispatch communication —
    the step-time delta between the two arms IS 18c's true value on this testbed.
    Bit-level safe: the shared-expert subgraph has no data dependence on dispatch;
    operators, operands, and internal order all identical, `torch.equal`-level equality
    (tests/test_ta2a_shared_overlap.py).
    """
    global _SHARED_OVERLAP
    if _SHARED_OVERLAP is None:
        _SHARED_OVERLAP = os.environ.get(_ENV_SHARED_OVERLAP, "1").strip() != "0"
    return _SHARED_OVERLAP


def reset_shared_overlap() -> None:
    """Forget the cached verdict. Test/debug hook; training code must not call it."""
    global _SHARED_OVERLAP
    _SHARED_OVERLAP = None


def _a2a_async(x, in_splits, out_splits, group=None):
    """Async raw all_to_all (Hop A only): returns (out, handle); the caller owns the wait.

    Outside the segments, outside autograd — its backward is replayed by hand by the
    vendor backward; building a graph here would actually be wrong. Async is 18c's
    vehicle: launch, compute the shared experts, then wait.
    """
    out = x.new_empty((sum(out_splits), *x.shape[1:]))
    handle = dist.all_to_all_single(out, x.contiguous(), out_splits, in_splits,
                                    group=group, async_op=True)
    return out, handle


def ta2a_permute_overlap(hidden_states, probs, routing_map, *, world, rank, rpn,
                         n_experts, intra_group, inter_group=None, groups_m=None,
                         save_tensors, run_shared_experts=None):
    """The dispatch half of the overlap 6-arg seam.

    Same math as `ta2a_permute` (the legacy 3-arg half) — plan/expansion/two hops. Only
    three things differ, all dictated by the vendor's hand-written backward contract (see
    the big comment above):
      1. Segment boundaries: detach at rx/rgate; Hop A launches async outside the segments;
      2. save_tensors: append 7 entries at the vendor's fixed positions;
      3. the shared-expert callback runs after Hop A launches and before the wait (the
         18c overlap comes free; `TERRACE_SHARED_OVERLAP=0` moves it after all dispatch
         collectives, giving the machine a control arm for "what is the overlap actually
         worth" — numerics bit-for-bit unchanged).

    Returns (permuted, tokens_per_expert, permuted_probs, share_experts_output, state).
    The caller (the usercustomize wrapper) rearranges into the vendor's 4-tuple and sets
    the splits on disp.
    """
    epr = n_experts // world
    slots = epr * rpn
    n_nodes = world // rpn
    dev = hidden_states.device
    my_local = rank % rpn
    H = hidden_states.shape[-1]
    pack_mode = _pk.pack_mode()       # read once; rationale at the same spot in the legacy half
    packing = pack_mode != "off"
    overlap_shared = shared_overlap_enabled()   # same; hot path, read once
    share_experts_output = None

    def _shared_now():
        """Run the shared-expert callback at the current position (None when no shared experts are configured).

        The callback body (usercustomize._ta2a_overlap_seam_permute.run_shared) carries
        its own enable_grad; the graph roots directly on the hidden_states leaf the
        vendor built — isomorphic to the vendor's
        `forward_func(shared_experts, (hidden_states))` (its detach_tensor returns an
        input that already is a requires_grad leaf as-is; no new leaf).
        """
        return run_shared_experts() if run_shared_experts is not None else None

    # The whole function body under explicit enable_grad: the seam is called inside
    # autograd.Function.forward where grad is off by default, yet the segment graphs must
    # get built — the vendor's forward_func does the same. Forgetting it is not an error:
    # backward_func sees grad_fn is None and just returns, and the gradients silently
    # vanish for the whole segment.
    with torch.enable_grad():
        # ---- Segment 1 (= vendor permute1 position): purely local, rooted on the hidden/probs leaves ----
        expert_idx, gates = routing_map_to_topk(routing_map, probs)
        T, k = expert_idx.shape
        u_src, _, node_counts, inverse = plan_ta2a(
            expert_idx, world, n_experts, rpn, groups_m=groups_m)
        n_rows = int(u_src.numel())
        # A3-lite: same as the legacy half — launch the counts exchange early; cover its
        # α₁₂₈ with the gather + packing.
        h_cnt = send = recv = None
        if async_counts_enabled():
            send, recv = _hopa_counts(world, n_nodes, rpn, my_local, dev, node_counts,
                                  inter_world=_group_world(inter_group))
            h_cnt = dist.all_to_all_single(recv, send, group=inter_group,
                                           async_op=True)
        h = hidden_states.view(-1, H)
        payload = h[u_src]
        if groups_m:
            assert k % groups_m == 0, f"k={k} not divisible by groups_m={groups_m}"
        quota = (k // groups_m) if groups_m else None
        # gate always payload.dtype, the same rounding point as the legacy half — see the
        # "gate precision" item in the big comment above (one internal commit went with
        # probs.dtype; the eqov alignment testbed falsified it, 2026-08-20).
        # The cast commutes elementwise with gather/reshape, bit-identical to the legacy
        # path (the vendor layer casts probs to the hidden dtype upstream of dispatch).
        # The graph still roots on the original-precision probs leaf; .to's backward is
        # an exact upcast, invisible to the vendor orchestration. When probs already is
        # payload.dtype, .to returns self, adds no graph node, zero change bit for bit.
        if quota is not None:
            # C1 quota wire format: same as the legacy half (see that branch's comment in
            # ta2a_permute and the bit-exact argument in _pack_quota_wire); applying the
            # cast before packing == the same rounding point the old sparse plane had at
            # index_put.
            mask, gate_rows = _pack_quota_wire(
                expert_idx, gates.to(payload.dtype), inverse, payload,
                n_rows, slots, quota, n_experts, sorted_rows=True)
        else:
            # Same as the legacy half: one modulo shared by both planes; sorted_rows is
            # guaranteed by routing_map_to_topk's ascending construction (mask bits
            # unchanged bit for bit, see build_expansion).
            slot_flat = expert_idx.reshape(-1) % slots
            mask = build_expansion(expert_idx, inverse, n_rows, world, n_experts,
                                   rpn, groups_m=groups_m, sorted_rows=True,
                                   slot_flat=slot_flat)
            gate_rows = payload.new_zeros(n_rows, slots)
            gate_rows[inverse, slot_flat] = gates.reshape(-1).to(payload.dtype)
        _dp.note("seam.payload", payload)
        _dp.note("seam.gate", gate_rows)
        _dp.note_int("seam.mask", mask)
        save_tensors.append(payload)      # ↔ permute1_graph
        save_tensors.append(gate_rows)    # ↔ permuted_probs_graph

        # ---- Hop A (outside the segments, async): exactly the hop the vendor backward replays by hand ----
        if h_cnt is None:
            send, recv = _hopa_counts(world, n_nodes, rpn, my_local, dev, node_counts,
                                  inter_world=_group_world(inter_group))
            dist.all_to_all_single(recv, send, group=inter_group)
        else:
            h_cnt.wait()
        send_l, recv_l = _splits_to_lists(send, recv)      # one sync, not two

        # mask is the id plane under either wire format (bitmask or the C1 slot table);
        # see the same spot in the legacy half.
        if pack_mode == "small":
            # A1'' (default form; rationale at the same spot in the legacy half and in
            # internal measurement records): merge only id ‖ gate, the payload takes its
            # own collective. Both launch async and jointly cover the shared-expert
            # segment — an overlap window as long as the full form's.
            with torch.no_grad():
                _sbuf, _lay = _pk.hopa_pack_small(gate_rows.detach(), mask)
            rx, h_rx = _a2a_async(payload.detach(), send_l, recv_l, group=inter_group)
            _rbuf, h_sm = _pk.hopa_exchange_async(_sbuf, send_l, recv_l,
                                                  group=inter_group)
            if overlap_shared:
                share_experts_output = _shared_now()
            h_rx.wait()
            h_sm.wait()
            with torch.no_grad():
                rgate, rmask = _pk.hopa_unpack_small(_rbuf, _lay)
            _pk.assert_not_aliased(_rbuf, rgate, rmask)
            _sbuf.untyped_storage().resize_(0)
            _rbuf.untyped_storage().resize_(0)
        elif pack_mode == "full":
            # A1′: three merged into one. Hop A sits outside the segments, outside
            # autograd (the vendor backward replays it by hand), so only the **data** is
            # merged here; the replay walks disp.input_splits/output_splits =
            # send_l/recv_l, and those remain **row counts** (dim 0 of the packed buffer
            # is exactly n_rows, no scaling needed) — semantics untouched to the letter
            # ⇒ the vendor's two replays (rx_d.grad / rgate_d.grad) run unmodified, each
            # still one a2a.
            # **Verdict testbed measured a net loss of 2.05 ms/call**; kept only for
            # reproducibility and A/B, no longer the default.
            with torch.no_grad():
                _sbuf, _lay = _pk.hopa_pack(payload.detach(), gate_rows.detach(), mask)
            _rbuf, h_rx = _pk.hopa_exchange_async(_sbuf, send_l, recv_l,
                                                  group=inter_group)
            # 18c: Hop A in flight; compute the shared experts now — same segment
            # position as the vendor's token_permutation.
            if overlap_shared:
                share_experts_output = _shared_now()
            h_rx.wait()
            with torch.no_grad():
                rx, rgate, rmask = _pk.hopa_unpack(_rbuf, _lay)
            # Confirm the three planes no longer point into the buffer before returning
            # the memory. On 2026-08-21 verdict-testbed rank61 died exactly here: with
            # R==1, `.contiguous()` is a no-op, the unpack results are views of _rbuf,
            # and after resize(0) they dangle — the crash lands a dozen operators later
            # at `owner = slot_idx // epr`
            # ("non-zero number of elements, but its data is not allocated yet").
            # Root-caused at ta2a_pack._own; these three pointer comparisons are a
            # tripwire against regression, negligible on the hot path.
            _pk.assert_not_aliased(_rbuf, rx, rgate, rmask)
            _sbuf.untyped_storage().resize_(0)      # on the wire already, receive side unpacked — return the memory now
            _rbuf.untyped_storage().resize_(0)
        else:
            rx, h_rx = _a2a_async(payload.detach(), send_l, recv_l, group=inter_group)
            rmask, h_rm = _a2a_async(mask, send_l, recv_l, group=inter_group)
            rgate, h_rg = _a2a_async(gate_rows.detach(), send_l, recv_l,
                                     group=inter_group)
            if overlap_shared:
                share_experts_output = _shared_now()
            h_rx.wait()
            h_rm.wait()
            h_rg.wait()
        _dp.note("seam.rx", rx)
        _dp.note("seam.rgate", rgate)
        _dp.note_int("seam.rmask", rmask)
        # The data is handed to the fabric; the graph's backward (gather/index_put) needs
        # only the indices, not the data — free the storage the vendor's way
        # (save_tensors keeps only the graph and the .grad slots).
        payload.untyped_storage().resize_(0)
        gate_rows.untyped_storage().resize_(0)

        # ---- Segment 2 (= vendor permute2 position): rooted on the rx_d/rgate_d detached leaves ----
        rx_d = rx.detach().requires_grad_(True)
        rgate_d = rgate.detach().requires_grad_(True)
        R = rx.shape[0]
        # K1 (AscendC kernel, 2026-08-20 second seam: the overlap 6-arg seam). The same
        # kernel and the same math as the legacy half (argument at the same spot in
        # ta2a_permute and in the header of
        # terrace/ops/ascendc/op_kernel/terrace_k1_arrival.cpp); the only difference is
        # it sits **inside** the permute2 segment graph: the inputs are the detached
        # leaves rx_d/rgate_d, so it goes through the segment-graph variant
        # k1_arrival_segment — the kernel runs once to produce the data, and
        # exp_rx / gate_pairs each hang one independent edge back to their own leaf
        # (fusing them into one node would make the vendor gmm's two .backward() calls
        # collide; see terrace/ops/__init__.py::_K1SendEdge). r_idx/slot_idx/i_send are
        # integer index/count planes and take no gradient. The seat contract (7+3),
        # detach boundaries, splits handover, and everything the vendor's hand-written
        # backward can see are unchanged.
        # The else branch is the pre-K1 chain **verbatim**, and the only path when the
        # kernel is absent (TERRACE_CUSTOM_OPS=0 / no .so): zero behavior change.
        # exp_rx/gate_pairs are hoisted above the collectives only so both branches enter
        # Hop B with the same five tensors — pure dataflow reorder, operators and
        # operands unchanged.
        _h_i = _i_recv_early = None      # A6's handles; the K1 branch does not send early, stays None
        if quota is not None and _tops.custom_ops_enabled():
            exp_rx, gate_pairs, r_idx, slot_idx, i_send = _tops.k1_arrival_segment(
                rx_d, rmask, rgate_d, quota, epr, rpn, my_local)
        else:
            if quota is not None:
                r_idx, slot_idx = _expand_arrival_quota(rmask)  # the slot table IS the pair table
            else:
                r_idx, slot_idx = _expand_arrival(rmask, slots, quota)
            owner = slot_idx // epr
            # A6: same as the legacy half — the histogram is permutation-blind; fire
            # Hop B's counts async right away and cover it with the sort and the two
            # gathers below. Pure scheduling reorder, bit-for-bit unchanged.
            i_send = fixed_hist(owner, rpn)          # fixed-length histogram, avoids bincount's hidden host sync
            if sync_probe_enabled():
                _ = i_send.tolist()   # discriminating probe: purely discarded, only pays one host sync (see sync_probe_enabled)
            if early_hopb_counts_enabled():
                _i_recv_early = torch.empty_like(i_send)
                _h_i = dist.all_to_all_single(_i_recv_early, i_send, group=intra_group,
                                              async_op=True)
            ordo = _stable_argsort_small(owner, rpn)
            r_idx, slot_idx = r_idx[ordo], slot_idx[ordo]
            exp_rx = rx_d[r_idx]
            # C1 fast path: the dense gate table is isomorphic to the pair enumeration,
            # so the 2-D gather degenerates to a flat gather (bit-for-bit equality argued
            # at the same spot in ta2a_permute). The gradient falls back through the view
            # onto the [R, quota] rgate_d.grad; the vendor replays by row-count splits,
            # blind to the narrower layout.
            gate_pairs = (rgate_d.reshape(-1)[ordo] if quota is not None
                          else rgate_d[r_idx, slot_idx])
        _dp.note("seam.exprx", exp_rx)
        _dp.note("seam.gpairs", gate_pairs)
        _dp.note_int("seam.slot", slot_idx)
        _dp.note_int("seam.isend", i_send)
        if _h_i is not None:
            _h_i.wait()                  # A6: already sent async above; just wait for it to land
            i_recv = _i_recv_early
        else:
            i_recv = torch.empty_like(i_send)
            dist.all_to_all_single(i_recv, i_send, group=intra_group)
        is_l, ir_l = _splits_to_lists(i_send, i_recv)

        # A2: the slot plane (int64, no gradient) and the gate plane merge into one
        # collective, sitting where the slot exchange used to be; the payload plane
        # exp_rx deliberately stays **unpacked** — merging costs two extra 100-MB-scale
        # HBM copies (+0.317 ms) to save 1 α₈ (0.058 ms), a 0.20 ms net loss on the
        # verdict testbed (arithmetic and the zero-copy follow-up in the Hop B section of
        # terrace/ta2a_pack.py). The gate path still hangs on **its own** independent
        # edge: the vendor gmm's hand-written backward enters this segment via two
        # separate .backward() calls on permute2_graph / permute2_prob_graph; any welding
        # of the two paths into one node hits "backward a second time", and the first
        # call would write zero gradients into the other path (internal engineering
        # records 2026-08-20 / _K1SendEdge). The else branch is the pre-packing chain
        # **verbatim** (including collective ordering): gate off, zero behavior change.
        if packing:
            with torch.no_grad():
                _bbuf = _pk.hopb_pack_meta(slot_idx, gate_pairs.detach())
                _rbb = _a2a_raw(_bbuf, is_l, ir_l, group=intra_group)
                my_slot, _mg_raw = _pk.hopb_unpack_meta(_rbb, gate_pairs.dtype)
            _bbuf.untyped_storage().resize_(0)
            _rbb.untyped_storage().resize_(0)
        else:
            # Metadata (int, no gradient) goes over the raw a2a first.
            my_slot = _a2a_raw(slot_idx, is_l, ir_l, group=intra_group)
        exp_j = my_slot - my_local * epr
        order = _stable_argsort_small(exp_j, epr)
        tokens_per_expert = fixed_hist(exp_j, epr)                 # permutation-blind, skips the gather

        # Hop B's two float paths use the differentiable _A2A / _PackedEdge and stay
        # inside the segment; when the vendor calls .backward() on permute2_(prob_)graph,
        # autograd reverses them automatically.
        node_rx = _a2a(exp_rx, is_l, ir_l, group=intra_group)
        # Same as the legacy half: the my_gate exchange is the delivery of
        # permuted_probs; Step 2 does not apply. Even harder here: the vendor gmm's
        # hand-written backward calls backward on permute2_prob_graph at its fixed seat
        # and replays rgate_d.grad along Hop A; the gating gradient must come back from
        # the expert rank through this chain.
        my_gate = (_pk.attach_edge(gate_pairs, _mg_raw, is_l, ir_l, intra_group)
                   if packing
                   else _a2a(gate_pairs, is_l, ir_l, group=intra_group))
        permuted = node_rx[order]
        pprobs = my_gate[order]
        _dp.note("seam.permuted", permuted)
        _dp.note("seam.pprobs", pprobs)
        _dp.note_int("seam.tpe", tokens_per_expert)
        # The landing copies' data is fully consumed within the segment (gather's
        # backward needs only r_idx); free them.
        rx.untyped_storage().resize_(0)
        rgate.untyped_storage().resize_(0)

        # The last 5 seats of the fixed-position contract. Seat 3 under zm=disable is
        # carried but never used by the vendor backward; we have no such semantic (it is
        # the vendor permute2's [ep, num_local] block table), so pass None: if some
        # ungated vendor path ever does use it, None raises AttributeError on the spot —
        # better than a plausibly-shaped fake table quietly computing wrong.
        save_tensors.append(None)         # ↔ num_global_tokens_per_local_expert_cpu
        save_tensors.append(rx_d)         # ↔ permute2_input_detach
        save_tensors.append(permuted)     # ↔ permute2_graph
        save_tensors.append(rgate_d)      # ↔ permute2_prob_detach
        save_tensors.append(pprobs)       # ↔ permute2_prob_graph

        # The 18c control arm: runs only when the gate is off. The position is
        # deliberately after **all** dispatch collectives — from here the critical path
        # is expert GEMM only (compute), with no communication left for it to hide
        # behind, which is what a true "serial" baseline means; moved merely after
        # Hop A's wait it could still be partly covered by Hop B, and the reading would
        # understate the overlap's value. The seat contract is unaffected (shared experts
        # take no seat); numerics bit-for-bit unchanged (no data dependence on dispatch).
        if not overlap_shared:
            share_experts_output = _shared_now()

    st = TA2AState()
    st.u_src, st.r_idx, st.order = u_src, r_idx, order
    st.R, st.T, st.hidden = R, h.shape[0], H
    st.send_l, st.recv_l, st.is_l, st.ir_l = send_l, recv_l, is_l, ir_l
    st.intra, st.inter, st.dtype = intra_group, inter_group, hidden_states.dtype
    return permuted, tokens_per_expert, pprobs, share_experts_output, st


def ta2a_unpermute_overlap(expert_out, st: TA2AState, save_tensors, out_shape=None):
    """The combine half of the overlap 6-arg seam. Returns a **single tensor** (the vendor contract for this seam).

    Same math as `ta2a_unpermute`; segment boundaries: Hop B's reverse (differentiable)
    hides inside the unpermute1 segment, Hop A's reverse sits outside the segments
    (detach at back), replayed by hand by the vendor moe-layer backward with
    (output_splits, input_splits). Appends the 3 seats of the fixed-position contract.
    """
    with torch.enable_grad():
        # ---- Segment (= vendor unpermute1 position): rooted on eo_d ----
        eo_d = expert_out.detach().requires_grad_(True)
        back_pairs = eo_d.new_empty(eo_d.shape)
        back_pairs[st.order] = eo_d
        ret = _a2a(back_pairs, st.ir_l, st.is_l, group=st.intra)   # Hop B reverse, differentiable
        red = ret.new_zeros(st.R, ret.shape[1])
        red.index_add_(0, st.r_idx, ret)
        _dp.note("seam.eout", expert_out)
        _dp.note("seam.ret", ret)
        _dp.note("seam.red", red)
        _dp.check_reduction("seam.red", red, ret, st.r_idx, st.R, _per(ret, st.R))
        save_tensors.append(eo_d)         # ↔ unpermute1_input_detach
        save_tensors.append(red)          # ↔ unpermute1_graph
        eo_d.untyped_storage().resize_(0)  # vendor's own trick: expert-output data no longer needed, the graph stays intact

        # ---- Hop A reverse (outside the segments, synchronous): the reverse pair = (recv, send) ----
        back = _a2a_raw(red.detach(), st.recv_l, st.send_l, group=st.inter)
        red.untyped_storage().resize_(0)

        # ---- Segment (= vendor unpermute2 position): rooted on back_d ----
        back_d = back.requires_grad_(True)   # what _a2a_raw returns is already a leaf
        y = back_d.new_zeros(st.T, st.hidden).index_add(0, st.u_src, back_d)
        _dp.note("seam.back", back_d)
        _dp.note("seam.y", y)
        _dp.check_reduction("seam.y", y, back_d, st.u_src, st.T, _per(back_d, st.T))
        if out_shape is not None:
            y = y.view(out_shape)
        save_tensors.append(back_d)       # ↔ unpermute2_input_detach
        back_d.untyped_storage().resize_(0)
    return y


def ta2a_enabled(ep_world: int, rpn: int = 8) -> bool:
    """Whether the EP group crosses nodes. If it doesn't (EP<=rpn) there is no fabric hop and T-A2A is a no-op by construction."""
    if os.environ.get("TERRACE_TA2A") != "1":
        return False
    return ep_world // rpn >= 2
