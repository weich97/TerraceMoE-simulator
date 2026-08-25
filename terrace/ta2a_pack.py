"""T-A2A collective packing (A1 / A2, 2026-08-21): merge several collectives that share
splits within the same hop into one. **Pure byte re-layout** -- no reduction order,
pairing order, sort key, dtype or rounding point moves.

Why this is the best-value cut right now (internal design record, not shipped with the repo):
splitting the w128 a2a curve into `α + β` gives α₁₂₈ = 0.45 ms, α₈ = 0.058 ms,
β₈ ≈ β₁₂₈ ≈ 110 GB/s -- **bytes are nearly free, collective count is expensive**. One
dispatch used to take 8 collectives (inter 4 + intra 4), fixed overhead
4α₁₂₈ + 4α₈ ≈ 2.03 ms, while the vendor alltoall_seq side takes only about 3 ≈ 1.35 ms.
Independent corroboration: the combine side has only 2 collectives and its measured phase
ties the vendor; dispatch has 8 and measures +6 ms behind.

## Collectives per dispatch after packing (with this module's gate open)

| Hop | before | after | α saved | extra paid (HBM + wire) | net |
|---|---|---|---|---|---|
| Hop A (inter, α₁₂₈) | counts + payload + id + gate = 4 | counts + **[id‖payload‖gate]** = 2 | 0.900 ms | 0.106 ms | **+0.794 ms** |
| Hop B (intra, α₈)   | counts + exp_rx + slot + gate = 4 | counts + exp_rx + **[slot‖gate]** = 3 | 0.058 ms | 0.002 ms | **+0.056 ms** |
| combine            | 2 | 2 | — | — | — |

Total dispatch **8 -> 5** collectives; on the verdict-testbed geometry (H=2048 / M=2 /
quota=3 / n_rows=8192 / pairs=24576 / bf16 / HBM effective 1275 GB/s, measured internal
hardware profile) **net −0.85 ms per dispatch**; the split-half seam's forward goes
10 -> 7, **backward 6 -> 6 unchanged** (see the "Differentiability" section).

**Why Hop B packs only the two scalar planes, never the payload plane**: packing saves α,
but every **large** plane packed in pays two extra HBM copies of itself. Hop A's α is
7.8x pricier and its payload 3.2x smaller ⇒ worth it; Hop B is the other way around
⇒ merging three into one pays +0.317 ms to save 0.116 ms, **net loss 0.202 ms**. The
arithmetic is in the Hop B section below. The only way to win back that 1α₈ is zero-copy
(have K1 write straight into the merged buffer), scheduled after A4
(`npu_alltoallv_gmm` swallows Hop B) -- once A4 lands, this item disappears on its own.

**The live-path evidence moved its observation point (A1′, 2026-08-21)**:
`tests/test_ta2a_quota_wire_bitparity.py` used to decide whether C1 had silently fallen
back by checking "is there a `[n_rows, quota]` int64 slot table / 1-D bitmask / wide-float
gate table on the Hop A wire". After packing, the three planes share one collective, so
the criterion now reads **the packer `hopa_pack`'s arguments and `HopALayout.id_w`** --
same three planes, same widths, just read one step earlier (packing only moves bytes; it
does not change the planes). The full argument that the discriminating power is
equivalent is in that file's header.

## Row layout and alignment

- **Hop A (int64 container)**: row = `[id(id_w words) | payload(H) | gate(gw) | pad]`,
  row width `W = id_w + ceil((H + gw) / (8 // itemsize))` words. The id plane sits at the
  **row head**; the float region's starting byte is always a multiple of 8 (the other way
  round it would drift with H/dtype and the int64 view could not be taken).
  Verdict-testbed quota arm: `id_w=3`, `W=516` words `=4128 B/row`, **2 B (+0.05%)** of
  overhead against the three original streams' 4126 B. **splits are row counts as-is, no
  scaling** -- dim 0 IS n_rows.
- **Hop B (int64 container)**: row = `[slot(1 word) | gate(1 float) | pad]` = 2 words.
  Again the int64 plane leads the row. 16 B/row against the two original streams' 10 B
  (bf16): 6 B of overhead -- +0.15% relative to the same hop's 4096 B payload row.
  splits are again row counts as-is.

## Differentiability: one forward, two **independent** backward edges

The two float paths each still hang off their own independent `autograd.Function` edge
(`_PackedEdge`); the backward is the very same line as the pre-packing
`ep_dist._A2A.backward`: `_a2a_raw(g, out_splits, in_splits, group)`
-- byte-for-byte the same call, gradients bit-identical, **backward collective count
unchanged**.

We do not weld the two paths into one fused node, for the same reason as K1 (internal
engineering record 2026-08-20 "a fused kernel feeding a hand-written backward must keep
the segment-graph edges separate" / `terrace/ops/__init__.py::_K1SendEdge`): the vendor
gmm's hand-written backward runs `.backward()` **twice** into the same segment, once for
`permute2_graph` and once for `permute2_prob_graph`; a fused node hits *backward through
the graph a second time* on the second call, and the first call would first write the
materialised zero gradients into the other path's `.grad`. So the **data** is packed and
received once outside the graph, and the **graph** is rebuilt from two disjoint edges,
isomorphic to the two pre-packing `_A2A` subgraphs.

## Gate

`TERRACE_TA2A_PACK`: unset / anything but "0" = on (default); "0" = off, both seams take
the pre-packing chain **verbatim** (zero behaviour change), for on-machine A/B and
one-command rollback. Read once at process start (the hot path never checks the
environment); tests flip the switch with `reset()` or by monkeypatching `pack_enabled`
directly.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

from .ep_dist import _a2a_raw

_ENV_SWITCH = "TERRACE_TA2A_PACK"
_MODE: str | None = None

# Three modes. **The default changed from full to small on 2026-08-22**, based on
# verdict-testbed measurements:
#
#   mode    Hop A collectives   planes packed into the Hop A container   measured dispatch/call (verdict testbed)
#   off     4                   —                                        12.171 ms (baseline)
#   full    2                   id + payload + gate                      14.222 ms (**+2.051, a loss**)
#   small   3                   id + gate (payload travels alone)        expected −0.43 (measurement pending)
#
# Why full loses: packing saves α, but every **large** plane packed in pays two extra HBM
# copies of itself. full saves 2α₁₂₈ + 1α₈ = 0.96 ms, yet has to copy the [n, 2048] bf16
# payload into the container and back out -- the real copy cost, back-solved on the
# verdict testbed, is ≈ 3.0 ms (originally estimated 0.106 ms, off by 28x).
#
# This module's Hop B section **had this right all along** ("merging three into one pays
# +0.317 ms to save 0.116 ms, net loss 0.202 ms"); we just underestimated Hop A's payload
# copy by 28x and so reached the opposite conclusion on the same question. small is Hop
# B's correct trade-off transplanted verbatim onto Hop A:
# **pack only the scalar planes of a few dozen bytes per row; a large payload always
# travels on its own collective.**
_MODE_OFF, _MODE_SMALL, _MODE_FULL = "off", "small", "full"
_ALIASES = {"0": _MODE_OFF, "off": _MODE_OFF, "no": _MODE_OFF,
            "1": _MODE_SMALL, "small": _MODE_SMALL, "meta": _MODE_SMALL,
            "full": _MODE_FULL, "2": _MODE_FULL, "all": _MODE_FULL}


def _env_mode() -> str:
    """The mode written in the environment. Read once per process lifetime (dispatch is a hot path)."""
    global _MODE
    if _MODE is None:
        raw = os.environ.get(_ENV_SWITCH)
        if raw is None:
            _MODE = _MODE_SMALL          # unset = the default mode, deliberately
        else:
            key = raw.strip().lower()
            if key not in _ALIASES:
                # **Unknown values must blow up; never fall back to the default silently.**
                # Audit measurement 2026-08-23: `smal` / `fulll` / `false` / `true` / ""
                # all silently became small. The most dangerous is `false` -- whoever
                # wrote it meant to turn packing off, and it actually turned packing
                # **on**; and the tier guard only checks "an assignment statement
                # exists", never the value, so a one-character typo in the tier table
                # can run one tier as another with the whole test suite still green.
                raise RuntimeError(
                    "%s=%r is not a valid mode. Valid values: %s. "
                    "**No silent fallback to the default** -- one typo in the tier "
                    "table would run a whole tier as another mode, and the readings "
                    "would look completely normal." % (_ENV_SWITCH, raw, sorted(_ALIASES)))
            _MODE = _ALIASES[key]
    return _MODE


def pack_enabled() -> bool:
    """Packing switch. **The historical hook tests use to flip the A/B control arm is monkeypatching this function.**"""
    return _env_mode() != _MODE_OFF


def pack_mode() -> str:
    """"off" / "small" / "full". The hot path reads this.

    `TERRACE_TA2A_PACK=1` now maps to **small** (not the historical full) -- the testbed
    scripts and existing runners all write 1, so they pick up the corrected mode
    automatically; running the losing tier requires writing full explicitly.

    **Must ask `pack_enabled()` first**: a large batch of existing tests monkeypatch
    that function to build the un-packed control arm. If this read the environment
    directly and bypassed it, those monkeypatches would **silently stop working** --
    the control arm would be packing too, the A/B bit-level reconciliation would
    compare two packed arms, every equivalence assertion would turn into a no-op, and
    the tests would stay green.
    (Actually stepped on this while wiring the small mode on 2026-08-22; the hopb
    counter {'hopb': 2} is what exposed it.)
    """
    if not pack_enabled():
        return _MODE_OFF
    return _env_mode()


def reset() -> None:
    """Forget the cached decision; the next pack_mode() re-reads the environment. Test/debug hook; training code must not call it."""
    global _MODE
    _MODE = None


def _ceil_mul(n: int, unit: int) -> int:
    return -(-n // unit) * unit


def _own(t: torch.Tensor) -> torch.Tensor:
    """Make contiguous and **guarantee independent storage**. Every unpacked plane must go through here.

    Why `.contiguous()` will not do (cause of death on verdict-testbed rank61, 2026-08-21):
    `.contiguous()` hands the **original view** straight back when the tensor is "already
    contiguous". For a column slice `buf[:, a:b]` of an `[R, W]` buffer, when **R <= 1**
    dim0's stride is ignored by the contiguity check, so the slice is judged contiguous
    -- `.contiguous()` becomes a no-op and the unpack result shares storage with the pack
    buffer. The caller then immediately returns the memory with
    `_rbuf.untyped_storage().resize_(0)` and all three planes dangle on the spot:

        RuntimeError: The tensor has a non-zero number of elements,
                      but its data is not allocated yet.
        (raised in ta2a_dispatch.py at `owner = slot_idx // epr`, slot_idx expanded from rmask)

    R==1 is almost never hit on the alignment testbed (4 nodes); on the verdict testbed
    (16 nodes, rows sliced finer, load_cv climbing from 1.0 to 1.4 over training) it is
    -- both pack-on runs on 08-21 died near iter 30.

    Cost: for R>1, `.contiguous()` had to copy once anyway and `clone` is also one copy,
    **equivalent**; only on that dangerous degenerate shape is there one extra copy
    (1 row, negligible).
    """
    return t.clone(memory_format=torch.contiguous_format)


# --------------------------------------------------------------------------------------
# Differentiability: independent edges that rebuild the graph after packing
# --------------------------------------------------------------------------------------

class _PackedEdge(torch.autograd.Function):
    """Hang **one** of the packed-and-received paths back onto its own send-side tensor.

    Zero work in the forward (the data already arrived outside the graph via one packed
    a2a; `recvd` is its unpacked result); the backward = that path's own reverse a2a,
    byte-for-byte the same line as the pre-packing `ep_dist._A2A.backward`:
    `_a2a_raw(g, out_splits, in_splits, group)`. So **gradients are bit-identical and
    the backward collective count is unchanged**; what is saved is forward collectives.

    One edge per path, disjoint -- the vendor gmm's hand-written backward runs
    `.backward()` twice over the permute2 segment, and a fused node would collide
    (the "Differentiability" section in the module header / _K1SendEdge).
    """

    @staticmethod
    def forward(ctx, src, recvd, in_splits, out_splits, group):
        ctx.a2a_splits = (in_splits, out_splits, group)
        return recvd            # data already produced; autograd aliases it and attaches grad_fn

    @staticmethod
    def backward(ctx, g):
        in_splits, out_splits, group = ctx.a2a_splits
        if g is None or not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        # The backward pair is (out, in), same order as _A2A.backward. The swapped
        # in/out is hard-coded here rather than passed by the caller, so no future call
        # site can hand the direction in reversed -- that would silently misroute while
        # the forward still looks plausible.
        return _a2a_raw(g, out_splits, in_splits, group), None, None, None, None


def attach_edge(src, recvd, in_splits, out_splits, group):
    """Hang `recvd` back onto `src`. With grad disabled, hand it out at zero cost (same branch structure as `_a2a`)."""
    if torch.is_grad_enabled():
        return _PackedEdge.apply(src, recvd, in_splits, out_splits, group)
    return recvd


# --------------------------------------------------------------------------------------
# Hop A: id ‖ payload ‖ gate (int64 container; the id plane leads the row for 8-byte alignment)
# --------------------------------------------------------------------------------------
#
# A1′ (2026-08-21, coordinator sign-off): the id plane is packed in too, so Hop A drops
# from 4 collectives to **2** (counts + one packed collective). The id plane used to
# travel alone because `tests/test_ta2a_quota_wire_bitparity.py::_wire_flags` decided
# whether C1 had silently fallen back by checking "is this tensor shape on the wire";
# that criterion now reads **the packer's arguments** (see that file's header), equally
# discriminating and no longer dependent on "one collective per plane".
#
# Layout rationale (same as Hop B): the int64 plane sits at the **row head**, row width
# is counted in int64 words, and the float region starts at word `id_w` -- its starting
# byte is always a multiple of 8. The other way round (float first) the in-row offset
# would drift with H and dtype, the int64 view could not be taken, and every row's int64
# plane would land on a 4-byte boundary.
#
# splits **are row counts as-is**, no scaling (dim 0 IS n_rows) -- cleaner even than the
# pre-A1′ version, which scaled by `F/gw` so the wire tensor width equalled the gate
# width; now the scaling is gone entirely.
# `st.send_l / st.recv_l` (the two lists handed to the vendor's
# `disp.input_splits/output_splits`) are therefore element-for-element the same as
# before packing; the vendor backward's two manual replays run as before, still one a2a
# each.


class HopALayout:
    """Row layout of the Hop A pack buffer. `pack` produces it; `unpack` reads by it.

    `id_w` is the number the live-path evidence now watches: C1 quota wire format =
    quota, the pre-change bitmask format = 1.
    """

    __slots__ = ("hidden", "gate_w", "id_w", "id_1d", "words", "dtype", "per_word")

    def __init__(self, hidden, gate_w, id_w, id_1d, dtype):
        self.hidden, self.gate_w = hidden, gate_w
        self.id_w, self.id_1d = id_w, id_1d
        self.dtype = dtype
        self.per_word = 8 // dtype.itemsize
        self.words = id_w + -(-(hidden + gate_w) // self.per_word)

    def __repr__(self):                                   # must be readable in error messages
        return (f"HopALayout(H={self.hidden}, gw={self.gate_w}, id_w={self.id_w}, "
                f"id_1d={self.id_1d}, words={self.words}, dtype={self.dtype})")


def hopa_layout(payload: torch.Tensor, gate_rows: torch.Tensor,
                ids: torch.Tensor) -> HopALayout:
    """Three planes -> row layout. `ids` is an `[n]` bitmask (general) or `[n, quota]` slot table (C1)."""
    if payload.dtype != gate_rows.dtype:
        raise RuntimeError(
            f"Hop A packing requires payload/gate to share a dtype, got {payload.dtype} vs "
            f"{gate_rows.dtype} -- the gate plane is derived from payload by contract (see _pack_quota_wire)")
    if ids.dtype != torch.int64:
        raise RuntimeError(f"Hop A packing's id plane must be int64, got {ids.dtype}")
    if ids.dim() not in (1, 2):
        raise RuntimeError(f"Hop A's id plane must be [n] or [n, quota], got {ids.shape}")
    id_1d = ids.dim() == 1
    id_w = 1 if id_1d else int(ids.shape[1])
    n = payload.shape[0]
    if ids.shape[0] != n or gate_rows.shape[0] != n:
        raise RuntimeError(
            f"Hop A's three planes must have equal row counts (they share splits), got "
            f"payload={n} gate={gate_rows.shape[0]} id={ids.shape[0]}")
    return HopALayout(int(payload.shape[1]), int(gate_rows.shape[1]), id_w, id_1d,
                      payload.dtype)


def hopa_pack(payload: torch.Tensor, gate_rows: torch.Tensor, ids: torch.Tensor):
    """(`[n, H]` float, `[n, gw]` float, `[n]`/`[n, quota]` int64) -> (`[n, W]` int64, layout).

    Copies only, no conversion of any kind: every bit of the three segments lands in
    the buffer verbatim.
    """
    lay = hopa_layout(payload, gate_rows, ids)
    n, pw, iw = payload.shape[0], lay.per_word, lay.id_w
    buf = torch.empty(n, lay.words, dtype=torch.int64, device=payload.device)
    buf[:, :iw] = ids.unsqueeze(1) if lay.id_1d else ids
    fv = buf.view(lay.dtype)                      # bitwise reinterpretation of a contiguous buffer; metadata-only op
    base = iw * pw
    fv[:, base:base + lay.hidden] = payload
    fv[:, base + lay.hidden:base + lay.hidden + lay.gate_w] = gate_rows
    tail = base + lay.hidden + lay.gate_w
    if tail < lay.words * pw:
        fv[:, tail:] = 0                          # receiver never reads it; zeroed only so uninitialised bits never hit the wire
    return buf, lay


def hopa_unpack(buf: torch.Tensor, lay: HopALayout):
    """`[n, W]` int64 -> (`[n, H]` payload, `[n, gw]` gate, id plane in its original shape), all three contiguous.

    Deliberately returns no strided views: the 2026-08-01 `torch.cat` gate packing lost
    tens of ms precisely because **the downstream gather ate non-contiguous slices**
    (the "DO NOT FUSE" comment in ta2a_fwd.py), not because packing itself was slow.
    Here we pay one extra copy of the arrival payload (≈34 MB on the verdict testbed,
    about 0.05 ms) so downstream operators (K1 kernel included) see **exactly the same
    contiguous tensors** as before packing.
    """
    pw, iw = lay.per_word, lay.id_w
    fv = buf.view(lay.dtype)
    base = iw * pw
    # `_own`, not `.contiguous()`: with R<=1 the column slice is judged already
    # contiguous, contiguous() is a no-op and hands back a view of buf; the caller then
    # resizes buf's storage to 0 and all three planes dangle.
    ids = _own(buf[:, 0]) if lay.id_1d else _own(buf[:, :iw])
    return (_own(fv[:, base:base + lay.hidden]),
            _own(fv[:, base + lay.hidden:base + lay.hidden + lay.gate_w]),
            ids)


class HopASmallLayout:
    """Row layout of the A1'' small-plane container: `[id(id_w words) | gate(gw) | pad]`; the payload is not in it.

    Same alignment rules as `HopALayout` (int64 plane leads the row, the float region's
    starting byte is always a multiple of 8), just without the hidden segment -- row
    width drops from 516 words (4128 B) to 4 words (32 B).
    """

    __slots__ = ("gate_w", "id_w", "id_1d", "words", "dtype", "per_word")

    def __init__(self, gate_w, id_w, id_1d, dtype):
        self.gate_w, self.id_w, self.id_1d = gate_w, id_w, id_1d
        self.dtype = dtype
        self.per_word = 8 // dtype.itemsize
        self.words = id_w + -(-gate_w // self.per_word)

    def __repr__(self):
        return (f"HopASmallLayout(gw={self.gate_w}, id_w={self.id_w}, "
                f"id_1d={self.id_1d}, words={self.words}, dtype={self.dtype})")


def hopa_small_layout(gate_rows: torch.Tensor, ids: torch.Tensor) -> HopASmallLayout:
    if ids.dtype != torch.int64:
        raise RuntimeError(f"Hop A small packing's id plane must be int64, got {ids.dtype}")
    if not gate_rows.is_floating_point():
        raise RuntimeError(f"Hop A small packing's gate plane must be floating point, got {gate_rows.dtype}")
    if ids.dim() not in (1, 2):
        raise RuntimeError(f"Hop A's id plane must be [n] or [n, quota], got {ids.shape}")
    if ids.shape[0] != gate_rows.shape[0]:
        raise RuntimeError(
            f"Hop A small packing's two planes must have equal row counts (they share "
            f"splits), got gate={gate_rows.shape[0]} id={ids.shape[0]}")
    id_1d = ids.dim() == 1
    return HopASmallLayout(int(gate_rows.shape[1]), 1 if id_1d else int(ids.shape[1]),
                           id_1d, gate_rows.dtype)


def hopa_pack_small(gate_rows: torch.Tensor, ids: torch.Tensor):
    """(`[n, gw]` float, `[n]`/`[n, quota]` int64) -> (`[n, Ws]` int64, layout).

    Verdict-testbed geometry: gw=quota=3, id_w=3, per_word=4 ⇒ Ws=4 words=**32 B/row**
    (against the full mode's 4128 B/row -- a 129x difference in copy volume, which is
    the entire reason A1'' exists).
    """
    lay = hopa_small_layout(gate_rows, ids)
    n, pw, iw = gate_rows.shape[0], lay.per_word, lay.id_w
    buf = torch.empty(n, lay.words, dtype=torch.int64, device=gate_rows.device)
    buf[:, :iw] = ids.unsqueeze(1) if lay.id_1d else ids
    fv = buf.view(lay.dtype)
    base = iw * pw
    fv[:, base:base + lay.gate_w] = gate_rows
    if base + lay.gate_w < lay.words * pw:
        fv[:, base + lay.gate_w:] = 0        # receiver never reads it; zeroed only so uninitialised bits never hit the wire
    return buf, lay


def hopa_unpack_small(buf: torch.Tensor, lay: HopASmallLayout):
    """`[n, Ws]` int64 -> (`[n, gw]` gate, id plane in its original shape). Both have independent storage (see `_own`)."""
    pw, iw = lay.per_word, lay.id_w
    fv = buf.view(lay.dtype)
    base = iw * pw
    ids = _own(buf[:, 0]) if lay.id_1d else _own(buf[:, :iw])
    return _own(fv[:, base:base + lay.gate_w]), ids


def hopa_exchange_raw(buf, in_splits, out_splits, group=None):
    """One raw a2a. splits are row counts as-is -- dim 0 IS n_rows, no scaling needed."""
    return _a2a_raw(buf, in_splits, out_splits, group)


def hopa_exchange_async(buf, in_splits, out_splits, group=None):
    """Async variant (the overlap seam sends Hop A outside the segment; 18c uses the flight time to compute the shared expert).

    Returns (recv_buf, handle); the caller waits, then calls `hopa_unpack` directly.
    """
    out = buf.new_empty((sum(out_splits), buf.shape[1]))
    handle = dist.all_to_all_single(out, buf, out_splits, in_splits, group=group,
                                    async_op=True)
    return out, handle


# --------------------------------------------------------------------------------------
# Hop B: slot ‖ gate (two **one-scalar-per-pair** metadata planes, int64 container)
# --------------------------------------------------------------------------------------
#
# Why Hop B does **not** pack exp_rx in as well (the accounting was done before
# sign-off; verdict-testbed geometry H=2048 / quota=3 / P=24576 / bf16 / HBM effective
# 1275 GB/s (measured internal hardware profile) / α₈=0.058 ms):
#
#   Option 1 (three into one, slot‖exp_rx‖gate): row width 514 words = 4112 B, wire
#     overhead only 6 B/row (+0.15%) -- **bytes are not the problem**; the problem is
#     pack+unpack pay 4 extra 100-MB-class HBM copies = 403.9 MB ⇒ **+0.317 ms**, while
#     the saving is 2α₈ = 0.116 ms ⇒ **net −0.202 ms**.
#   Option 2 (pack only the two scalar planes): pay 1.28 MB HBM + 129 KB on the wire
#     ⇒ +0.002 ms, save 1α₈ = 0.058 ms ⇒ **net +0.056 ms**.
#
# Take option 2. This is exactly the physics the 2026-08-01 "torch.cat gate packing"
# crashed into: **packing saves α, but every large plane packed in pays two extra HBM
# copies of itself**. Hop A's ledger runs the other way (α₁₂₈ is 7.8x pricier, payload
# 3x smaller: +0.105 ms buys 0.450 ms, net +0.344 ms), hence pack there, not here.
#
# The only correct way to win back Hop B's remaining 1α₈ is **zero-copy**: have the step
# that produces exp_rx (the K1 kernel, or the else branch's gather) write straight into
# the merged buffer's payload view, instead of materialising a [pairs, H] first and
# copying. That touches K1's output contract and the permute2 segment-graph edges
# (_K1SendEdge), is a separate piece of work, and is worth 0.058 ms -- not in this cut.


def hopb_meta_words() -> int:
    """Row width of the Hop B metadata pack buffer: 1 word of slot id + 1 word holding 1 gate float."""
    return 2


def hopb_pack_meta(slot_idx: torch.Tensor, gate_pairs: torch.Tensor) -> torch.Tensor:
    """(`[P]` int64 slot ids, `[P]` float gates) -> `[P, 2]` int64 contiguous buffer.

    The int64 plane sits at the **row head**: row width is counted in int64 words, the
    float region starts at word 1, and its starting byte is always a multiple of 8 --
    the other way round (float first) the in-row offset would drift with dtype and the
    int64 view could not be taken.
    """
    if slot_idx.dtype != torch.int64:
        raise RuntimeError(f"Hop B packing's id plane must be int64, got {slot_idx.dtype}")
    if slot_idx.shape != gate_pairs.shape:
        raise RuntimeError(
            f"Hop B packing requires both planes to share a shape (one scalar per pair), "
            f"got {tuple(slot_idx.shape)} vs {tuple(gate_pairs.shape)}")
    if not gate_pairs.is_floating_point():
        raise RuntimeError(f"Hop B packing's gate plane must be floating point, got {gate_pairs.dtype}")
    P = slot_idx.numel()
    per_word = 8 // gate_pairs.element_size()
    buf = torch.empty(P, hopb_meta_words(), dtype=torch.int64, device=slot_idx.device)
    buf[:, 0] = slot_idx
    fv = buf.view(gate_pairs.dtype)               # bitwise reinterpretation of a contiguous buffer; metadata-only op
    fv[:, per_word] = gate_pairs
    fv[:, per_word + 1:] = 0                      # receiver never reads it
    return buf


def hopb_unpack_meta(buf: torch.Tensor, dtype):
    """`[P, 2]` int64 -> (`[P]` int64 slot ids, `[P]` float gates)."""
    per_word = 8 // dtype.itemsize
    # Same as hopa_unpack: with P<=1, `.contiguous()` is a no-op and the two planes
    # would share buf's storage. Hop B has no resize(0) right behind it today, but keep
    # the same contract, so a future resize does not become the way we find out.
    return _own(buf[:, 0]), _own(buf.view(dtype)[:, per_word])


def assert_not_aliased(buf: torch.Tensor, *planes: torch.Tensor) -> None:
    """Call once before resizing `buf`'s storage to 0: no unpacked plane may still point into buf.

    Three pointer comparisons, negligible on the hot path. It stays because when this
    pit reopens it is **silent** (the dangling plane only blows up at the first real
    read downstream, by which time the stack is far from the scene -- on 08-21 it blew
    up at `owner = slot_idx // epr`, a dozen-plus operators away from the pack site).
    """
    sp = buf.untyped_storage().data_ptr()
    for i, p in enumerate(planes):
        if p is not None and p.numel() and p.untyped_storage().data_ptr() == sp:
            raise RuntimeError(
                f"Unpacked plane {i} is still in shared storage with the pack buffer "
                f"(shape={tuple(p.shape)}, buf={tuple(buf.shape)}) -- releasing the "
                f"memory would leave it dangling. See ta2a_pack._own")
