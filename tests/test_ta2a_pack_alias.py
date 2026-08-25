# -*- coding: utf-8 -*-
"""Unpacked planes must not share storage with the pack buffer (regression test for
the 2026-08-21 verdict-testbed rank61 crash).

What happened:
  Both shots of the verdict testbed's pack-on arm died around iter 30 with
      RuntimeError: The tensor has a non-zero number of elements,
                    but its data is not allocated yet.
      at terrace/ta2a_dispatch.py `owner = slot_idx // epr`
  The other 15 machines sat in a collective waiting for the already-dead rank,
  which presented as "the whole frame hung for 40 minutes".

Root cause:
  `hopa_unpack` used `.contiguous()` to make the column slices contiguous. When
  PyTorch checks contiguity, **the stride of any dimension of size 1 is ignored** --
  so at R==1, `buf[:, a:b]` is judged already contiguous and `.contiguous()` hands
  the view back unchanged. The caller then runs
  `_rbuf.untyped_storage().resize_(0)` to return the memory, all three planes
  dangle on the spot, and nothing blows up until the first real read a dozen
  operators later.

  Why the alignment bed (4 nodes) never hit it and the verdict testbed (16 nodes)
  did: rows get split 16 ways instead of 4, and with load_cv climbing from 1.0 to
  1.4 over training, the probability of some rank receiving exactly 1 row goes from
  negligible to bound-to-happen.
  Why iter 30 and not iter 1: routing has to specialize first before the
  distribution gets skewed enough.

The fix: `ta2a_pack._own()` uses `clone(memory_format=contiguous_format)`, which
      **always** allocates new storage; the call site adds an `assert_not_aliased`
      tripwire before returning the memory.

This test is a pure-CPU shape-level reproduction; it needs no NPU and no
distributed setup.
"""
import pytest
import torch

from terrace import ta2a_pack as pk


# Verdict-testbed geometry: H=2048, quota=3 (k=6 / M=2), slots=24 (384 experts /
# 128 EP / 8 rpn)
JUDGMENT = dict(hidden=2048, quota=3, slots=24)

# R=1 is the dangerous shape (dim0 size 1 -> stride ignored -> slice judged
# contiguous); R=0 and R>=2 are pinned down too, so nobody ever "fixes only the
# 1 case".
ROW_COUNTS = [0, 1, 2, 3, 17]


def _storage_ptr(t):
    return t.untyped_storage().data_ptr()


def _aliases(plane, buf):
    """Whether plane shares storage with buf. Empty tensors do not count (they have
    no data that could dangle)."""
    return plane.numel() > 0 and _storage_ptr(plane) == _storage_ptr(buf)


@pytest.mark.parametrize("rows", ROW_COUNTS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_hopa_unpack_never_aliases_buffer(rows, dtype):
    H, q = JUDGMENT["hidden"], JUDGMENT["quota"]
    payload = torch.zeros(rows, H, dtype=dtype)
    gate = torch.zeros(rows, q, dtype=dtype)
    ids = torch.zeros(rows, q, dtype=torch.int64)

    buf, lay = pk.hopa_pack(payload, gate, ids)
    rx, rgate, rmask = pk.hopa_unpack(buf, lay)

    for name, plane in (("payload", rx), ("gate", rgate), ("id", rmask)):
        assert not _aliases(plane, buf), (
            "rows=%d dtype=%s: %s plane still shares storage with the pack "
            "buffer -- it will dangle after the caller's resize(0)"
            % (rows, dtype, name)
        )


@pytest.mark.parametrize("rows", ROW_COUNTS)
def test_hopa_unpack_never_aliases_bitmask_format(rows):
    """The bitmask wire format (1-D id plane) takes a different slice path; aliasing
    is just as forbidden there."""
    H, slots = JUDGMENT["hidden"], JUDGMENT["slots"]
    payload = torch.zeros(rows, H, dtype=torch.bfloat16)
    gate = torch.zeros(rows, slots, dtype=torch.bfloat16)
    ids = torch.zeros(rows, dtype=torch.int64)

    buf, lay = pk.hopa_pack(payload, gate, ids)
    assert lay.id_1d, "this group should take the 1-D id plane"
    for plane in pk.hopa_unpack(buf, lay):
        assert not _aliases(plane, buf), (
            "rows=%d: unpacked plane aliases the buffer under the bitmask format"
            % rows)


@pytest.mark.parametrize("pairs", ROW_COUNTS)
def test_hopb_unpack_never_aliases_buffer(pairs):
    slot = torch.zeros(pairs, dtype=torch.int64)
    gate = torch.zeros(pairs, dtype=torch.bfloat16)
    buf = pk.hopb_pack_meta(slot, gate)
    for plane in pk.hopb_unpack_meta(buf, torch.bfloat16):
        assert not _aliases(plane, buf), (
            "pairs=%d: Hop B unpacked plane aliases the buffer" % pairs)


@pytest.mark.parametrize("rows", ROW_COUNTS)
def test_planes_survive_freeing_the_buffer(rows):
    """End-to-end reproduction of the incident shape: unpack -> shrink the buffer's
    storage to 0 -> all three planes still readable with correct values.

    This is the one actually written against the incident: the tests above only
    check pointers; this one checks "does it still work after the memory is
    returned".
    """
    H, q = JUDGMENT["hidden"], JUDGMENT["quota"]
    payload = torch.arange(rows * H, dtype=torch.float32).reshape(rows, H).to(torch.bfloat16)
    gate = torch.full((rows, q), 0.5, dtype=torch.bfloat16)
    ids = torch.arange(rows * q, dtype=torch.int64).reshape(rows, q)

    buf, lay = pk.hopa_pack(payload, gate, ids)
    rx, rgate, rmask = pk.hopa_unpack(buf, lay)
    pk.assert_not_aliased(buf, rx, rgate, rmask)

    buf.untyped_storage().resize_(0)          # the exact line from the incident

    assert torch.equal(rx, payload), "payload plane reads back wrong after the memory is returned"
    assert torch.equal(rgate, gate), "gate plane reads back wrong after the memory is returned"
    assert torch.equal(rmask, ids), "id plane reads back wrong after the memory is returned"
    # first real downstream use: this is the exact statement that blew up in the incident
    _ = rmask // 3


def test_assert_not_aliased_actually_catches_an_alias():
    """Tripwire self-check: hand-build an aliased plane; assert_not_aliased must
    catch it.

    Without this, a mis-written tripwire (say, the numel check inverted) becomes a
    mute that never fires -- exactly how idle_watch died on 08-22.
    """
    buf = torch.zeros(4, 8, dtype=torch.int64)
    aliased = buf[:, :2]                       # an explicit view, no copy
    assert _aliases(aliased, buf), \
        "the hand-built alias plane does not actually alias; the test is void"
    with pytest.raises(RuntimeError, match="shared storage"):
        pk.assert_not_aliased(buf, aliased)
    # a real copy must pass
    pk.assert_not_aliased(buf, aliased.clone())


def test_contiguous_would_have_been_a_noop_at_one_row():
    """Pin "why .contiguous() is not enough" down as an executable fact.

    If PyTorch ever changes its contiguity check, or someone reverts _own to
    .contiguous(), this test blows up first, prompting a re-check of whether the
    argument in the comments still holds.
    """
    buf = torch.zeros(1, 516, dtype=torch.int64)
    sliced = buf[:, :3]
    assert sliced.is_contiguous(), (
        "a column slice at R==1 is no longer judged contiguous -- the necessity "
        "argument for _own needs re-verification"
    )
    assert sliced.contiguous().untyped_storage().data_ptr() == buf.untyped_storage().data_ptr(), (
        ".contiguous() no longer hands the view back unchanged -- same as above"
    )
    assert sliced.clone().untyped_storage().data_ptr() != buf.untyped_storage().data_ptr()

    two = torch.zeros(2, 516, dtype=torch.int64)[:, :3]
    assert not two.is_contiguous(), "R>=2 should be non-contiguous (the control group)"
