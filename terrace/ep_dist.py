"""On-device expert-parallel forward via dist.all_to_all_single (docs/02).

One process per die = one EP rank holding E/W experts. This is the real
distributed path whose output must match the CPU-verified emulate_ep_experts
(旧 EP 参考实现(未随仓发布)) and the single-process reference. Metadata (token counts) and
payloads are exchanged with all_to_all_single over HCCL.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .layer import grouped_mm


def _a2a_raw(x, in_splits, out_splits, group=None):
    """Accepts split counts as tensors OR as already-materialised python lists.

    Each `.sum()` / `.tolist()` on a device tensor is a device->host sync that drains the
    dispatch queue. This function ran three of them per call (`int(out_splits.sum())` and
    two `.tolist()`), and T-A2A calls it six times in the forward and six more in the
    backward -- so the splits were being re-materialised on the gradient path even though
    they are the SAME counts the forward already computed. Taking lists here lets the
    caller convert once. `sum(list)` == `int(tensor.sum())` for the same counts; nothing
    numeric changes.

    `torch.is_tensor`, not `isinstance(..., list)`: a tuple must not fall through to
    `.tolist()` and blow up.
    """
    if torch.is_tensor(out_splits):
        out_splits = out_splits.tolist()
    if torch.is_tensor(in_splits):
        in_splits = in_splits.tolist()
    out = x.new_empty((sum(out_splits), *x.shape[1:]))
    dist.all_to_all_single(out, x.contiguous(), out_splits, in_splits, group=group)
    return out


class _A2A(torch.autograd.Function):
    """Differentiable all_to_all_single: backward is the transpose exchange
    (swap input/output splits). Needed for EP *training*."""

    @staticmethod
    def forward(ctx, x, in_splits, out_splits, group=None):
        # Materialise the splits ONCE, here, and stash the LISTS -- `backward` reuses them
        # for the transpose exchange, so stashing tensors made the gradient path re-run
        # `.tolist()` on counts the forward had already read. The swap in `backward` is
        # unchanged and now costs zero device->host syncs.
        if torch.is_tensor(in_splits):
            in_splits = in_splits.tolist()
        if torch.is_tensor(out_splits):
            out_splits = out_splits.tolist()
        ctx.in_splits, ctx.out_splits, ctx.group = in_splits, out_splits, group
        # `group` MUST be forwarded. It was dropped here while the backward passed it
        # correctly, which no benchmark could ever catch: every performance run is under
        # `torch.no_grad()`, and `_a2a` only routes through this autograd Function when
        # grad is enabled -- so the whole measured path went through the plain
        # `_a2a_raw(..., group)` and was fine. With grad on, T-A2A's intra-node exchange
        # would hand rpn-sized splits to the world group and die with "Number of tensor
        # splits not equal to group size" on the first training step. Found by
        # tests/test_ta2a_grad.py, the first test that enabled gradients at all.
        return _a2a_raw(x, in_splits, out_splits, group)

    @staticmethod
    def backward(ctx, g):
        return _a2a_raw(g, ctx.out_splits, ctx.in_splits, ctx.group), None, None, None


def _a2a(x, in_splits, out_splits, group=None):
    """Autograd-aware for float payloads; plain for integer metadata.

    `group` restricts the exchange to a subgroup (T-A2A uses the intra-node group for its
    demand routing). The backward must use the SAME group, or gradients travel a
    different communication domain than the forward did.
    """
    if x.is_floating_point() and torch.is_grad_enabled():
        return _A2A.apply(x, in_splits, out_splits, group)
    return _a2a_raw(x, in_splits, out_splits, group)


def ep_moe_forward(x_local, expert_idx, gates, w13_shard, w2_shard,
                   world: int, n_experts: int) -> torch.Tensor:
    """Routed-expert output for this rank's local tokens under EP.

    x_local [T, h]; expert_idx/gates [T, k] (this rank's routing decisions);
    w13_shard [epr, h, 2d], w2_shard [epr, d, h] (this rank's expert shard).
    Returns y [T, h] — the routed contribution (add shared separately).
    """
    epr = n_experts // world
    T, k = expert_idx.shape
    dev = x_local.device
    src = torch.arange(T, device=dev).repeat_interleave(k)
    eid = expert_idx.reshape(-1)
    gate = gates.reshape(-1)
    dest = torch.div(eid, epr, rounding_mode="floor")

    perm = torch.argsort(dest, stable=True)
    send = torch.bincount(dest[perm], minlength=world)
    payload = x_local[src[perm]]                          # [T*k, h] grouped by dest rank
    local_eid = (eid[perm] % epr).to(torch.int32)
    gate_p, src_p = gate[perm], src[perm]

    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)                    # learn per-rank recv counts
    # Materialise once. The BASELINE gets this too, deliberately: T-A2A's call sites were
    # hoisted in the same change, and optimising only the arm under test would inflate the
    # speedup by making the comparison unfair rather than by making the schedule better.
    send_l, recv_l = send.tolist(), recv.tolist()

    rx = _a2a(payload, send_l, recv_l)                    # tokens destined here
    reid = _a2a(local_eid, send_l, recv_l).long()        # their local expert ids

    order = torch.argsort(reid, stable=True)
    counts = torch.bincount(reid[order], minlength=epr)
    a, b = grouped_mm(rx[order], w13_shard, counts).chunk(2, dim=-1)
    ye = grouped_mm(F.silu(a) * b, w2_shard, counts)
    ye = ye[torch.argsort(order, stable=True)]           # back to arrival order

    back = _a2a(ye, recv_l, send_l)      # results returned, in perm order -- reversed pair
    y = x_local.new_zeros(T, x_local.shape[1])
    return y.index_add(0, src_p, gate_p[:, None] * back)
