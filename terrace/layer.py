"""Terraced-MoE layer (design notes: docs/01-troute-design.md and docs/02-ta2a-design.md).

Two interchangeable expert backends producing identical results:
  * "loop"    — reference: iterate experts, index_add. Correctness oracle, CPU.
  * "grouped" — permute tokens into per-expert contiguous groups, one grouped
                matmul per projection, unpermute. The permute/scatter logic is
                shared; only the inner GEMM differs (npu_grouped_matmul on device,
                a portable per-group loop off device) so the two backends match.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routing import Router, TRouteConfig, expert_load, t_route, update_bias_


class SwiGLU(torch.nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.w13 = torch.nn.Linear(hidden, 2 * inter, bias=False)
        self.w2 = torch.nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(a) * b)


def _grouped_mm_fwd(x: torch.Tensor, w: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Block-diagonal matmul forward (no autograd). x [N,in], w [E,in,out] -> [N,out]."""
    if x.device.type == "npu":
        group_list = counts.cumsum(0).to(torch.int64)
        return torch.ops.npu.npu_grouped_matmul(
            [x], [w], group_list=group_list, split_item=2, group_type=0)[0]
    out = x.new_empty(x.shape[0], w.shape[-1])
    off = 0
    for g in range(w.shape[0]):
        n = int(counts[g])
        if n:
            out[off:off + n] = x[off:off + n] @ w[g]
        off += n
    return out


class _GroupedMM(torch.autograd.Function):
    """Explicit autograd for the block-diagonal matmul. The vendor
    npu_grouped_matmul provides no gradients on this stack, so we compute them:
    grad_x = grouped_mm(grad_y, wᵀ, counts);  grad_w[g] = x_gᵀ @ grad_y_g."""

    @staticmethod
    def forward(ctx, x, w, counts):
        ctx.save_for_backward(x, w, counts)
        return _grouped_mm_fwd(x, w, counts)

    @staticmethod
    def backward(ctx, gy):
        x, w, counts = ctx.saved_tensors
        gy = gy.contiguous()
        gx = _grouped_mm_fwd(gy, w.transpose(-1, -2).contiguous(), counts)
        gw = torch.zeros_like(w)
        off = 0
        for g in range(w.shape[0]):
            n = int(counts[g])
            if n:
                gw[g] = x[off:off + n].transpose(0, 1) @ gy[off:off + n]
            off += n
        return gx, gw, None


def grouped_mm(x: torch.Tensor, w: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Block-diagonal matmul with correct gradients on CPU and NPU (rows of x are
    pre-sorted into groups sized by `counts`; group g multiplies w[g])."""
    return _GroupedMM.apply(x, w, counts)


class TerracedMoE(torch.nn.Module):
    def __init__(self, hidden, d_expert, d_shared, cfg: TRouteConfig,
                 n_shared: int = 1, bias_gamma: float = 1e-3):
        super().__init__()
        self.cfg = cfg
        self.bias_gamma = bias_gamma
        self.routing_mode = "full"
        self.expert_backend = "loop"        # "loop" | "grouped"
        self.router = Router(hidden, cfg)
        self.shared = SwiGLU(hidden, n_shared * d_shared)
        e, h, d = cfg.n_experts, hidden, d_expert
        self.w13 = torch.nn.Parameter(torch.empty(e, h, 2 * d))
        self.w2 = torch.nn.Parameter(torch.empty(e, d, h))
        torch.nn.init.normal_(self.w13, std=h ** -0.5)
        torch.nn.init.normal_(self.w2, std=d ** -0.5)

    def _experts_loop(self, x, expert_idx, gates, y):
        for e in expert_idx.unique().tolist():
            tok, slot = (expert_idx == e).nonzero(as_tuple=True)
            a, b = (x[tok] @ self.w13[e]).chunk(2, dim=-1)
            ye = (F.silu(a) * b) @ self.w2[e]
            y = y.index_add(0, tok, gates[tok, slot, None] * ye)
        return y

    def _experts_grouped(self, x, expert_idx, gates, y):
        n_tok, k = expert_idx.shape
        src = torch.arange(n_tok, device=x.device).repeat_interleave(k)   # source token per (tok,slot)
        eid = expert_idx.reshape(-1)
        gate = gates.reshape(-1)
        order = torch.argsort(eid)                                        # group by expert
        src, eid, gate = src[order], eid[order], gate[order]
        counts = torch.bincount(eid, minlength=self.cfg.n_experts)
        xs = x[src]                                                       # [N*k, h] tokens in expert order
        a, b = grouped_mm(xs, self.w13, counts).chunk(2, dim=-1)
        ye = grouped_mm(F.silu(a) * b, self.w2, counts)                  # [N*k, h]
        return y.index_add(0, src, gate[:, None] * ye)

    def _experts(self, x, expert_idx, gates, y):
        if self.expert_backend == "grouped":
            return self._experts_grouped(x, expert_idx, gates, y)
        return self._experts_loop(x, expert_idx, gates, y)

    def forward(self, x: torch.Tensor):
        """x: [..., hidden] -> (y same shape, stats). Leading dims flattened."""
        lead = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        expert_idx, gates, group_idx = self.router(x, self.routing_mode)
        y = self._experts(x, expert_idx, gates, self.shared(x))
        load = expert_load(expert_idx, self.cfg.n_experts)
        if self.training:
            update_bias_(self.router.bias, load, self.bias_gamma)
        stats = {"expert_load": load, "group_idx": group_idx, "gates": gates,
                 "expert_idx": expert_idx}
        return y.reshape(*lead, -1), stats


class HierarchicalMoE(nn.Module):
    """A1 — terraced compute-depth experts (docs/01 ADR-8).

    tier-0: every token through fine experts of width d0 (T-Route).
    tier-1: only the hardest fraction p of tokens (lowest tier-0 top-1 confidence)
            through wider experts of width d1, residual-added.
    iso-FLOPs with a flat MoE of width d_e when  d_e = d0 + p * d1  (routing cost
    negligible). Fixed p keeps the traffic deterministic (no threshold).
    """

    def __init__(self, hidden, d0, d1, p, cfg: TRouteConfig,
                 n_shared=1, d_shared=None, bias_gamma=1e-3):
        super().__init__()
        self.p, self.cfg = p, cfg
        self.tier0 = TerracedMoE(hidden, d0, d_shared or (2 * d0), cfg,
                                 n_shared=n_shared, bias_gamma=bias_gamma)
        self.tier1 = TerracedMoE(hidden, d1, d1, cfg, n_shared=0, bias_gamma=bias_gamma)
        self.tier1.shared = None                       # tier-1 is routed-only

    @property
    def expert_backend(self):
        return self.tier0.expert_backend

    @expert_backend.setter
    def expert_backend(self, v):
        self.tier0.expert_backend = v; self.tier1.expert_backend = v

    @property
    def routing_mode(self):
        return self.tier0.routing_mode

    @routing_mode.setter
    def routing_mode(self, v):
        self.tier0.routing_mode = v; self.tier1.routing_mode = v

    def forward(self, x):
        lead = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        T = x.shape[0]
        # tier-0: all tokens
        aff0 = torch.sigmoid(x @ self.tier0.router.weight.t())
        e0, g0, grp0 = t_route(aff0, self.tier0.router.bias, self.cfg, self.routing_mode)
        y = self.tier0._experts(x, e0, g0, self.tier0.shared(x))
        # difficulty = 1 - top-1 confidence; pick hardest fixed fraction
        n_hard = max(1, int(self.p * T))
        hard = aff0.max(-1).values.argsort()[:n_hard]       # lowest confidence
        xh = x[hard]
        e1, g1, _ = self.tier1.router(xh, self.routing_mode)
        yh = self.tier1._experts(xh, e1, g1, xh.new_zeros(xh.shape[0], x.shape[1]))
        y = y.index_add(0, hard, yh)                        # residual for hard tokens
        if self.training:
            update_bias_(self.tier0.router.bias, expert_load(e0, self.cfg.n_experts), self.tier0.bias_gamma)
            update_bias_(self.tier1.router.bias, expert_load(e1, self.cfg.n_experts), self.tier1.bias_gamma)
        return y.reshape(*lead, -1), {"expert_load": expert_load(e0, self.cfg.n_experts),
                                      "group_idx": grp0, "gates": g0, "n_hard": n_hard,
                                      "expert_idx": e0}
