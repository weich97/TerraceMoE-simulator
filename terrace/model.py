"""Trainable Terrace decoder for the accuracy-ablation ladder.

Small dense-attention + Terraced-MoE decoder used for the T-Route ablation.
Single-device (no EP): the routing decision — the object under test — is
identical whether experts are colocated or distributed, so a single-die run
faithfully measures routing quality (loss, load balance) at fixed FLOPs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import HierarchicalMoE, SwiGLU, TerracedMoE
from .routing import TRouteConfig


@dataclass
class ModelArgs:
    vocab: int = 32000
    dim: int = 512
    n_layers: int = 12
    n_heads: int = 8
    seq_len: int = 1024
    # MoE
    n_experts: int = 64
    n_groups: int = 16
    top_k: int = 4
    top_groups: int = 2
    d_expert: int = 512
    d_shared: int = 1024
    n_shared: int = 1
    dense_first: int = 1          # leading dense FFN layers
    bias_gamma: float = 1e-3
    routing_mode: str = "full"    # full | group_limited | quota_only | global_topk
    balance_aux: float = 1e-3     # seq-level balance aux (early-training only)
    expert_backend: str = "loop"  # loop | grouped
    arch: str = "flat"            # flat | hier (A1 terraced compute-depth experts)
    d0: int = 256                 # hier tier-0 width
    d1: int = 1024                # hier tier-1 width (deep, hard tokens)
    p_hard: float = 0.25          # hier fraction of tokens routed to tier-1


def rmsnorm(x, w, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class Attention(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.nh, self.hd = a.n_heads, a.dim // a.n_heads
        self.wqkv = nn.Linear(a.dim, 3 * a.dim, bias=False)
        self.wo = nn.Linear(a.dim, a.dim, bias=False)
        self.register_buffer("qn", torch.ones(self.hd), persistent=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = self.wqkv(x).chunk(3, -1)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(B, T, -1))


def _rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, a: ModelArgs, moe: bool):
        super().__init__()
        self.dim = a.dim
        self.n1 = nn.Parameter(torch.ones(a.dim))
        self.n2 = nn.Parameter(torch.ones(a.dim))
        self.attn = Attention(a)
        if moe:
            cfg = TRouteConfig(a.n_experts, a.n_groups, a.top_k, a.top_groups)
            if a.arch == "hier":
                self.ffn = HierarchicalMoE(a.dim, a.d0, a.d1, a.p_hard, cfg,
                                           n_shared=a.n_shared, d_shared=a.d_shared,
                                           bias_gamma=a.bias_gamma)
            else:
                self.ffn = TerracedMoE(a.dim, a.d_expert, a.d_shared, cfg,
                                       n_shared=a.n_shared, bias_gamma=a.bias_gamma)
            self.ffn.routing_mode = a.routing_mode
            self.ffn.expert_backend = a.expert_backend
            self.is_moe = True
        else:
            self.ffn = SwiGLU(a.dim, 4 * a.dim)
            self.is_moe = False

    def forward(self, x, cos, sin):
        x = x + self.attn(rmsnorm(x, self.n1), cos, sin)
        if self.is_moe:
            y, stats = self.ffn(rmsnorm(x, self.n2))
            return x + y, stats
        return x + self.ffn(rmsnorm(x, self.n2)), None


class TerraceLM(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.args = a
        self.tok = nn.Embedding(a.vocab, a.dim)
        self.blocks = nn.ModuleList(
            Block(a, moe=(i >= a.dense_first and i < a.n_layers - 1))
            for i in range(a.n_layers))
        self.nf = nn.Parameter(torch.ones(a.dim))
        self.head = nn.Linear(a.dim, a.vocab, bias=False)
        hd = a.dim // a.n_heads
        freqs = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        t = torch.arange(a.seq_len).float()
        ang = torch.outer(t, freqs)
        self.register_buffer("cos", ang.cos()[None, None], persistent=False)
        self.register_buffer("sin", ang.sin()[None, None], persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx)
        cos, sin = self.cos[..., :T, :], self.sin[..., :T, :]
        loads, routes, gate_l, idx_l = [], [], [], []
        for blk in self.blocks:
            x, stats = blk(x, cos, sin)
            if stats is not None:
                loads.append(stats["expert_load"])
                gate_l.append(stats["gates"])
                idx_l.append(stats["expert_idx"])
                if getattr(self, "collect_routes", False) and "expert_idx" in stats:
                    routes.append(stats["expert_idx"])
        logits = self.head(rmsnorm(x, self.nf))
        if targets is None:
            return logits, {}
        lm = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        aux = (_balance_aux(loads, gate_l, idx_l, self.args)
               if self.args.balance_aux else lm.new_zeros(()))
        return lm + self.args.balance_aux * aux, {"lm_loss": lm.detach(), "aux": aux.detach(),
                                                  "loads": loads, "routes": routes}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _balance_aux(loads, gates, idxs, a: ModelArgs):
    """Differentiable balance loss: n_experts * sum_i f_i * P_i (DeepSeek form).

    f_i (routed-token share) comes from bincount and carries no gradient; the gradient
    flows through P_i, the mean gate-probability mass each expert receives. The previous
    version computed sum_i f_i^2 from counts alone -- zero gradient w.r.t. every parameter,
    so the knob raised the reported loss without exerting any balancing pressure
    (2026-08-02 full-repo review, H3). Scope of the old bug: this reference stack only;
    the upstream training-stack arms all trained with Megatron's own seq_aux_loss and are unaffected.
    """
    tot = a.n_experts
    aux = None
    for load, g, idx in zip(loads, gates, idxs):
        f = (load / load.sum().clamp_min(1)).detach()
        p = torch.zeros_like(load).index_add(
            0, idx.reshape(-1), g.reshape(-1).to(load.dtype)) / max(idx.shape[0], 1)
        term = (f * p).sum() * tot
        aux = term if aux is None else aux + term
    return aux / max(len(loads), 1)
