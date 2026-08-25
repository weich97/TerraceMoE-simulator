"""T-Route: terraced group-quota routing (design notes: docs/01-troute-design.md).

Selection uses bias-corrected affinities (aux-loss-free balancing); gate values
use raw affinities. Structural guarantees, per token:

  * exactly ``top_k`` experts, all distinct;
  * experts span exactly ``top_groups`` distinct groups (fan-out bound);
  * exactly ``top_k // top_groups`` experts per selected group (quota).

Group-level load balance is statistical, driven by the bias update; the
fan-out bound and quota hold unconditionally.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TRouteConfig:
    n_experts: int      # E
    n_groups: int       # N_g (group boundary = TTD group tier, e.g. one node)
    top_k: int          # k
    top_groups: int     # M
    group_score_pool: int = 2   # group score = mean of top-`pool` affinities in group

    def __post_init__(self):
        if self.n_experts % self.n_groups:
            raise ValueError("n_experts must divide evenly into n_groups")
        if self.top_k % self.top_groups:
            raise ValueError("top_k must divide evenly into top_groups")
        if self.per_group_k > self.group_size:
            raise ValueError("per-group quota exceeds group size")

    @property
    def group_size(self) -> int:
        return self.n_experts // self.n_groups

    @property
    def per_group_k(self) -> int:
        return self.top_k // self.top_groups


def t_route(
    affinity: torch.Tensor,   # [T, E], sigmoid affinities in (0, 1)
    bias: torch.Tensor,       # [E], balancing bias (selection only)
    cfg: TRouteConfig,
    mode: str = "full",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (expert_idx [T, k], gates [T, k], group_idx [T, M]).

    Ablation modes (internal quality-equivalence gate; records not shipped):
      full          — group-limited (top-M groups) + per-group equal quota (T-Route)
      group_limited — top-M groups, then free top-k within them (no quota)
      quota_only    — all groups eligible, equal quota per group (MoGE, M=N_g)
      global_topk   — plain top-k over all experts (unconstrained baseline)
    """
    # Explicit whitelist. The dispatch below ends in a bare `else` for "full", so before
    # this check any misspelling -- 'Full', 'quota', '', None -- fell through and silently
    # ran full T-Route. Four ablation arms silently collapsing into one identical arm is
    # the exact signature of the 2026-07-x vendor-branch incident (internal engineering
    # records: 48 arms of a false-positive quality gate), and a typo in a launch script must be a crash, not a fifth
    # unnamed mode.
    _MODES = ("full", "group_limited", "quota_only", "global_topk")
    if mode not in _MODES:
        raise ValueError(f"unknown routing mode {mode!r}; expected one of {_MODES}")
    n_tok = affinity.shape[0]
    g = cfg.group_size
    s_sel = (affinity + bias)

    if mode == "global_topk":
        expert_idx = s_sel.topk(cfg.top_k, dim=-1).indices
        group_idx = (expert_idx // g)
    elif mode == "quota_only":
        expert_idx, group_idx = _quota_route(s_sel, cfg, cfg.n_groups, cfg.top_k // cfg.n_groups)
    elif mode == "group_limited":
        sg = s_sel.view(n_tok, cfg.n_groups, g)
        gsc = sg.topk(min(cfg.group_score_pool, g), -1).values.mean(-1)
        group_idx = gsc.topk(cfg.top_groups, -1).indices
        cand = sg.gather(1, group_idx[..., None].expand(-1, -1, g)).reshape(n_tok, -1)
        local = cand.topk(cfg.top_k, -1).indices                            # free within groups
        base = (group_idx * g)[..., None]                                   # [T, M, 1]
        flat_base = base.expand(-1, -1, g).reshape(n_tok, -1)
        expert_idx = flat_base.gather(1, local) + (local % g)
    else:  # full T-Route
        sg = s_sel.view(n_tok, cfg.n_groups, g)
        gsc = sg.topk(min(cfg.group_score_pool, g), -1).values.mean(-1)
        group_idx = gsc.topk(cfg.top_groups, -1).indices
        picked = sg.gather(1, group_idx[..., None].expand(-1, -1, g))
        local = picked.topk(cfg.per_group_k, -1).indices
        expert_idx = (group_idx[..., None] * g + local).reshape(n_tok, cfg.top_k)

    gates = affinity.gather(1, expert_idx)
    gates = gates / gates.sum(-1, keepdim=True)
    return expert_idx, gates, group_idx


def _quota_route(s_sel, cfg, n_sel_groups, per_group):
    if per_group < 1:
        raise ValueError(f"quota_only needs top_k ({cfg.top_k}) divisible by "
                         f"n_groups ({cfg.n_groups}); got {per_group} experts/group")
    n_tok, g = s_sel.shape[0], cfg.group_size
    sg = s_sel.view(n_tok, cfg.n_groups, g)
    local = sg.topk(per_group, -1).indices                                  # [T, G, per_group]
    grp = torch.arange(cfg.n_groups, device=s_sel.device)[None, :, None]
    expert_idx = (grp * g + local).reshape(n_tok, -1)
    group_idx = torch.arange(cfg.n_groups, device=s_sel.device)[None].expand(n_tok, -1)
    return expert_idx, group_idx


def expert_load(expert_idx: torch.Tensor, n_experts: int) -> torch.Tensor:
    return torch.bincount(expert_idx.reshape(-1), minlength=n_experts).float()


@torch.no_grad()
def update_bias_(bias: torch.Tensor, load: torch.Tensor, gamma: float) -> torch.Tensor:
    """Aux-loss-free balancing step: nudge bias toward under-loaded experts."""
    err = load.mean() - load
    bias.add_(gamma * torch.sign(err))
    return bias


class Router(torch.nn.Module):
    def __init__(self, hidden_size: int, cfg: TRouteConfig):
        super().__init__()
        self.cfg = cfg
        self.weight = torch.nn.Parameter(torch.empty(cfg.n_experts, hidden_size))
        torch.nn.init.normal_(self.weight, std=hidden_size ** -0.5)
        self.register_buffer("bias", torch.zeros(cfg.n_experts))

    def forward(self, hidden: torch.Tensor, mode: str = "full"):
        affinity = torch.sigmoid(hidden @ self.weight.t())
        return t_route(affinity, self.bias, self.cfg, mode)
