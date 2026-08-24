"""Deterministic A2A traffic-matrix tools.

T-Route makes dispatch traffic a static object: with node-level dedup each token
contributes exactly `top_groups` node-to-node messages. These helpers materialize
that matrix from routing decisions so schedules (and later, replay backends) can
consume it.
"""
from __future__ import annotations

import torch

from terrace.routing import TRouteConfig


def dispatch_traffic(
    group_idx: torch.Tensor,   # [T, M] selected groups per token
    src_card: torch.Tensor,    # [T] source card of each token (0..EP-1)
    cfg: TRouteConfig,
    cards_per_group: int,
) -> torch.Tensor:
    """Node-level dispatch message counts with node dedup → [N_g, N_g] tokens."""
    n_g = cfg.n_groups
    src_node = (src_card // cards_per_group).repeat_interleave(cfg.top_groups)
    dst_node = group_idx.reshape(-1)
    flat = src_node * n_g + dst_node
    return torch.bincount(flat, minlength=n_g * n_g).reshape(n_g, n_g).float()


def traffic_bytes(matrix_tokens: torch.Tensor, hidden: int, c_bytes: float = 1.0) -> torch.Tensor:
    return matrix_tokens * hidden * c_bytes


def uniformity_cv(matrix: torch.Tensor) -> float:
    """Coefficient of variation across matrix entries (0 = perfectly uniform)."""
    m = matrix.reshape(-1)
    return (m.std(unbiased=False) / m.mean()).item()
