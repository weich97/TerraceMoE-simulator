"""The reference stack's balance aux must actually balance.

WHY THIS EXISTS
The previous _balance_aux computed sum_i f_i^2 from bincount loads alone. Counts carry no
gradient, so the aux raised the reported loss without exerting any pressure on any
parameter -- a knob wired to nothing (2026-08-02 full-repo review, finding H3). The
upstream-training-stack arms are unaffected (they use Megatron's seq_aux_loss); this covers the
pure-PyTorch reference stack that the open-source release ships.

Two properties, each of which the old implementation violated or could not demonstrate:
  1. the aux carries gradient back to the router inputs;
  2. it is larger for skewed routing than for near-uniform routing.
"""
from __future__ import annotations

import torch

from terrace.model import ModelArgs, _balance_aux


def _routed(T: int, E: int, k: int, skew: float, seed: int):
    """Mimic the router's output shape: differentiable gates over top-k expert picks."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(T, E, generator=g, requires_grad=True)
    # A per-expert additive tilt concentrates the top-k picks on high-index experts
    # without touching the differentiable path (detached, like a routing decision).
    idx = (logits.detach() + skew * torch.arange(E, dtype=torch.float32)).topk(k, -1).indices
    aff = torch.softmax(logits, -1)
    gates = aff.gather(1, idx)
    gates = gates / gates.sum(-1, keepdim=True)
    load = torch.bincount(idx.reshape(-1), minlength=E).float()
    return logits, load, gates, idx


def test_aux_carries_gradient():
    a = ModelArgs()
    logits, load, gates, idx = _routed(T=64, E=a.n_experts, k=4, skew=0.0, seed=0)
    aux = _balance_aux([load], [gates], [idx], a)
    assert aux.requires_grad, "aux detached from the graph -- the dead-knob bug is back"
    aux.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0, "aux gradient w.r.t. router inputs is zero"


def test_aux_larger_when_skewed():
    a = ModelArgs()
    vals = []
    for skew in (0.0, 4.0):
        _, load, gates, idx = _routed(T=256, E=a.n_experts, k=4, skew=skew, seed=1)
        vals.append(float(_balance_aux([load], [gates], [idx], a).detach()))
    assert vals[1] > vals[0], (
        f"skewed routing must score a higher aux than near-uniform (got {vals})")


def test_aux_multi_layer_average():
    a = ModelArgs()
    l1 = _routed(T=64, E=a.n_experts, k=4, skew=0.0, seed=2)
    l2 = _routed(T=64, E=a.n_experts, k=4, skew=0.0, seed=3)
    one = _balance_aux([l1[1]], [l1[2]], [l1[3]], a)
    two = _balance_aux([l1[1], l2[1]], [l1[2], l2[2]], [l1[3], l2[3]], a)
    # Averaged over layers, not summed: adding a layer must not double the aux scale.
    assert float(two.detach()) < 2 * float(one.detach())
