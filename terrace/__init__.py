# -*- coding: utf-8 -*-
"""TerraceMoE (public release): hierarchy-aligned MoE routing (T-Route) and two-hop all-to-all (T-A2A).

This package contains the method itself and nothing else: the routing constraints, the
two-hop communication chain, and their reference implementations and tests.
Training-framework integration, measurement tooling, and experiment data are not here
(see "Repository layout" in the README).

The public __init__ is thinner than the internal one: the internal version also exports
the planning/co-design modules (plan / ttd etc.), which carry measured parameters of a
specific platform and were removed at the release boundary — nothing in this package
loses any functionality without them, and the routing and communication chain have zero
dependence on platform data.
"""
from .routing import Router, TRouteConfig, expert_load, t_route, update_bias_
from .layer import SwiGLU, TerracedMoE

__all__ = [
    "Router", "TRouteConfig", "expert_load", "t_route", "update_bias_",
    "SwiGLU", "TerracedMoE",
]
