# -*- coding: utf-8 -*-
"""TerraceMoE(公开版):层级对齐的 MoE 路由(T-Route)与两跳 all-to-all(T-A2A)。

本包只含方法本身:路由约束、两跳通信链、以及它们的参考实现与测试。
训练框架接入、测量工具、实验数据都不在此(见 README「仓库结构」)。

公开版的 __init__ 比内部版瘦:内部版还导出规划/共设计模块(plan / ttd 等),
那些装着具体平台的实测参数,按发布边界剔除 —— 缺它们不影响本包的一切功能,
路由与通信链对平台数据零依赖。
"""
from .routing import Router, TRouteConfig, expert_load, t_route, update_bias_
from .layer import SwiGLU, TerracedMoE

__all__ = [
    "Router", "TRouteConfig", "expert_load", "t_route", "update_bias_",
    "SwiGLU", "TerracedMoE",
]
