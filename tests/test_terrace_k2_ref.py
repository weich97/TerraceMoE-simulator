"""K2(发送侧融合打包链)CPU 参考 vs 现链:逐位对账。

k2_pack_ref(terrace/ops/k2_ref.py)是 AscendC kernel 两遍计数排序的直译
(独立实现,不复用现链原语),现链臂是 quota 快路径发送段的**原文组合**:
plan_ta2a(groups_m=M)+ hidden[u_src] 去重 gather + _pack_quota_wire。
两臂五个输出(payload / mask 槽号表 / gate_rows / u_src / node_counts)全部
torch.equal(逐位,不设容差)—— 这是 K2 kernel 上机位级验收的 CPU 侧地基:
kernel 与 k2_pack_ref 同一数学对象(论证见 kernel 文件头),k2_pack_ref 与
现链在这里钉死,传递即 kernel == 现链。

覆盖轴(≥8 例来自几何 x 行序 x dtype 的笛卡尔积):
  - 几何:slots 4/16/24/32(跨 build_expansion 的 24 位 float32 掩码边界 ——
    对 C1 槽号表本身无关,但保证几何谱与 quota_wire 回归床一致)、quota 1
    (k==M,槽号表宽 1)、M=1(单节点整包)、T=1、奇数 T/quota=2;
  - 行序:升序(接缝入口,routing_map_to_topk 按构造)与乱序(融合前向入口,
    打包侧行内 argsort 分支)。现链臂对升序行走 sorted_rows=True 的免排序
    分支、乱序行走 =False 的 argsort 分支 —— 参考实现单一路径必须同时逐位
    等于两者;
  - dtype:fp32 / bf16 / fp16(gate 平面纯位搬运,一个 dtype 都不许漂)。

另钉两条契约:gates/hidden dtype 失配两臂都必须大声死(C1 圆整点契约,
一次内部提交 漂移缺陷的进场口);槽号表每行升序是线上契约(到达侧不再排序),
参考臂单独断言,不只靠与现链的巧合一致。
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terrace.ops.k2_ref import k2_pack_ref
from terrace.ta2a import plan_ta2a
from terrace.ta2a_fwd import _pack_quota_wire


def _chain(hidden, expert_idx, gates, world, n_experts, rpn, groups_m, sorted_rows):
    """现链原文组合(ta2a_moe_forward / ta2a_permute 的 quota 支路逐行同构)。"""
    u_src, _, node_counts, inverse = plan_ta2a(expert_idx, world, n_experts, rpn,
                                               groups_m=groups_m)
    n_rows = u_src.numel()
    payload = hidden[u_src]
    slots = (n_experts // world) * rpn
    quota = expert_idx.shape[1] // groups_m
    mask, gate_rows = _pack_quota_wire(expert_idx, gates, inverse, payload,
                                       n_rows, slots, quota, n_experts,
                                       sorted_rows=sorted_rows)
    return payload, mask, gate_rows, u_src, node_counts


def _equal_quota(T, k, n_experts, n_nodes, m, seed, sort_rows):
    """T-Route 等额配额路由:每 token 恰 m 个节点、每节点恰 k//m 个专家。"""
    g = torch.Generator().manual_seed(seed)
    per, quota = n_experts // n_nodes, k // m
    rows = []
    for _ in range(T):
        nodes = torch.randperm(n_nodes, generator=g)[:m]
        rows.append(torch.cat([torch.arange(a * per, (a + 1) * per)[
            torch.randperm(per, generator=g)[:quota]] for a in nodes]))
    idx = torch.stack(rows)
    return torch.sort(idx, dim=1).values if sort_rows else idx


# (world, rpn, n_experts, T, k, M) —— slots = (E//world)*rpn 标注在行尾。
GEOMETRIES = [
    (4, 2, 8, 8, 4, 2),        # slots 4,分布回归床同几何
    (32, 8, 64, 48, 4, 2),     # slots 16,对齐床几何
    (32, 8, 96, 32, 8, 4),     # slots 24,f32 掩码最后宽度
    (32, 8, 128, 32, 8, 2),    # slots 32,旧掩码 int64 精确路径的几何
    (16, 8, 32, 5, 2, 2),      # quota 1:k==M,槽号表宽 1
    (16, 4, 16, 12, 4, 1),     # M=1:单节点整包,4 节点
    (8, 4, 8, 1, 2, 2),        # T=1
    (64, 8, 128, 33, 6, 3),    # 奇数 T,quota 2
]


@pytest.mark.parametrize("world,rpn,E,T,k,m", GEOMETRIES)
@pytest.mark.parametrize("sort_rows", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_k2_ref_bitwise_equals_chain(world, rpn, E, T, k, m, sort_rows, dtype):
    n_nodes = world // rpn
    quota = k // m
    for seed in range(2):
        idx = _equal_quota(T, k, E, n_nodes, m, 500 + seed, sort_rows)
        g = torch.Generator().manual_seed(seed)
        hidden = torch.randn(T, 16, generator=g).to(dtype)
        gates = torch.rand(T, k, generator=g).to(dtype)

        ref = k2_pack_ref(hidden, idx, gates, world, E, rpn, m)
        chain = _chain(hidden, idx, gates, world, E, rpn, m,
                       sorted_rows=sort_rows)

        names = ("payload", "mask", "gate_rows", "u_src", "node_counts")
        for name, a, b in zip(names, ref, chain):
            assert a.dtype == b.dtype, f"{name}: dtype {a.dtype} != {b.dtype}"
            assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
            assert torch.equal(a, b), (
                f"{name} 参考实现与现链不逐位相等 "
                f"(geom w{world}/rpn{rpn}/E{E}/T{T}/k{k}/M{m}, "
                f"sorted={sort_rows}, {dtype}, seed={seed})")
        # 形状/静态性:n_rows = T*M 是 K2 免同步的本钱
        assert ref[0].shape == (T * m, 16)
        assert ref[1].shape == (T * m, quota) and ref[1].dtype == torch.int64
        # 槽号表每行升序是 C1 线上契约(到达侧不再排序),单独钉住
        assert torch.equal(ref[1], torch.sort(ref[1], dim=1).values)
        # 计数守恒:每节点行数之和 == 发送行数
        assert int(ref[4].sum()) == T * m


def test_k2_ref_gradients_flow_like_the_chain():
    """参考实现是纯 gather 组合,autograd 的伴随(index_add / scatter)即现链
    反向 —— 嫁接后的 TerraceK2PackFn.backward 用同一组原语(内部嫁接记录(未随仓发布))。
    这里钉:payload/gate_rows 载梯度、与现链臂的梯度逐位同。"""
    world, rpn, E, T, k, m = 4, 2, 8, 8, 4, 2
    idx = _equal_quota(T, k, E, world // rpn, m, 700, False)
    g = torch.Generator().manual_seed(3)
    hidden_r = torch.randn(T, 16, generator=g).requires_grad_(True)
    gates_r = torch.rand(T, k, generator=g).requires_grad_(True)
    hidden_c = hidden_r.detach().clone().requires_grad_(True)
    gates_c = gates_r.detach().clone().requires_grad_(True)

    pr, _, gr, _, _ = k2_pack_ref(hidden_r, idx, gates_r, world, E, rpn, m)
    pc, _, gc, _, _ = _chain(hidden_c, idx, gates_c, world, E, rpn, m,
                             sorted_rows=False)
    gp = torch.randn(pr.shape, generator=g)
    gg = torch.randn(gr.shape, generator=g)
    (pr * gp).sum().backward(retain_graph=True)
    (gr * gg).sum().backward()
    (pc * gp).sum().backward(retain_graph=True)
    (gc * gg).sum().backward()
    assert torch.equal(hidden_r.grad, hidden_c.grad)
    assert torch.equal(gates_r.grad, gates_c.grad)


def test_k2_ref_keeps_dtype_mismatch_loud():
    """gates/hidden dtype 失配必须大声死 —— 参考臂与现链臂同一失效形态
    (_pack_quota_wire 在 index_put 处 RuntimeError;gate 平面从 payload 派生,
    从 gates 派生就会静默把更宽的 gate 平面送上线,一次内部提交 的进场口)。"""
    world, rpn, E, T, k, m = 4, 2, 8, 8, 4, 2
    idx = _equal_quota(T, k, E, world // rpn, m, 11, True)
    hidden = torch.randn(T, 16, dtype=torch.bfloat16)
    gates = torch.rand(T, k)                                  # fp32:失配
    with pytest.raises(RuntimeError):
        k2_pack_ref(hidden, idx, gates, world, E, rpn, m)
    with pytest.raises(RuntimeError):
        _chain(hidden, idx, gates, world, E, rpn, m, sorted_rows=True)


def test_k2_ref_rejects_out_of_range_expert_ids():
    """越界专家号:现链在 plan 的 scatter 处大声死,参考实现同样 fail loud
    (kernel 侧不能 raise,以跳过写 + zeros 收容代之,不承诺位级 —— 见 kernel
    文件头;参考实现是规格,规格必须拒绝)。"""
    world, rpn, E, T, k, m = 4, 2, 8, 4, 4, 2
    idx = _equal_quota(T, k, E, world // rpn, m, 13, True)
    idx[0, 0] = E                                             # 越界
    hidden = torch.randn(T, 16)
    gates = torch.rand(T, k)
    with pytest.raises(ValueError, match="out of range"):
        k2_pack_ref(hidden, idx, gates, world, E, rpn, m)
