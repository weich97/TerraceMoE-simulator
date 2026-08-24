"""两条拆半接缝(legacy 3 参 vs overlap 6 参)同输入下前向与全部梯度**逐位相等**。

Why this file exists(eqov 对齐床 FAIL 的修复锁,2026-08-20):
overlap 接缝的 T-A2A 在对齐床上相对厂商 overlap 单调漂移(1e-5@20 → 1.38e-4@100,
校准比 10.6×,界 3×),而 legacy 接缝同床 ≤2e-5。漂移猎捕(逐位对照两条拆半路径)
定位到唯一数值差异:一次内部提交 的 gate 平面走 probs.dtype(fp32),而 legacy 半程是
payload.dtype(床上 bf16)。fp32 gate 经厂商 gmm 的 probs 乘法把 expert_out 提升到
fp32,连带 combine 的两跳回程与两级 index_add 归约全部在 fp32 里做 —— dispatch 出口
token 平面仍逐位相等(所以逐层 ASSERT 零触发),但每一处归约的舍入都偏离 legacy
已验路径,逐步放大。修复:gate 在进 gate_rows 处圆整到 payload.dtype(与 legacy
同一圆整点;cast∘gather == gather∘cast,逐元素同值)。

所以本文件锁两个不变量,全部 torch.equal(逐位),不设容差:
  1. fp32 均匀精度:两接缝的前向输出与 hidden/probs/w13/w2 四路梯度逐位相等
     (同一串运算换调用边界,不该有任何舍入差);
  2. 床口径(bf16 载荷/权重 + fp32 router probs):overlap 接缝收 fp32 probs
     (厂商 overlap 层交来的就是 router 精度),legacy 接缝收 bf16 probs(厂商
     legacy 层在 dispatch 上游已转 hidden dtype —— 这是硬事实:legacy 半程的
     gate_rows 是 payload.dtype,fp32 probs 会当场死在 index_put 的 dtype 匹配
     检查上,而 legacy 门 100 步跑通了)。两臂前向与 hidden/w13/w2 梯度逐位相等;
     probs 梯度一个落 bf16 叶子一个落 fp32 叶子,唯一合法差异是 .to 反向的精确
     upcast(bf16 -> fp32 无损),所以断言 overlap 的 fp32 梯度 == legacy 的 bf16
     梯度 upcast 后逐位相等 —— 仍是零容差。

overlap 臂的反向按厂商手写编排逐步复刻(与 tests/test_ta2a_overlap_seam.py 同一
顺序);legacy 臂走普通 autograd。gloo / CPU,world 4、rpn 2(2 节点,有真实跨节点跳)。
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORLD, RPN, T, K, M, E, H, D = 4, 2, 8, 4, 2, 8, 6, 4


def _routing(gen, n_nodes, per, quota):
    """T-Route 等额配额:恰好 M 个节点,每个节点恰好 K/M 个专家。"""
    rows = []
    for _ in range(T):
        gs = torch.randperm(n_nodes, generator=gen)[:M]
        rows.append(torch.cat([torch.arange(a * per, (a + 1) * per)[
            torch.randperm(per, generator=gen)[:quota]] for a in gs]))
    return torch.stack(rows)


def _bitdiff(a, b):
    """None = 逐位相等;否则返回 (dtype 对, max|Δ|) 供报错定位。"""
    if a is None or b is None:
        return ("missing", None)
    if a.dtype != b.dtype:
        return (f"{a.dtype} vs {b.dtype}",
                float((a.float() - b.float()).abs().max()))
    if torch.equal(a, b):
        return None
    return (str(a.dtype), float((a.float() - b.float()).abs().max()))


def _run(rank, world, q, payload_dtype_name):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # 两个床各用一个端口,避免与其它 dist 测试文件撞车(已用 29577/29591/29613)。
    os.environ.setdefault(
        "MASTER_PORT", "29623" if payload_dtype_name == "float32" else "29627")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from terrace.layer import grouped_mm
        from terrace.ep_dist import _a2a_raw
        from terrace.ta2a_fwd import init_ta2a_groups
        from terrace.ta2a_dispatch import (ta2a_permute, ta2a_unpermute,
                                           ta2a_permute_overlap,
                                           ta2a_unpermute_overlap)

        pdt = getattr(torch, payload_dtype_name)
        intra = init_ta2a_groups(world, RPN)
        epr = E // world
        n_nodes, per, quota = world // RPN, E // (world // RPN), K // M

        g = torch.Generator().manual_seed(11 + rank)
        x0 = torch.randn(T, H, generator=g).to(pdt)
        idx = _routing(g, n_nodes, per, quota)
        gates0 = torch.rand(T, K, generator=g)          # fp32:router 输出精度
        w13_0 = (torch.randn(epr, H, 2 * D, generator=g) / (H ** 0.5)).to(pdt)
        w2_0 = (torch.randn(epr, D, H, generator=g) / (D ** 0.5)).to(pdt)
        G = torch.randn(T, H, generator=g).to(pdt)      # 上游梯度

        routing_map = torch.zeros(T, E, dtype=torch.bool)
        routing_map[torch.arange(T).unsqueeze(1), idx] = True
        probs_dense = torch.zeros(T, E)
        probs_dense[torch.arange(T).unsqueeze(1), idx] = gates0   # fp32 稠密

        # ---- legacy 臂:普通 autograd。probs 按厂商 legacy 层的口径转 payload
        # dtype 后进接缝(fp32 时 .to 是恒等,两床同一行代码)。----
        hidL = x0.clone().requires_grad_(True)
        # 先 clone 再 .to:fp32 时 .to 是恒等,直接对 probs_dense 置 requires_grad
        # 会让下面 overlap 臂的 clone 变成非叶子(.grad 不落),两臂必须各自成叶。
        probL = probs_dense.clone().to(pdt).requires_grad_(True)
        w13L = w13_0.clone().requires_grad_(True)
        w2L = w2_0.clone().requires_grad_(True)
        permL, tpeL, ppL, stL = ta2a_permute(
            hidL, probL, routing_map, world=world, rank=rank, rpn=RPN,
            n_experts=E, intra_group=intra, inter_group=None, groups_m=M)
        a, b = grouped_mm(permL, w13L, tpeL).chunk(2, dim=-1)
        eoL = grouped_mm(F.silu(a) * b, w2L, tpeL) * ppL.unsqueeze(-1)
        outL = ta2a_unpermute(eoL, stL, hidL)
        outL.backward(G)

        # ---- overlap 臂:6 参接缝两半 + 厂商手写 backward 的逐步编排
        # (与 test_ta2a_overlap_seam.py 同一顺序)。probs 保持 fp32 进接缝
        # (厂商 overlap 层交来的就是 router 精度)。----
        hidO = x0.clone().requires_grad_(True)
        probO = probs_dense.clone().requires_grad_(True)
        save = []
        permO, tpeO, ppO, _share, stO = ta2a_permute_overlap(
            hidO, probO, routing_map, world=world, rank=rank, rpn=RPN,
            n_experts=E, intra_group=intra, inter_group=None, groups_m=M,
            save_tensors=save, run_shared_experts=None)
        w13O = w13_0.clone().requires_grad_(True)
        w2O = w2_0.clone().requires_grad_(True)
        dI = permO.detach().requires_grad_(True)
        pI = ppO.detach().requires_grad_(True)
        a, b = grouped_mm(dI, w13O, tpeO).chunk(2, dim=-1)
        eoO = grouped_mm(F.silu(a) * b, w2O, tpeO) * pI.unsqueeze(-1)
        outO = ta2a_unpermute_overlap(eoO, stO, save)
        (p1g, p1p, _ncpu, p2ind, p2g, p2pd, p2pg, u1ind, u1g, u2ind) = save

        fwd = {
            "permuted": _bitdiff(permL.detach(), permO.detach()),
            "pprobs": _bitdiff(ppL.detach(), ppO.detach()),
            "out": _bitdiff(outL.detach(), outO.detach()),
            "tpe": None if torch.equal(tpeL, tpeO) else ("tpe", -1.0),
        }

        outO.backward(G)                                   # unpermute2 段
        grad_red = _a2a_raw(u2ind.grad, stO.send_l, stO.recv_l)
        u1g.backward(grad_red)                             # unpermute1 段
        eoO.backward(u1ind.grad)                           # experts 等价物
        p2pg.backward(pI.grad)                             # gmm 内:先 prob 路
        ggr = _a2a_raw(p2pd.grad, stO.recv_l, stO.send_l)
        p2g.backward(dI.grad)                              # gmm 内:再 token 路
        gpl = _a2a_raw(p2ind.grad, stO.recv_l, stO.send_l)
        torch.autograd.backward([p1g, p1p], grad_tensors=[gpl, ggr])

        # probs 梯度:legacy 落在 pdt 叶子上,overlap 落在 fp32 叶子上。唯一合法
        # 差异是 .to 反向的精确 upcast(pdt -> fp32 无损,fp32 时二者同 dtype),
        # 所以 upcast 后仍要求逐位相等 —— 零容差,不是近似比较。
        bwd = {
            "hidden": _bitdiff(hidL.grad, hidO.grad),
            "probs": _bitdiff(probL.grad.float(), probO.grad.float()
                              if probO.grad is not None else None),
            "w13": _bitdiff(w13L.grad, w13O.grad),
            "w2": _bitdiff(w2L.grad, w2O.grad),
        }
        q.put({"rank": rank, "status": "ok", "fwd": fwd, "bwd": bwd})
    except Exception:                                      # noqa: BLE001
        import traceback
        q.put({"rank": rank, "status": "err", "trace": traceback.format_exc()})
    finally:
        dist.destroy_process_group()


def _spawn(payload_dtype_name):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_run, args=(r, WORLD, q, payload_dtype_name))
             for r in range(WORLD)]
    for p in procs:
        p.start()
    out = [q.get(timeout=240) for _ in range(WORLD)]
    for p in procs:
        p.join(timeout=60)
    for r in out:
        assert r["status"] == "ok", f"rank {r['rank']}:\n{r.get('trace')}"
    return out


@pytest.fixture(scope="module")
def parity_fp32():
    return _spawn("float32")


@pytest.fixture(scope="module")
def parity_bf16():
    return _spawn("bfloat16")


@pytest.mark.timeout(300)
def test_fp32_forward_bitwise_equal(parity_fp32):
    """fp32 均匀精度:同一串运算换调用边界,前向必须逐位相等。"""
    for r in parity_fp32:
        for name, d in r["fwd"].items():
            assert d is None, f"rank {r['rank']}: 前向 {name} 不逐位相等 {d}"


@pytest.mark.timeout(300)
def test_fp32_all_grads_bitwise_equal(parity_fp32):
    """fp32 均匀精度:厂商编排复刻出的四路梯度与 autograd 逐位相等。"""
    for r in parity_fp32:
        for name, d in r["bwd"].items():
            assert d is None, f"rank {r['rank']}: 梯度 {name} 不逐位相等 {d}"


@pytest.mark.timeout(300)
def test_bed_dtype_forward_bitwise_equal(parity_bf16):
    """床口径(bf16 载荷 + fp32 probs):gate 平面在同一点圆整后前向逐位相等。

    修复前的失效形态(锚定,防回归):pprobs 是 fp32(dtype 就不同),expert_out
    被乘法提升到 fp32,combine 全程 fp32,out 与 legacy 差 1e-2 量级 —— 正是
    eqov 床 1.38e-4@100 漂移的注入源。
    """
    for r in parity_bf16:
        for name, d in r["fwd"].items():
            assert d is None, f"rank {r['rank']}: 前向 {name} 不逐位相等 {d}"


@pytest.mark.timeout(300)
def test_bed_dtype_all_grads_bitwise_equal(parity_bf16):
    """床口径:hidden/w13/w2 逐位相等;probs 梯度 upcast 后逐位相等(见 _run 注释)。"""
    for r in parity_bf16:
        for name, d in r["bwd"].items():
            assert d is None, f"rank {r['rank']}: 梯度 {name} 不逐位相等 {d}"
