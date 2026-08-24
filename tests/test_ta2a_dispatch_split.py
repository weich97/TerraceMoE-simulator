"""拆半必须与闭合前向逐元素相等 —— 否则接进厂商 MoE 层的是另一个模型。

Why this file exists(#18(a)):`ta2a_moe_forward` 是一个闭合前向(调度+专家+回收),
已被 tests/test_ta2a.py 锁死等于 `ep_moe_forward`。要挂进 Megatron 的 MoE 层,必须把它
按专家计算切成 `token_permutation` / `token_unpermutation` 两半,中间夹厂商自己的
grouped GEMM。切开的地方正是最容易出错的地方:

- 中间量(u_src / r_idx / order / 四组 split 计数)要跨两次调用传递,漏一个就静默错行;
- **gate 施加点会变**:闭合前向在专家处自己乘,厂商则在 experts.py:241 用
  permuted_probs 乘 —— 照搬就会乘两次,而两次 gate 的输出依然"看起来像个正常的 loss";
- 返回路径的两对 split 计数是**反着**用的(ir,is / recv,send),写顺了方向不报错,
  只是把行送到别的 rank。

所以本文件不测"能不能跑",测的是**拆半后组合起来是否等于原前向**。
gloo / CPU,world 4、rpn 2(=2 节点,有跨节点跳),跑得起单元测试。
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


def _run(rank, world, q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from terrace.layer import grouped_mm
        from terrace.ta2a_fwd import ta2a_moe_forward, init_ta2a_groups
        from terrace.ta2a_dispatch import ta2a_permute, ta2a_unpermute

        intra = init_ta2a_groups(world, RPN)
        epr = E // world
        n_nodes, per, quota = world // RPN, E // (world // RPN), K // M

        g = torch.Generator().manual_seed(11 + rank)
        x = torch.randn(T, H, generator=g, dtype=torch.float32)
        idx = _routing(g, n_nodes, per, quota)
        gates = torch.rand(T, K, generator=g, dtype=torch.float32)
        w13 = torch.randn(epr, H, 2 * D, generator=g, dtype=torch.float32) / (H ** 0.5)
        w2 = torch.randn(epr, D, H, generator=g, dtype=torch.float32) / (D ** 0.5)

        # 参照:闭合前向(已被 test_ta2a.py 锁死 == ep_moe_forward)
        ref = ta2a_moe_forward(x, idx, gates, w13, w2, world, E, RPN, groups_m=M)

        # 被测:厂商形状的稠密 routing_map + probs -> 两半 + 中间夹"厂商的"专家计算
        routing_map = torch.zeros(T, E, dtype=torch.bool)
        probs = torch.zeros(T, E, dtype=torch.float32)
        routing_map[torch.arange(T).unsqueeze(1), idx] = True
        probs[torch.arange(T).unsqueeze(1), idx] = gates

        permuted, tpe, pprobs, st = ta2a_permute(
            x, probs, routing_map, world=world, rank=rank, rpn=RPN,
            n_experts=E, intra_group=intra, groups_m=M)
        # 厂商 experts.py 的等价物:先按 permuted_probs 加权,再做 grouped GEMM
        a, b = grouped_mm(permuted, w13, tpe).chunk(2, dim=-1)
        ye = grouped_mm(F.silu(a) * b, w2, tpe) * pprobs.unsqueeze(-1)
        got = ta2a_unpermute(ye, st, x)

        q.put((rank, "ok", float((got - ref).abs().max()), int(tpe.sum()), tpe.tolist()))
    except Exception as e:                                    # noqa: BLE001
        import traceback
        q.put((rank, "err", traceback.format_exc(), 0, []))
    finally:
        dist.destroy_process_group()


def test_split_halves_equal_the_closed_forward():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_run, args=(r, WORLD, q)) for r in range(WORLD)]
    for p in procs:
        p.start()
    out = [q.get(timeout=180) for _ in range(WORLD)]
    for p in procs:
        p.join(timeout=60)

    for rank, status, payload, total, tpe in out:
        assert status == "ok", f"rank {rank}:\n{payload}"
        # float32 下拆半只是把同一串运算换了个调用边界,不该有可见误差
        assert payload < 1e-5, f"rank {rank} 与闭合前向差 {payload}"
        # tokens_per_expert 是数学必然量:T-A2A 不改「哪个专家处理哪些对」,
        # 全局所有 rank 的行数之和必须等于 (token, expert) 对总数 = world * T * K / world
        assert sum(tpe) == total


def test_total_pairs_are_conserved():
    """所有 rank 的 tokens_per_expert 加起来 == 全局 (token, expert) 对总数。

    这是"接对了没有"的必然量探针:少了就是丢了专家(快路径静默丢专家是审计 1.4 的原案),
    多了就是把某些对复制了(早期 combine 按 (token,expert) 累加的 bug)。
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_run, args=(r, WORLD, q)) for r in range(WORLD)]
    for p in procs:
        p.start()
    out = [q.get(timeout=180) for _ in range(WORLD)]
    for p in procs:
        p.join(timeout=60)
    assert all(s == "ok" for _, s, _, _, _ in out), [p for _, s, p, _, _ in out if s != "ok"]
    assert sum(t for _, _, _, t, _ in out) == WORLD * T * K
