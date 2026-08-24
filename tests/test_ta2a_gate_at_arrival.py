"""Step 2(BACKWARD-PLAN):门控在**到达 rank** 施加,必须与「专家 rank 施加」逐位相等。

Why this file exists(2026-08-20 实施 Step 2 时的回归锁):
`ta2a_moe_forward` 自 2026-08-20 起不再把每对 (row, slot) 的门控经节点内交换送到
专家 rank(旧 `my_gate = _a2a(...)`),而是留在到达 rank,在专家结果返程落地时乘
(`red.index_add_(0, r_idx, ret * gate_pairs)`)。变换的合法性论证只有一句话:
逐元素乘法与数据搬运可交换 —— `ye * gate`(专家侧)与 `ret * gate`(到达侧)是
同一批操作数的同一次乘法,交换只移动行。**这句话必须永远被测试钉住**,否则未来
任何触碰配对顺序(ordo 重排、r_idx/slot_idx 的来源)的改动都可能让 gate 对错行,
而错行的前向依然形状正确、loss 依然像个正常的 loss。

对照臂就是「改前公式」本身:legacy 拆半接缝(ta2a_permute → 厂商式 `* pprobs` →
ta2a_unpermute)在专家 rank 施加门控 —— 与 2026-08-20 之前的融合前向逐字节同构。
所以本文件同时也把「融合前向 == 拆半组合」从 1e-5 容差(test_ta2a_dispatch_split)
收紧到零容差、且盖到全部四路梯度与 bf16。

三组断言,全部 torch.equal(逐位,不设容差):
  1. 前向:两臂 y 逐位相等(fp32 与 bf16;groups_m=M 的 topk 支路与 =None 的
     nonzero 支路都要 —— 两支路的配对顺序不同,门控对位都不许错);
  2. 四路梯度:x / gates / w13 / w2 逐位相等(gates 对照取稠密 probs 叶子在
     路由位置上的 gather,gather 反向只散不并,数值不变);
  3. 集合通信计数:融合前向 forward 9 次、backward 5 次;对照臂 forward **8** 次、
     backward 6 次 —— Step 2 的省工是融合臂那一进一出各一次,对照臂的 10 -> 8 是
     A1''/A2 并包(逐项账见下方 SEAM_FWD_A2A 的注释)。两个方向都钉
     精确值:回涨 = 优化被静默回退,再降 = 有交换被静默丢弃
     (BACKWARD-PLAN Step 2 的 Measure 一项要求此断言)。

**并包之后本文件的证据地位更强了**:融合前向臂(ta2a_fwd)一字未动、三条平面各走
各的;拆半接缝臂走 A1/A2 并包。第 1、2 组断言因此同时是「并包 == 未并包」的逐位
证明 —— 前向与四路梯度,fp32 与 bf16,quota 支路与通用支路各一遍。

历史口径,防止再混淆:被 2026-08-01 实测否决并回滚的是 **torch.cat 门控并包**
(一次内部提交 → 一次内部提交,整载荷复制是慢的物理原因;BACKWARD-PLAN「Do not re-propose」
第一条)。Step 2 与它不同构:零复制、零新增张量,只删一对交换。2026-08-13 文档写的
「Step 2 已被实测否决」张冠李戴,Step 2 在 2026-08-20 之前从未被实施或计时过。
2026-08-21 的 A1′/A2 并包与 08-01 那次也不同构:①不用 torch.cat,收侧解包回**连续**
张量,下游 gather 见到的形状与并包前逐字节相同(08-01 慢的物理原因是下游吃非连续
切片,不是并包本身);②省的是 α(条数),不是 β(字节)—— α₁₂₈=0.45 ms 的拆分
是 08-20/21 才测出来的,08-01 那次没有这个账本。

gloo / CPU,world 4、rpn 2(2 节点,有真实跨节点跳)。
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

# Step 2 之后融合前向的集合通信条数(融合前向**未并包**,是本文件的不动参照臂)。
FUSED_FWD_A2A, FUSED_BWD_A2A = 9, 5

# 拆半接缝(ta2a_dispatch)的条数。**2026-08-22 起默认形态是 A1''(small),10/6 -> 8/6**。
#
# 历史:2026-08-21 的 A1′(full)是 10/6 -> 7/6,估值 −0.85 ms/次。判决床实测把它
# 推翻了:dispatch 14.222 vs 未并包 12.171,**净亏 2.05 ms/次**
# (内部实测记录)。反推出打包/解包那两趟载荷 HBM 拷贝的真实
# 成本 ≈ 3.0 ms,而当初估的是 0.106 ms —— 差 28 倍。
#
# small 形态的逐项账(前向,判决床几何):
#   Hop A  4 -> 3:counts + payload(自己一条)+ [id‖gate](省 1 条,α₁₂₈=0.450 ms;
#          容器 32 B/行 而不是 4128 B/行,拷贝可忽略 ⇒ 净 ≈ −0.43 ms)
#   Hop B  4 -> 3:counts + exp_rx + [slot‖gate](省 1 条,α₈=0.058 ms;净 +0.056 ms)
# full 形态(TERRACE_TA2A_PACK=full)仍可跑,条数是 7/6,保留只为复现那次读数与 A/B。
#          载荷面 exp_rx 有意不并:并它要多付 0.317 ms 拷贝去省 0.116 ms,净亏
#          0.20 ms —— 与 2026-08-01 torch.cat 并包翻车同一条物理(算式见 ta2a_pack)。
#   combine 2 -> 2:不动(本就只有 2 条,实测相位与厂商打平)
# 反向 6 -> 6 **有意不变**:两条 float 路各挂一条独立的边(terrace/ta2a_pack.py
# ::_PackedEdge),反向仍是并包前的同几句 _a2a_raw —— 融合成一枚节点会撞厂商 gmm
# 对 permute2 段的两次 .backward()(内部工程记录 2026-08-20),而 overlap 接缝的 Hop A
# 反向本就由厂商手工重放、我们改不了。所以反向条数不许降,也不许涨。
# 断言仍然钉**精确值**(不是 <=):计数回涨 = 并包被静默回退,计数再降 = 有交换被
# 静默丢弃,两个方向都必须当场红。
SEAM_FWD_A2A, SEAM_BWD_A2A = 8, 6


def _routing(gen, n_nodes, per, quota):
    """T-Route 等额配额:恰好 M 个节点,每个节点恰好 K/M 个专家。

    每行升序排序:routing_map_to_topk 用 nonzero 恢复索引,天然升序 —— 两臂必须
    喂进**同一**[T, K] 排列,否则输入就不逐位相同,断言测的是别的东西。
    """
    rows = []
    for _ in range(T):
        gs = torch.randperm(n_nodes, generator=gen)[:M]
        rows.append(torch.cat([torch.arange(a * per, (a + 1) * per)[
            torch.randperm(per, generator=gen)[:quota]] for a in gs]))
    return torch.sort(torch.stack(rows), dim=1).values


def _bitdiff(a, b):
    """None = 逐位相等;否则 (说明, max|Δ|) 供报错定位。"""
    if a is None or b is None:
        return ("missing", None)
    if a.dtype != b.dtype:
        return (f"{a.dtype} vs {b.dtype}", float((a.float() - b.float()).abs().max()))
    if torch.equal(a, b):
        return None
    return (str(a.dtype), float((a.float() - b.float()).abs().max()))


def _run(rank, world, q, payload_dtype_name):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # 计数黄金值量的是**出厂配置**下的条数,所以这里显式清掉并包闸门的环境覆盖:
    # 操作员在集群上用 TERRACE_TA2A_PACK=0 做 A/B 时,本文件不该跟着变红
    # (并包开/关两条路径的逐位等价由 tests/test_ta2a_pack_bitparity.py 单独把守)。
    os.environ.pop("TERRACE_TA2A_PACK", None)
    # 已占用:29577/29591/29613/29623/29627。本床两个 dtype 各一个。
    os.environ.setdefault(
        "MASTER_PORT", "29641" if payload_dtype_name == "float32" else "29645")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from terrace.layer import grouped_mm
        from terrace.ta2a_fwd import ta2a_moe_forward, init_ta2a_groups
        from terrace.ta2a_dispatch import ta2a_permute, ta2a_unpermute

        pdt = getattr(torch, payload_dtype_name)
        intra = init_ta2a_groups(world, RPN)
        epr = E // world
        n_nodes, per, quota = world // RPN, E // (world // RPN), K // M

        # 计数器:两臂各自的 all_to_all_single 条数(forward / backward 分开)。
        # terrace.ep_dist / ta2a_fwd / ta2a_dispatch 持有的是同一个模块对象,
        # 改模块属性对所有调用点生效。
        counter = [0]
        real_a2a = dist.all_to_all_single

        def counting_a2a(*args, **kwargs):
            counter[0] += 1
            return real_a2a(*args, **kwargs)

        report = {}
        for gm in (M, None):
            g = torch.Generator().manual_seed(11 + rank)
            x0 = torch.randn(T, H, generator=g).to(pdt)
            idx = _routing(g, n_nodes, per, quota)
            gates0 = torch.rand(T, K, generator=g).to(pdt)
            w13_0 = (torch.randn(epr, H, 2 * D, generator=g) / (H ** 0.5)).to(pdt)
            w2_0 = (torch.randn(epr, D, H, generator=g) / (D ** 0.5)).to(pdt)
            G = torch.randn(T, H, generator=g).to(pdt)

            # ---- 新臂:融合前向,门控在到达 rank 施加(Step 2 之后的代码)----
            xN = x0.clone().requires_grad_(True)
            gN = gates0.clone().requires_grad_(True)
            w13N = w13_0.clone().requires_grad_(True)
            w2N = w2_0.clone().requires_grad_(True)
            dist.all_to_all_single = counting_a2a
            try:
                counter[0] = 0
                yN = ta2a_moe_forward(xN, idx, gN, w13N, w2N, world, E, RPN,
                                      groups_m=gm)
                fused_fwd = counter[0]
                counter[0] = 0
                yN.backward(G)
                fused_bwd = counter[0]
            finally:
                dist.all_to_all_single = real_a2a

            # ---- 对照臂 = 改前公式:legacy 拆半,门控经 my_gate 交换到专家 rank,
            # 由「厂商」(这里内联其 experts.py 等价物)在专家侧乘 ----
            routing_map = torch.zeros(T, E, dtype=torch.bool)
            routing_map[torch.arange(T).unsqueeze(1), idx] = True
            probs_dense = torch.zeros(T, E, dtype=pdt)
            probs_dense[torch.arange(T).unsqueeze(1), idx] = gates0

            xR = x0.clone().requires_grad_(True)
            pR = probs_dense.clone().requires_grad_(True)
            w13R = w13_0.clone().requires_grad_(True)
            w2R = w2_0.clone().requires_grad_(True)
            dist.all_to_all_single = counting_a2a
            try:
                counter[0] = 0
                perm, tpe, pprobs, st = ta2a_permute(
                    xR, pR, routing_map, world=world, rank=rank, rpn=RPN,
                    n_experts=E, intra_group=intra, inter_group=None, groups_m=gm)
                a, b = grouped_mm(perm, w13R, tpe).chunk(2, dim=-1)
                ye = grouped_mm(F.silu(a) * b, w2R, tpe) * pprobs.unsqueeze(-1)
                yR = ta2a_unpermute(ye, st, xR)
                seam_fwd = counter[0]
                counter[0] = 0
                yR.backward(G)
                seam_bwd = counter[0]
            finally:
                dist.all_to_all_single = real_a2a

            tag = f"gm={gm}"
            report[tag] = {
                "y": _bitdiff(yN.detach(), yR.detach()),
                "x": _bitdiff(xN.grad, xR.grad),
                # 稠密 probs 叶子的 gather 反向只把 [T,K] 梯度散到路由位置,数值
                # 不变,所以在路由位置上取回即可与融合臂的 gates 梯度逐位对账。
                "gates": _bitdiff(gN.grad, None if pR.grad is None
                                  else pR.grad.gather(1, idx)),
                "w13": _bitdiff(w13N.grad, w13R.grad),
                "w2": _bitdiff(w2N.grad, w2R.grad),
                "counts": (fused_fwd, fused_bwd, seam_fwd, seam_bwd),
            }
        q.put({"rank": rank, "status": "ok", "report": report})
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
def arrival_fp32():
    return _spawn("float32")


@pytest.fixture(scope="module")
def arrival_bf16():
    return _spawn("bfloat16")


def _assert_bitwise(results, dtype_label):
    for r in results:
        for tag, rep in r["report"].items():
            for name in ("y", "x", "gates", "w13", "w2"):
                assert rep[name] is None, (
                    f"rank {r['rank']} {dtype_label} {tag}: {name} 到达侧施门与"
                    f"专家侧施门不逐位相等 {rep[name]} —— 配对对位被破坏")


def _assert_counts(results, dtype_label):
    for r in results:
        for tag, rep in r["report"].items():
            ff, fb, sf, sb = rep["counts"]
            assert (ff, fb) == (FUSED_FWD_A2A, FUSED_BWD_A2A), (
                f"rank {r['rank']} {dtype_label} {tag}: 融合前向集合通信数 "
                f"fwd={ff}/bwd={fb},应为 {FUSED_FWD_A2A}/{FUSED_BWD_A2A} —— "
                f"多了即 Step 2 被静默回退,少了即有交换被静默丢弃")
            assert (sf, sb) == (SEAM_FWD_A2A, SEAM_BWD_A2A), (
                f"rank {r['rank']} {dtype_label} {tag}: 对照臂集合通信数 "
                f"fwd={sf}/bwd={sb},应为 {SEAM_FWD_A2A}/{SEAM_BWD_A2A} —— "
                f"多了即 A1′/A2 并包被静默回退(或 C1/Step 2 形态变了),"
                f"少了即有交换被静默丢弃")


@pytest.mark.timeout(300)
def test_fp32_gate_position_bitwise_equal(arrival_fp32):
    """fp32:到达 rank 施门的前向与四路梯度 == 专家 rank 施门,逐位。"""
    _assert_bitwise(arrival_fp32, "fp32")


@pytest.mark.timeout(300)
def test_bf16_gate_position_bitwise_equal(arrival_bf16):
    """bf16(生产载荷精度):同上,逐位。"""
    _assert_bitwise(arrival_bf16, "bf16")


@pytest.mark.timeout(300)
def test_fp32_collective_counts(arrival_fp32):
    """省工被计数钉住:融合前向 9/5(未并包),拆半接缝 8/6(A1''/A2 并包后)。"""
    _assert_counts(arrival_fp32, "fp32")


@pytest.mark.timeout(300)
def test_bf16_collective_counts(arrival_bf16):
    _assert_counts(arrival_bf16, "bf16")
