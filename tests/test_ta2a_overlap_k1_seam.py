"""K1 接进 overlap 6 参接缝(permute2 段图内部)的位级契约与接线证明。

Why this file exists(字节审计 2026-08-20 的头号刀):K1 kernel 早已存在,legacy 3 参
接缝与融合前向都接了,但**判决床走的是 overlap 6 参接缝**,那一处此前有意未接 ——
on 臂 dispatch 余肉 6.2–8.8ms/调用的大头正是这段到达展开(256 MiB 名义内存流量 +
十余算子)。接进去的难点不是数学(与 legacy 分支同一枚 kernel、同一套数学),而是
**段图**:

  - 厂商 gmm 的手写 backward 对 permute2_graph 与 permute2_prob_graph 分**两次**
    .backward() 进同一段。K1 是一枚同时产出两路的融合节点,直接用会
    (a) 第二次撞 "backward through the graph a second time"(没有 retain_graph 可给),
    (b) 第一次把 materialize 出的零梯度先写进另一路的 .grad。
    所以接入点走 terrace.ops.k1_arrival_segment:kernel 在图外跑一次出数据,
    两个 float 输出各挂一条**独立**的边回自己的 detach 叶 —— 与现链两条互不相交的
    gather 子图同构。
  - 席位契约(7+3)、detach 边界、splits 交接、厂商手写反向可见的一切不许变。

本文件锁四层,全部 torch.equal(逐位),不设容差:
  1. 段图版包装器 k1_arrival_segment 的数据面 == 现链原文(多几何 x 多 dtype);
  2. 两条边确实**独立**:按厂商顺序分两次 .backward() 不炸,且两路梯度与现链
     autograd(同样分两次)逐位相等;整数平面(r_idx/slot_idx/i_send)不带梯度;
  3. 接缝级:overlap 接缝强制走 K1 后,前向 + 四路叶子梯度 + 厂商要读的四个
     detach 叶 .grad,与**未强制**的同进程基线趟逐位相等(= 改前 vs 改后回归;
     基线趟就是 K1 前的现链原文);同一趟里 overlap 臂与 legacy 臂逐位相等
     (= 跨接缝 bitparity 在 K1 打开后仍成立);
  4. 活路径证据:强制趟里 overlap 入口的段图版恰被调 1 次、kernel 恰被调 2 次
     (legacy 接缝 1 + overlap 段图 1),基线趟两个计数都必须为 0 —— 等价断言对
     「闸门从未打开」是盲的(内部工程记录)。

gloo / CPU,world 4、rpn 2(2 节点,有真实跨节点跳);两个床:fp32 均匀精度与
床口径(bf16 载荷/权重 + fp32 router probs)。
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

import terrace.ops as tops  # noqa: E402
from terrace.ta2a_fwd import (_expand_arrival_quota,  # noqa: E402
                              _stable_argsort_small)

WORLD, RPN, T, K, M, E, H, D = 4, 2, 8, 4, 2, 8, 6, 4

NAMES = ("send_buf", "gate_pairs", "r_idx", "slot_idx", "i_send")


def _chain(rx, rslot, rgate, quota, epr, rpn):
    """现组合链原文(接入点 else 分支的逐行照抄)—— K1 的功能规格。"""
    r_idx, slot_idx = _expand_arrival_quota(rslot)
    owner = slot_idx // epr
    ordo = _stable_argsort_small(owner, rpn)
    r_idx, slot_idx = r_idx[ordo], slot_idx[ordo]
    i_send = torch.bincount(owner, minlength=rpn)
    return rx[r_idx], rgate.reshape(-1)[ordo], r_idx, slot_idx, i_send


def _mk(R, quota, epr, rpn, Hh, dtype, seed):
    """C1 线格式的到达面:每行升序不重复槽号(_pack_quota_wire 的构造)。"""
    g = torch.Generator().manual_seed(seed)
    slots = epr * rpn
    assert quota <= slots
    scores = torch.rand(R, slots, generator=g)
    rslot = torch.sort(torch.topk(scores, quota, dim=1).indices,
                       dim=1).values.to(torch.int64)
    rx = torch.randn(R, Hh, generator=g).to(dtype)
    rgate = torch.rand(R, quota, generator=g).to(dtype)
    return rx, rslot, rgate


# 几何覆盖同 test_terrace_k1_arrival 的口径(含对齐床 slots=16 与退化 quota=1)。
GEOMS = [
    (16, 2, 2, 8, 64),     # 对齐床几何(slots 16)
    (64, 1, 2, 8, 32),     # quota=1:r_idx == ordo 的退化
    (33, 3, 2, 4, 48),     # 非 2 幂 R/quota
    (7, 2, 8, 4, 96),      # epr>rpn
    (40, 5, 3, 4, 40),     # quota 不整除 slots(slots 12)
]


# ======================================================================================
# 1. 段图版包装器的数据面 == 现链原文
# ======================================================================================

@pytest.mark.parametrize("geom", GEOMS, ids=lambda g: f"R{g[0]}q{g[1]}e{g[2]}r{g[3]}")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_segment_forward_bitwise_equals_chain(geom, dtype):
    """k1_arrival_segment 只是给 k1_arrival 换了挂图方式,数据必须逐位不变。"""
    R, quota, epr, rpn, Hh = geom
    rx, rslot, rgate = _mk(R, quota, epr, rpn, Hh, dtype, 300)
    want = _chain(rx, rslot, rgate, quota, epr, rpn)
    got = tops.k1_arrival_segment(rx, rslot, rgate, quota, epr, rpn, my_local=1)
    for name, w, g in zip(NAMES, want, got):
        assert w.dtype == g.dtype and w.shape == g.shape, name
        assert torch.equal(w, g), f"段图版 {name} 与现链不逐位相等"


def test_segment_integer_planes_carry_no_grad():
    """r_idx/slot_idx/i_send 是整数索引/计数平面:不挂图、不被 materialize 出假梯度。"""
    R, quota, epr, rpn, Hh = 16, 2, 2, 8, 32
    rx, rslot, rgate = _mk(R, quota, epr, rpn, Hh, torch.float32, 301)
    rx.requires_grad_(True)
    rgate.requires_grad_(True)
    send_buf, gate_pairs, r_idx, slot_idx, i_send = tops.k1_arrival_segment(
        rx, rslot, rgate, quota, epr, rpn)
    assert send_buf.grad_fn is not None and gate_pairs.grad_fn is not None, \
        "两条边没挂上图 —— 厂商 backward_func 见 grad_fn is None 会直接 return"
    assert send_buf.grad_fn is not gate_pairs.grad_fn, \
        "两路共用一个节点 = 厂商两次 .backward() 会撞车"
    for name, t in zip(NAMES[2:], (r_idx, slot_idx, i_send)):
        assert not t.requires_grad and t.grad_fn is None, name
        assert t.dtype == torch.int64, name


# ======================================================================================
# 2. 两条边独立:厂商顺序的两次 .backward() 不炸,且逐位等于现链
# ======================================================================================

@pytest.mark.parametrize("geom", GEOMS[:3], ids=lambda g: f"R{g[0]}q{g[1]}")
def test_segment_split_adjoints_survive_two_backwards(geom):
    """厂商 gmm 先 prob 路、后 token 路,各一次 .backward():段图版必须两次都活,
    且两路梯度与现链 autograd(同样分两次)逐位相等。

    这是本件的核心失效形态锚点:直接用融合的 TerraceK1ArrivalFn 会在第二次
    .backward() 抛 "Trying to backward through the graph a second time"。
    """
    R, quota, epr, rpn, Hh = geom
    rx0, rslot, rgate0 = _mk(R, quota, epr, rpn, Hh, torch.float32, 302)
    g = torch.Generator().manual_seed(7)
    gs = torch.randn(R * quota, Hh, generator=g)
    gg = torch.randn(R * quota, generator=g)

    # 现链参照:两条 gather 子图互不相交,分两次反向本就成立
    rxC, rgC = rx0.clone().requires_grad_(True), rgate0.clone().requires_grad_(True)
    sC, pC, *_ = _chain(rxC, rslot, rgC, quota, epr, rpn)
    (pC * gg).sum().backward()
    (sC * gs).sum().backward()

    rxS, rgS = rx0.clone().requires_grad_(True), rgate0.clone().requires_grad_(True)
    sS, pS, *_ = tops.k1_arrival_segment(rxS, rslot, rgS, quota, epr, rpn)
    (pS * gg).sum().backward()          # 厂商顺序:先 prob 路
    assert rxS.grad is None, "gate 路的反向不许往 token 路的叶子写(零梯度也不行)"
    (sS * gs).sum().backward()          # 再 token 路 —— 融合节点会死在这一行

    assert torch.equal(rxS.grad, rxC.grad), "token 路梯度与现链不逐位相等"
    assert torch.equal(rgS.grad, rgC.grad), "gate 路梯度与现链不逐位相等"


def test_segment_backward_bitwise_in_bed_dtype():
    """床口径 dtype(bf16 载荷)下两路梯度同样逐位 —— index_add_ 与 index 伴随
    在 bf16 上也必须同一归约。"""
    R, quota, epr, rpn, Hh = 16, 2, 2, 8, 6
    rx0, rslot, rgate0 = _mk(R, quota, epr, rpn, Hh, torch.bfloat16, 303)
    g = torch.Generator().manual_seed(8)
    gs = torch.randn(R * quota, Hh, generator=g).to(torch.bfloat16)
    gg = torch.randn(R * quota, generator=g).to(torch.bfloat16)

    rxC, rgC = rx0.clone().requires_grad_(True), rgate0.clone().requires_grad_(True)
    sC, pC, *_ = _chain(rxC, rslot, rgC, quota, epr, rpn)
    (pC.float() * gg.float()).sum().backward()
    (sC.float() * gs.float()).sum().backward()

    rxS, rgS = rx0.clone().requires_grad_(True), rgate0.clone().requires_grad_(True)
    sS, pS, *_ = tops.k1_arrival_segment(rxS, rslot, rgS, quota, epr, rpn)
    (pS.float() * gg.float()).sum().backward()
    (sS.float() * gs.float()).sum().backward()

    assert torch.equal(rxS.grad, rxC.grad)
    assert torch.equal(rgS.grad, rgC.grad)


# ======================================================================================
# 3+4. 接缝级:强制 K1 的 overlap 接缝 == 基线趟 == 同趟 legacy 臂;活路径计数
# ======================================================================================

def _routing(gen, n_nodes, per, quota):
    rows = []
    for _ in range(T):
        gs = torch.randperm(n_nodes, generator=gen)[:M]
        rows.append(torch.cat([torch.arange(a * per, (a + 1) * per)[
            torch.randperm(per, generator=gen)[:quota]] for a in gs]))
    return torch.stack(rows)


def _bitdiff(a, b):
    """None = 逐位相等;否则返回 (dtype 说明, max|Δ|) 供报错定位。"""
    if a is None or b is None:
        return ("missing", None)
    if a.dtype != b.dtype:
        return (f"{a.dtype} vs {b.dtype}", float((a.float() - b.float()).abs().max()))
    if torch.equal(a, b):
        return None
    return (str(a.dtype), float((a.float() - b.float()).abs().max()))


def _run(rank, world, q, dtype_name):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # 已占用:29577/29591/29613/29623/29627/29641/29645/29661/29665/29677。
    os.environ.setdefault(
        "MASTER_PORT", "29681" if dtype_name == "float32" else "29685")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        import terrace.ops as ops_mod
        from terrace.layer import grouped_mm
        from terrace.ep_dist import _a2a_raw
        from terrace.ta2a_fwd import init_ta2a_groups
        from terrace.ta2a_dispatch import (ta2a_permute, ta2a_unpermute,
                                           ta2a_permute_overlap,
                                           ta2a_unpermute_overlap)

        pdt = getattr(torch, dtype_name)
        intra = init_ta2a_groups(world, RPN)
        epr = E // world
        n_nodes, per, quota = world // RPN, E // (world // RPN), K // M

        counts = {"k1": 0, "seg": 0}
        in_seg = [False]
        real_enabled = ops_mod.custom_ops_enabled
        real_k1 = ops_mod.k1_arrival
        real_seg = ops_mod.k1_arrival_segment

        def fake_k1(rx, rslot, rgate, quota_, epr_, rpn_, my_local=0):
            # 接线断言:接入点交给 kernel 的几何必须自洽(参数错位在等价断言里
            # 可能显现为对的 —— 比如 epr/rpn 同值几何 —— 这里直接钉死)。
            assert rx.dim() == 2 and rslot.shape == rgate.shape
            assert rslot.shape[1] == quota_ == quota
            assert epr_ == epr and rpn_ == RPN
            assert my_local == rank % RPN
            assert rslot.dtype == torch.int64 and rgate.dtype == rx.dtype
            if in_seg[0]:
                # 段图版必须在**图外**取数据:kernel 输出自带 grad_fn 就等于又把两路
                # 焊回一个节点,两条独立的边白挂。
                assert not torch.is_grad_enabled()
                assert not rx.requires_grad and not rgate.requires_grad
                assert rx.grad_fn is None and rgate.grad_fn is None
            counts["k1"] += 1
            return ops_mod.k1_arrival_ref(rx, rslot, rgate, quota_, epr_, rpn_,
                                          my_local)

        def counting_seg(rx, rslot, rgate, quota_, epr_, rpn_, my_local=0):
            # 段图版的入参必须就是 permute2 的两个 detach 叶(席位 4 与席位 6)。
            assert rx.requires_grad and rgate.requires_grad
            assert rx.grad_fn is None and rgate.grad_fn is None
            counts["seg"] += 1
            in_seg[0] = True
            try:
                return real_seg(rx, rslot, rgate, quota_, epr_, rpn_, my_local)
            finally:
                in_seg[0] = False

        def one_pass(forced):
            if forced:
                ops_mod.custom_ops_enabled = lambda: True
                ops_mod.k1_arrival = fake_k1
                ops_mod.k1_arrival_segment = counting_seg
            try:
                g = torch.Generator().manual_seed(41 + rank)
                x0 = torch.randn(T, H, generator=g).to(pdt)
                idx = _routing(g, n_nodes, per, quota)
                gates0 = torch.rand(T, K, generator=g)       # fp32:router 输出精度
                w13_0 = (torch.randn(epr, H, 2 * D, generator=g) / (H ** 0.5)).to(pdt)
                w2_0 = (torch.randn(epr, D, H, generator=g) / (D ** 0.5)).to(pdt)
                G = torch.randn(T, H, generator=g).to(pdt)   # 上游梯度
                routing_map = torch.zeros(T, E, dtype=torch.bool)
                routing_map[torch.arange(T).unsqueeze(1), idx] = True
                probs_dense = torch.zeros(T, E)
                probs_dense[torch.arange(T).unsqueeze(1), idx] = gates0

                # ---- legacy 3 参接缝(普通 autograd);probs 按厂商 legacy 层
                # 的口径转 payload dtype 后进接缝 ----
                hidL = x0.clone().requires_grad_(True)
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

                # ---- overlap 6 参接缝 + 厂商手写 backward 的逐步复刻 ----
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
                (p1g, p1p, ncpu, p2ind, p2g, p2pd, p2pg, u1ind, u1g, u2ind) = save
                seats = {
                    "n": len(save),
                    "num_cpu_is_none": ncpu is None,
                    "leaves": all(t.requires_grad and t.grad_fn is None
                                  for t in (p2ind, p2pd, u1ind, u2ind)),
                    "graphs": all(t.grad_fn is not None
                                  for t in (p1g, p1p, p2g, p2pg, u1g)),
                }

                outO.backward(G)                            # unpermute2 段
                grad_red = _a2a_raw(u2ind.grad, stO.send_l, stO.recv_l)
                u1g.backward(grad_red)                      # unpermute1 段
                eoO.backward(u1ind.grad)                    # experts 等价物
                p2pg.backward(pI.grad)                      # gmm 内:先 prob 路
                _ggr = _a2a_raw(p2pd.grad, stO.recv_l, stO.send_l)
                p2g.backward(dI.grad)                       # gmm 内:再 token 路
                _gpl = _a2a_raw(p2ind.grad, stO.recv_l, stO.send_l)
                torch.autograd.backward([p1g, p1p], grad_tensors=[_gpl, _ggr])

                return {
                    "L.perm": permL.detach(), "L.pp": ppL.detach(),
                    "L.tpe": tpeL, "L.out": outL.detach(),
                    "L.gx": hidL.grad, "L.gp": probL.grad,
                    "L.gw13": w13L.grad, "L.gw2": w2L.grad,
                    "O.perm": permO.detach(), "O.pp": ppO.detach(),
                    "O.tpe": tpeO, "O.out": outO.detach(),
                    "O.gx": hidO.grad, "O.gp": probO.grad,
                    "O.gw13": w13O.grad, "O.gw2": w2O.grad,
                    # 厂商手工重放真正读的四个 detach 叶 —— K1 两条边的直接产物
                    "O.p2in": p2ind.grad, "O.p2prob": p2pd.grad,
                    "O.u1in": u1ind.grad, "O.u2in": u2ind.grad,
                }, seats
            finally:
                ops_mod.custom_ops_enabled = real_enabled
                ops_mod.k1_arrival = real_k1
                ops_mod.k1_arrival_segment = real_seg

        base, _base_seats = one_pass(forced=False)
        counts_baseline = dict(counts)
        forced, seats = one_pass(forced=True)

        regress = {name: _bitdiff(base[name], forced[name]) for name in base}
        cross = {}
        for a_name, b_name in (("O.perm", "L.perm"), ("O.pp", "L.pp"),
                               ("O.tpe", "L.tpe"), ("O.out", "L.out"),
                               ("O.gx", "L.gx"), ("O.gw13", "L.gw13"),
                               ("O.gw2", "L.gw2")):
            cross[a_name] = _bitdiff(forced[a_name], forced[b_name])
        # probs 梯度:legacy 落 pdt 叶子、overlap 落 fp32 叶子,唯一合法差异是 .to
        # 反向的精确 upcast(无损),所以 upcast 后仍要求逐位相等 —— 零容差。
        cross["O.gp"] = _bitdiff(forced["O.gp"].float(), forced["L.gp"].float())

        q.put({"rank": rank, "status": "ok", "regress": regress, "cross": cross,
               "seats": seats, "counts": counts, "counts_baseline": counts_baseline})
    except Exception:                                      # noqa: BLE001
        import traceback
        q.put({"rank": rank, "status": "err", "trace": traceback.format_exc()})
    finally:
        dist.destroy_process_group()


def _spawn(dtype_name):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_run, args=(r, WORLD, q, dtype_name))
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
def seam_fp32():
    return _spawn("float32")


@pytest.fixture(scope="module")
def seam_bf16():
    return _spawn("bfloat16")


@pytest.mark.timeout(300)
def test_overlap_k1_matches_pre_k1_chain_fp32(seam_fp32):
    """改前 vs 改后:强制 K1 的 overlap 接缝与同进程基线趟(= K1 前的现链原文)
    前向、四路叶子梯度、四个 detach 叶 .grad 全部逐位相等。"""
    for r in seam_fp32:
        for name, d in r["regress"].items():
            assert d is None, f"rank {r['rank']}: {name} 强制 K1 与基线趟不逐位相等 {d}"


@pytest.mark.timeout(300)
def test_overlap_k1_matches_legacy_seam_fp32(seam_fp32):
    """跨接缝 bitparity 在 K1 打开后仍成立(两条接缝同趟、同输入、同 kernel)。"""
    for r in seam_fp32:
        for name, d in r["cross"].items():
            assert d is None, f"rank {r['rank']}: {name} overlap 与 legacy 不逐位相等 {d}"


@pytest.mark.timeout(300)
def test_overlap_k1_is_live(seam_fp32):
    """活路径证据:强制趟里 overlap 入口的段图版恰 1 次、kernel 恰 2 次
    (legacy 接缝 1 + overlap 段图 1);基线趟两个计数都必须为 0。等价断言对
    『闸门从未打开』是盲的,这条不通过上面两条什么都没证明(内部工程记录)。"""
    for r in seam_fp32:
        assert r["counts_baseline"] == {"k1": 0, "seg": 0}, \
            f"rank {r['rank']}: 基线趟闸门竟然开着 {r['counts_baseline']}"
        assert r["counts"]["seg"] == 1, (
            f"rank {r['rank']}: overlap 段图版被调 {r['counts']['seg']} 次,预期 1"
            f" —— 接入点没走 K1 分支")
        assert r["counts"]["k1"] == 2, (
            f"rank {r['rank']}: kernel 被调 {r['counts']['k1']} 次,预期 2"
            f"(legacy 接缝 1 + overlap 段图 1)")


@pytest.mark.timeout(300)
def test_overlap_k1_keeps_save_tensors_contract(seam_fp32):
    """席位契约零改动:K1 打开后仍是 10 席、第 3 席 None、4 个 detach 叶、5 个带图。"""
    for r in seam_fp32:
        s = r["seats"]
        assert s["n"] == 10, f"rank {r['rank']}: save_tensors 席位数 {s['n']} != 10"
        assert s["num_cpu_is_none"], f"rank {r['rank']}: 第 3 席应为 None"
        assert s["leaves"], f"rank {r['rank']}: detach 叶子席位不是叶子"
        assert s["graphs"], f"rank {r['rank']}: 图席位缺 grad_fn(梯度会静默消失)"


@pytest.mark.timeout(300)
def test_overlap_k1_matches_pre_k1_chain_bed_dtype(seam_bf16):
    """床口径(bf16 载荷/权重 + fp32 router probs):改前 vs 改后同样逐位。"""
    for r in seam_bf16:
        for name, d in r["regress"].items():
            assert d is None, f"rank {r['rank']}: {name} 强制 K1 与基线趟不逐位相等 {d}"


@pytest.mark.timeout(300)
def test_overlap_k1_matches_legacy_seam_bed_dtype(seam_bf16):
    """床口径:跨接缝 bitparity 在 K1 打开后仍成立(gate 平面同一圆整点不被 K1 改)。"""
    for r in seam_bf16:
        for name, d in r["cross"].items():
            assert d is None, f"rank {r['rank']}: {name} overlap 与 legacy 不逐位相等 {d}"


@pytest.mark.timeout(300)
def test_overlap_k1_is_live_bed_dtype(seam_bf16):
    for r in seam_bf16:
        assert r["counts_baseline"] == {"k1": 0, "seg": 0}
        assert r["counts"] == {"k1": 2, "seg": 1}, \
            f"rank {r['rank']}: 计数 {r['counts']}"


# ======================================================================================
# 工程纪律:本文件自身 LF + py_compile
# ======================================================================================

def test_this_file_compiles_and_is_lf():
    import py_compile
    py_compile.compile(__file__, doraise=True)
    with open(__file__, "rb") as f:
        assert b"\r\n" not in f.read()
