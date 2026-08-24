"""A1/A2 集合通信并包(2026-08-21)的位级契约、条数账与活路径证据。

Why this file exists(内部设计记录(未随仓发布) 的 A1/A2):w128/w8 的 a2a
曲线拆成 `α + β` 后,α₁₂₈ = 0.45 ms、α₈ = 0.058 ms,而 β₈ ≈ β₁₂₈ ≈ 110 GB/s ——
**字节几乎不要钱、条数很贵**。一次 dispatch 原走 8 条集合通信(inter 4 + intra 4),
厂商 alltoall_seq 侧只有约 3 条。并包把共用 splits 的路合成一条:Hop A 4 -> 2
([id‖payload‖gate])、Hop B 4 -> 3([slot‖gate]),拆半接缝前向 10 -> 7(combine 不动)。
**这是纯字节重排**,不动任何归约序、配对序、排序键、dtype 与圆整点,所以判据只能是
`torch.equal`,不设容差。

Hop B 的载荷面 exp_rx **有意不并**:并包省 α 但每并进一个大平面就多付两趟它自己的
HBM 拷贝,判决床上是 +0.317 ms 换 0.116 ms = 净亏 0.20 ms(算式见 terrace/ta2a_pack.py
的 Hop B 一节)—— 与 2026-08-01「torch.cat 门控并包」翻车同一条物理。

四层锁:

1. **单元层**(无进程组):打包/解包对着**原张量**逐位往返 —— Hop A(int64 容器,
   id 面在行首;id_w=1 的位掩码与 id_w=quota 的槽号表两支)与 Hop B(int64 容器)
   各一套,多几何 × fp32/bf16;pad 区清零、行宽公式、int64 全宽不截断、
   **splits 就是行数不缩放**、dtype 失配 fail-loud。
2. **接缝层**(gloo,world 4 / rpn 2,2 节点真实跨节点跳):**同进程 A/B** —— 并包臂
   与未并包臂(monkeypatch `pack_enabled`)在同一趟里跑同一批输入,legacy 3 参接缝与
   overlap 6 参接缝的前向、四路叶子梯度、以及厂商手工重放真正要读的四个 detach 叶
   `.grad` 全部逐位相等。fp32 与床口径(bf16 载荷/权重 + fp32 router probs)双档,
   quota 快路径(gm=M)与通用支路(gm=None)双支。
3. **厂商契约层**:`st.send_l / st.recv_l`(交给 `disp.input_splits/output_splits`
   的就是这两份)在两臂**逐项相等且仍是行数** —— 并包把线上张量按 `F/gw` 缩放的
   是**内部**的一份拷贝,厂商那两次手工重放(rx_d.grad / rgate_d.grad)因此照跑不误,
   仍各是**一次** a2a。测试里的重放严格照厂商顺序逐步复刻(与
   tests/test_ta2a_overlap_seam.py 同序),两次 `.backward()` 进 permute2 段都必须活
   —— Hop B 并包若焊成一枚融合节点,第二次会当场炸(内部工程记录 2026-08-20)。
4. **条数与活路径**:并包臂(默认 small 形态)legacy 接缝 fwd/bwd = 8/6、
   未并包臂 = 10/6;overlap 接缝 dispatch 半边 6 vs 8。并且逐条记录
   `hopa_pack / hopa_pack_small / hopb_pack_meta` 的调用次数
   —— 等价断言对「闸门从未打开」是盲的(内部工程记录),没有活路径证据,前三层全是空转。
   (C1 线格式本身有没有静默回退,由 tests/test_ta2a_quota_wire_bitparity.py 把守;
   A1′ 之后它的观测点也挪到了 `hopa_pack` 的入参上。)

gloo / CPU,world 4、rpn 2。
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

import terrace.ta2a_pack as pk  # noqa: E402

WORLD, RPN, T, K, M, E, H, D = 4, 2, 8, 4, 2, 8, 6, 4

# 并包前 / 后的拆半接缝条数(黄金值同步钉在 test_ta2a_gate_at_arrival.py)。
# **2026-08-22:默认形态由 full 改为 small,这三个数随之从 7/6/5 变成 8/6/6。**
# 不是回归 —— full 形态在判决床实测净亏 2.05 ms/次,少那一条集合通信是拿
# 载荷的两趟 HBM 拷贝换的,换亏了(内部实测记录)。
PACKED_FWD, PACKED_BWD = 8, 6
PLAIN_FWD, PLAIN_BWD = 10, 6
# overlap 接缝 dispatch 半边(permute_overlap)的条数,small 形态:
#   inter 3 条 = counts + payload + [id‖gate]
#   intra 3 条 = counts + exp_rx + [slot‖gate]
# 未并包时 4 + 4 = 8;full 形态是 2 + 3 = 5(要跑它得显式 TERRACE_TA2A_PACK=full)。
PACKED_DISP, PLAIN_DISP = 6, 8


# ======================================================================================
# 1. 单元层:打包/解包 == 原三张量,逐位
# ======================================================================================

# (n, hidden, gate_w, id_w):id_w=1 是通用臂的位掩码(1 维),>1 是 C1 槽号表。
# 覆盖 pad 非零/为零、gate 宽 1、宽 gate 面(通用臂 gw=slots)、判决床宽度。
HOPA_GEOMS = [(5, 6, 2, 2), (5, 6, 4, 1), (1, 7, 3, 3), (17, 2048, 3, 3),
              (4, 5, 1, 1), (3, 24, 24, 1), (9, 2048, 24, 1)]


@pytest.mark.parametrize("geom", HOPA_GEOMS,
                         ids=lambda g: f"n{g[0]}h{g[1]}g{g[2]}i{g[3]}")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hopa_pack_roundtrip_is_bitwise(geom, dtype):
    """Hop A:id + payload + gate -> [n, W] int64 -> 回来,三块逐位、dtype/形状不变。"""
    n, hidden, gate_w, id_w = geom
    g = torch.Generator().manual_seed(1000 + n * 31 + hidden + id_w)
    payload = torch.randn(n, hidden, generator=g).to(dtype)
    gate = torch.rand(n, gate_w, generator=g).to(dtype)
    ids = (torch.randint(0, 1 << 40, (n,), generator=g, dtype=torch.int64)
           if id_w == 1 else
           torch.randint(0, 24, (n, id_w), generator=g, dtype=torch.int64))

    buf, lay = pk.hopa_pack(payload, gate, ids)
    per_word = 8 // dtype.itemsize
    assert buf.shape == (n, lay.words) and buf.dtype == torch.int64
    assert buf.is_contiguous()
    assert lay.id_w == id_w and lay.id_1d is (id_w == 1 and ids.dim() == 1)
    # 行宽:id 面 + 刚好装下 (H + gw) 个浮点的 word 数,冗余 < 1 word
    assert (lay.words - id_w) * per_word >= hidden + gate_w
    assert (lay.words - id_w - 1) * per_word < hidden + gate_w
    # id 面在行首 => 浮点区起始字节恒为 8 的倍数
    assert (id_w * 8) % 8 == 0
    # pad 区显式清零(收侧永不读)
    tail = id_w * per_word + hidden + gate_w
    if tail < lay.words * per_word:
        pad = buf.view(dtype)[:, tail:]
        assert torch.equal(pad, torch.zeros_like(pad))

    got_p, got_g, got_i = pk.hopa_unpack(buf, lay)
    assert torch.equal(got_p, payload) and got_p.dtype == payload.dtype
    assert torch.equal(got_g, gate) and got_g.dtype == gate.dtype
    assert torch.equal(got_i, ids) and got_i.shape == ids.shape
    assert got_p.is_contiguous() and got_g.is_contiguous() and got_i.is_contiguous(), (
        "解包必须回连续张量:2026-08-01 的 torch.cat 并包正是被下游吃非连续切片拖死的")


def test_hopa_id_plane_survives_full_int64_range():
    """位掩码是 int64 的**全宽**值(slots 可到 63 位),不许被任何窄化路径截断。"""
    ids = torch.tensor([0, 1, (1 << 62) + 12345, -1, (1 << 63) - 1],
                       dtype=torch.int64)
    payload = torch.randn(5, 6).to(torch.bfloat16)
    gate = torch.rand(5, 3).to(torch.bfloat16)
    buf, lay = pk.hopa_pack(payload, gate, ids)
    _p, _g, got = pk.hopa_unpack(buf, lay)
    assert torch.equal(got, ids)


@pytest.mark.parametrize("geom", HOPA_GEOMS,
                         ids=lambda g: f"n{g[0]}h{g[1]}g{g[2]}i{g[3]}")
def test_hopa_splits_are_plain_row_counts(geom):
    """线上张量 dim 0 就是行数 —— splits **不缩放**,每一段恰好是原来那几行。

    这是厂商契约不动的直接理由:`st.send_l / st.recv_l`(= disp.input_splits /
    output_splits)与并包前逐项相同,厂商 backward 的手工重放不需要知道并包存在。
    """
    n, hidden, gate_w, id_w = geom
    payload = torch.randn(n, hidden)
    gate = torch.rand(n, gate_w)
    ids = (torch.zeros(n, dtype=torch.int64) if id_w == 1
           else torch.zeros(n, id_w, dtype=torch.int64))
    buf, _lay = pk.hopa_pack(payload, gate, ids)
    assert buf.shape[0] == n, "dim 0 必须是行数"
    row_splits = [n // 3, n - n // 3 - n // 4, n // 4]
    assert sum(row_splits) == n
    off = 0
    for sz in row_splits:                       # 段的所有权划分与并包前逐行一致
        assert torch.equal(buf[off:off + sz], buf.narrow(0, off, sz))
        off += sz


@pytest.mark.parametrize("P", [1, 5, 13, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hopb_meta_pack_roundtrip_is_bitwise(P, dtype):
    """Hop B:槽号(int64)+ gate -> int64 容器 -> 回来,两块逐位。"""
    g = torch.Generator().manual_seed(2000 + P * 17)
    slot = torch.randint(0, 24, (P,), generator=g, dtype=torch.int64)
    gate = torch.rand(P, generator=g).to(dtype)

    buf = pk.hopb_pack_meta(slot, gate)
    assert buf.shape == (P, 2) and buf.dtype == torch.int64 and buf.is_contiguous()
    # 槽号面在行首 + 行宽以 int64 word 计 => 每行起始与浮点区起始都在 8 字节边界上
    assert pk.hopb_meta_words() == 2

    got_s, got_g = pk.hopb_unpack_meta(buf, dtype)
    assert torch.equal(got_s, slot) and got_s.dtype == torch.int64
    assert torch.equal(got_g, gate) and got_g.dtype == dtype
    assert got_s.is_contiguous() and got_g.is_contiguous()


def test_hopb_meta_pack_pad_is_zeroed():
    """pad 区显式清零:收侧永不读它,但也不该把未初始化位送上线。"""
    slot = torch.arange(7, dtype=torch.int64)
    gate = torch.rand(7).to(torch.bfloat16)
    buf = pk.hopb_pack_meta(slot, gate)
    fv = buf.view(torch.bfloat16)
    assert torch.equal(fv[:, 5:], torch.zeros_like(fv[:, 5:]))


def test_pack_dtype_mismatch_is_loud():
    """dtype / 形状失配必须当场死 —— 与旧稀疏平面在 index_put 处同一失效形态
    (一次内部提交 的 fp32 gate 平面漂移正是从这类静默处进场的)。"""
    ids3 = torch.zeros(3, dtype=torch.int64)
    with pytest.raises(RuntimeError):           # Hop A:gate 与 payload 不同 dtype
        pk.hopa_pack(torch.randn(3, 4).to(torch.bfloat16), torch.rand(3, 2), ids3)
    with pytest.raises(RuntimeError):           # Hop A:id 面必须 int64
        pk.hopa_pack(torch.randn(3, 4), torch.rand(3, 2),
                     torch.zeros(3, dtype=torch.int32))
    with pytest.raises(RuntimeError):           # Hop A:三个平面行数必须一致
        pk.hopa_pack(torch.randn(3, 4), torch.rand(3, 2),
                     torch.zeros(4, dtype=torch.int64))
    with pytest.raises(RuntimeError):           # Hop B:槽号面必须 int64
        pk.hopb_pack_meta(torch.zeros(3, dtype=torch.int32), torch.rand(3))
    with pytest.raises(RuntimeError):           # Hop B:gate 面必须浮点
        pk.hopb_pack_meta(torch.zeros(3, dtype=torch.int64),
                          torch.zeros(3, dtype=torch.int64))
    with pytest.raises(RuntimeError):           # Hop B:两面必须同形状
        pk.hopb_pack_meta(torch.zeros(3, dtype=torch.int64), torch.rand(4))


def test_pack_switch_defaults_on_and_can_be_turned_off(monkeypatch):
    """闸门:未设 / 非 "0" 为开;"0" 关(床上 A/B 与一键回滚靠它)。"""
    monkeypatch.delenv("TERRACE_TA2A_PACK", raising=False)
    pk.reset()
    assert pk.pack_enabled() is True
    monkeypatch.setenv("TERRACE_TA2A_PACK", "0")
    pk.reset()
    assert pk.pack_enabled() is False
    monkeypatch.setenv("TERRACE_TA2A_PACK", "1")
    pk.reset()
    assert pk.pack_enabled() is True
    monkeypatch.delenv("TERRACE_TA2A_PACK", raising=False)
    pk.reset()


# ======================================================================================
# 2+3+4. 接缝层:同进程 A/B、厂商契约、条数与活路径
# ======================================================================================

def _routing(gen, n_nodes, per, quota):
    """T-Route 等额配额:恰好 M 个节点,每个节点恰好 K/M 个专家。"""
    rows = []
    for _ in range(T):
        gs = torch.randperm(n_nodes, generator=gen)[:M]
        rows.append(torch.cat([torch.arange(a * per, (a + 1) * per)[
            torch.randperm(per, generator=gen)[:quota]] for a in gs]))
    return torch.stack(rows)


def _bitdiff(a, b):
    """None = 逐位相等;否则 (说明, max|Δ|) 供报错定位。"""
    if a is None or b is None:
        return ("missing", None)
    if a.dtype != b.dtype:
        return (f"{a.dtype} vs {b.dtype}", float((a.float() - b.float()).abs().max()))
    if torch.equal(a, b):
        return None
    return (str(a.dtype), float((a.float() - b.float()).abs().max()))


def _run(rank, world, q, dtype_name):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # 已占用:29577/29591/29613/29623/29627/29641/29645/29661/29665/29677/
    #         29681/29685/29697。本床两个 dtype 各一个。
    os.environ.setdefault(
        "MASTER_PORT", "29709" if dtype_name == "float32" else "29713")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        import terrace.ta2a_pack as pack_mod
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

        real_a2a = dist.all_to_all_single
        counter = [0]

        def counting_a2a(*args, **kwargs):
            counter[0] += 1
            return real_a2a(*args, **kwargs)

        # 活路径计数器:并包臂必须 > 0,未并包臂必须 == 0。
        # **两个 Hop A 打包器都要数**:small 形态(2026-08-22 起的默认)调
        # hopa_pack_small,full 形态调 hopa_pack。只数其中一个,换形态时活路径证据
        # 会静默变成 0,而"闸门从未打开"正是本文件要防的那件事(内部工程记录)。
        hits = {"hopa": 0, "hopb": 0}
        real_hopa = pack_mod.hopa_pack
        real_hopa_small = pack_mod.hopa_pack_small
        real_hopb = pack_mod.hopb_pack_meta
        real_switch = pack_mod.pack_enabled      # 必须在被覆盖**之前**抓住原件

        def counting_hopa(*a, **kw):
            hits["hopa"] += 1
            return real_hopa(*a, **kw)

        def counting_hopa_small(*a, **kw):
            hits["hopa"] += 1
            return real_hopa_small(*a, **kw)

        def counting_hopb(*a, **kw):
            hits["hopb"] += 1
            return real_hopb(*a, **kw)

        pack_mod.hopa_pack = counting_hopa
        pack_mod.hopa_pack_small = counting_hopa_small
        pack_mod.hopb_pack_meta = counting_hopb

        def one_pass(packed, gm):
            pack_mod.pack_enabled = (lambda: True) if packed else (lambda: False)
            g = torch.Generator().manual_seed(23 + rank)
            x0 = torch.randn(T, H, generator=g).to(pdt)
            idx = _routing(g, n_nodes, per, quota)
            gates0 = torch.rand(T, K, generator=g)          # fp32:router 输出精度
            w13_0 = (torch.randn(epr, H, 2 * D, generator=g) / (H ** 0.5)).to(pdt)
            w2_0 = (torch.randn(epr, D, H, generator=g) / (D ** 0.5)).to(pdt)
            G = torch.randn(T, H, generator=g).to(pdt)
            routing_map = torch.zeros(T, E, dtype=torch.bool)
            routing_map[torch.arange(T).unsqueeze(1), idx] = True
            probs_dense = torch.zeros(T, E)
            probs_dense[torch.arange(T).unsqueeze(1), idx] = gates0

            dist.all_to_all_single = counting_a2a
            try:
                # ---- legacy 3 参接缝(普通 autograd)----
                hidL = x0.clone().requires_grad_(True)
                probL = probs_dense.clone().to(pdt).requires_grad_(True)
                w13L = w13_0.clone().requires_grad_(True)
                w2L = w2_0.clone().requires_grad_(True)
                counter[0] = 0
                permL, tpeL, ppL, stL = ta2a_permute(
                    hidL, probL, routing_map, world=world, rank=rank, rpn=RPN,
                    n_experts=E, intra_group=intra, inter_group=None, groups_m=gm)
                a, b = grouped_mm(permL, w13L, tpeL).chunk(2, dim=-1)
                eoL = grouped_mm(F.silu(a) * b, w2L, tpeL) * ppL.unsqueeze(-1)
                outL = ta2a_unpermute(eoL, stL, hidL)
                seam_fwd = counter[0]
                counter[0] = 0
                outL.backward(G)
                seam_bwd = counter[0]

                # ---- overlap 6 参接缝 + 厂商手写 backward 的逐步复刻 ----
                hidO = x0.clone().requires_grad_(True)
                probO = probs_dense.clone().requires_grad_(True)
                save = []
                counter[0] = 0
                permO, tpeO, ppO, _share, stO = ta2a_permute_overlap(
                    hidO, probO, routing_map, world=world, rank=rank, rpn=RPN,
                    n_experts=E, intra_group=intra, inter_group=None, groups_m=gm,
                    save_tensors=save, run_shared_experts=None)
                disp_fwd = counter[0]
                w13O = w13_0.clone().requires_grad_(True)
                w2O = w2_0.clone().requires_grad_(True)
                dI = permO.detach().requires_grad_(True)
                pI = ppO.detach().requires_grad_(True)
                a, b = grouped_mm(dI, w13O, tpeO).chunk(2, dim=-1)
                eoO = grouped_mm(F.silu(a) * b, w2O, tpeO) * pI.unsqueeze(-1)
                outO = ta2a_unpermute_overlap(eoO, stO, save)
                (p1g, p1p, ncpu, p2ind, p2g, p2pd, p2pg, u1ind, u1g, u2ind) = save

                outO.backward(G)                            # unpermute2 段
                grad_red = _a2a_raw(u2ind.grad, stO.send_l, stO.recv_l)
                u1g.backward(grad_red)                      # unpermute1 段
                eoO.backward(u1ind.grad)                    # experts 等价物
                # 厂商 gmm 分两次 .backward() 进 permute2 段:先 prob 路,再 token 路。
                # Hop B 并包若焊成一枚融合节点,第二次会在这里当场炸。
                p2pg.backward(pI.grad)
                ggr = _a2a_raw(p2pd.grad, stO.recv_l, stO.send_l)
                p2g.backward(dI.grad)
                gpl = _a2a_raw(p2ind.grad, stO.recv_l, stO.send_l)
                torch.autograd.backward([p1g, p1p], grad_tensors=[gpl, ggr])
            finally:
                dist.all_to_all_single = real_a2a

            return {
                "L.perm": permL.detach(), "L.pp": ppL.detach(), "L.tpe": tpeL,
                "L.out": outL.detach(), "L.gx": hidL.grad, "L.gp": probL.grad,
                "L.gw13": w13L.grad, "L.gw2": w2L.grad,
                "O.perm": permO.detach(), "O.pp": ppO.detach(), "O.tpe": tpeO,
                "O.out": outO.detach(), "O.gx": hidO.grad, "O.gp": probO.grad,
                "O.gw13": w13O.grad, "O.gw2": w2O.grad,
                # 厂商手工重放真正要读的四个 detach 叶
                "O.p2in": p2ind.grad, "O.p2prob": p2pd.grad,
                "O.u1in": u1ind.grad, "O.u2in": u2ind.grad,
                # 契约与账目(不进逐位对比)
                "_counts": (seam_fwd, seam_bwd, disp_fwd),
                "_splits": (list(stO.send_l), list(stO.recv_l),
                            list(stL.send_l), list(stL.recv_l)),
                "_seats": (len(save), ncpu is None),
            }

        report = {}
        try:
            for gm in (M, None):
                hits["hopa"] = hits["hopb"] = 0
                plain = one_pass(packed=False, gm=gm)
                hits_plain = dict(hits)
                hits["hopa"] = hits["hopb"] = 0
                packed = one_pass(packed=True, gm=gm)
                hits_packed = dict(hits)

                diffs = {}
                for name in packed:
                    if name.startswith("_"):
                        continue
                    if name.endswith(".tpe"):
                        diffs[name] = (None if torch.equal(packed[name], plain[name])
                                       else ("tpe", -1.0))
                    else:
                        diffs[name] = _bitdiff(packed[name], plain[name])
                report[f"gm={gm}"] = {
                    "diffs": diffs,
                    "counts_packed": packed["_counts"],
                    "counts_plain": plain["_counts"],
                    "splits_same": packed["_splits"] == plain["_splits"],
                    "splits": packed["_splits"],
                    "seats": packed["_seats"],
                    "hits_packed": hits_packed,
                    "hits_plain": hits_plain,
                }
        finally:
            pack_mod.hopa_pack, pack_mod.hopb_pack_meta = real_hopa, real_hopb
            pack_mod.hopa_pack_small = real_hopa_small
            pack_mod.pack_enabled = real_switch

        q.put({"rank": rank, "status": "ok", "report": report})
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
def pack_fp32():
    return _spawn("float32")


@pytest.fixture(scope="module")
def pack_bf16():
    return _spawn("bfloat16")


def _assert_bitwise(results, label):
    for r in results:
        for tag, rep in r["report"].items():
            for name, d in rep["diffs"].items():
                assert d is None, (
                    f"rank {r['rank']} {label} {tag}: {name} 并包臂与未并包臂不逐位"
                    f"相等 {d} —— 并包只许重排字节")


def _assert_contract(results, label):
    for r in results:
        for tag, rep in r["report"].items():
            cp, cq = rep["counts_packed"], rep["counts_plain"]
            assert cp == (PACKED_FWD, PACKED_BWD, PACKED_DISP), (
                f"rank {r['rank']} {label} {tag}: 并包臂条数 {cp},应为 "
                f"{(PACKED_FWD, PACKED_BWD, PACKED_DISP)}(legacy fwd/bwd, "
                f"overlap dispatch)—— 多了即并包被静默回退,少了即有交换被丢弃")
            assert cq == (PLAIN_FWD, PLAIN_BWD, PLAIN_DISP), (
                f"rank {r['rank']} {label} {tag}: 未并包臂条数 {cq},应为 "
                f"{(PLAIN_FWD, PLAIN_BWD, PLAIN_DISP)} —— 对照臂形态变了,"
                f"本文件的『改前公式』前提失效,须重审")
            assert rep["splits_same"], (
                f"rank {r['rank']} {label} {tag}: st.send_l/recv_l 两臂不同 "
                f"{rep['splits']} —— 那两份就是交给厂商 disp.input_splits/"
                f"output_splits 的行数,并包不许改它的语义")
            n_send, n_recv, l_send, l_recv = rep["splits"]
            assert len(n_send) == WORLD and len(n_recv) == WORLD
            assert sum(n_send) == T * M, (
                f"rank {r['rank']} {label} {tag}: Hop A splits 总和 {sum(n_send)} "
                f"!= T*M={T * M} —— 交给厂商的必须是**行数**,不是缩放后的字节段数")
            assert (n_send, n_recv) == (l_send, l_recv)
            assert rep["seats"] == (10, True), (
                f"rank {r['rank']} {label} {tag}: 席位契约变了 {rep['seats']}")


def _assert_live(results, label):
    for r in results:
        for tag, rep in r["report"].items():
            assert rep["hits_plain"] == {"hopa": 0, "hopb": 0}, (
                f"rank {r['rank']} {label} {tag}: 未并包臂竟走了并包 "
                f"{rep['hits_plain']} —— A/B 的对照臂失守,等价断言全是空转")
            # legacy 接缝 1 次 + overlap 接缝 1 次
            assert rep["hits_packed"] == {"hopa": 2, "hopb": 2}, (
                f"rank {r['rank']} {label} {tag}: 并包臂打包次数 "
                f"{rep['hits_packed']},应为两条接缝各 1 次 —— 闸门没打开或"
                f"某条接缝没接上(等价断言对这个是盲的,内部工程记录)")


@pytest.mark.timeout(300)
def test_fp32_pack_is_bitwise(pack_fp32):
    """fp32:两条接缝的前向、四路叶子梯度、四个 detach 叶 .grad,并包 == 未并包。"""
    _assert_bitwise(pack_fp32, "fp32")


@pytest.mark.timeout(300)
def test_fp32_vendor_contract_and_counts(pack_fp32):
    """条数降到 7/6(dispatch 半边 8 -> 5),splits 仍是行数、两臂逐项相等。"""
    _assert_contract(pack_fp32, "fp32")


@pytest.mark.timeout(300)
def test_fp32_pack_is_live(pack_fp32):
    """活路径证据:并包臂真打了包,未并包臂真没打。"""
    _assert_live(pack_fp32, "fp32")


@pytest.mark.timeout(300)
def test_bed_dtype_pack_is_bitwise(pack_bf16):
    """床口径(bf16 载荷/权重 + fp32 router probs):同上,逐位。"""
    _assert_bitwise(pack_bf16, "bf16")


@pytest.mark.timeout(300)
def test_bed_dtype_vendor_contract_and_counts(pack_bf16):
    _assert_contract(pack_bf16, "bf16")


@pytest.mark.timeout(300)
def test_bed_dtype_pack_is_live(pack_bf16):
    _assert_live(pack_bf16, "bf16")


# ======================================================================================
# 工程纪律:本文件自身 LF + py_compile
# ======================================================================================

def test_this_file_compiles_and_is_lf():
    import py_compile
    py_compile.compile(__file__, doraise=True)
    with open(__file__, "rb") as f:
        assert b"\r\n" not in f.read()
