"""K1(terrace_k1_arrival,到达侧融合链)的 CPU 侧位级契约与接入开关语义。

kernel 本体跑在集群 NPU 上(位级验证:README §3.4 的 k1 冒烟命令 +
内部基准脚本(未随仓发布));本文件把守的是**本地能证的全部**:

  1. 可执行规格:terrace.ops.k1_arrival_ref(纯 torch,kernel 语义的逐位镜像)
     对着现组合链**原文**(_expand_arrival_quota + owner 稳定桶排 + bincount +
     两处 gather,逐行照抄)逐位对账 —— 多几何 x 多 dtype,含退化输入
     (C1 zeros 收容行、R=0);反向(组合链伴随)同样逐位。
  2. 降级路径:无 .so 时 k1_arrival 包装器走参考实现,结果与现链逐位不变;
     TerraceK1ArrivalFn.backward 的公式(kernel 路径将来真正跑的反向)对着
     autograd 的组合链反向逐位对账。
  3. 开关语义:k1_arrival 列入 _REQUIRED_OPS(旧 .so 整体降级,不半注册);
     "0" 显式关、"require" 无 .so 必炸 —— 与 passthrough 同一套契约。
  4. 接线证明(gloo,world 4 / rpn 2):把 terrace.ops.k1_arrival 换成计数的
     参考实现、强制打开闸门,融合前向与 legacy 3 参接缝的前向 + 全部梯度必须与
     未接 K1 的现链逐位相等,且假 kernel 的调用计数 > 0(活路径证据 ——
     等价断言对"闸门从未打开"是盲的,内部工程记录 的静默失效纪律)。

为什么参考实现足以在 CPU 上代表 kernel:两者是同一个数学对象(稳定计数排序),
两遍法(计数 -> 前缀游标 -> 展开写行)与稳定升序 argsort 的逐位一致性论证见
terrace/ops/ascendc/op_kernel/terrace_k1_arrival.cpp 文件头;CPU 单测通不出设备
行为(内部工程记录),所以 NPU 位级另有集群冒烟把守,本文件不冒充它。
"""
from __future__ import annotations

import os
import sys
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import terrace.ops as tops  # noqa: E402
from terrace.ta2a_fwd import (_expand_arrival_quota,  # noqa: E402
                              _stable_argsort_small)


@pytest.fixture()
def clean_ops(monkeypatch):
    """每例独立判定(与 test_terrace_ops_scaffold 同款):清环境、清缓存。"""
    monkeypatch.delenv("TERRACE_CUSTOM_OPS", raising=False)
    monkeypatch.delenv("TERRACE_OPS_LIB", raising=False)
    tops.reset()
    yield monkeypatch
    tops.reset()


def _chain(rx, rslot, rgate, quota, epr, rpn):
    """现组合链原文(ta2a_fwd.ta2a_moe_forward / ta2a_dispatch.ta2a_permute 的
    到达段 quota 分支,即接入点 else 分支的逐行照抄)—— K1 的功能规格。"""
    r_idx, slot_idx = _expand_arrival_quota(rslot)
    owner = slot_idx // epr
    ordo = _stable_argsort_small(owner, rpn)
    r_idx, slot_idx = r_idx[ordo], slot_idx[ordo]
    i_send = torch.bincount(owner, minlength=rpn)
    return rx[r_idx], rgate.reshape(-1)[ordo], r_idx, slot_idx, i_send


def _mk(R, quota, epr, rpn, H, dtype, seed, degenerate=False):
    """C1 线格式的到达面:每行升序不重复槽号(_pack_quota_wire 的构造),或
    degenerate= True 时全零行(zeros 收容:掉队行 = 槽 0 / gate 0)。"""
    g = torch.Generator().manual_seed(seed)
    slots = epr * rpn
    assert quota <= slots
    if degenerate:
        rslot = torch.zeros(R, quota, dtype=torch.int64)
    else:
        scores = torch.rand(R, slots, generator=g)
        rslot = torch.sort(torch.topk(scores, quota, dim=1).indices,
                           dim=1).values.to(torch.int64)
    rx = torch.randn(R, H, generator=g).to(dtype)
    rgate = torch.rand(R, quota, generator=g).to(dtype)
    return rx, rslot, rgate


NAMES = ("send_buf", "gate_pairs", "r_idx", "slot_idx", "i_send")

# 几何覆盖:quota 1/2/3/4/5、epr 1..8、rpn 1..16、slots 跨 4..63、H 非 2 幂。
GEOMS = [
    # (R, quota, epr, rpn, H)
    (16, 2, 2, 8, 64),     # 对齐床几何(slots 16)
    (64, 1, 2, 8, 32),     # quota=1:r_idx == ordo 的退化
    (33, 3, 2, 4, 48),     # 非 2 幂 R/quota
    (128, 4, 4, 8, 16),    # slots 32
    (7, 2, 8, 4, 96),      # epr>rpn
    (1, 1, 1, 1, 24),      # 最小几何
    (256, 2, 2, 16, 8),    # rpn 16
    (40, 5, 3, 4, 40),     # quota 不整除 slots 的奇数几何(slots 12)
    (48, 7, 9, 7, 8),      # slots 63:现链 int64 掩码上界几何
]


# ======================================================================================
# 1. 可执行规格:参考实现 == 现链原文,逐位(多几何 x 多 dtype >= 8 例)
# ======================================================================================

@pytest.mark.parametrize("geom", GEOMS, ids=lambda g: f"R{g[0]}q{g[1]}e{g[2]}r{g[3]}")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_ref_bitwise_equals_chain(geom, dtype):
    R, quota, epr, rpn, H = geom
    for seed in range(2):
        rx, rslot, rgate = _mk(R, quota, epr, rpn, H, dtype, 100 + seed)
        want = _chain(rx, rslot, rgate, quota, epr, rpn)
        got = tops.k1_arrival_ref(rx, rslot, rgate, quota, epr, rpn)
        for name, w, g in zip(NAMES, want, got):
            assert w.dtype == g.dtype and w.shape == g.shape, name
            assert torch.equal(w, g), f"{name} 与现链不逐位相等 (seed {seed})"


def test_ref_bitwise_on_degenerate_rows():
    """C1 zeros 收容:全零槽行(slot 0 x quota)是线上合法输入,照样逐位。"""
    R, quota, epr, rpn, H = 24, 2, 2, 8, 32
    rx, rslot, rgate = _mk(R, quota, epr, rpn, H, torch.float32, 5, degenerate=True)
    for name, w, g in zip(NAMES, _chain(rx, rslot, rgate, quota, epr, rpn),
                          tops.k1_arrival_ref(rx, rslot, rgate, quota, epr, rpn)):
        assert torch.equal(w, g), name


def test_ref_bitwise_on_empty_arrival():
    """R=0(某 rank 一行未收到):五个输出空/零,与现链同形同值。"""
    quota, epr, rpn, H = 2, 2, 8, 16
    rx = torch.zeros(0, H)
    rslot = torch.zeros(0, quota, dtype=torch.int64)
    rgate = torch.zeros(0, quota)
    want = _chain(rx, rslot, rgate, quota, epr, rpn)
    got = tops.k1_arrival_ref(rx, rslot, rgate, quota, epr, rpn)
    for name, w, g in zip(NAMES, want, got):
        assert torch.equal(w, g), name
    assert got[4].shape == (rpn,) and int(got[4].sum()) == 0


def test_ref_backward_bitwise_equals_chain_autograd():
    """参考实现的反向(gather 伴随)与现链 autograd 逐位相等。"""
    R, quota, epr, rpn, H = 12, 2, 2, 4, 8
    rx0, rslot, rgate0 = _mk(R, quota, epr, rpn, H, torch.float32, 9)
    g = torch.Generator().manual_seed(1)
    gs = torch.randn(R * quota, H, generator=g)
    gg = torch.randn(R * quota, generator=g)

    rx1, rg1 = rx0.clone().requires_grad_(True), rgate0.clone().requires_grad_(True)
    s1, p1, *_ = _chain(rx1, rslot, rg1, quota, epr, rpn)
    ((s1 * gs).sum() + (p1 * gg).sum()).backward()
    rx2, rg2 = rx0.clone().requires_grad_(True), rgate0.clone().requires_grad_(True)
    s2, p2, *_ = tops.k1_arrival_ref(rx2, rslot, rg2, quota, epr, rpn)
    ((s2 * gs).sum() + (p2 * gg).sum()).backward()
    assert torch.equal(rx1.grad, rx2.grad)
    assert torch.equal(rg1.grad, rg2.grad)


# ======================================================================================
# 2. 降级路径:无 .so 时包装器 == 现链;kernel 路径的 backward 公式逐位
# ======================================================================================

def test_wrapper_falls_back_bitwise_without_so(clean_ops):
    assert tops.custom_ops_enabled() is False, "本地无 CANN 前提失效?"
    R, quota, epr, rpn, H = 16, 2, 2, 8, 64
    rx, rslot, rgate = _mk(R, quota, epr, rpn, H, torch.bfloat16, 3)
    want = _chain(rx, rslot, rgate, quota, epr, rpn)
    got = tops.k1_arrival(rx, rslot, rgate, quota, epr, rpn, my_local=1)
    for name, w, g in zip(NAMES, want, got):
        assert torch.equal(w, g), f"降级路径 {name} 与现链不逐位相等"


def test_fn_backward_formula_bitwise_equals_chain_autograd():
    """TerraceK1ArrivalFn.backward 的公式(kernel 路径在集群上真正跑的反向,
    ordo 重算 + 两处 index_add)对着现链 autograd 逐位对账 —— 不需要 .so:
    公式是纯组合链,拿一个手搓 ctx 直接调静态方法。"""
    R, quota, epr, rpn, H = 20, 3, 2, 4, 16
    rx0, rslot, rgate0 = _mk(R, quota, epr, rpn, H, torch.float32, 21)
    g = torch.Generator().manual_seed(2)
    gs = torch.randn(R * quota, H, generator=g)
    gg = torch.randn(R * quota, generator=g)

    rx1, rg1 = rx0.clone().requires_grad_(True), rgate0.clone().requires_grad_(True)
    s1, p1, r_idx, _, _ = _chain(rx1, rslot, rg1, quota, epr, rpn)
    ((s1 * gs).sum() + (p1 * gg).sum()).backward()

    ctx = types.SimpleNamespace(
        saved_tensors=(rslot, r_idx),
        k1_geom=(quota, epr, rpn, rx0.shape, rgate0.shape),
        needs_input_grad=(True, False, True, False, False, False, False))
    grad_rx, _, grad_rgate, *_ = tops.TerraceK1ArrivalFn.backward(ctx, gs, gg,
                                                                  None, None, None)
    assert torch.equal(grad_rx, rx1.grad)
    assert torch.equal(grad_rgate, rg1.grad)


# ======================================================================================
# 3. 开关语义(k1 特有件;通用契约归 test_terrace_ops_scaffold)
# ======================================================================================

def test_k1_is_in_required_ops():
    """旧的仅含 passthrough 的 .so 必须整体降级 —— 半新半旧的算子集比慢更糟。"""
    assert "k1_arrival" in tops._REQUIRED_OPS


def test_switch_off_uses_ref_without_load_attempt(clean_ops):
    clean_ops.setenv("TERRACE_CUSTOM_OPS", "0")
    rx, rslot, rgate = _mk(8, 2, 2, 4, 16, torch.float32, 11)
    got = tops.k1_arrival(rx, rslot, rgate, 2, 2, 4)
    want = _chain(rx, rslot, rgate, 2, 2, 4)
    for w, g in zip(want, got):
        assert torch.equal(w, g)
    assert tops.status().requested == "0"


def test_switch_require_fails_hard_on_k1_call(clean_ops):
    clean_ops.setenv("TERRACE_CUSTOM_OPS", "require")
    rx, rslot, rgate = _mk(8, 2, 2, 4, 16, torch.float32, 12)
    with pytest.raises(RuntimeError, match="TERRACE_CUSTOM_OPS"):
        tops.k1_arrival(rx, rslot, rgate, 2, 2, 4)


# ======================================================================================
# 4. 接线证明(gloo 分布层):强制 kernel 路径 == 现链,且假 kernel 确实被调用
# ======================================================================================

WORLD, RPN, T, K, M, E, H, D = 4, 2, 8, 4, 2, 8, 6, 4


def _routing(gen, n_nodes, per, quota):
    rows = []
    for _ in range(T):
        gs = torch.randperm(n_nodes, generator=gen)[:M]
        rows.append(torch.cat([torch.arange(a * per, (a + 1) * per)[
            torch.randperm(per, generator=gen)[:quota]] for a in gs]))
    return torch.stack(rows)


def _run(rank, world, q):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # 已占用:29577/29591/29613/29623/29627/29641/29645/29661/29665。
    os.environ.setdefault("MASTER_PORT", "29677")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        import terrace.ops as ops_mod
        from terrace.layer import grouped_mm
        from terrace.ta2a_fwd import ta2a_moe_forward, init_ta2a_groups
        from terrace.ta2a_dispatch import ta2a_permute, ta2a_unpermute

        intra = init_ta2a_groups(world, RPN)
        epr = E // world
        n_nodes, per, quota = world // RPN, E // (world // RPN), K // M

        calls = [0]
        real_enabled, real_k1 = ops_mod.custom_ops_enabled, ops_mod.k1_arrival

        def fake_k1(rx, rslot, rgate, quota_, epr_, rpn_, my_local=0):
            # 接线断言:接入点交给 kernel 的几何必须自洽(参数错位在等价断言
            # 里可能显现为对的 —— 比如 epr/rpn 同值几何 —— 这里直接钉死)。
            assert rx.dim() == 2 and rslot.shape == rgate.shape
            assert rslot.shape[1] == quota_ == quota
            assert epr_ == epr and rpn_ == RPN
            assert my_local == rank % RPN
            assert rslot.dtype == torch.int64 and rgate.dtype == rx.dtype
            calls[0] += 1
            return ops_mod.k1_arrival_ref(rx, rslot, rgate, quota_, epr_, rpn_,
                                          my_local)

        def one_pass(forced):
            if forced:
                ops_mod.custom_ops_enabled = lambda: True
                ops_mod.k1_arrival = fake_k1
            try:
                g = torch.Generator().manual_seed(31 + rank)
                x0 = torch.randn(T, H, generator=g)
                idx = _routing(g, n_nodes, per, quota)
                gates0 = torch.rand(T, K, generator=g)
                w13_0 = torch.randn(epr, H, 2 * D, generator=g) / (H ** 0.5)
                w2_0 = torch.randn(epr, D, H, generator=g) / (D ** 0.5)
                G = torch.randn(T, H, generator=g)
                routing_map = torch.zeros(T, E, dtype=torch.bool)
                routing_map[torch.arange(T).unsqueeze(1), idx] = True
                probs_dense = torch.zeros(T, E)
                probs_dense[torch.arange(T).unsqueeze(1), idx] = gates0

                # 入口 1:融合前向(ta2a_fwd 的接入点)
                xF = x0.clone().requires_grad_(True)
                gF = gates0.clone().requires_grad_(True)
                w13F = w13_0.clone().requires_grad_(True)
                w2F = w2_0.clone().requires_grad_(True)
                yF = ta2a_moe_forward(xF, idx, gF, w13F, w2F, world, E, RPN,
                                      groups_m=M)
                yF.backward(G)

                # 入口 2:legacy 3 参接缝(ta2a_dispatch 的接入点)
                hidL = x0.clone().requires_grad_(True)
                probL = probs_dense.clone().requires_grad_(True)
                w13L = w13_0.clone().requires_grad_(True)
                w2L = w2_0.clone().requires_grad_(True)
                permL, tpeL, ppL, stL = ta2a_permute(
                    hidL, probL, routing_map, world=world, rank=rank, rpn=RPN,
                    n_experts=E, intra_group=intra, inter_group=None, groups_m=M)
                a, b = grouped_mm(permL, w13L, tpeL).chunk(2, dim=-1)
                eoL = grouped_mm(F.silu(a) * b, w2L, tpeL) * ppL.unsqueeze(-1)
                outL = ta2a_unpermute(eoL, stL, hidL)
                outL.backward(G)

                return {"yF": yF.detach(), "xF": xF.grad, "gF": gF.grad,
                        "w13F": w13F.grad, "w2F": w2F.grad,
                        "yL": outL.detach(), "xL": hidL.grad, "pL": probL.grad,
                        "w13L": w13L.grad, "w2L": w2L.grad,
                        "tpe": tpeL, "perm": permL.detach(), "pp": ppL.detach()}
            finally:
                ops_mod.custom_ops_enabled = real_enabled
                ops_mod.k1_arrival = real_k1

        base = one_pass(forced=False)
        calls_baseline = calls[0]           # 必须仍为 0:未强制时假 kernel 不可达
        forced = one_pass(forced=True)
        diffs = {}
        for name in base:
            same = torch.equal(base[name], forced[name])
            diffs[name] = None if same else str(base[name].dtype)
        q.put({"rank": rank, "status": "ok", "diffs": diffs,
               "calls": calls[0], "calls_baseline": calls_baseline})
    except Exception:                                      # noqa: BLE001
        import traceback
        q.put({"rank": rank, "status": "err", "trace": traceback.format_exc()})
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def k1_wiring():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_run, args=(r, WORLD, q)) for r in range(WORLD)]
    for p in procs:
        p.start()
    out = [q.get(timeout=240) for _ in range(WORLD)]
    for p in procs:
        p.join(timeout=60)
    for r in out:
        assert r["status"] == "ok", f"rank {r['rank']}:\n{r.get('trace')}"
    return out


@pytest.mark.timeout(300)
def test_k1_forced_path_bitwise_equals_chain(k1_wiring):
    """强制 kernel 路径(假 kernel = 参考实现)后,两个接入点的前向与全部梯度
    必须与现链逐位相等 —— 接入点的参数接线、张量交接、autograd 缝合全对。"""
    for r in k1_wiring:
        for name, d in r["diffs"].items():
            assert d is None, f"rank {r['rank']}: {name} 强制 K1 路径与现链不逐位相等 ({d})"


@pytest.mark.timeout(300)
def test_k1_forced_path_is_live(k1_wiring):
    """活路径证据:融合前向 + legacy 接缝各过一次接入点,假 kernel 恰被调 2 次;
    未强制的基线趟必须 0 次(闸门默认关死)。等价断言对『闸门从未打开』是盲的,
    这条不通过,上面那条什么都没证明(内部工程记录)。"""
    for r in k1_wiring:
        assert r["calls_baseline"] == 0, f"rank {r['rank']}: 基线趟闸门竟然开着"
        assert r["calls"] == 2, (
            f"rank {r['rank']}: 假 kernel 被调 {r['calls']} 次,预期 2"
            f"(融合前向 1 + legacy 接缝 1)—— 接入点没走 kernel 分支")


# ======================================================================================
# 工程纪律:本文件自身 LF + py_compile(ops 目录文件归 test_terrace_ops_scaffold 锁)
# ======================================================================================

def test_this_file_compiles_and_is_lf():
    import py_compile
    py_compile.compile(__file__, doraise=True)
    with open(__file__, "rb") as f:
        assert b"\r\n" not in f.read()


def test_gm_ub_gm_kernels_declare_both_vecin_and_vecout():
    """做 GM→UB→GM 的 kernel **必须同时有 VECIN 和 VECOUT 队列**。

    2026-08-24 实测出来的坑(先在 passthrough 上,再在 K1 上照同一形状发现):
    AscendC 里 `TQue` 的**位置**决定它同步哪两条流水 ——
        VECIN  的 EnQue/DeQue 配 MTE2 -> V
        VECOUT 的 EnQue/DeQue 配 V    -> MTE3
    只声明 VECIN、然后直接从 VECIN 的 LocalTensor 发 MTE3(DataCopy 到 GM),
    那道 MTE2→MTE3 的屏障**没有人插**:搬出去的可能是还没搬完的内容,
    也可能已被下一轮 AllocTensor 复用覆盖。

    症状是**静默的数据错**,不是编译错也不是崩溃 —— passthrough 上表现为
    「输出大部分是零」(8x256 错 2047/2048),而算子 build/load/execute 全绿。
    K1 的 CopyRow 当时逐字照抄了 passthrough 的错误样板,注释还写着「同款」。
    **护栏定在源码层,因为这一类错编译器不报、加载不报、只有逐位比对才现形。**
    """
    import glob
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kdir = os.path.join(root, "terrace/ops/ascendc/op_kernel")
    if not os.path.isdir(kdir):
        pytest.skip("kernel 目录不在")
    checked = 0
    for path in sorted(glob.glob(os.path.join(kdir, "*.cpp"))):
        src = open(path, encoding="utf-8").read()
        # 只管"把 UB 里的东西 DataCopy 回 GM"的 kernel;纯读入/纯标量写的不管
        writes_gm = re.search(r"DataCopy\(\s*\w*[Gg]m", src) is not None
        if not writes_gm:
            continue
        checked += 1
        name = os.path.basename(path)
        assert "QuePosition::VECIN" in src, "%s 写 GM 却没有 VECIN 队列" % name
        assert "QuePosition::VECOUT" in src, (
            "%s 有 GM→UB→GM 的搬运却**没有 VECOUT 队列** —— "
            "MTE2/MTE3 之间的屏障是队列框架按位置配对插的,少了 VECOUT 就没人插,"
            "写出去的是脏数据。这是 2026-08-24 在 passthrough 上实测到的静默错。" % name)
    assert checked >= 2, "只扫到 %d 个写 GM 的 kernel,护栏可能没扫到东西" % checked


def test_custom_ops_default_is_off_not_on():
    """**未设 TERRACE_CUSTOM_OPS 时算子必须是关的。**

    2026-08-24 的事故:默认是 "1",于是 `.so` 第一次编译成功那一刻
    (07:25,k1-rebuild),未过逐位校验的 K1 kernel 自动进入训练 dispatch 路径。
    此后每一发 T-A2A on 臂在第 0 步全 128 rank 同时炸:
        RuntimeError: Split sizes dosen't match total dim 0 size
    (K1 的 slot_idx 索引算错 -> i_send 错 -> Hop B 的 splits 对不上)
    r4 与 isub 两发判决床白烧,而 runner 还报"收工"、写了 DONE 旗。

    bitcheck 的判决行一直写着「K1 不得上床」,但**没有任何机制执行它** ——
    唯一的闸是「.so 能不能 dlopen」。编译成功 ≠ 算对了。

    这条护栏钉住的就是那个默认值:**没人签字,kernel 不上路径。**
    """
    import importlib
    saved = os.environ.pop("TERRACE_CUSTOM_OPS", None)
    try:
        importlib.reload(tops)
        assert tops._normalized_switch() == "0", (
            "TERRACE_CUSTOM_OPS 未设时归一化成了 %r —— 必须是 '0'。"
            "默认开 = 任何一次成功编译都会把未验证的 kernel 送进训练路径。"
            % tops._normalized_switch())
        # 反向对照:显式索取时必须真的开,否则这条护栏就把功能锁死了
        os.environ["TERRACE_CUSTOM_OPS"] = "1"
        assert tops._normalized_switch() == "1"
        os.environ["TERRACE_CUSTOM_OPS"] = "require"
        assert tops._normalized_switch() == "require"
    finally:
        os.environ.pop("TERRACE_CUSTOM_OPS", None)
        if saved is not None:
            os.environ["TERRACE_CUSTOM_OPS"] = saved
        importlib.reload(tops)
